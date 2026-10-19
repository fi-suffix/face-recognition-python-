import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

load_dotenv()

LARAVEL_API_URL = os.getenv("LARAVEL_API_URL", "http://localhost:8000/api")
LARAVEL_API_KEY = os.getenv("LARAVEL_API_KEY", "")

PLAYBACK_FFMPEG_PATH = os.getenv("PLAYBACK_FFMPEG_PATH", "")
PLAYBACK_WIDTH = int(os.getenv("PLAYBACK_WIDTH", "640"))
PLAYBACK_HEIGHT = int(os.getenv("PLAYBACK_HEIGHT", "360"))
PLAYBACK_FPS = int(os.getenv("PLAYBACK_FPS", "15"))
PLAYBACK_CRF = int(os.getenv("PLAYBACK_CRF", "27"))
PLAYBACK_LOOKAHEAD_MIN = int(os.getenv("PLAYBACK_LOOKAHEAD_MIN", "20"))
PLAYBACK_MAX_LOOKAHEAD_MIN = int(os.getenv("PLAYBACK_MAX_LOOKAHEAD_MIN", "60"))
PLAYBACK_HLS_SEGMENT = int(os.getenv("PLAYBACK_HLS_SEGMENT", "2"))
PLAYBACK_HLS_LIST_SIZE = int(os.getenv("PLAYBACK_HLS_LIST_SIZE", "16"))
PLAYBACK_SESSION_TTL_S = int(os.getenv("PLAYBACK_SESSION_TTL_S", "120"))
PLAYBACK_TMP_DIR = os.getenv("PLAYBACK_TMP_DIR", "")
PLAYBACK_DVR_TZ = os.getenv("PLAYBACK_DVR_TZ", "Asia/Jakarta")

DEFAULT_TIMEOUT_MS = "8000000"
READY_TIMEOUT_S = 20.0

FFMPEG_FALLBACKS = [
    r"C:\Users\padil\Downloads\ffmpeg-2026-09-07-git-ecc7eb519e-full_build\bin\ffmpeg.exe",
]

playback_router = APIRouter(prefix="/api/v1/playback", tags=["playback"])


class PlaybackSessionCreate(BaseModel):
    camera_id: int
    start_iso: str
    lookahead_min: Optional[int] = None


class PlaybackSeek(BaseModel):
    start_iso: str
    lookahead_min: Optional[int] = None


class PlaybackAvailabilityRequest(BaseModel):
    camera_id: int
    date: str


_sessions: dict = {}
_sessions_lock = threading.Lock()
_cleanup_started = False


def _ffmpeg_bin() -> Optional[str]:
    found = shutil.which("ffmpeg")
    if found:
        return found
    candidates = []
    if PLAYBACK_FFMPEG_PATH:
        candidates.append(PLAYBACK_FFMPEG_PATH)
    candidates.extend(FFMPEG_FALLBACKS)
    for cand in candidates:
        if cand and os.path.isfile(cand):
            return cand
    return None


def _tmp_root() -> Path:
    if PLAYBACK_TMP_DIR:
        return Path(PLAYBACK_TMP_DIR)
    return Path(os.environ.get("TEMP", "/tmp")) / "dvr_playback"


def _to_epoch(iso: str) -> float:
    dt = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.astimezone()
    return dt.timestamp()


def _dvr_tz() -> ZoneInfo:
    try:
        return ZoneInfo(PLAYBACK_DVR_TZ)
    except Exception:
        return ZoneInfo("Asia/Jakarta")


def _fmt_z(epoch: float) -> str:
    # DVR Hikvision membaca angka starttime/endtime sebagai JAM LOKAL-nya sendiri,
    # meskipun menerima suffix "Z". Jadi kita kirim jam lokal DVR (bukan UTC).
    return datetime.fromtimestamp(epoch, _dvr_tz()).strftime("%Y%m%dT%H%M%S") + "Z"


def _fetch_camera(camera_id: int) -> Optional[dict]:
    headers = {"Authorization": f"Bearer {LARAVEL_API_KEY}"} if LARAVEL_API_KEY else {}
    try:
        resp = requests.get(
            f"{LARAVEL_API_URL}/face-recognition/cameras/{camera_id}",
            headers=headers,
            timeout=5,
        )
    except requests.RequestException:
        return None
    if resp.status_code != 200:
        return None
    return resp.json().get("camera")


def _archive_track(live_url: str) -> Optional[str]:
    match = re.search(r"/Streaming/Channels/(\d+)", live_url)
    if not match:
        return None
    stream = match.group(1)
    if len(stream) < 3:
        return None
    channel = int(stream[:-2])
    if channel < 1:
        return None
    return f"{channel}01"


def _archive_url(live_url: str, start_epoch: float, end_epoch: float) -> str:
    track = _archive_track(live_url)
    if not track:
        raise ValueError("Kamera tidak memiliki track arsip DVR")
    _, _, rest = live_url.partition("//")
    hostport, _, pathpart = rest.partition("/")
    pathpart = "/" + pathpart
    new_path = re.sub(r"/Streaming/Channels/[^/?]+", f"/Streaming/tracks/{track}", pathpart, count=1)
    query = f"starttime={_fmt_z(start_epoch)}&endtime={_fmt_z(end_epoch)}"
    return f"rtsp://{hostport}{new_path}?{query}"


def _ffmpeg_cmd(url: str, out_dir: Path) -> list:
    return [
        _ffmpeg_bin(),
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        DEFAULT_TIMEOUT_MS,
        "-rtsp_flags",
        "prefer_tcp",
        "-i",
        url,
        "-vf",
        f"scale={PLAYBACK_WIDTH}:{PLAYBACK_HEIGHT},fps={PLAYBACK_FPS}",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(PLAYBACK_CRF),
        "-g",
        str(max(1, PLAYBACK_FPS * PLAYBACK_HLS_SEGMENT)),
        "-keyint_min",
        str(max(1, PLAYBACK_FPS * PLAYBACK_HLS_SEGMENT)),
        "-sc_threshold",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-profile:v",
        "main",
        "-an",
        "-f",
        "hls",
        "-hls_time",
        str(PLAYBACK_HLS_SEGMENT),
        "-hls_list_size",
        str(PLAYBACK_HLS_LIST_SIZE),
        "-hls_flags",
        "delete_segments+omit_endlist+independent_segments",
        "-hls_segment_filename",
        str(out_dir / "index_%03d.ts"),
        str(out_dir / "index.m3u8"),
    ]


def _spawn_session(start_epoch: float, archive_url: str) -> dict:
    sid = uuid.uuid4().hex[:16]
    sdir = _tmp_root() / sid
    sdir.mkdir(parents=True, exist_ok=True)
    errf = sdir / "ffmpeg.err.log"
    err_handle = open(errf, "w", encoding="utf-8", errors="replace")
    proc = subprocess.Popen(
        _ffmpeg_cmd(archive_url, sdir),
        stdout=subprocess.DEVNULL,
        stderr=err_handle,
    )
    err_handle.close()
    return {"sid": sid, "dir": sdir, "proc": proc}


def _stderr_tail(session: dict, lines: int = 6) -> str:
    try:
        path = session["dir"] / "ffmpeg.err.log"
        if not path.exists():
            return ""
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        return " | ".join(content.splitlines()[-lines:])
    except Exception:
        return ""


def _wait_ready(session: dict, timeout_s: float = READY_TIMEOUT_S) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if (session["dir"] / "index.m3u8").exists():
            if session["proc"].poll() is None or os.path.getsize(session["dir"] / "index.m3u8") > 0:
                return
        if session["proc"].poll() is not None:
            raise RuntimeError(_stderr_tail(session) or "ffmpeg exited sebelum menghasilkan stream")
        time.sleep(0.4)
    raise TimeoutError("DVR tidak mengirim data arsip (timeout) — coba lagi / pastikan rekaman ada")


def _terminate_all() -> None:
    with _sessions_lock:
        for sid, session in list(_sessions.items()):
            if session["proc"].poll() is None:
                try:
                    session["proc"].kill()
                except Exception:
                    pass
            _sessions.pop(sid, None)


def _terminate(sid: str) -> None:
    with _sessions_lock:
        session = _sessions.pop(sid, None)
    if not session:
        return
    if session["proc"].poll() is None:
        try:
            session["proc"].kill()
        except Exception:
            pass


def _cleanup_worker() -> None:
    while True:
        time.sleep(15)
        now = time.time()
        with _sessions_lock:
            expired = []
            for sid, session in _sessions.items():
                alive = session["proc"].poll() is None
                if not alive or (now - session["last_touch"]) > PLAYBACK_SESSION_TTL_S:
                    expired.append(sid)
            for sid in expired:
                session = _sessions.pop(sid, None)
                if session and session["proc"].poll() is None:
                    try:
                        session["proc"].kill()
                    except Exception:
                        pass
                if session:
                    shutil.rmtree(session["dir"], ignore_errors=True)


def _ensure_cleanup() -> None:
    global _cleanup_started
    if _cleanup_started:
        return
    _cleanup_started = True
    threading.Thread(target=_cleanup_worker, daemon=True).start()


def _respond_session(session: dict, camera: dict) -> dict:
    return {
        "sid": session["sid"],
        "camera_id": camera.get("id"),
        "camera_name": camera.get("name"),
        "start_ts": session["start_ts"],
        "end_ts": session["end_ts"],
        "window_min": session["window_min"],
        "width": PLAYBACK_WIDTH,
        "height": PLAYBACK_HEIGHT,
        "fps": PLAYBACK_FPS,
        "segment_s": PLAYBACK_HLS_SEGMENT,
        "ttl_s": PLAYBACK_SESSION_TTL_S,
        "stream_url": f"/api/v1/playback/session/{session['sid']}/index.m3u8",
    }


def _start(start_iso: str, camera: dict, lookahead_min: Optional[int]) -> dict:
    if not _archive_track(camera.get("rtsp_url") or ""):
        raise HTTPException(
            status_code=400,
            detail={"playable": False, "message": "Kamera tidak memiliki track arsip DVR"},
        )
    lookahead = lookahead_min or PLAYBACK_LOOKAHEAD_MIN
    lookahead = max(1, min(lookahead, PLAYBACK_MAX_LOOKAHEAD_MIN))
    start_epoch = _to_epoch(start_iso)
    end_epoch = start_epoch + lookahead * 60
    archive_url = _archive_url(camera["rtsp_url"], start_epoch, end_epoch)

    session = _spawn_session(start_epoch, archive_url)
    _ensure_cleanup()

    with _sessions_lock:
        if session["sid"] in _sessions:
            _sessions[session["sid"]]["dir"] = session["dir"]
        _sessions[session["sid"]] = session
    session.update({"start_ts": start_epoch, "end_ts": end_epoch, "window_min": lookahead, "last_touch": time.time(), "camera_id": camera.get("id")})

    try:
        _wait_ready(session)
    except (RuntimeError, TimeoutError) as e:
        _terminate(session["sid"])
        shutil.rmtree(session["dir"], ignore_errors=True)
        raise HTTPException(status_code=400, detail={"playable": False, "message": str(e)[:240]})

    return session


@playback_router.post("/session")
def create_session(payload: PlaybackSessionCreate):
    camera = _fetch_camera(payload.camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    _terminate_all()
    session = _start(payload.start_iso, camera, payload.lookahead_min)
    return _respond_session(session, camera)


@playback_router.get("/session/{sid}/index.m3u8")
def get_playlist(sid: str):
    session = _sessions.get(sid)
    if not session:
        raise HTTPException(status_code=404, detail="Session tidak ditemukan")
    playlist = session["dir"] / "index.m3u8"
    if not playlist.exists():
        raise HTTPException(status_code=404, detail="Playlist belum siap")
    session["last_touch"] = time.time()
    return Response(content=playlist.read_bytes(), media_type="application/vnd.apple.mpegurl")


@playback_router.get("/session/{sid}/{file}")
def get_segment(sid: str, file: str):
    if not re.fullmatch(r"[\w\-]+\.ts", file):
        raise HTTPException(status_code=400, detail="Invalid segment")
    session = _sessions.get(sid)
    if not session:
        raise HTTPException(status_code=404, detail="Session tidak ditemukan")
    path = session["dir"] / file
    if not path.exists():
        raise HTTPException(status_code=404, detail="Segment tidak ditemukan")
    session["last_touch"] = time.time()
    return FileResponse(path, media_type="video/mp2t")


@playback_router.post("/session/{sid}/seek")
def seek_session(sid: str, payload: PlaybackSeek):
    session = _sessions.get(sid)
    if not session:
        raise HTTPException(status_code=404, detail="Session tidak ditemukan")
    camera = _fetch_camera(session["camera_id"])
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    _terminate(sid)
    shutil.rmtree(session["dir"], ignore_errors=True)
    new_session = _start(payload.start_iso, camera, payload.lookahead_min)
    new_session["camera_id"] = session["camera_id"]
    return _respond_session(new_session, camera)


@playback_router.post("/session/{sid}/stop")
def stop_session(sid: str):
    _terminate(sid)
    return {"ok": True}


@playback_router.get("/availability")
def availability(camera_id: int, date: str):
    camera = _fetch_camera(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not _archive_track(camera.get("rtsp_url") or ""):
        return {"available": False, "reason": "no_dvr"}
    if not _ffmpeg_bin():
        return {"available": False, "reason": "no_ffmpeg"}
    try:
        probe_local = datetime.strptime(date, "%Y-%m-%d").astimezone()
    except ValueError:
        raise HTTPException(status_code=400, detail="Tanggal tidak valid (YYYY-MM-DD)")
    probe_epoch = probe_local.timestamp() + 12 * 3600
    archive_url = _archive_url(camera["rtsp_url"], probe_epoch, probe_epoch + 60)
    cmd = [
        _ffmpeg_bin(),
        "-hide_banner",
        "-loglevel",
        "quiet",
        "-rtsp_transport",
        "tcp",
        "-timeout",
        DEFAULT_TIMEOUT_MS,
        "-i",
        archive_url,
        "-frames:v",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
    except subprocess.TimeoutExpired:
        return {"available": False, "reason": "dvr_not_responding"}
    if proc.returncode == 0:
        return {"available": True, "reason": "ok"}
    if "400" in (proc.stderr or ""):
        return {"available": False, "reason": "no_recording"}
    return {"available": False, "reason": "error"}