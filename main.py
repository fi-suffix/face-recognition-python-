import os
import urllib.request
import cv2
import numpy as np
import requests
import logging
import queue
import threading
import time
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, HttpUrl
import uvicorn
from dotenv import load_dotenv
import concurrent.futures

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
LARAVEL_API_URL = os.getenv("LARAVEL_API_URL", "http://localhost:8000/api")
LARAVEL_API_KEY = os.getenv("LARAVEL_API_KEY", "")
FACE_RECOGNITION_THRESHOLD = float(os.getenv("FACE_RECOGNITION_THRESHOLD", "0.45"))
CAMERA_RECONNECT_INTERVAL = int(os.getenv("CAMERA_RECONNECT_INTERVAL", "5"))
LOG_COOLDOWN = float(os.getenv("LOG_COOLDOWN", "3"))
# Minimum seconds between recognition passes per face track (keeps the video fluid)
RECOGNITION_INTERVAL = float(os.getenv("RECOGNITION_INTERVAL", "2.0"))
# Stream width & fps proven on the field laptop (5 cameras, ~60% CPU)
MAX_STREAM_WIDTH = int(os.getenv("MAX_STREAM_WIDTH", "640"))
STREAM_FPS = int(os.getenv("STREAM_FPS", "10"))

# Performance tuning
DETECTION_INTERVAL = float(os.getenv("DETECTION_INTERVAL", "0.25"))  # seconds between YuNet passes per camera
DETECT_FRAME_WIDTH = int(os.getenv("DETECT_FRAME_WIDTH", "640"))     # detection runs on a downscaled frame
# Upscale detection frame to catch small/far faces (1.0 = off); boxes are mapped back
DETECT_UPSCALE = float(os.getenv("DETECT_UPSCALE", "1.5"))
# Crop margin ratio applied around the face bbox before embedding
FACE_CROP_MARGIN = float(os.getenv("FACE_CROP_MARGIN", "0.2"))
LOOP_FPS = int(os.getenv("LOOP_FPS", "15"))                          # target capture-loop FPS per camera
YUNET_INPUT_WIDTH = int(os.getenv("YUNET_INPUT_WIDTH", "640"))
YUNET_INPUT_HEIGHT = int(os.getenv("YUNET_INPUT_HEIGHT", "480"))
# Alias kept for the older FACE_DETECTION_CONFIDENCE var name coming from dashboard/team .env
YUNET_CONFIDENCE_THRESHOLD = float(os.getenv("FACE_DETECTION_CONFIDENCE", os.getenv("YUNET_CONFIDENCE_THRESHOLD", "0.5")))
GOOD_FRAME_CONFIDENCE = float(os.getenv("GOOD_FRAME_CONFIDENCE", "0.6"))  # only recognize high-confidence crops
MIN_FACE_WIDTH = int(os.getenv("MIN_FACE_WIDTH", "20"))              # min face width (in detect-frame px)
RECOGNITION_WORKERS = int(os.getenv("RECOGNITION_WORKERS", "2"))
# Recompute ALL stored embeddings with aligned SFace at startup (migration switch)
RECOMPUTE_EMBEDDINGS_ON_START = (os.getenv("RECOMPUTE_EMBEDDINGS_ON_START", "false").lower() in ("1", "true", "yes"))
RTSP_OPEN_TIMEOUT_MS = int(os.getenv("RTSP_OPEN_TIMEOUT_MS", "10000"))
RTSP_READ_TIMEOUT_MS = int(os.getenv("RTSP_READ_TIMEOUT_MS", "10000"))

# Model files (OpenCV Zoo YuNet + SFace)
MODEL_DIR = Path(__file__).resolve().parent / "models"
YUNET_PATH = MODEL_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_PATH = MODEL_DIR / "face_recognition_sface_2021dec.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

# Global storage for camera streams and face embeddings
camera_streams: Dict[str, Dict] = {}
face_embeddings_cache: Dict[int, List[np.ndarray]] = {}
employee_data_cache: Dict[int, Dict] = {}

# Vectorized matcher state
embedding_matrix: Optional[np.ndarray] = None  # shape (N, D), rows normalized
embedding_labels: List[int] = []               # aligned with matrix rows
last_log: Dict[int, Dict[Any, tuple]] = {}     # camera_id -> fingerprint -> (emp_id, last_log_ts)

# Per-camera face tracks (id continuity for drawing + recognition gating)
camera_tracks: Dict[int, List[dict]] = {}
track_lock = threading.Lock()

# Recognition work (embedding + snapshot + logging) is done off the frame loop
# so detection/streaming never blocks on the CNN or on Laravel HTTP calls.
recognition_queue: "queue.Queue" = queue.Queue(maxsize=64)

# Small thread pool for recognition (kept small so the capture loops keep CPU)
recognition_executor = concurrent.futures.ThreadPoolExecutor(max_workers=RECOGNITION_WORKERS)
# Guard for shared cv2 models (YuNet/SFace are NOT thread-safe; multiple camera
# threads + API workers call them concurrently)
model_lock = threading.Lock()


def now_ms() -> float:
    return datetime.now().timestamp()


# Pydantic models
class CameraConfig(BaseModel):
    id: int
    name: str
    rtsp_url: str
    location: str
    status: str = "active"
    username: Optional[str] = None
    password: Optional[str] = None
    reconnect_interval: int = 5


class FaceDetectionRequest(BaseModel):
    camera_id: int
    image_base64: str


class FaceRecognitionResult(BaseModel):
    camera_id: int
    employee_id: Optional[int]
    employee_name: Optional[str]
    confidence: float
    status: str  # "recognized" or "unknown"
    timestamp: str
    snapshot_path: Optional[str] = None


class TestRtspRequest(BaseModel):
    rtsp_url: str
    username: Optional[str] = None
    password: Optional[str] = None


def ensure_models():
    """Auto-download YuNet & SFace ONNX models from OpenCV Zoo if missing."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in [
        ("face_detection_yunet_2023mar.onnx", YUNET_URL),
        ("face_recognition_sface_2021dec.onnx", SFACE_URL),
    ]:
        path = MODEL_DIR / name
        if path.exists() and path.stat().st_size > 1000:
            continue
        logger.info(f"Downloading OpenCV Zoo model: {name} ...")
        try:
            with urllib.request.urlopen(url, timeout=180) as resp:
                MODEL_DIR.joinpath(name).write_bytes(resp.read())
            logger.info(f"Downloaded {name}")
        except Exception as e:
            logger.error(f"Failed to download {name}: {e}")


# Initialize face detection models (OpenCV 5.x compatible)
def init_face_models():
    global face_detector, face_recognizer
    face_detector = None
    face_recognizer = None
    try:
        ensure_models()

        if YUNET_PATH.exists():
            face_detector = cv2.FaceDetectorYN_create(
                str(YUNET_PATH), "", (YUNET_INPUT_WIDTH, YUNET_INPUT_HEIGHT),
                YUNET_CONFIDENCE_THRESHOLD, 0.3, 5000
            )
            logger.info(f"YuNet face detector loaded successfully (input: {YUNET_INPUT_WIDTH}x{YUNET_INPUT_HEIGHT}, conf: {YUNET_CONFIDENCE_THRESHOLD})")
        else:
            logger.warning("YuNet model not found, using fallback detection")

        if SFACE_PATH.exists():
            face_recognizer = cv2.FaceRecognizerSF_create(str(SFACE_PATH), "")
            logger.info("SFace face recognizer loaded successfully")
        else:
            logger.warning("SFace model not found, using fallback recognition")

        logger.info("Face detection models initialized")
    except Exception as e:
        logger.error(f"Failed to initialize face models: {e}")
        face_detector = None
        face_recognizer = None


def rebuild_embedding_matrix():
    """Precompute a normalized (N, D) embedding matrix once instead of nested loops."""
    global embedding_matrix, embedding_labels
    rows, labels = [], []
    for emp_id, embeddings in face_embeddings_cache.items():
        for emb in embeddings:
            norm = np.linalg.norm(emb)
            if norm > 1e-6:
                rows.append(emb / norm)
                labels.append(emp_id)
    if rows:
        embedding_matrix = np.vstack(rows)  # (N, D)
    else:
        embedding_matrix = None
    embedding_labels = labels
    logger.info(f"Embedding matrix rebuilt with {len(labels)} samples from {len(face_embeddings_cache)} employees")


def load_employee_embeddings():
    """Load face embeddings from Laravel API"""
    global face_embeddings_cache, employee_data_cache
    try:
        headers = {"Authorization": f"Bearer {LARAVEL_API_KEY}"} if LARAVEL_API_KEY else {}
        response = requests.get(f"{LARAVEL_API_URL}/face-recognition/face-embeddings", headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            face_embeddings_cache = {}
            employee_data_cache = {}
            for emp in data.get("employees", []):
                emp_id = emp["id"]
                employee_data_cache[emp_id] = emp
                embeddings = []
                for emb in emp.get("embeddings", []):
                    try:
                        vec = emb.get("embedding")
                        if vec is None:
                            continue
                        embeddings.append(np.array(vec, dtype=np.float32))
                    except Exception:
                        continue
                if embeddings:
                    face_embeddings_cache[emp_id] = embeddings

            # Migrate legacy (non-SFace) embeddings so recognition stays accurate.
            needs_check = (RECOMPUTE_EMBEDDINGS_ON_START or face_recognizer is not None) and not RECOMPUTE_EMBEDDINGS_TRIED[0]
            if needs_check:
                has_legacy = False
                for emp in data.get("employees", []):
                    for emb in emp.get("embeddings", []):
                        vec = emb.get("embedding")
                        if vec is None or len(vec) != SFACE_DIM:
                            has_legacy = True
                            break
                    if has_legacy:
                        break
                if has_legacy or RECOMPUTE_EMBEDDINGS_ON_START:
                    if has_legacy:
                        logger.info("Detected legacy or missing embeddings, recomputing with SFace ...")
                    else:
                        logger.info("RECOMPUTE_EMBEDDINGS_ON_START=true, recomputing embeddings with aligned SFace ...")
                    RECOMPUTE_EMBEDDINGS_TRIED[0] = True
                    recompute_embeddings(force=RECOMPUTE_EMBEDDINGS_ON_START)
                    load_employee_embeddings()
                    return

            rebuild_embedding_matrix()
            logger.info(f"Loaded embeddings for {len(face_embeddings_cache)} employees")
        else:
            logger.warning(f"Failed to load embeddings: {response.status_code}")
    except Exception as e:
        logger.error(f"Error loading employee embeddings: {e}")


SFACE_DIM = 128
RECOMPUTE_EMBEDDINGS_TRIED = [False]


def recompute_embeddings(force: bool = False):
    """Recompute stored photo embeddings with SFace when they use the old fallback format.

    Photos are fetched from the Laravel storage (same machine in dev), re-encoded with
    aligned SFace, and persisted back via the Laravel API. Pass force=True to recompute
    even embeddings that already have the right dimension (e.g. after an alignment upgrade).
    """
    if face_recognizer is None:
        return {"message": "sface unavailable", "recomputed": 0, "failed": 0}

    headers = {"Authorization": f"Bearer {LARAVEL_API_KEY}"} if LARAVEL_API_KEY else {}
    recomputed = 0
    failed = 0
    storage_url = LARAVEL_API_URL.split("/api")[0].rstrip("/") + "/storage"

    try:
        resp = requests.get(f"{LARAVEL_API_URL}/face-recognition/face-embeddings", headers=headers, timeout=15)
        if resp.status_code != 200:
            return {"message": "failed to fetch employees", "recomputed": 0, "failed": 0}

        for emp in resp.json().get("employees", []):
            for photo in emp.get("embeddings", []):
                emb = photo.get("embedding") or []
                if len(emb) == SFACE_DIM and not force:
                    continue
                image_path = photo.get("image_path")
                if not image_path:
                    continue
                try:
                    img_resp = requests.get(f"{storage_url}/{image_path}", timeout=10)
                    img_resp.raise_for_status()
                    nparr = np.frombuffer(img_resp.content, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        failed += 1
                        continue
                    faces = detect_faces_yunet(frame)
                    if not faces:
                        failed += 1
                        continue
                    x, y, w, h, lm, _ = max(faces, key=lambda f: f[2] * f[3])
                    if w <= 0 or h <= 0:
                        failed += 1
                        continue
                    face_img, lm_rel = crop_face(frame, x, y, w, h, lm)
                    if face_img is None:
                        failed += 1
                        continue
                    embedding = extract_face_embedding_sface(face_img, lm_rel)
                    upd = requests.put(
                        f"{LARAVEL_API_URL}/face-recognition/employee-photos/{photo['id']}/embedding",
                        json={"embedding": embedding.tolist()},
                        headers=headers,
                        timeout=10,
                    )
                    if upd.status_code in (200, 201):
                        recomputed += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                    logger.warning(f"Embedding recompute failed for photo {photo.get('id')}: {e}")
    except Exception as e:
        logger.error(f"Embedding recompute error: {e}")

    logger.info(f"Embedding recompute finished: {recomputed} recomputed, {failed} failed")
    return {"message": "done", "recomputed": recomputed, "failed": failed}

def crop_face(frame: np.ndarray, x: int, y: int, w: int, h: int, lm):
    """Crop a face with margin (FACE_CROP_MARGIN), returning (crop, landmarks relative to crop)."""
    mx = max(12, int(w * FACE_CROP_MARGIN))
    my = max(12, int(h * FACE_CROP_MARGIN))
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(frame.shape[1], x + w + mx)
    y1 = min(frame.shape[0], y + h + my)
    if x1 - x0 < 12 or y1 - y0 < 12:
        return None, None
    face_img = frame[y0:y1, x0:x1]
    if face_img.size == 0:
        return None, None
    lm_rel = None
    if lm is not None:
        lm_rel = lm - np.array([x0, y0], dtype=np.float32)
    return face_img, lm_rel


def align_face(face_img: np.ndarray, landmarks) -> np.ndarray:
    """Align the face to the SFace 112x112 template using the 5 YuNet landmarks."""
    if landmarks is None or len(landmarks) != 5:
        return face_img
    src = np.float32(landmarks)
    tpl = np.float32([
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ])
    mat, _ = cv2.estimateAffinePartial2D(src, tpl)
    if mat is None:
        return face_img
    aligned = cv2.warpAffine(face_img, mat, (112, 112), borderValue=0.0)
    return aligned


def detect_faces_yunet(frame: np.ndarray) -> List[tuple]:
    """Detect faces using YuNet (OpenCV 5.x).

    Supports DETECT_UPSCALE: the frame is enlarged before detection to catch
    small/far faces; bounding boxes & landmarks are mapped back to the space of
    the input `frame`. Shared model access is guarded by model_lock.

    Returns list of (x, y, w, h, landmarks(5x2 float32 or None), confidence).
    """
    if face_detector is None:
        return detect_faces_fallback(frame)

    scale = DETECT_UPSCALE if DETECT_UPSCALE > 1.0 else 1.0
    det_frame = frame
    if scale > 1.0:
        dh, dw = frame.shape[:2]
        det_frame = cv2.resize(
            frame,
            (int(dw * scale), int(dh * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

    h, w = det_frame.shape[:2]
    with model_lock:
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(det_frame)

    results = []
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            confidence = float(face[14])
            if confidence > YUNET_CONFIDENCE_THRESHOLD:
                lm = None
                if face.shape[0] >= 14:
                    lm = face[4:14].reshape(5, 2).astype(np.float32)
                if scale > 1.0:
                    x = int(round(x / scale))
                    y = int(round(y / scale))
                    w = int(round(w / scale))
                    h = int(round(h / scale))
                    if lm is not None:
                        lm = lm / scale
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                if w > MIN_FACE_WIDTH and h > 10:
                    results.append((x, y, w, h, lm, confidence))
    return results


def detect_faces_fallback(frame: np.ndarray) -> List[tuple]:
    """Fallback face detection without ONNX models (skin-color + geometry heuristics)"""
    results = []
    h, w = frame.shape[:2]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 48, 0])
    upper_skin = np.array([20, 255, 255])

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w < 60 or h < 60:
            continue
        aspect_ratio = float(w) / h
        if not (0.5 < aspect_ratio < 1.5):
            continue
        area_ratio = (w * h) / (frame.shape[0] * frame.shape[1])
        if area_ratio < 0.005:
            continue
        results.append((x, y, w, h))

    if not results:
        return []

    results = sorted(results, key=lambda r: r[2] * r[3], reverse=True)
    kept = []
    for (x, y, w, h) in results:
        keep = True
        for (kx, ky, kw, kh) in kept:
            xx1 = max(x, kx)
            yy1 = max(y, ky)
            xx2 = min(x + w, kx + kw)
            yy2 = min(y + h, ky + kh)
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            union = w * h + kw * kh - inter
            iou = inter / union if union > 0 else 0
            if iou > 0.3:
                keep = False
                break
        if keep:
            kept.append((x, y, w, h))

    return [(x, y, w, h, None, None) for (x, y, w, h) in kept]


def extract_face_embedding_sface(face_img: np.ndarray, landmarks=None) -> np.ndarray:
    """Extract face embedding using SFace (OpenCV 5.x), aligned via landmarks."""
    if face_recognizer is None:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (100, 100))
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        cv2.normalize(hist, hist)
        features = []
        features.extend(hist.flatten())
        hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
        hsv_small = cv2.resize(hsv, (50, 50))
        for i in range(3):
            mean, std = cv2.meanStdDev(hsv_small[:, :, i])
            features.extend([float(mean[0, 0]), float(std[0, 0])])
        lap = cv2.Laplacian(gray, cv2.CV_32F)
        features.extend([float(np.mean(lap)), float(np.std(lap))])
        return np.array(features, dtype=np.float32)

    aligned_face = align_face(face_img, landmarks)
    aligned_face = cv2.resize(aligned_face, (112, 112))
    embedding = face_recognizer.feature(aligned_face)
    return embedding.flatten()


def recognize_face(face_embedding: np.ndarray) -> tuple:
    """Recognize face by checking against the precomputed embedding matrix."""
    norm = np.linalg.norm(face_embedding)
    if norm < 1e-6 or embedding_matrix is None or embedding_matrix.shape[0] == 0:
        return None, 0.0

    q = face_embedding / norm
    sims = embedding_matrix @ q  # (N,) cosine similarities
    idx = int(np.argmax(sims))
    best_confidence = float(sims[idx])

    if best_confidence >= FACE_RECOGNITION_THRESHOLD:
        return embedding_labels[idx], best_confidence
    return None, best_confidence


def should_log(camera_id: int, fp: Any, emp_id: Optional[int], t: float) -> bool:
    """Limit detection logs to avoid DB spamming while still capturing identity changes."""
    fp_map = last_log.setdefault(camera_id, {})
    prev = fp_map.get(fp)
    if prev is None:
        fp_map[fp] = (emp_id, t)
        return True
    prev_emp, prev_time = prev
    if prev_emp != emp_id and (t - prev_time) >= 1.0:
        fp_map[fp] = (emp_id, t)
        return True
    if (t - prev_time) >= LOG_COOLDOWN:
        fp_map[fp] = (emp_id, t)
        return True
    return False


def prune_last_log():
    """Periodically clean up stale track entries so the LRU dict doesn't grow forever."""
    t = now_ms()
    for cam_id in list(last_log.keys()):
        fp_map = last_log[cam_id]
        stale = [fp for fp, (_, ts) in fp_map.items() if t - ts > 300]
        for fp in stale:
            del fp_map[fp]
        if len(fp_map) > 1000:
            items = sorted(fp_map.items(), key=lambda kv: kv[1][1])
            for fp, _ in items[: max(0, len(items) - 500)]:
                del fp_map[fp]


def send_detection_log(result: FaceRecognitionResult):
    """Send detection log to Laravel API"""
    try:
        headers = {
            "Authorization": f"Bearer {LARAVEL_API_KEY}",
            "Content-Type": "application/json"
        } if LARAVEL_API_KEY else {"Content-Type": "application/json"}

        payload = {
            "camera_id": result.camera_id,
            "employee_id": result.employee_id,
            "employee_name": result.employee_name,
            "confidence": result.confidence,
            "status": result.status,
            "timestamp": result.timestamp,
            "snapshot_path": result.snapshot_path
        }

        response = requests.post(
            f"{LARAVEL_API_URL}/face-recognition/detection-logs",
            json=payload,
            headers=headers,
            timeout=5
        )
        if response.status_code != 201:
            logger.warning(f"Failed to send log: {response.status_code} - {response.text}")
    except Exception as e:
        logger.error(f"Error sending detection log: {e}")


def save_snapshot(camera_id: int, frame: np.ndarray, status: str, timestamp: str) -> str:
    try:
        dt = datetime.fromisoformat(timestamp)
        snapshot_dir = Path("snapshots") / str(camera_id) / dt.strftime("%Y-%m-%d")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot_name = f"{dt.strftime('%H-%M-%S-%f')}_{status}.jpg"
        snapshot_path = snapshot_dir / snapshot_name
        cv2.imwrite(str(snapshot_path), frame)
        return str(snapshot_path).replace("\\", "/")
    except Exception as e:
        logger.error(f"Failed to save snapshot: {e}")
        return None


def build_rtsp_url(camera: CameraConfig) -> str:
    """Build RTSP URL with credentials if provided"""
    if camera.username and camera.password:
        url = camera.rtsp_url
        if url.startswith("rtsp://"):
            url = url.replace("rtsp://", f"rtsp://{camera.username}:{camera.password}@")
        return url
    return camera.rtsp_url


def match_track(tracks: List[dict], bbox: tuple, t: float, max_age_ms: float = 3000.0) -> tuple:
    """Find the tracked face whose box overlaps the new detection by IoU."""
    bx, by, bw, bh = bbox
    best_iou = 0.0
    best_i = None
    for i, tr in enumerate(tracks):
        if t - tr.get("ts", 0.0) > max_age_ms:
            continue
        tx, ty, tw, th = tr.get("bbox", (0, 0, 0, 0))
        xi1 = max(bx, tx)
        yi1 = max(by, ty)
        xi2 = min(bx + bw, tx + tw)
        yi2 = min(by + bh, ty + th)
        inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        union = bw * bh + tw * th - inter
        iou = inter / union if union > 0 else 0
        if iou > best_iou:
            best_iou = iou
            best_i = i
    return best_i, best_iou


def prune_tracks(camera_id: int, t: float, max_age_ms: float = 5000.0, max_items: int = 80):
    tracks = camera_tracks.get(camera_id, [])
    if not tracks:
        return
    alive = [tr for tr in tracks if t - tr.get("ts", 0.0) <= max_age_ms]
    if len(alive) > max_items:
        alive = alive[-max_items:]
    camera_tracks[camera_id] = alive

def process_camera_stream(camera: CameraConfig, stop_event: threading.Event):
    """Process camera stream in background.

    The capture loop reads + downscales + encodes frames at a capped fps. Face
    detection runs time-gated on a small frame, and recognition (SFace + snapshot
    + Laravel log) is off-loaded to the worker pool. Nothing here copies the full
    frame per detected face, so a walking person no longer starves the loop.
    """
    logger.info(f"Starting stream processing for camera {camera.id}: {camera.name}")

    rtsp_url = build_rtsp_url(camera)
    is_webcam = False
    try:
        webcam_index = int(rtsp_url)
        is_webcam = True
    except ValueError:
        pass

    # Force TCP transport for RTSP and fail fast instead of hanging 30s per attempt
    conn_url = rtsp_url
    if not is_webcam and rtsp_url.startswith("rtsp://"):
        conn_url = rtsp_url + "?tcp"

    reconnect_interval = camera.reconnect_interval or CAMERA_RECONNECT_INTERVAL
    cap = None
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    frames = 0
    loop_dt = 1.0 / max(1, LOOP_FPS)
    last_detect_ts = 0.0
    cached_faces = []
    det_frame = None

    while camera_streams.get(camera.id, {}).get("running", False) and not stop_event.is_set():
        loop_start = time.time()
        try:
            if cap is None or not cap.isOpened():
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error(f"Max reconnect attempts reached for camera {camera.id}")
                    break

                logger.info(f"Connecting to camera {camera.id}: {rtsp_url}")

                if is_webcam:
                    cap = cv2.VideoCapture(webcam_index)
                else:
                    cap = cv2.VideoCapture(conn_url)
                    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS)
                    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS)

                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                reconnect_attempts += 1

                if not cap.isOpened():
                    logger.warning(f"Failed to open camera {camera.id}, retrying in {reconnect_interval}s")
                    time.sleep(reconnect_interval)
                    continue

                reconnect_attempts = 0
                logger.info(f"Camera {camera.id} connected successfully")

            ret, frame = cap.read()
            if not ret:
                logger.warning(f"Failed to read frame from camera {camera.id}")
                cap.release()
                cap = None
                time.sleep(0.5)
                continue

            frames += 1

            # Downscale large frames for cheaper streaming (tiered)
            h, w = frame.shape[:2]
            if w > MAX_STREAM_WIDTH:
                scale = MAX_STREAM_WIDTH / float(w)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (MAX_STREAM_WIDTH, new_h), interpolation=cv2.INTER_AREA)
            elif w > 640:
                scale = 640.0 / float(w)
                new_h = int(h * scale)
                frame = cv2.resize(frame, (640, new_h), interpolation=cv2.INTER_AREA)

            t = now_ms()

            # Detection is time-gated (not frame-gated) and runs on a small frame
            if (t - last_detect_ts) >= (DETECTION_INTERVAL * 1000.0):
                last_detect_ts = t
                th, tw = frame.shape[:2]
                det_frame = frame
                if tw > DETECT_FRAME_WIDTH:
                    dscale = DETECT_FRAME_WIDTH / float(tw)
                    det_frame = cv2.resize(frame, (DETECT_FRAME_WIDTH, int(th * dscale)), interpolation=cv2.INTER_AREA)
                if face_detector is not None:
                    cached_faces = detect_faces_yunet(det_frame)
                else:
                    cached_faces = detect_faces_fallback(det_frame)

            faces = cached_faces if cached_faces else []
            disp_scale = 1.0
            if det_frame is not None and det_frame is not frame:
                disp_scale = frame.shape[1] / float(det_frame.shape[1])

            tracks = camera_tracks.setdefault(camera.id, [])
            frames_due = []

            for (x, y, w, h, lm, fconf) in faces:
                good = (fconf is None or fconf >= GOOD_FRAME_CONFIDENCE)
                idx, _ = match_track(tracks, (x, y, w, h), t)
                if idx is not None:
                    tr = tracks[idx]
                    tr["bbox"] = (x, y, w, h)
                    tr["ts"] = t
                    emp_id = tr.get("emp_id")
                    confidence = tr.get("confidence", 0.0)
                    recognize_now = good and w >= MIN_FACE_WIDTH and (
                        t - tr.get("last_recog_ts", 0.0)) >= (RECOGNITION_INTERVAL * 1000.0)
                else:
                    emp_id = None
                    confidence = 0.0
                    tr = {"bbox": (x, y, w, h), "ts": t, "emp_id": None, "confidence": 0.0, "last_recog_ts": 0.0}
                    with track_lock:
                        tracks.append(tr)
                    recognize_now = good and w >= MIN_FACE_WIDTH

                if recognize_now:
                    frames_due.append((x, y, w, h, lm, fconf, t))
                    tr["last_recog_ts"] = t

                dx, dy = int(x * disp_scale), int(y * disp_scale)
                dw, dh = int(w * disp_scale), int(h * disp_scale)
                emp_data = employee_data_cache.get(emp_id, {}) if emp_id else {}
                name = emp_data.get("name", "")
                status = "recognized" if emp_id else "unknown"
                color = (0, 255, 0) if status == "recognized" else (0, 0, 255)
                label = f"{name or 'Unknown'} ({confidence:.2f})"
                cv2.rectangle(frame, (dx, dy), (dx + dw, dy + dh), color, 2)
                cv2.putText(frame, label, (dx, max(20, dy - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Off-load recognition (SFace + snapshot + Laravel log) to the workers.
            # Only the small detection frame is copied, once per detection pass.
            if frames_due and face_recognizer is not None:
                try:
                    recognition_queue.put_nowait((camera.id, det_frame.copy(), frames_due))
                except queue.Full:
                    logger.warning(f"Recognition queue full for camera {camera.id}, dropping batch")

            # Store cached JPEG for MJPEG streaming / snapshot polling
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                camera_streams[camera.id]["latest_jpg"] = buf.tobytes()
            camera_streams[camera.id]["last_update"] = now_ms()

            if frames % 200 == 0:
                prune_last_log()
                prune_tracks(camera.id, now_ms())

            elapsed = time.time() - loop_start
            if elapsed < loop_dt:
                time.sleep(loop_dt - elapsed)

        except Exception as e:
            logger.error(f"Error processing camera {camera.id}: {e}")
            if cap:
                cap.release()
                cap = None
            time.sleep(reconnect_interval)

    if cap:
        cap.release()
    logger.info(f"Stopped stream processing for camera {camera.id}")


def recognition_worker():
    """Consumer for recognition_queue: embedding, result cache, snapshot + Laravel log.

    Runs off the camera frame loop so the CNN and HTTP calls never stall the video.
    """
    while True:
        task = recognition_queue.get()
        if task is None:
            break
        recognition_executor.submit(process_recognition_task, task)


def process_recognition_task(task):
    """Process a recognition batch (frame + list of faces) in the thread pool."""
    camera_id, det_frame, faces = task
    t = now_ms()
    try:
        timestamp = datetime.now().isoformat()
        for (x, y, w, h, lm, fconf, ts) in faces:
            face_img, lm_rel = crop_face(det_frame, x, y, w, h, lm)
            if face_img is None:
                continue

            embedding = extract_face_embedding_sface(face_img, lm_rel)
            emp_id, confidence = recognize_face(embedding)

            with track_lock:
                tr = camera_tracks.get(camera_id)
                if tr is not None:
                    idx, _ = match_track(tr, (x, y, w, h), t)
                    if idx is not None and tr[idx].get("last_recog_ts", -1e9) <= ts:
                        tr[idx]["emp_id"] = emp_id
                        tr[idx]["confidence"] = confidence

            status = "recognized" if emp_id else "unknown"
            emp_data = employee_data_cache.get(emp_id, {}) if emp_id else {}
            name = emp_data.get("name", "")

            # Logging dedup: limit writes to Laravel DB
            fp = ("emp", emp_id) if emp_id else ("unk", int(x // 64), int(y // 64))
            if should_log(camera_id, fp, emp_id, t):
                snapshot_path = save_snapshot(camera_id, det_frame, status, timestamp)
                result = FaceRecognitionResult(
                    camera_id=camera_id,
                    employee_id=emp_id,
                    employee_name=name or None,
                    confidence=float(confidence),
                    status=status,
                    timestamp=timestamp,
                    snapshot_path=snapshot_path
                )
                send_detection_log(result)
    except Exception as e:
        logger.error(f"Recognition worker error for camera {camera_id}: {e}")


def start_camera_thread(camera: CameraConfig):
    """Start camera processing thread (restarts cleanly if already present)"""
    if camera.id in camera_streams:
        old = camera_streams[camera.id]
        old_thread = old.get("thread")
        if old_thread and old_thread.is_alive():
            old["running"] = False
            old["stop_event"].set()
            old_thread.join(timeout=3)

    stop_event = threading.Event()
    camera_streams[camera.id] = {
        "config": camera,
        "running": True,
        "stop_event": stop_event,
        "latest_jpg": None,
        "last_update": None,
    }
    thread = threading.Thread(target=process_camera_stream, args=(camera, stop_event), daemon=True)
    thread.start()
    camera_streams[camera.id]["thread"] = thread


# Lifespan handlers (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_face_models()
    load_employee_embeddings()

    # Recognition worker (embedding / snapshot / logging off the frame loop)
    recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
    recognition_thread.start()

    # Load cameras from Laravel
    try:
        headers = {"Authorization": f"Bearer {LARAVEL_API_KEY}"} if LARAVEL_API_KEY else {}
        response = requests.get(f"{LARAVEL_API_URL}/face-recognition/cameras", headers=headers, timeout=10)
        if response.status_code == 200:
            cameras = response.json().get("cameras", [])
            for cam_data in cameras:
                if cam_data.get("status") == "active":
                    camera = CameraConfig(**cam_data)
                    start_camera_thread(camera)
    except Exception as e:
        logger.error(f"Failed to load cameras on startup: {e}")

    yield

    # Shutdown
    for cam_id in camera_streams:
        camera_streams[cam_id]["running"] = False
        camera_streams[cam_id].get("stop_event", threading.Event()).set()
    try:
        recognition_queue.put_nowait(None)
    except queue.Full:
        pass
    # Shutdown thread pool
    recognition_executor.shutdown(wait=True)


app = FastAPI(
    title="Facial Recognition CCTV Service",
    description="Computer Vision service for face detection and recognition",
    version="2.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# API Endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "facial-recognition",
        "models": {
            "yunet": face_detector is not None,
            "sface": face_recognizer is not None,
        },
        "cameras": {cid: s.get("running") for cid, s in camera_streams.items()},
        "employees": len(face_embeddings_cache),
    }


@app.post("/cameras/{camera_id}/start")
async def start_camera(camera_id: int):
    try:
        headers = {"Authorization": f"Bearer {LARAVEL_API_KEY}"} if LARAVEL_API_KEY else {}
        response = requests.get(f"{LARAVEL_API_URL}/face-recognition/cameras/{camera_id}", headers=headers, timeout=5)
        if response.status_code != 200:
            raise HTTPException(status_code=404, detail="Camera not found")

        cam_data = response.json()["camera"]
        camera = CameraConfig(**cam_data)

        if camera_id in camera_streams and camera_streams[camera_id].get("running"):
            return {"message": "Camera already running"}

        start_camera_thread(camera)
        return {"message": f"Camera {camera_id} started"}
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch camera: {e}")


@app.post("/cameras/{camera_id}/stop")
async def stop_camera(camera_id: int):
    if camera_id in camera_streams:
        camera_streams[camera_id]["running"] = False
        camera_streams[camera_id].get("stop_event", threading.Event()).set()
        return {"message": f"Camera {camera_id} stopped"}
    raise HTTPException(status_code=404, detail="Camera not running")


@app.get("/cameras/{camera_id}/stream")
async def camera_stream(camera_id: int):
    """MJPEG stream for live monitoring (serves cached JPEG frames)"""
    if camera_id not in camera_streams:
        raise HTTPException(status_code=404, detail="Camera not running")

    def generate_frames():
        frame_interval = 1.0 / STREAM_FPS
        while camera_streams.get(camera_id, {}).get("running", False):
            jpg = camera_streams[camera_id].get("latest_jpg")
            if jpg is not None:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')
            time.sleep(frame_interval)

    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/cameras/{camera_id}/snapshot")
async def camera_snapshot(camera_id: int):
    """Single JPEG snapshot for polling-based browser playback"""
    if camera_id not in camera_streams:
        raise HTTPException(status_code=404, detail="Camera not running")
    jpg = camera_streams[camera_id].get("latest_jpg")
    if jpg is None:
        raise HTTPException(status_code=503, detail="No frame available")
    return Response(content=jpg, media_type="image/jpeg")


@app.post("/recognize")
def recognize_face_endpoint(request: FaceDetectionRequest):
    """Recognize face from base64 image"""
    import base64

    try:
        img_data = base64.b64decode(request.image_base64)
        nparr = np.frombuffer(img_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        if face_detector is not None:
            faces = detect_faces_yunet(frame)
        else:
            faces = detect_faces_fallback(frame)

        results = []
        for (x, y, w, h, lm, _) in faces:
            x, y, w, h = int(x), int(y), int(w), int(h)
            face_img, lm_rel = crop_face(frame, x, y, w, h, lm)
            if face_img is None:
                continue

            embedding = extract_face_embedding_sface(face_img, lm_rel)
            emp_id, confidence = recognize_face(embedding)

            if emp_id:
                emp_data = employee_data_cache.get(emp_id, {})
                results.append({
                    "employee_id": emp_id,
                    "employee_name": emp_data.get("name", "Unknown"),
                    "confidence": float(confidence),
                    "status": "recognized",
                    "bbox": [x, y, w, h]
                })
            else:
                results.append({
                    "employee_id": None,
                    "employee_name": None,
                    "confidence": float(confidence),
                    "status": "unknown",
                    "bbox": [x, y, w, h]
                })

        return {"camera_id": request.camera_id, "detections": results}
    except Exception as e:
        logger.error(f"Recognition error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/test-detect")
def test_detect_endpoint(file: UploadFile = File(...)):
    """Test face detection from uploaded image file"""
    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        if face_detector is not None:
            faces = detect_faces_yunet(frame)
        else:
            faces = detect_faces_fallback(frame)

        for (x, y, w, h, lm, _) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "FACE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        output_path = Path("test_results") / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)

        return {
            "faces_detected": len(faces),
            "faces": [{"bbox": [int(x), int(y), int(w), int(h)]} for (x, y, w, h, _, _) in faces],
            "annotated_image": str(output_path),
            "model": "yunet" if face_detector is not None else "fallback"
        }
    except Exception as e:
        logger.error(f"Test detect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-embedding")
def extract_embedding_endpoint(file: UploadFile = File(...)):
    """Extract embedding from uploaded face image"""
    try:
        contents = file.file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")

        if face_detector is not None:
            faces = detect_faces_yunet(frame)
        else:
            faces = detect_faces_fallback(frame)

        if not faces:
            return {"embedding": None, "faces_detected": 0}

        # Pick the largest face if multiple are detected
        x, y, w, h, lm, _ = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = int(x), int(y), int(w), int(h)
        face_img, lm_rel = crop_face(frame, x, y, w, h, lm)
        if face_img is None:
            return {"embedding": None, "faces_detected": 0}
        embedding = extract_face_embedding_sface(face_img, lm_rel)

        return {
            "embedding": embedding.tolist(),
            "faces_detected": len(faces),
            "bbox": [x, y, w, h],
            "model": "sface" if face_recognizer is not None else "histogram"
        }
    except Exception as e:
        logger.error(f"Embedding extraction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/reload-embeddings")
async def reload_embeddings():
    load_employee_embeddings()
    return {"message": "Embeddings reloaded", "count": len(face_embeddings_cache)}


@app.post("/recompute-embeddings")
async def recompute_embeddings_endpoint(force: bool = True):
    """Recompute stored embeddings using aligned SFace (migrates old embeddings)."""
    result = recompute_embeddings(force=force)
    load_employee_embeddings()
    return result


@app.get("/cameras/status")
async def cameras_status():
    status = {}
    for cam_id, data in camera_streams.items():
        status[cam_id] = {
            "name": data["config"].name,
            "running": data.get("running", False),
            "rtsp_url": data["config"].rtsp_url
        }
    return {"cameras": status}


@app.post("/test-rtsp")
def test_rtsp_endpoint(req: TestRtspRequest):
    """Probe an RTSP camera (dashboard 'Test Connection'). Pakai ffmpeg CLI agar
    tidak pernah memblokir/menggantung service saat URL tidak terjangkau."""
    from urllib.parse import quote
    import shutil
    import subprocess

    url = req.rtsp_url
    if url.startswith("rtsp://") and req.username and req.password:
        user = quote(req.username, safe="")
        pwd = quote(req.password, safe="")
        url = url.replace("rtsp://", f"rtsp://{user}:{pwd}@", 1)

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        for cand in [r"C:\Users\padil\Downloads\ffmpeg-2026-09-07-git-ecc7eb519e-full_build\bin\ffmpeg.exe"]:
            if os.path.isfile(cand):
                ffmpeg = cand
                break

    if ffmpeg:
        cmd = [
            ffmpeg, "-hide_banner", "-loglevel", "error",
            "-rtsp_transport", "tcp",
            "-i", url,
            "-frames:v", "1",
            "-f", "null", "-",
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=12)
            if proc.returncode == 0:
                return {"success": True, "message": "Koneksi berhasil - stream video aktif"}

            detail = (proc.stderr or proc.stdout or "").strip().splitlines()
            detail = detail[-1] if detail else "koneksi ditolak"
            return {"success": False, "message": f"Gagal terhubung ({detail[:160]})"}
        except subprocess.TimeoutExpired:
            return {"success": False, "message": "Timeout - perangkat tidak merespon dalam 12 detik"}
        except Exception as e:
            return {"success": False, "message": f"Error: {e}"}

    # Fallback: probe via OpenCV di thread terpisah (tidak memblokir API worker)
    result_box: dict = {}

    def probe_cv():
        cap = None
        try:
            conn_url = url + ("?tcp" if url.startswith("rtsp://") else "")
            cap = cv2.VideoCapture(conn_url)
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            if not cap.isOpened():
                result_box["r"] = {"success": False, "message": "Gagal membuka koneksi RTSP"}
                return
            ok, frame = cap.read()
            if not ok:
                result_box["r"] = {"success": False, "message": "Terkoneksi tapi gagal membaca frame"}
                return
            h, w = frame.shape[:2]
            result_box["r"] = {"success": True, "message": f"Koneksi berhasil - video {w}x{h}", "width": w, "height": h}
        except Exception as e:
            result_box["r"] = {"success": False, "message": f"Error: {e}"}
        finally:
            if cap is not None:
                cap.release()

    threading.Thread(target=probe_cv, daemon=True).start()
    waited = 0.0
    while "r" not in result_box and waited < 12:
        time.sleep(0.5)
        waited += 0.5
    if "r" in result_box:
        return result_box["r"]
    return {"success": False, "message": "Timeout - perangkat tidak merespon dalam 12 detik"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)