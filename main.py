import os
import math
import uuid
import collections
import urllib.request
import cv2
import numpy as np
import requests
import logging
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, Response
from pydantic import BaseModel, HttpUrl
import uvicorn
from dotenv import load_dotenv
from urllib.parse import urlsplit, urlunsplit, quote

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
RECOGNITION_EVERY_N_FRAMES = int(os.getenv("RECOGNITION_EVERY_N_FRAMES", "2"))
# Minimum seconds between recognition passes per camera (keeps the video fluid)
RECOGNITION_INTERVAL = float(os.getenv("RECOGNITION_INTERVAL", "2.0"))
MAX_STREAM_WIDTH = int(os.getenv("MAX_STREAM_WIDTH", "640"))
STREAM_FPS = int(os.getenv("STREAM_FPS", "10"))
# Sensitivitas deteksi wajah (cocok untuk kamera tinggi / wajah kecil)
DETECT_UPSCALE = float(os.getenv("DETECT_UPSCALE", "1.0"))
FACE_CROP_MARGIN = float(os.getenv("FACE_CROP_MARGIN", "0.2"))

# Performance tuning
DETECT_EVERY_N_FRAMES = int(os.getenv("DETECT_EVERY_N_FRAMES", "2"))  # Run YuNet every N frames
YUNET_INPUT_WIDTH = int(os.getenv("YUNET_INPUT_WIDTH", "640"))
YUNET_INPUT_HEIGHT = int(os.getenv("YUNET_INPUT_HEIGHT", "480"))
YUNET_CONFIDENCE_THRESHOLD = float(os.getenv("YUNET_CONFIDENCE_THRESHOLD", "0.5"))
RECOGNITION_WORKERS = int(os.getenv("RECOGNITION_WORKERS", "2"))
RTSP_OPEN_TIMEOUT_MS = int(os.getenv("RTSP_OPEN_TIMEOUT_MS", "10000"))
RTSP_READ_TIMEOUT_MS = int(os.getenv("RTSP_READ_TIMEOUT_MS", "10000"))
# CLAHE pada frame deteksi untuk video burik (1 = nyala, hanya memengaruhi deteksi)
DETECT_ENHANCE = os.getenv("DETECT_ENHANCE", "1") == "1"
# Wajah lebih kecil dari ini (px, di lebar stream) tidak di-recognisi/di-log (buang noise)
MIN_FACE_SIZE = int(os.getenv("MIN_FACE_SIZE", "24"))
# Maksimal wajah yang diproses recognition per jendela gate (ambil yang terbesar)
RECOGNITION_MAX_FACES = int(os.getenv("RECOGNITION_MAX_FACES", "2"))

# --- Event emitter: rekognisi -> POST /api/internal/recognition-events (Laravel) ---
# Token internal Laravel (dari Setting api_integration.internal_token / env AI_INTERNAL_TOKEN)
LARAVEL_INTERNAL_TOKEN = os.getenv("LARAVEL_INTERNAL_TOKEN", "")
EVENT_EMIT_ENABLED = os.getenv("EVENT_EMIT_ENABLED", "1") == "1"
EVENT_EMIT_RETRIES = int(os.getenv("EVENT_EMIT_RETRIES", "3"))

# --- Phase 2: cheap recognition pipeline (per-camera, recognition_enabled only) ---
USE_GO2RTC_SOURCE = os.getenv("USE_GO2RTC_SOURCE", "0") == "1"
GO2RTC_RTSP_PORT = int(os.getenv("GO2RTC_RTSP_PORT", "8554"))
CV2_THREADS = int(os.getenv("CV2_THREADS", "1"))
DETECTION_INTERVAL = float(os.getenv("DETECTION_INTERVAL", "0.5"))
MOTION_ENABLED = os.getenv("MOTION_ENABLED", "1") == "1"
MOTION_WIDTH = int(os.getenv("MOTION_WIDTH", "160"))
MOTION_PIXEL_DIFF = int(os.getenv("MOTION_PIXEL_DIFF", "20"))
MOTION_AREA_RATIO = float(os.getenv("MOTION_AREA_RATIO", "0.02"))
MOTION_GRACE_SECONDS = float(os.getenv("MOTION_GRACE_SECONDS", "1.0"))
FACE_QUALITY_MIN_WIDTH = int(os.getenv("FACE_QUALITY_MIN_WIDTH", "60"))
FACE_QUALITY_MIN_SCORE = float(os.getenv("FACE_QUALITY_MIN_SCORE", "0.6"))
FACE_FRONTAL_CHECK = os.getenv("FACE_FRONTAL_CHECK", "1") == "1"
FACE_FRONTAL_MAX_ROLL_DEG = float(os.getenv("FACE_FRONTAL_MAX_ROLL_DEG", "20"))
FACE_FRONTAL_MIN_EYE_DIST = float(os.getenv("FACE_FRONTAL_MIN_EYE_DIST", "0.18"))
RECOGNITION_VOTES = int(os.getenv("RECOGNITION_VOTES", "3"))
RECOGNITION_VOTE_MAX = int(os.getenv("RECOGNITION_VOTE_MAX", "5"))
VERIFY_INTERVAL = float(os.getenv("VERIFY_INTERVAL", "4.0"))
IDENTITY_SWITCH_DISAGREEMENTS = int(os.getenv("IDENTITY_SWITCH_DISAGREEMENTS", "2"))
TRACK_IOU = float(os.getenv("TRACK_IOU", "0.3"))
TRACK_MAX_AGE = float(os.getenv("TRACK_MAX_AGE", "2.0"))

# --- Fall Detection Configuration ---
FALL_DETECTION_ENABLED = os.getenv("FALL_DETECTION_ENABLED", "1") == "1"
FALL_DETECTION_INTERVAL = float(os.getenv("FALL_DETECTION_INTERVAL", "1.0"))
FALL_ANGLE_THRESHOLD = float(os.getenv("FALL_ANGLE_THRESHOLD", "45"))
FALL_DURATION_THRESHOLD = float(os.getenv("FALL_DURATION_THRESHOLD", "3.0"))
FALL_CONFIDENCE_THRESHOLD = float(os.getenv("FALL_CONFIDENCE_THRESHOLD", "0.5"))
ALERT_SOUND_ENABLED = os.getenv("ALERT_SOUND_ENABLED", "1") == "1"
ALERT_SOUND_PATH = os.getenv("ALERT_SOUND_PATH", "")
ALERT_REPEAT_COUNT = int(os.getenv("ALERT_REPEAT_COUNT", "3"))

# Import fall detection modules
from fall_detector import FallDetector, FallState
from alert_sound import play_emergency_alert

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

# Cache of last recognition result per camera (single face approx tracking)
last_results: Dict[int, tuple] = {}

# Vectorized matcher state
embedding_matrix: Optional[np.ndarray] = None  # shape (N, D), rows normalized
embedding_labels: List[int] = []               # aligned with matrix rows
last_log: Dict[int, Dict[Any, tuple]] = {}     # camera_id -> fingerprint -> (emp_id, last_log_ts)

# Recognition work (embedding + snapshot + logging) is done off the frame loop
# so detection/streaming never blocks on the CNN or on Laravel HTTP calls.
recognition_queue: "queue.Queue" = queue.Queue(maxsize=64)  # Increased buffer

# Thread pool for parallel recognition processing
import concurrent.futures
recognition_executor = concurrent.futures.ThreadPoolExecutor(max_workers=RECOGNITION_WORKERS)

# YuNet & SFace masing-masing tidak thread-safe: akses setInputSize/detect (YuNet)
# dan feature (SFace) diserial-kan dengan lock terpisah supaya deteksi tidak
# diblokir recognition saat frame ramai (beberapa thread kamera + worker SFace).
yunet_lock = threading.Lock()
sface_lock = threading.Lock()


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
    recognition_enabled: bool = False


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
            if face_recognizer is not None and not RECOMPUTE_EMBEDDINGS_TRIED[0]:
                has_legacy = False
                for emp in data.get("employees", []):
                    for emb in emp.get("embeddings", []):
                        vec = emb.get("embedding")
                        if vec is None or len(vec) != SFACE_DIM:
                            has_legacy = True
                            break
                    if has_legacy:
                        break
                if has_legacy:
                    logger.info("Detected legacy or missing embeddings, recomputing with SFace ...")
                    RECOMPUTE_EMBEDDINGS_TRIED[0] = True
                    recompute_embeddings()
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


def recompute_embeddings():
    """Recompute stored photo embeddings with SFace when they use the old fallback format.

    Photos are fetched from the Laravel storage (same machine in dev), re-encoded with
    SFace, and persisted back via the Laravel API.
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
                if len(emb) == SFACE_DIM:
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
                    x, y, w, h = faces[0]
                    if w <= 0 or h <= 0:
                        failed += 1
                        continue
                    face_img = crop_face(frame, x, y, w, h)
                    embedding = extract_face_embedding_sface(face_img)
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


def detect_faces_yunet(frame: np.ndarray) -> List[tuple]:
    """Detect faces using YuNet (OpenCV 5.x).

    Lebih sensitif untuk kamera tinggi / wajah kecil:
    - ambang skor memakai YUNET_CONFIDENCE_THRESHOLD (.env, default 0.5);
    - frame diperbesar DETECT_UPSCALE sebelum dideteksi (koordinat dipetakan kembali).
    """
    if face_detector is None:
        return detect_faces_fallback(frame)

    scale = DETECT_UPSCALE
    det_frame = frame
    if scale > 1.0:
        dh, dw = frame.shape[:2]
        det_frame = cv2.resize(
            frame,
            (int(dw * scale), int(dh * scale)),
            interpolation=cv2.INTER_LINEAR,
        )

    # Kontras lokal (CLAHE) untuk video burik/low-light — hanya pada frame deteksi
    if DETECT_ENHANCE:
        try:
            ycrcb = cv2.cvtColor(det_frame, cv2.COLOR_BGR2YCrCb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
            det_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        except Exception as e:
            logger.debug(f"CLAHE enhance skipped: {e}")

    h, w = det_frame.shape[:2]
    with yunet_lock:
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(det_frame)

    results = []
    if faces is not None:
        for face in faces:
            x, y, w, h = face[:4].astype(int)
            confidence = face[14]
            if confidence > YUNET_CONFIDENCE_THRESHOLD:
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)
                if w > 10 and h > 10:
                    results.append((x, y, w, h))
    return results


def crop_face(frame: np.ndarray, x: int, y: int, w: int, h: int) -> np.ndarray:
    """Crop wajah dengan margin proporsional (lebih stabil untuk wajah kecil / miring)."""
    mx = int(w * FACE_CROP_MARGIN)
    my = int(h * FACE_CROP_MARGIN)
    x0 = max(0, x - mx)
    y0 = max(0, y - my)
    x1 = min(frame.shape[1], x + w + mx)
    y1 = min(frame.shape[0], y + h + my)
    if x1 <= x0 or y1 <= y0:
        return frame[y:y + h, x:x + w]
    return frame[y0:y1, x0:x1]


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

    return kept


def extract_face_embedding_sface(face_img: np.ndarray) -> np.ndarray:
    """Extract face embedding using SFace (OpenCV 5.x)"""
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

    aligned_face = cv2.resize(face_img, (112, 112))
    with sface_lock:
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


def _apply_credentials(url: str, username: Optional[str], password: Optional[str]) -> str:
    """Sisipkan kredensial RTSP tanpa menduplikasi userinfo yang sudah ada di URL.
    Kredensial terpisah (username/password) menang jika URL sudah mengandung userinfo lama."""
    if not url.startswith("rtsp://"):
        return url
    if not username or not password:
        return url
    parts = urlsplit(url)
    hostport = parts.netloc.rsplit("@", 1)[-1]
    user = quote(username, safe="")
    pwd = quote(password, safe="")
    return urlunsplit(parts._replace(netloc=f"{user}:{pwd}@{hostport}"))


def build_rtsp_url(camera: CameraConfig) -> str:
    """Build RTSP URL with credentials if provided"""
    return _apply_credentials(camera.rtsp_url, camera.username, camera.password)


def process_camera_stream(camera: CameraConfig, stop_event: threading.Event):
    """Process camera stream in background"""
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
    rec_enabled = bool(camera.recognition_enabled)
    cap = None
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    frames = 0
    last_scale = 1.0
    last_recognition = 0.0
    detect_frame_counter = 0
    cached_faces = []
    read_interval = 1.0 / STREAM_FPS
    last_read_time = 0.0

    while camera_streams.get(camera.id, {}).get("running", False) and not stop_event.is_set():
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

                cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
                reconnect_attempts += 1

                if not cap.isOpened():
                    logger.warning(f"Failed to open camera {camera.id}, retrying in {reconnect_interval}s")
                    time.sleep(reconnect_interval)
                    continue

                reconnect_attempts = 0
                logger.info(f"Camera {camera.id} connected successfully")

            # Throttle read ke STREAM_FPS agar CPU tidak decode semua frame kamera
            elapsed = now_ms() - last_read_time
            wait = read_interval - elapsed
            if wait > 0:
                time.sleep(wait)

            ret, frame = cap.read()
            if ret:
                last_read_time = now_ms()
            if not ret:
                logger.warning(f"Failed to read frame from camera {camera.id}")
                cap.release()
                cap = None
                time.sleep(0.5)
                continue

            frames += 1

            # Downscale large frames for cheaper detection + streaming (tiered)
            h, w = frame.shape[:2]
            if w > MAX_STREAM_WIDTH:
                last_scale = MAX_STREAM_WIDTH / float(w)
                new_h = int(h * last_scale)
                frame = cv2.resize(frame, (MAX_STREAM_WIDTH, new_h), interpolation=cv2.INTER_AREA)
            elif w > 640:  # Second tier: downscale to 640 for detection
                last_scale = 640.0 / float(w)
                new_h = int(h * last_scale)
                frame = cv2.resize(frame, (640, new_h), interpolation=cv2.INTER_AREA)
            else:
                last_scale = 1.0

            # Detect faces only every DETECT_EVERY_N_FRAMES to reduce CPU load
            detect_frame_counter += 1
            if detect_frame_counter >= DETECT_EVERY_N_FRAMES:
                detect_frame_counter = 0
                if face_detector is not None:
                    cached_faces = detect_faces_yunet(frame)
                else:
                    cached_faces = detect_faces_fallback(frame)
            faces = cached_faces

            # Run recognition off-thread, time-gated, to keep the video fluid.
            # Only cameras with recognition_enabled do recognition/logging; the
            # rest stay video-only (boxes are still drawn from cached_faces).
            run_recognition = rec_enabled and (frames % RECOGNITION_EVERY_N_FRAMES == 0)

            t = now_ms()

            # Buat SATU batch recognition per jendela gate: pilih wajah terbesar yang
            # memenuhi MIN_FACE_SIZE (buang noise video burik), maks RECOGNITION_MAX_FACES.
            batch_faces = []
            rec_frame = None
            if run_recognition and (t - last_recognition) >= RECOGNITION_INTERVAL:
                cands = [f for f in faces if f[2] >= MIN_FACE_SIZE and f[3] >= MIN_FACE_SIZE]
                cands.sort(key=lambda f: f[2] * f[3], reverse=True)
                batch_faces = cands[:RECOGNITION_MAX_FACES]
                if batch_faces:
                    rec_frame = frame.copy()
                    last_recognition = t

            for (x, y, w, h) in faces:
                face_img = crop_face(frame, x, y, w, h)
                if face_img.size == 0:
                    continue

                # Reuse the latest recognition result for this face (approx by position)
                emp_id, confidence = None, 0.0
                matched = False
                prev = last_results.get(camera.id)
                if prev is not None:
                    px, py, pw, ph, prev_emp, prev_conf, prev_ts = prev
                    if (abs(px - x) <= w * 1.5 and abs(py - y) <= h * 1.5
                            and (t - prev_ts) < 3.0):
                        emp_id, confidence = prev_emp, prev_conf
                        matched = True

                # Enqueue hanya untuk wajah terpilih di batch ini; skip bila hasil
                # cache masih segar (kotak belum banyak bergerak).
                if rec_frame is not None and (x, y, w, h) in batch_faces:
                    if not (matched and (t - prev_ts) < 1.5):
                        try:
                            recognition_queue.put_nowait((
                                camera.id, rec_frame,
                                int(x), int(y), int(w), int(h), t
                            ))
                        except queue.Full:
                            logger.warning(f"Recognition queue full for camera {camera.id}, dropping task")

                # Draw bounding box + label immediately (no waiting on the CNN)
                status = "recognized" if emp_id else "unknown"
                emp_data = employee_data_cache.get(emp_id, {}) if emp_id else {}
                name = emp_data.get("name", "")
                color = (0, 255, 0) if status == "recognized" else (0, 0, 255)
                label = f"{name or 'Unknown'} ({confidence:.2f})"
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                cv2.putText(frame, label, (x, max(20, y-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # Store cached JPEG for MJPEG streaming (encode once, serve many clients)
            ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if ok:
                camera_streams[camera.id]["latest_jpg"] = buf.tobytes()
            camera_streams[camera.id]["latest_frame"] = frame
            camera_streams[camera.id]["last_update"] = now_ms()

            if frames % 300 == 0:
                prune_last_log()

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
    """Consumer for both recognition queues (old video-loop path + v2 pipeline).
    Runs off the camera frame loop so CNN / HTTP never stall the video."""
    queues = (
        (recognition_queue, process_recognition_task),
        (recognition_queue_v2, process_recognition_v2_task),
    )
    while True:
        for q, fn in queues:
            try:
                task = q.get(timeout=0.2)
                if task is None:
                    return
                recognition_executor.submit(fn, task)
                break
            except queue.Empty:
                continue


def process_recognition_task(task):
    """Process a single recognition task (runs in thread pool)"""
    camera_id, frame, x, y, w, h, ts = task
    try:
        face_img = frame[y:y+h, x:x+w]
        if face_img.size == 0:
            return

        embedding = extract_face_embedding_sface(face_img)
        emp_id, confidence = recognize_face(embedding)
        last_results[camera_id] = (x, y, w, h, emp_id, confidence, ts)

        timestamp = datetime.now().isoformat()
        status = "recognized" if emp_id else "unknown"
        emp_data = employee_data_cache.get(emp_id, {}) if emp_id else {}
        name = emp_data.get("name", "")

        # Logging dedup: limit writes to Laravel DB
        fp = ("emp", emp_id) if emp_id else ("unk", int(x // 64), int(y // 64))
        if should_log(camera_id, fp, emp_id, ts):
            snapshot_path = save_snapshot(camera_id, frame, status, timestamp)
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
        "latest_frame": None,
        "last_update": None,
    }
    if camera.recognition_enabled:
        # Recognition cameras use the cheap v2 pipeline: a dedicated reader
        # thread + the shared analyzer loop. Video path stays intact.
        state = CameraAIState()
        with ai_states_lock:
            ai_states[camera.id] = state
        thread = threading.Thread(target=process_recognition_camera, args=(camera, stop_event, state), daemon=True)
    else:
        # Video-only cameras keep the classic loop, now without recognition.
        with ai_states_lock:
            ai_states.pop(camera.id, None)
        thread = threading.Thread(target=process_camera_stream, args=(camera, stop_event), daemon=True)
    thread.start()
    camera_streams[camera.id]["thread"] = thread
    ensure_analyzer()


# ===========================================================================
# Phase 2: cheap per-camera recognition pipeline (recognition_enabled only)
# Video path (/stream, /snapshot, go2rtc) is intentionally NOT touched.
# 1 reader thread per enabled camera: reads continuously, keeps ONLY the
#   latest frame in a small slot (never sleeps; only the AI is throttled).
# 1 shared analyzer loop: round-robins enabled cameras, cheap motion gate,
#   YuNet at DETECTION_INTERVAL, an IoU tracker, and "recognize-once" per
#   track with majority vote + periodic verify. Embedding runs in the shared
#   recognition_executor (synchronized by sface_lock).
# ===========================================================================

UNKNOWN_IDENTITY = -1  # sentinel for a committed-but-unmatched person


class LatestFrameSlot:
    """Thread-safe single-frame slot (reader writes, analyzer reads latest)."""

    def __init__(self):
        self._frame = None
        self._ts = 0.0
        self._seq = 0
        self._lock = threading.Lock()

    def set(self, frame, ts):
        with self._lock:
            self._frame = frame
            self._ts = ts
            self._seq += 1

    def get(self):
        with self._lock:
            if self._frame is None:
                return None, 0.0, 0
            return self._frame, self._ts, self._seq


class StageStats:
    """Rolling per-stage timing statistics (avg/max over a window)."""

    def __init__(self, window: int = 60):
        self._data: Dict[str, list] = {}
        self._window = window
        self._lock = threading.Lock()

    def add(self, stage: str, ms: float):
        with self._lock:
            q = self._data.setdefault(stage, [])
            q.append(ms)
            if len(q) > self._window:
                q.pop(0)

    def snapshot(self) -> Dict[str, tuple]:
        with self._lock:
            out = {}
            for stage, q in self._data.items():
                out[stage] = (sum(q) / len(q), max(q), len(q))
            return out

    def counts(self) -> Dict[str, int]:
        with self._lock:
            return {k: len(v) for k, v in self._data.items()}


class Track:
    # committed: None = uncommitted, int = employee id, UNKNOWN_IDENTITY = unknown
    __slots__ = (
        "id", "bbox", "last_seen", "born", "votes", "committed", "committed_sim",
        "flip_tally", "last_attempt", "last_verify", "in_flight",
    )

    def __init__(self, track_id: int, bbox, now: float):
        self.id = track_id
        self.bbox = bbox
        self.last_seen = now
        self.born = now
        self.votes: List[tuple] = []           # list of (emp_id|None, sim)
        self.committed: Optional[Any] = None
        self.committed_sim: float = 0.0
        self.flip_tally = 0
        self.last_attempt = 0.0
        self.last_verify = 0.0
        self.in_flight = False


class CameraAIState:
    def __init__(self):
        self.slot = LatestFrameSlot()
        self.lock = threading.Lock()
        self.tracks: Dict[int, Track] = {}
        self.next_track_id = 1
        self.prev_small = None
        self.motion_seen_at = 0.0
        self.last_analyzed_at = 0.0
        self.last_jpg_ts = 0.0
        self.frames_pushed = 0
        self.analyze_frames = 0
        self.summary_ts = 0.0
        self.stats = StageStats()
        self.src_fps_queue = collections.deque(maxlen=100)
        self.last_rows: List[np.ndarray] = []  # latest raw detections (overlay)


# state per camera id; analyzer iterates this (guarded by ai_states_lock)
ai_states: Dict[int, CameraAIState] = {}
ai_states_lock = threading.Lock()
_analyzer_started = False
_analyzer_start_lock = threading.Lock()

# --- Fall Detection ---
# Global fall detector (shared across cameras for resource efficiency)
fall_detector: Optional[FallDetector] = None
# Queue for fall detection tasks
fall_detection_queue: "queue.Queue" = queue.Queue(maxsize=32)
# Track which cameras have had fall alerts (cooldown)
fall_alert_cooldown: Dict[int, float] = {}  # camera_id -> last_alert_timestamp
FALL_ALERT_COOLDOWN_SECONDS = 30.0  # Don't spam same camera


def _recognition_source_url(camera: CameraConfig) -> str:
    """AI source: go2rtc local restream cam{ID}_sub when USE_GO2RTC_SOURCE=1,
    otherwise the raw camera URL."""
    if USE_GO2RTC_SOURCE:
        return f"rtsp://127.0.0.1:{GO2RTC_RTSP_PORT}/cam{camera.id}_sub"
    return build_rtsp_url(camera)


def _downscale_for_ai(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > MAX_STREAM_WIDTH:
        scale = MAX_STREAM_WIDTH / float(w)
        return cv2.resize(frame, (MAX_STREAM_WIDTH, int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def _motion_gate(state: CameraAIState, frame: np.ndarray) -> bool:
    """Cheap frame-difference at ~160px width. Pure pixel work, no CNN."""
    h, w = frame.shape[:2]
    small_w = max(MOTION_WIDTH, 32)
    small_h = max(32, int(h * small_w / float(max(w, 1))))
    small = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    prev = state.prev_small
    state.prev_small = gray.copy()
    if prev is None:
        return False
    diff = cv2.absdiff(gray, prev)
    ratio = float(np.count_nonzero(diff > MOTION_PIXEL_DIFF)) / diff.size
    motion = ratio > MOTION_AREA_RATIO
    if motion:
        state.motion_seen_at = time.perf_counter()
    return motion


def detect_faces_yunet_v2(frame: np.ndarray) -> List[np.ndarray]:
    """YuNet returning FULL rows (bbox + 5 landmarks + score), coordinates in
    the original frame space. Used by the recognition pipeline."""
    if face_detector is None:
        return []
    scale = DETECT_UPSCALE
    det_frame = frame
    if scale > 1.0:
        dh, dw = frame.shape[:2]
        det_frame = cv2.resize(frame, (int(dw * scale), int(dh * scale)), interpolation=cv2.INTER_LINEAR)
    if DETECT_ENHANCE:
        try:
            ycrcb = cv2.cvtColor(det_frame, cv2.COLOR_BGR2YCrCb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
            det_frame = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        except Exception:
            pass

    h, w = det_frame.shape[:2]
    with yunet_lock:
        face_detector.setInputSize((w, h))
        _, faces = face_detector.detect(det_frame)

    rows = []
    if faces is not None:
        for face in faces:
            if face[14] <= YUNET_CONFIDENCE_THRESHOLD:
                continue
            row = face.copy()
            if scale > 1.0:
                row[0:14] /= scale
            x, y, fw, fh = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            if fw <= 10 or fh <= 10 or x < 0 or y < 0:
                continue
            rows.append(row)
    return rows


def _frontal_ok(row: np.ndarray) -> bool:
    """Rough frontal-ness using the two eye landmarks. Rejects heavy profiles."""
    if not FACE_FRONTAL_CHECK:
        return True
    w = float(row[2])
    if w <= 1:
        return True
    rex, rey = float(row[4]), float(row[5])
    lex, ley = float(row[6]), float(row[7])
    dx = lex - rex
    dy = ley - rey
    eye_dist = math.hypot(dx, dy)
    if eye_dist < FACE_FRONTAL_MIN_EYE_DIST * w:  # eyes ~same x => profile
        return False
    roll = abs(math.degrees(math.atan2(dy, dx)))
    return roll <= FACE_FRONTAL_MAX_ROLL_DEG


def _face_quality_ok(row: np.ndarray) -> bool:
    """Skip faces that are too small / low-score / not frontal. Never waste an
    embedding on a bad face. Recognition happens on the live frame, so these
    values are at display resolution (typically 640px wide)."""
    if row[2] < FACE_QUALITY_MIN_WIDTH or row[3] < FACE_QUALITY_MIN_WIDTH:
        return False
    if row[14] < FACE_QUALITY_MIN_SCORE:
        return False
    return _frontal_ok(row)


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1, y1 = max(ax, bx), max(ay, by)
    x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _update_tracks(state: CameraAIState, rows: List[np.ndarray], now: float):
    """Assign detections to tracks by IoU/centroid; prune stale tracks."""
    with state.lock:
        det_rows = sorted(rows, key=lambda r: r[2] * r[3], reverse=True)
        used = set()
        assigned: Dict[int, np.ndarray] = {}
        for row in det_rows:
            box = (int(row[0]), int(row[1]), int(row[2]), int(row[3]))
            best_id, best_iou = None, TRACK_IOU
            for tid, tr in state.tracks.items():
                if tid in used:
                    continue
                iou = _iou(box, tr.bbox)
                if iou > best_iou:
                    best_id, best_iou = tid, iou
            if best_id is None:
                tid = state.next_track_id
                state.next_track_id += 1
                state.tracks[tid] = Track(tid, box, now)
                best_id = tid
            else:
                tr = state.tracks[best_id]
                tr.bbox = box
                tr.last_seen = now
                used.add(best_id)
            assigned[best_id] = row

        stale = [tid for tid, tr in state.tracks.items() if (now - tr.last_seen) > TRACK_MAX_AGE]
        for tid in stale:
            del state.tracks[tid]

        _schedule_recognition(state, assigned, now)


def _aligned_face(frame: np.ndarray, row: np.ndarray) -> Optional[np.ndarray]:
    """alignCrop (face alignment) when SFace is available; fallback to a padded
    crop resize. Always returns a 112x112 BGR patch for SFace.embedding."""
    if face_recognizer is not None:
        try:
            with sface_lock:
                aligned = face_recognizer.alignCrop(frame, row.reshape(1, -1))
            return aligned
        except Exception:
            pass
    x, y, w, h = int(row[0]), int(row[1]), int(row[2]), int(row[3])
    face_img = crop_face(frame, x, y, w, h)
    if face_img.size == 0:
        return None
    return cv2.resize(face_img, (112, 112))


def _schedule_recognition(state: CameraAIState, assigned: Dict[int, np.ndarray], now: float):
    """Queue recognition for eligible tracks: uncommitted (vote collecting) or
    due for re-verify. Only one attempt in flight per track at a time."""
    for tid, row in assigned.items():
        if not _face_quality_ok(row):
            continue
        tr = state.tracks.get(tid)
        if tr is None or tr.in_flight:
            continue
        if tr.committed is None:
            if now - tr.last_attempt < (DETECTION_INTERVAL * 2.0):
                continue
        else:
            if now - tr.last_verify < VERIFY_INTERVAL:
                continue
        try:
            recognition_queue_v2.put_nowait((state, tid, row))
            tr.in_flight = True
            tr.last_attempt = now
        except queue.Full:
            logger.warning(f"[ai] recognition v2 queue full, dropping task for track {tid}")


recognition_queue_v2: "queue.Queue" = queue.Queue(maxsize=64)

# Bounded queue of events to emit into Laravel. A dedicated emitter thread
# POSTs them so the recognition pipeline never blocks on Laravel (constraint:
# python must never block on laravel). Dropping events under overload is
# acceptable; the rules engine cooldown in Laravel is the dedup authority.
event_emit_queue: "queue.Queue" = queue.Queue(maxsize=64)


def _emit_recognition_event(state: CameraAIState, tr: Track, frame: np.ndarray):
    """Queue a recognition event (identity just committed or changed) for the
    emitter thread. Produces the facts only; Laravel decides what's an alarm."""
    if not EVENT_EMIT_ENABLED:
        return
    cam_id = next((cid for cid, st in ai_states.items() if st is state), None)
    if cam_id is None or tr.committed is None:
        return

    emp_id = None if tr.committed == UNKNOWN_IDENTITY else tr.committed
    event_type = "known" if emp_id is not None else "unknown"

    # bbox crop with margin for the snapshot (same visual as the old pipeline)
    x, y, w, h = (int(v) for v in tr.bbox)
    margin = FACE_CROP_MARGIN
    cw = int(w * (1 + 2 * margin))
    ch = int(h * (1 + 2 * margin))
    fx = max(0, x + w // 2 - cw // 2)
    fy = max(0, y + h // 2 - ch // 2)
    fx2 = min(frame.shape[1], fx + cw)
    fy2 = min(frame.shape[0], fy + ch)
    crop = frame[fy:fy2, fx:fx2]
    ok, buf = cv2.imencode('.jpg', crop, [cv2.IMWRITE_JPEG_QUALITY, 90]) if crop.size else (False, None)

    item = {
        "event_uuid": uuid.uuid4().hex,
        "type": event_type,
        "camera_id": cam_id,
        "employee_id": emp_id,
        "similarity": round(float(tr.committed_sim), 4),
        "track_id": f"{cam_id}t{tr.id}",
        "bbox": [x, y, w, h],
        "occurred_at": datetime.now(timezone.utc),
        "snapshot_bytes": buf.tobytes() if ok else None,
    }
    try:
        event_emit_queue.put_nowait(item)
        logger.info(
            f"[emit] queued camera={cam_id} track={tr.id} type={event_type} "
            f"emp={emp_id} sim={item['similarity']:.3f}"
        )
    except queue.Full:
        logger.warning("[emit] queue full, dropping event (pipeline must stay responsive)")


def event_emitter_worker():
    """Persistence thread: emits recognition events to Laravel with retry."""
    while True:
        item = event_emit_queue.get()
        if item is None:
            break
        _send_recognition_event(item)


def _send_recognition_event(item: dict):
    snap_bytes = item.pop("snapshot_bytes", None)
    payload = {k: v for k, v in item.items() if v is not None}
    payload["occurred_at"] = item["occurred_at"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    body = []
    for key, val in payload.items():
        if key == "bbox" and isinstance(val, (list, tuple)):
            for box_val in val:
                body.append(("bbox[]", float(box_val)))
        else:
            body.append((key, val))
    url = f"{LARAVEL_API_URL.rstrip('/')}/internal/recognition-events"
    headers = {"X-Internal-Token": LARAVEL_INTERNAL_TOKEN} if LARAVEL_INTERNAL_TOKEN else {}
    files = {"snapshot": ("face.jpg", snap_bytes, "image/jpeg")} if snap_bytes else None

    for attempt in range(EVENT_EMIT_RETRIES):
        try:
            resp = requests.post(url, data=body, files=files, headers=headers, timeout=15)
            if resp.status_code in (200, 201):
                return True
            logger.warning(
                f"[emit] attempt {attempt + 1}/{EVENT_EMIT_RETRIES} HTTP {resp.status_code}: {resp.text[:200]}"
            )
        except Exception as e:
            logger.warning(f"[emit] attempt {attempt + 1}/{EVENT_EMIT_RETRIES} error: {e}")
        time.sleep(2 * (attempt + 1))
    logger.error(f"[emit] FAILED camera={payload.get('camera_id')} type={payload.get('type')} uuid={payload.get('event_uuid')}")
    return False


def _commit_track(state: CameraAIState, tr: Track, now: float, frame: np.ndarray):
    """Vote logic for a freshly collected track. Only ambiguous 'all below
    threshold' or clear-majority results ever commit; anything unclear keeps
    collecting up to RECOGNITION_VOTE_MAX."""
    known = [(e, s) for e, s in tr.votes if e is not None]
    if len(known) >= math.ceil(RECOGNITION_VOTES / 2.0):
        counts: Dict[int, int] = {}
        sims: Dict[int, float] = {}
        for e, s in known:
            counts[e] = counts.get(e, 0) + 1
            sims[e] = sims.get(e, 0.0) + s
        emp = max(counts, key=lambda e: (counts[e], sims[e] / counts[e]))
        tr.committed = emp
        tr.committed_sim = sims[emp] / counts[emp]
        tr.votes = []
        tr.flip_tally = 0
        _notify_identity(state, tr, frame)
        return
    if len(tr.votes) >= RECOGNITION_VOTES and len(known) == 0:
        # N good attempts, every one below threshold -> genuine unknown
        tr.committed = UNKNOWN_IDENTITY
        tr.committed_sim = max((s for _, s in tr.votes), default=0.0)
        tr.votes = []
        tr.flip_tally = 0
        _notify_identity(state, tr, frame)
        return
    if len(tr.votes) >= RECOGNITION_VOTE_MAX:
        # mixed votes, no clear majority -> drop, stay uncommitted (no event)
        tr.votes = []


def _verify_track(state: CameraAIState, tr: Track, result, now: float, frame: np.ndarray):
    """Re-verify a committed track. Identity only changes after consecutive
    disagreement (never on a single bad frame)."""
    emp_id, sim = result
    same = (
        (emp_id is not None and tr.committed not in (None, UNKNOWN_IDENTITY) and emp_id == tr.committed)
        or (emp_id is None and tr.committed == UNKNOWN_IDENTITY)
    )
    if same:
        tr.flip_tally = 0
        if emp_id == tr.committed:
            tr.committed_sim = sim
        return
    tr.flip_tally += 1
    if tr.flip_tally >= IDENTITY_SWITCH_DISAGREEMENTS:
        tr.committed = emp_id if emp_id is not None else UNKNOWN_IDENTITY
        tr.committed_sim = sim if emp_id is not None else 0.0
        tr.flip_tally = 0
        tr.votes = []
        _notify_identity(state, tr, frame)


def process_recognition_v2_task(task):
    """Runs in the shared recognition executor. Align+embed+match happen OUTSIDE
    the state lock; commit/verify under it. Timings feed the 10s summary."""
    state, tid, row = task
    cam_id = None
    for cid, st in list(ai_states.items()):
        if st is state:
            cam_id = cid
            break

    frame, _, _ = state.slot.get()
    if frame is None:
        with state.lock:
            tr = state.tracks.get(tid)
            if tr is not None:
                tr.in_flight = False
        return

    try:
        t_align = time.perf_counter()
        aligned = _aligned_face(frame, row)
        if aligned is None:
            with state.lock:
                tr = state.tracks.get(tid)
                if tr is not None:
                    tr.in_flight = False
            return
        t_embed = time.perf_counter()
        embedding = extract_face_embedding_sface(aligned)
        t_match = time.perf_counter()
        emp_id, sim = recognize_face(embedding)
        t_end = time.perf_counter()

        state.stats.add("align", (t_embed - t_align) * 1000)
        state.stats.add("embed", (t_match - t_embed) * 1000)
        state.stats.add("match", (t_end - t_match) * 1000)

        now = time.perf_counter()
        with state.lock:
            tr = state.tracks.get(tid)
            if tr is None:
                return
            tr.in_flight = False
            if tr.committed is None:
                tr.votes.append((emp_id, float(sim)))
                _commit_track(state, tr, now, frame)
            else:
                _verify_track(state, tr, (emp_id, float(sim)), now, frame)
    except Exception as e:
        with state.lock:
            tr = state.tracks.get(tid)
            if tr is not None:
                tr.in_flight = False
        logger.error(f"Recognition v2 worker error camera={cam_id} track={tid}: {e}")


def _notify_identity(state: CameraAIState, tr: Track, frame: np.ndarray):
    """Identity committed or changed -> log + enqueue event for Laravel."""
    cam_id = next((cid for cid, st in ai_states.items() if st is state), None)
    if cam_id is None:
        return
    known = tr.committed not in (None, UNKNOWN_IDENTITY)
    label = "unknown" if not known else f"employee_{tr.committed}"
    logger.info(
        f"[ai] camera={cam_id} track={tr.id} identity committed -> {label} "
        f"(sim={tr.committed_sim:.3f})"
    )
    _emit_recognition_event(state, tr, frame)


def _analyze_camera(state: CameraAIState, now: float):
    frame, ts, _ = state.slot.get()
    if frame is None:
        return

    t_detect0 = time.perf_counter()
    motion = _motion_gate(state, frame) if MOTION_ENABLED else True

    # warmup: run detection on the first frames right after start (static scene)
    warmup = state.analyze_frames < 2
    do_detect = (now - state.last_analyzed_at) >= DETECTION_INTERVAL and (
        warmup or motion or (now - state.motion_seen_at) < MOTION_GRACE_SECONDS or not MOTION_ENABLED
    )
    if not do_detect:
        return

    state.last_analyzed_at = now
    rows = detect_faces_yunet_v2(frame)
    state.stats.add("detect", (time.perf_counter() - t_detect0) * 1000)
    state.analyze_frames += 1
    _update_tracks(state, rows, now)
    state.last_rows = rows  # overlay source for the video stream

    # --- Fall Detection (if enabled) ---
    if FALL_DETECTION_ENABLED:
        _analyze_fall_detection(state, frame, now)


def analyzer_loop(stop_event: threading.Event):
    """Single shared loop: round-robins enabled cameras over their LATEST frame
    slots so one slow camera never stalls another."""
    while not stop_event.is_set():
        with ai_states_lock:
            cam_ids = list(ai_states.keys())
        for cam_id in cam_ids:
            state = ai_states.get(cam_id)
            if state is None:
                continue
            if not camera_streams.get(cam_id, {}).get("running", False):
                continue
            now = time.perf_counter()
            _analyze_camera(state, now)

            # 10-second summary per camera
            if now - state.summary_ts >= 10.0:
                state.summary_ts = now
                _print_summary(cam_id, state)
        time.sleep(0.005)


def _print_summary(cam_id: int, state: CameraAIState):
    snap = state.stats.snapshot()
    def fmt(stage):
        avg, mx, n = snap.get(stage, (0.0, 0.0, 0))
        return f"{stage}={avg:.1f}/{mx:.1f}ms" if n else f"{stage}=--"

    now = time.perf_counter()
    with state.lock:
        n_tracks = len(state.tracks)
    if len(state.src_fps_queue) >= 2:
        src_fps = (len(state.src_fps_queue) - 1) / (state.src_fps_queue[-1] - state.src_fps_queue[0])
    else:
        src_fps = 0.0
    fps = state.analyze_frames / 10.0 if state.analyze_frames else 0.0
    logger.info(
        f"[ai-summary] cam{cam_id} source={src_fps:.1f}fps ai={fps:.1f}fps tracks={n_tracks} "
        f"{fmt('read')} {fmt('detect')} {fmt('align')} {fmt('embed')} {fmt('match')}"
    )


# ===========================================================================
# Fall Detection Functions
# ===========================================================================

def _init_fall_detector():
    """Initialize global fall detector."""
    global fall_detector
    if fall_detector is None and FALL_DETECTION_ENABLED:
        fall_detector = FallDetector(
            angle_threshold=FALL_ANGLE_THRESHOLD,
            duration_threshold=FALL_DURATION_THRESHOLD,
            confidence_threshold=FALL_CONFIDENCE_THRESHOLD,
            enabled=True
        )
        logger.info(f"Fall detector initialized: angle={FALL_ANGLE_THRESHOLD}°, duration={FALL_DURATION_THRESHOLD}s")


def _analyze_fall_detection(state: CameraAIState, frame: np.ndarray, now: float):
    """Analyze frame for fall detection."""
    global fall_detector, fall_alert_cooldown

    if fall_detector is None:
        return

    # Get camera_id from state
    cam_id = None
    for cid, st in ai_states.items():
        if st is state:
            cam_id = cid
            break
    if cam_id is None:
        return

    # Check cooldown
    last_alert = fall_alert_cooldown.get(cam_id, 0)
    if (now - last_alert) < FALL_ALERT_COOLDOWN_SECONDS:
        return

    try:
        h, w = frame.shape[:2]
        landmarks = fall_detector.detect_pose(frame)

        if landmarks is not None:
            events = fall_detector.update(landmarks, h, w)

            for event in events:
                if event.get("type") == "fall":
                    logger.warning(f"[FALL] Camera {cam_id}: {event}")
                    _emit_emergency_event(cam_id, frame, event)
                    fall_alert_cooldown[cam_id] = now

    except Exception as e:
        logger.debug(f"Fall detection error for camera {cam_id}: {e}")


def _emit_emergency_event(camera_id: int, frame: np.ndarray, event_data: dict):
    """Emit emergency event to Laravel and play audio alert."""
    try:
        # Create snapshot
        ok, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85]) if frame.size else (False, None)

        item = {
            "event_uuid": uuid.uuid4().hex,
            "camera_id": camera_id,
            "type": event_data.get("type", "fall"),
            "severity": event_data.get("severity", "critical"),
            "confidence": round(event_data.get("confidence", 0.5), 3),
            "fallen_duration": round(event_data.get("fallen_duration", 0), 2),
            "body_angle": round(event_data.get("body_angle", 0), 1),
            "track_id": f"fall_{camera_id}_{event_data.get('track_id', 1)}",
            "bbox": event_data.get("bbox"),
            "occurred_at": datetime.now(timezone.utc),
            "snapshot_bytes": buf.tobytes() if ok else None,
        }

        # Queue for emitter thread
        try:
            emergency_emit_queue.put_nowait(item)
            logger.info(f"[emergency] Event queued for camera {camera_id}")
        except queue.Full:
            logger.warning("[emergency] Queue full, dropping event")

        # Play audio alert immediately (in background thread)
        if ALERT_SOUND_ENABLED:
            threading.Thread(
                target=play_emergency_alert,
                args=(ALERT_REPEAT_COUNT,),
                kwargs={"sound_path": ALERT_SOUND_PATH or None, "enabled": ALERT_SOUND_ENABLED},
                daemon=True
            ).start()

    except Exception as e:
        logger.error(f"[emergency] Failed to emit event: {e}")


def _send_emergency_event(item: dict):
    """Send emergency event to Laravel API."""
    snap_bytes = item.pop("snapshot_bytes", None)
    payload = {k: v for k, v in item.items() if v is not None}
    payload["occurred_at"] = item["occurred_at"].strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    body = []
    for key, val in payload.items():
        if key == "bbox" and isinstance(val, (list, tuple)):
            for box_val in val:
                body.append(("bbox[]", float(box_val)))
        else:
            body.append((key, val))

    url = f"{LARAVEL_API_URL.rstrip('/')}/internal/emergency-events"
    headers = {"X-Internal-Token": LARAVEL_INTERNAL_TOKEN} if LARAVEL_INTERNAL_TOKEN else {}
    files = {"snapshot": ("emergency.jpg", snap_bytes, "image/jpeg")} if snap_bytes else None

    for attempt in range(EVENT_EMIT_RETRIES):
        try:
            resp = requests.post(url, data=body, files=files, headers=headers, timeout=15)
            if resp.status_code in (200, 201):
                logger.info(f"[emergency] Event sent successfully to Laravel")
                return True
            logger.warning(f"[emergency] HTTP {resp.status_code}: {resp.text[:200]}")
        except Exception as e:
            logger.warning(f"[emergency] attempt {attempt + 1}/{EVENT_EMIT_RETRIES} error: {e}")
        time.sleep(2 * (attempt + 1))

    logger.error(f"[emergency] FAILED to send event uuid={payload.get('event_uuid')}")
    return False


# Queue and worker for emergency events
emergency_emit_queue: "queue.Queue" = queue.Queue(maxsize=32)


def emergency_emitter_worker():
    """Worker thread to send emergency events to Laravel."""
    while True:
        item = emergency_emit_queue.get()
        if item is None:
            break
        _send_emergency_event(item)


def ensure_analyzer():
    global _analyzer_started
    with _analyzer_start_lock:
        if _analyzer_started:
            return True
        th = threading.Thread(target=analyzer_loop, args=(threading.Event(),), daemon=True)
        th.start()
        _analyzer_started = True
        return True


def process_recognition_camera(camera: CameraConfig, stop_event: threading.Event, state: CameraAIState):
    """Reader thread for recognition cameras: reads continuously (no sleep),
    keeps only the latest frame, and still feeds the MJPEG snapshot path."""
    rtsp_url = _recognition_source_url(camera)
    logger.info(f"[ai] camera {camera.id} reader start ({rtsp_url})")

    conn_url = rtsp_url
    if rtsp_url.startswith("rtsp://"):
        conn_url = rtsp_url + "?tcp"

    reconnect_interval = camera.reconnect_interval or CAMERA_RECONNECT_INTERVAL
    cap = None
    reconnect_attempts = 0
    max_reconnect_attempts = 10

    while camera_streams.get(camera.id, {}).get("running", False) and not stop_event.is_set():
        try:
            if cap is None or not cap.isOpened():
                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error(f"[ai] camera {camera.id} max reconnect attempts reached")
                    break
                cap = cv2.VideoCapture(conn_url)
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, RTSP_OPEN_TIMEOUT_MS)
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, RTSP_READ_TIMEOUT_MS)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                reconnect_attempts += 1
                if not cap.isOpened():
                    logger.warning(f"[ai] camera {camera.id} open failed, retrying in {reconnect_interval}s")
                    time.sleep(reconnect_interval)
                    continue
                reconnect_attempts = 0

            t_read0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                logger.warning(f"[ai] camera {camera.id} read failed")
                cap.release()
                cap = None
                time.sleep(0.5)
                continue
            state.stats.add("read", (time.perf_counter() - t_read0) * 1000)

            frame = _downscale_for_ai(frame)
            now = time.perf_counter()
            state.slot.set(frame, now)
            state.frames_pushed += 1
            state.src_fps_queue.append(now)

            # MJPEG fallback feed at STREAM_FPS (video path unchanged)
            if now - state.last_jpg_ts >= 1.0 / STREAM_FPS:
                state.last_jpg_ts = now
                draw_frame = frame.copy()
                with state.lock:
                    active_tracks = list(state.tracks.values())
                for tr in active_tracks:
                    x, y, w, h = (int(v) for v in tr.bbox)
                    if tr.committed in (None, UNKNOWN_IDENTITY):
                        # unknown or still collecting votes -> red
                        color = (0, 0, 255)
                        emp_id = tr.committed
                        label = "Unknown" if tr.committed == UNKNOWN_IDENTITY else "Tracking…"
                    else:
                        color = (0, 255, 0)
                        emp_id = tr.committed
                        emp_data = employee_data_cache.get(emp_id, {})
                        label = f"{emp_data.get('name', f'employee_{emp_id}')} ({tr.committed_sim:.2f})"
                    cv2.rectangle(draw_frame, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(draw_frame, label, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                ok, buf = cv2.imencode('.jpg', draw_frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    camera_streams[camera.id]["latest_jpg"] = buf.tobytes()
                camera_streams[camera.id]["latest_frame"] = draw_frame
                camera_streams[camera.id]["last_update"] = now

        except Exception as e:
            logger.error(f"[ai] camera {camera.id} reader error: {e}")
            if cap:
                cap.release()
                cap = None
            time.sleep(reconnect_interval)

    if cap:
        cap.release()
    logger.info(f"[ai] camera {camera.id} reader stopped")


# Lifespan handlers (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    cv2.setNumThreads(CV2_THREADS)  # keep OpenCV from spawning extra threads (dev i5 2c/4t)
    init_face_models()
    load_employee_embeddings()

    # Initialize Fall Detector
    if FALL_DETECTION_ENABLED:
        _init_fall_detector()
        logger.info("Fall detection enabled")

    # Recognition worker (embedding / snapshot / logging off the frame loop)
    recognition_thread = threading.Thread(target=recognition_worker, daemon=True)
    recognition_thread.start()

    # Event emitter: recognition events -> Laravel internal API (off the pipeline)
    if EVENT_EMIT_ENABLED:
        emitter_thread = threading.Thread(target=event_emitter_worker, daemon=True)
        emitter_thread.start()

    # Emergency emitter: fall detection events -> Laravel internal API
    if FALL_DETECTION_ENABLED and EVENT_EMIT_ENABLED:
        emergency_thread = threading.Thread(target=emergency_emitter_worker, daemon=True)
        emergency_thread.start()
        logger.info("Emergency event emitter started")

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
    if EVENT_EMIT_ENABLED:
        try:
            event_emit_queue.put_nowait(None)
        except queue.Full:
            pass
    # Shutdown emergency queue
    try:
        emergency_emit_queue.put_nowait(None)
    except queue.Full:
        pass
    # Shutdown thread pool
    recognition_executor.shutdown(wait=True)


app = FastAPI(
    title="Facial Recognition CCTV Service",
    description="Computer Vision service for face detection and recognition",
    version="1.0.0",
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
    fall_stats = fall_detector.get_stats() if fall_detector else {"enabled": False}
    return {
        "status": "healthy",
        "service": "facial-recognition",
        "version": "1.1.0",
        "models": {
            "yunet": face_detector is not None,
            "sface": face_recognizer is not None,
        },
        "fall_detection": {
            "enabled": FALL_DETECTION_ENABLED,
            "status": "active" if fall_detector else "disabled",
            "stats": fall_stats
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
        with ai_states_lock:
            ai_states.pop(camera_id, None)
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
            if jpg is None:
                frame = camera_streams[camera_id].get("latest_frame")
                if frame is not None:
                    ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
                    if ok:
                        jpg = buffer.tobytes()
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
        frame = camera_streams[camera_id].get("latest_frame")
        if frame is None:
            raise HTTPException(status_code=503, detail="No frame available")
        ok, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
        if not ok:
            raise HTTPException(status_code=500, detail="Encode failed")
        jpg = buffer.tobytes()
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
        for (x, y, w, h) in faces:
            x, y, w, h = int(x), int(y), int(w), int(h)
            face_img = crop_face(frame, x, y, w, h)
            if face_img.size == 0:
                continue

            embedding = extract_face_embedding_sface(face_img)
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
    import base64

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

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "FACE", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        output_path = Path("test_results") / f"test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(output_path), frame)

        return {
            "faces_detected": len(faces),
            "faces": [{"bbox": [int(x), int(y), int(w), int(h)]} for (x, y, w, h) in faces],
            "annotated_image": str(output_path),
            "model": "yunet" if face_detector is not None else "fallback"
        }
    except Exception as e:
        logger.error(f"Test detect error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/extract-embedding")
def extract_embedding_endpoint(file: UploadFile = File(...)):
    """Extract embedding from uploaded face image"""
    import base64

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
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        x, y, w, h = int(x), int(y), int(w), int(h)
        face_img = crop_face(frame, x, y, w, h)
        embedding = extract_face_embedding_sface(face_img)

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
async def recompute_embeddings_endpoint():
    """Recompute stored embeddings using SFace (migrates legacy histogram features)."""
    result = recompute_embeddings()
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


class TestRtspRequest(BaseModel):
    rtsp_url: str
    username: Optional[str] = None
    password: Optional[str] = None


@app.post("/test-rtsp")
def test_rtsp_endpoint(req: TestRtspRequest):
    """Probe an RTSP camera (dashboard 'Test Connection'). Pakai ffmpeg CLI agar
    tidak pernah memblokir/menggantung service saat URL tidak terjangkau."""
    import shutil
    import subprocess

    url = req.rtsp_url
    url = _apply_credentials(url, req.username, req.password)

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