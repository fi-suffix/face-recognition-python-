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
MAX_STREAM_WIDTH = int(os.getenv("MAX_STREAM_WIDTH", "960"))
STREAM_FPS = int(os.getenv("STREAM_FPS", "20"))

# Performance tuning
DETECT_EVERY_N_FRAMES = int(os.getenv("DETECT_EVERY_N_FRAMES", "3"))  # Run YuNet every N frames
YUNET_INPUT_WIDTH = int(os.getenv("YUNET_INPUT_WIDTH", "640"))
YUNET_INPUT_HEIGHT = int(os.getenv("YUNET_INPUT_HEIGHT", "480"))
YUNET_CONFIDENCE_THRESHOLD = float(os.getenv("YUNET_CONFIDENCE_THRESHOLD", "0.5"))
RECOGNITION_WORKERS = int(os.getenv("RECOGNITION_WORKERS", "4"))
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
                    face_img = frame[y:y+h, x:x+w]
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
    """Detect faces using YuNet (OpenCV 5.x)"""
    if face_detector is None:
        return detect_faces_fallback(frame)

    h, w = frame.shape[:2]
    face_detector.setInputSize((w, h))
    _, faces = face_detector.detect(frame)

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
    cap = None
    reconnect_attempts = 0
    max_reconnect_attempts = 10
    frames = 0
    last_scale = 1.0
    last_recognition = 0.0
    detect_frame_counter = 0
    cached_faces = []

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

            ret, frame = cap.read()
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

            # Run recognition off-thread, time-gated, to keep the video fluid
            run_recognition = (frames % RECOGNITION_EVERY_N_FRAMES == 0)

            t = now_ms()
            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                if face_img.size == 0:
                    continue

                # Reuse the latest recognition result for this face (approx by position)
                emp_id, confidence = None, 0.0
                prev = last_results.get(camera.id)
                if prev is not None:
                    px, py, pw, ph, prev_emp, prev_conf, prev_ts = prev
                    if (abs(px - x) <= w * 1.5 and abs(py - y) <= h * 1.5
                            and (t - prev_ts) < 3.0):
                        emp_id, confidence = prev_emp, prev_conf

                # Enqueue recognition (SFace + snapshot + Laravel log) for the worker
                if run_recognition and (t - last_recognition) >= RECOGNITION_INTERVAL:
                    try:
                        recognition_queue.put_nowait((
                            camera.id, frame.copy(),
                            int(x), int(y), int(w), int(h), t
                        ))
                        last_recognition = t
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
            camera_streams[camera.id]["latest_frame"] = frame.copy()
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
    """Consumer for recognition_queue: embedding, result cache, snapshot + Laravel log.

    Runs off the camera frame loop so the CNN and HTTP calls never stall the video.
    """
    while True:
        task = recognition_queue.get()
        if task is None:
            break
        # Submit to thread pool for parallel processing
        recognition_executor.submit(process_recognition_task, task)


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
            face_img = frame[y:y+h, x:x+w]
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
        face_img = frame[y:y+h, x:x+w]
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


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)