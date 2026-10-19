"""
Fall Detection Module using MediaPipe Pose Estimation.

Detects falls by analyzing body pose from video frames.
Supports multi-person tracking and detects falls in 3 phases:
  1. RAPID_DESCENT - body center moving down fast (velocity-based)
  2. LYING_DOWN   - body angle exceeds threshold for sustained period
  3. LYING_STILL  - lying + no movement for confirmation
"""

import math
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class PoseLandmark:
    """MediaPipe Pose landmark indices."""
    NOSE = 0
    LEFT_EYE_INNER = 1
    LEFT_EYE = 2
    LEFT_EYE_OUTER = 3
    RIGHT_EYE_INNER = 4
    RIGHT_EYE = 5
    RIGHT_EYE_OUTER = 6
    LEFT_EAR = 7
    RIGHT_EAR = 8
    MOUTH_LEFT = 9
    MOUTH_RIGHT = 10
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_PINKY = 17
    RIGHT_PINKY = 18
    LEFT_INDEX = 19
    RIGHT_INDEX = 20
    LEFT_THUMB = 21
    RIGHT_THUMB = 22
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    LEFT_HEEL = 29
    RIGHT_HEEL = 30
    LEFT_FOOT_INDEX = 31
    RIGHT_FOOT_INDEX = 32

    # Landmark groups for visibility checks
    KEY_UPPER = {NOSE, LEFT_EAR, RIGHT_EAR, LEFT_SHOULDER, RIGHT_SHOULDER,
                 LEFT_ELBOW, RIGHT_ELBOW, LEFT_WRIST, RIGHT_WRIST}
    KEY_MIDDLE = {LEFT_HIP, RIGHT_HIP, LEFT_KNEE, RIGHT_KNEE}
    KEY_LOWER = {LEFT_ANKLE, RIGHT_ANKLE, LEFT_HEEL, RIGHT_HEEL}


class FallState(Enum):
    """Possible states for fall detection."""
    UNKNOWN = "unknown"
    STANDING = "standing"
    SITTING = "sitting"
    CROUCHING = "crouching"
    LYING = "lying"
    FALLING = "falling"
    LYING_STILL = "lying_still"


@dataclass
class PersonTrack:
    """Tracks a person through frames for fall detection.

    Stores pose history for velocity/angle analysis.
    """
    track_id: int
    # Current keypoints (normalized 0-1)
    shoulder_mid: Tuple[float, float] = field(default=(0.5, 0.5))
    hip_mid: Tuple[float, float] = field(default=(0.5, 0.5))
    head: Tuple[float, float] = field(default=(0.5, 0.5))
    ankle_mid: Tuple[float, float] = field(default=(0.5, 0.5))
    knee_mid: Tuple[float, float] = field(default=(0.5, 0.5))
    # Full body center (weighted average)
    body_center: Tuple[float, float] = field(default=(0.5, 0.5))
    # Body angle (degrees from vertical; 0=upright, 90=horizontal, 180=upside-down)
    body_angle: float = 0.0
    # Pose visibility score (fraction of key landmarks above confidence)
    visibility: float = 0.0
    # State machine
    current_state: FallState = FallState.UNKNOWN
    # Timing
    last_seen: float = 0.0
    state_since: float = 0.0     # When current_state started
    # History for velocity
    position_history: deque = field(default_factory=lambda: deque(maxlen=30))
    angle_history: deque = field(default_factory=lambda: deque(maxlen=20))
    # Fall detection
    fall_since: Optional[float] = None       # When entered LYING/LYING_STILL
    descent_since: Optional[float] = None     # When rapid descent detected
    fall_confirmed: bool = False              # Already sent alert
    descent_detected: bool = False            # Rapid descent in progress
    # Movement
    movement_threshold: float = 0.02
    # Min confidence required for each landmark group
    min_visibility_upper: float = 0.50
    min_visibility_middle: float = 0.45
    min_visibility_lower: float = 0.40

    def is_visible_enough(self) -> bool:
        """Check if enough landmarks are visible for reliable fall detection."""
        return self.visibility >= 0.40

    def update_keypoints(self, landmarks: List, frame_height: int, frame_width: int,
                         timestamp: float):
        """Update tracked position from pose landmarks.

        Args:
            landmarks: List of MediaPipe Landmark objects
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels
            timestamp: Current time in seconds
        """
        try:
            def get_lm(idx):
                """Get normalized (x, y) and visibility for a landmark."""
                lm = landmarks[idx]
                return (lm.x, lm.y), lm.visibility if hasattr(lm, 'visibility') else 1.0

            (l_sh, v_sh), v1 = get_lm(PoseLandmark.LEFT_SHOULDER)
            (r_sh, v_sh2), v2 = get_lm(PoseLandmark.RIGHT_SHOULDER)
            (l_hi, v_hi), v3 = get_lm(PoseLandmark.LEFT_HIP)
            (r_hi, v_hi2), v4 = get_lm(PoseLandmark.RIGHT_HIP)
            (nose, v_nose), _ = get_lm(PoseLandmark.NOSE)
            (l_an, v_an), v5 = get_lm(PoseLandmark.LEFT_ANKLE)
            (r_an, v_an2), v6 = get_lm(PoseLandmark.RIGHT_ANKLE)
            (l_kn, v_kn), v7 = get_lm(PoseLandmark.LEFT_KNEE)
            (r_kn, v_kn2), v8 = get_lm(PoseLandmark.RIGHT_KNEE)

            self.shoulder_mid = ((l_sh + r_sh) / 2, (v_sh + v_sh2) / 2)
            self.hip_mid = ((l_hi + r_hi) / 2, (v_hi + v_hi2) / 2)
            self.head = nose
            self.ankle_mid = ((l_an + r_an) / 2, (v_an + v_an2) / 2)
            self.knee_mid = ((l_kn + r_kn) / 2, (v_kn + v_kn2) / 2)

            # Compute full body center (weighted average of upper + middle + lower)
            self.body_center = (
                (nose[0] + l_sh + r_sh + l_hi + r_hi + l_an + r_an) / 6,
                (nose[1] + v_sh + v_sh2 + v_hi + v_hi2 + v_an + v_an2) / 6
            )

            # Compute body angle: vector from hip_mid to shoulder_mid
            # dy positive = body tilting from vertical
            dx = self.shoulder_mid[0] - self.hip_mid[0]
            dy = self.hip_mid[1] - self.shoulder_mid[1]  # invert so upright = small
            self.body_angle = math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))

            # Visibility score (simple average of available landmarks)
            visibilities = [v_sh, v_sh2, v_hi, v_hi2, v_nose, v_an, v_an2, v7, v8]
            self.visibility = sum(v for v in visibilities if v is not None) / len(visibilities)

            # Store history
            self.position_history.append((self.body_center[0], self.body_center[1], timestamp))
            self.angle_history.append((self.body_angle, timestamp))
            if len(self.angle_history) > 20:
                self.angle_history.pop(0)

            self.last_seen = timestamp
            self.state_since = getattr(self, 'state_since', 0.0) or timestamp

        except (IndexError, AttributeError) as e:
            logger.debug(f"Failed to update keypoints: {e}")

    def compute_velocity(self) -> Tuple[float, float]:
        """Compute body center velocity (pixels/sec normalized).

        Returns (vx, vy) where vy>0 means moving DOWN (falling).
        """
        if len(self.position_history) < 3:
            return 0.0, 0.0
        # Use last 5 positions for smoothing
        recent = list(self.position_history)[-5:]
        dt = recent[-1][2] - recent[0][2]
        if dt < 0.05:  # less than 50ms
            return 0.0, 0.0
        dx = recent[-1][0] - recent[0][0]
        dy = recent[-1][1] - recent[0][1]
        return dx / dt, dy / dt

    def compute_angle_velocity(self) -> float:
        """Compute rate of body angle change (degrees/sec)."""
        if len(self.angle_history) < 3:
            return 0.0
        recent = list(self.angle_history)[-5:]
        dt = recent[-1][1] - recent[0][1]
        if dt < 0.05:
            return 0.0
        d_angle = recent[-1][0] - recent[0][0]
        return d_angle / dt

    def is_moving(self) -> bool:
        """Check if person is still moving significantly."""
        vx, vy = self.compute_velocity()
        speed = math.sqrt(vx * vx + vy * vy)
        return speed > self.movement_threshold

    def is_head_below_hip(self) -> bool:
        """Check if head is significantly below hip level (pixel coords)."""
        # In pixel coords, larger y = lower on screen
        return self.head[1] > self.hip_mid[1] + 0.05  # 5% threshold

    def is_kneeling(self) -> bool:
        """Check if person is kneeling (knees below hips but not fully lying)."""
        return (self.knee_mid[1] > self.hip_mid[1] + 0.08 and
                self.body_angle < 70)

    def get_height_ratio(self) -> float:
        """Get person height ratio (distance from ankle to head in frame).

        A falling person shrinks as they get closer to the ground.
        Returns ratio of current height to initial standing height (approx).
        """
        if not self.position_history:
            return 1.0
        # Normalized: standing ≈ 0.4-0.5 of frame height, lying ≈ 0.1-0.15
        # Use angle as proxy: standing ≈ 0°, lying ≈ 90°
        return math.cos(math.radians(self.body_angle))


class FallDetector:
    """
    Fall detection using MediaPipe Pose estimation.

    Detects falls using a multi-phase approach:
    1. RAPID_DESCENT: Body center moving downward fast (velocity-based)
    2. ANGLE_CHANGE: Body angle increasing rapidly toward horizontal
    3. LYING_STILL: Person is horizontal + stationary for confirmation period

    Improvements over v1:
    - Uses model_complexity=2 for better landmark accuracy (33 landmarks)
    - Tracks multiple people
    - Velocity-based descent detection
    - Lower duration threshold (1.5s default)
    - Wider angle threshold (60°)
    - Height ratio analysis
    """

    def __init__(
        self,
        angle_threshold: float = 60.0,
        duration_threshold: float = 1.5,
        confidence_threshold: float = 0.45,
        descent_velocity_threshold: float = 0.08,  # normalized units/sec
        enabled: bool = True,
        min_person_height: float = 0.15,  # min person height in frame (normalized)
        rapid_descent_min_frames: int = 3,
        lying_still_min_frames: int = 5,
    ):
        self.angle_threshold = angle_threshold
        self.duration_threshold = duration_threshold
        self.confidence_threshold = confidence_threshold
        self.descent_velocity_threshold = descent_velocity_threshold
        self.enabled = enabled
        self.min_person_height = min_person_height
        self.rapid_descent_min_frames = rapid_descent_min_frames
        self.lying_still_min_frames = lying_still_min_frames

        self._pose = None
        self._mp_pose = None
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_track_id = 1
        self.detections = 0
        self.false_positives_prevented = 0
        self.descent_false_positives = 0

        logger.info(
            f"FallDetector: angle={angle_threshold}°, duration={duration_threshold}s, "
            f"descent_vel={descent_velocity_threshold}, conf={confidence_threshold}, "
            f"enabled={enabled}"
        )

    def _init_pose(self):
        """Lazy initialization of MediaPipe Pose with complexity=2."""
        if self._pose is None:
            try:
                import mediapipe as mp
                self._mp_pose = mp
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=2,        # Use all 33 landmarks (not 0 or 1)
                    smooth_landmarks=True,
                    min_detection_confidence=self.confidence_threshold,
                    min_tracking_confidence=self.confidence_threshold,
                    enable_segmentation=False,
                )
                logger.info("MediaPipe Pose initialized (model_complexity=2, 33 landmarks)")
            except ImportError:
                logger.error("MediaPipe not installed. Run: pip install mediapipe")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe Pose: {e}")
                self.enabled = False

    def detect_pose(self, frame: np.ndarray) -> Optional[List]:
        """Detect pose landmarks in a frame.

        Returns list of landmark objects with x, y, z, visibility.
        """
        if not self.enabled:
            return None
        self._init_pose()
        if self._pose is None:
            return None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._pose.process(rgb)
            if results.pose_landmarks is None:
                return None
            return results.pose_landmarks.landmark
        except Exception as e:
            logger.debug(f"Pose detection error: {e}")
            return None

    def _assign_to_track(self, landmarks: List, frame_h: int, frame_w: int,
                         timestamp: float) -> Tuple[Optional[PersonTrack], Optional[int]]:
        """Assign detected person to an existing track or create a new one.

        Uses body center distance for matching (IoU-like for poses).
        Returns (track, is_new_track_id or None).
        """
        # Compute current body center
        try:
            l_sh = landmarks[PoseLandmark.LEFT_SHOULDER]
            r_sh = landmarks[PoseLandmark.RIGHT_SHOULDER]
            l_hi = landmarks[PoseLandmark.LEFT_HIP]
            r_hi = landmarks[PoseLandmark.RIGHT_HIP]
            nose = landmarks[PoseLandmark.NOSE]

            cx = (nose.x + l_sh.x + r_sh.x + l_hi.x + r_hi.x) / 5
            cy = (nose.y + l_sh.y + r_sh.y + l_hi.y + r_hi.y) / 5
        except (IndexError, AttributeError):
            return None, None

        best_track = None
        best_dist = 0.20  # max normalized distance to match
        best_tid = None

        for tid, track in self.tracks.items():
            tcx, tcy = track.body_center
            dist = math.sqrt((cx - tcx) ** 2 + (cy - tcy) ** 2)
            if dist < best_dist:
                best_dist = dist
                best_track = track
                best_tid = tid

        if best_track is not None:
            return best_track, None

        # Create new track
        track = PersonTrack(track_id=self.next_track_id)
        self.next_track_id += 1
        self.tracks[track.track_id] = track
        return track, track.track_id

    def determine_state(self, person: PersonTrack) -> FallState:
        """Determine person's current state based on pose analysis."""
        angle = person.body_angle
        head_below = person.is_head_below_hip()
        kneeling = person.is_kneeling()
        vx, vy = person.compute_velocity()
        descent_speed = vy  # positive = moving down
        angle_vel = person.compute_angle_velocity()

        # Phase 1: Rapid downward descent
        if descent_speed > self.descent_velocity_threshold and not person.fall_confirmed:
            # Check if also angle changing
            if angle_vel > 15.0 or angle > 30:
                return FallState.FALLING

        # Phase 2: Angle-based classification
        if angle < 35:
            return FallState.STANDING
        elif angle < 60:
            if head_below:
                return FallState.SITTING
            elif kneeling:
                return FallState.CROUCHING
            return FallState.STANDING
        else:
            # angle >= 60° → likely lying
            if not person.is_moving():
                return FallState.LYING_STILL
            return FallState.LYING

    def update(self, landmarks: List, frame_height: int, frame_width: int,
               timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        """Update fall detection with new pose data.

        Args:
            landmarks: List of MediaPipe Landmark objects
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels
            timestamp: Current time (defaults to time.time())

        Returns:
            List of event dicts if fall detected
        """
        events = []
        if not self.enabled or landmarks is None:
            return events

        if timestamp is None:
            timestamp = time.time()

        try:
            # Assign to track
            track, is_new = self._assign_to_track(landmarks, frame_height, frame_width, timestamp)
            if track is None:
                return events

            # Update keypoints
            track.update_keypoints(landmarks, frame_height, frame_width, timestamp)

            # Skip if visibility too low
            if not track.is_visible_enough():
                return events

            # Determine current state
            new_state = self.determine_state(track)

            # State transition logic
            if new_state != track.current_state:
                track.current_state = new_state
                track.state_since = timestamp
                logger.debug(
                    f"[fall-track:{track.track_id}] state {new_state.value} "
                    f"(angle={track.body_angle:.1f}°, vel={track.compute_velocity()[1]:.4f})"
                )

            now = timestamp

            # --- Fall detection logic ---

            # A. LYING_STILL detection (primary)
            if new_state in (FallState.LYING, FallState.LYING_STILL):
                if track.fall_since is None:
                    track.fall_since = now
                    track.descent_since = getattr(track, 'descent_since', None) or now

                lying_duration = now - track.fall_since
                # Require sustained lying + no rapid descent (was a real fall, not just sitting)
                is_stationary = not track.is_moving()

                if lying_duration >= self.duration_threshold and is_stationary:
                    # Confidence based on: duration, angle, stillness, visibility
                    angle_factor = min(1.0, track.body_angle / self.angle_threshold)
                    duration_factor = min(1.0, lying_duration / max(1.0, self.duration_threshold * 2))
                    vis_factor = track.visibility
                    confidence = (angle_factor * 0.4 + duration_factor * 0.3 + vis_factor * 0.3)

                    if confidence >= 0.55 and not track.fall_confirmed:
                        events.append({
                            "type": "fall",
                            "severity": "critical",
                            "confidence": round(confidence, 3),
                            "fallen_duration": round(lying_duration, 2),
                            "body_angle": round(track.body_angle, 1),
                            "track_id": track.track_id,
                            "velocity_y": round(track.compute_velocity()[1], 4),
                            "angle_velocity": round(track.compute_angle_velocity(), 2),
                            "head_below_hip": track.is_head_below_hip(),
                            "is_stationary": is_stationary,
                        })
                        track.fall_confirmed = True
                        self.detections += 1
                        logger.warning(
                            f"[FALL DETECTED] track={track.track_id} "
                            f"duration={lying_duration:.1f}s angle={track.body_angle:.1f}° "
                            f"confidence={confidence:.2f}"
                        )

            # B. Rapid descent detection (early warning)
            elif new_state == FallState.FALLING:
                if track.descent_since is None:
                    track.descent_since = now
                descent_duration = now - track.descent_since
                # Fire if descent continues for 0.5s+ without confirming as lying
                if descent_duration >= 0.5 and not track.fall_confirmed:
                    events.append({
                        "type": "fall_warning",
                        "severity": "warning",
                        "confidence": 0.5,
                        "fallen_duration": round(descent_duration, 2),
                        "body_angle": round(track.body_angle, 1),
                        "track_id": track.track_id,
                        "velocity_y": round(track.compute_velocity()[1], 4),
                        "angle_velocity": round(track.compute_angle_velocity(), 2),
                        "head_below_hip": track.is_head_below_hip(),
                    })
                    logger.warning(
                        f"[FALL WARNING] track={track.track_id} "
                        f"descent={descent_duration:.1f}s vel={track.compute_velocity()[1]:.4f}"
                    )
                    # Reset fall_since so the main alarm fires if they stay down
                    track.fall_since = None

            # C. Recovery: person stands up → reset
            elif track.current_state in (FallState.STANDING, FallState.SITTING, FallState.CROUCHING):
                if track.fall_since is not None:
                    lying_dur = now - track.fall_since
                    if lying_dur < self.duration_threshold:
                        self.false_positives_prevented += 1
                        logger.debug(
                            f"[fall] False positive prevented: person recovered after {lying_dur:.1f}s"
                        )
                track.fall_since = None
                track.descent_since = None
                track.fall_confirmed = False
                track.descent_detected = False

            self._cleanup_old_tracks(now)

        except Exception as e:
            logger.error(f"Fall detection update error: {e}")

        return events

    def _cleanup_old_tracks(self, now: float):
        """Remove tracks that haven't been seen recently."""
        max_age = 5.0  # 5 seconds without update → remove
        to_remove = [tid for tid, track in self.tracks.items()
                     if (now - track.last_seen) > max_age]
        for tid in to_remove:
            del self.tracks[tid]

    def reset(self):
        """Reset detector state."""
        self.tracks.clear()
        self.next_track_id = 1
        self.detections = 0
        self.false_positives_prevented = 0
        logger.info("FallDetector reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "enabled": self.enabled,
            "active_tracks": len(self.tracks),
            "total_detections": self.detections,
            "false_positives_prevented": self.false_positives_prevented,
            "angle_threshold": self.angle_threshold,
            "duration_threshold": self.duration_threshold,
            "descent_velocity_threshold": self.descent_velocity_threshold,
        }
