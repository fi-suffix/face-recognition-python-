"""
Fall Detection Module using MediaPipe Pose Estimation.

This module detects falls by analyzing body pose from video frames.
When a person is detected lying down for a sustained period,
an emergency event is triggered.
"""

import math
import time
import logging
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field
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


class FallState(Enum):
    """Possible states for fall detection."""
    UNKNOWN = "unknown"
    STANDING = "standing"
    SITTING = "sitting"
    LYING = "lying"
    FALLING = "falling"


@dataclass
class PersonTrack:
    """Tracks a person through frames for fall detection."""
    track_id: int
    shoulder_mid: Tuple[float, float] = field(default=(0, 0))
    hip_mid: Tuple[float, float] = field(default=(0, 0))
    head: Tuple[float, float] = field(default=(0, 0))
    ankle_mid: Tuple[float, float] = field(default=(0, 0))
    current_state: FallState = FallState.UNKNOWN
    fallen_since: Optional[float] = None
    last_seen: float = 0
    last_positions: List[Tuple[float, float]] = field(default_factory=list)
    movement_threshold: float = 0.02

    def update_keypoints(self, landmarks: List, frame_height: int, frame_width: int):
        """Update tracked position from pose landmarks."""
        try:
            left_shoulder = landmarks[PoseLandmark.LEFT_SHOULDER]
            right_shoulder = landmarks[PoseLandmark.RIGHT_SHOULDER]
            left_hip = landmarks[PoseLandmark.LEFT_HIP]
            right_hip = landmarks[PoseLandmark.RIGHT_HIP]
            nose = landmarks[PoseLandmark.NOSE]
            left_ankle = landmarks[PoseLandmark.LEFT_ANKLE]
            right_ankle = landmarks[PoseLandmark.RIGHT_ANKLE]

            self.shoulder_mid = (
                (left_shoulder.x + right_shoulder.x) / 2,
                (left_shoulder.y + right_shoulder.y) / 2
            )
            self.hip_mid = (
                (left_hip.x + right_hip.x) / 2,
                (left_hip.y + right_hip.y) / 2
            )
            self.head = (nose.x, nose.y)
            self.ankle_mid = (
                (left_ankle.x + right_ankle.x) / 2,
                (left_ankle.y + right_ankle.y) / 2
            )

            self.last_positions.append((self.shoulder_mid[0], self.shoulder_mid[1]))
            if len(self.last_positions) > 30:
                self.last_positions.pop(0)

            self.last_seen = time.time()
        except (IndexError, AttributeError) as e:
            logger.debug(f"Failed to update keypoints: {e}")

    def is_moving(self) -> bool:
        """Check if person is still moving."""
        if len(self.last_positions) < 5:
            return True
        positions = self.last_positions[-10:]
        total_movement = 0
        for i in range(1, len(positions)):
            dx = positions[i][0] - positions[i-1][0]
            dy = positions[i][1] - positions[i-1][1]
            total_movement += math.sqrt(dx*dx + dy*dy)
        avg_movement = total_movement / (len(positions) - 1) if len(positions) > 1 else 0
        return avg_movement > self.movement_threshold


class FallDetector:
    """
    Fall detection using MediaPipe Pose estimation.
    """

    def __init__(
        self,
        angle_threshold: float = 45.0,
        duration_threshold: float = 3.0,
        confidence_threshold: float = 0.5,
        enabled: bool = True
    ):
        self.angle_threshold = angle_threshold
        self.duration_threshold = duration_threshold
        self.confidence_threshold = confidence_threshold
        self.enabled = enabled
        self._pose = None
        self._mp_pose = None
        self.tracks: Dict[int, PersonTrack] = {}
        self.next_track_id = 1
        self.detections = 0
        self.false_positives_prevented = 0
        logger.info(f"FallDetector: angle={angle_threshold}, duration={duration_threshold}s, enabled={enabled}")

    def _init_pose(self):
        """Lazy initialization of MediaPipe Pose."""
        if self._pose is None:
            try:
                import mediapipe as mp
                self._mp_pose = mp
                self._pose = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    smooth_landmarks=True,
                    min_detection_confidence=self.confidence_threshold,
                    min_tracking_confidence=self.confidence_threshold
                )
                logger.info("MediaPipe Pose initialized")
            except ImportError:
                logger.error("MediaPipe not installed. Run: pip install mediapipe")
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize MediaPipe Pose: {e}")
                self.enabled = False

    def detect_pose(self, frame: np.ndarray) -> Optional[List]:
        """Detect pose landmarks in a frame."""
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

    def calculate_body_angle(self, person: PersonTrack) -> float:
        """Calculate body angle relative to vertical (0=upright, 90=horizontal)."""
        try:
            dx = person.shoulder_mid[0] - person.hip_mid[0]
            dy = person.shoulder_mid[1] - person.hip_mid[1]
            angle_rad = math.atan2(abs(dx), abs(dy))
            return math.degrees(angle_rad)
        except (ZeroDivisionError, AttributeError):
            return 0.0

    def is_head_below_hips(self, person: PersonTrack) -> bool:
        """Check if head is below hip level."""
        return person.head[1] > person.hip_mid[1]

    def determine_state(self, person: PersonTrack) -> FallState:
        """Determine person's current state."""
        angle = self.calculate_body_angle(person)
        head_below = self.is_head_below_hips(person)

        if angle < 20:
            return FallState.STANDING
        elif angle < 60:
            return FallState.SITTING if head_below else FallState.STANDING
        else:
            return FallState.LYING

    def update(self, landmarks: List, frame_height: int, frame_width: int) -> List[Dict[str, Any]]:
        """Update fall detection with new pose data."""
        events = []
        if not self.enabled or landmarks is None:
            return events

        try:
            if len(self.tracks) == 0:
                track = PersonTrack(track_id=self.next_track_id)
                self.next_track_id += 1
                self.tracks[track.track_id] = track

            track = list(self.tracks.values())[0]
            track.update_keypoints(landmarks, frame_height, frame_width)
            new_state = self.determine_state(track)
            now = time.time()

            if new_state == FallState.LYING:
                if track.fallen_since is None:
                    track.fallen_since = now
                fallen_duration = now - track.fallen_since
                if fallen_duration >= self.duration_threshold and not track.is_moving():
                    events.append({
                        "type": "fall",
                        "severity": "critical",
                        "confidence": min(1.0, fallen_duration / 10.0),
                        "fallen_duration": fallen_duration,
                        "body_angle": self.calculate_body_angle(track),
                        "track_id": track.track_id
                    })
                    self.detections += 1
                    logger.warning(f"FALL DETECTED! Duration: {fallen_duration:.1f}s")
            elif track.fallen_since is not None:
                duration = now - track.fallen_since
                if duration < self.duration_threshold:
                    self.false_positives_prevented += 1
                track.fallen_since = None

            track.current_state = new_state
            self._cleanup_old_tracks()
        except Exception as e:
            logger.error(f"Fall detection update error: {e}")

        return events

    def _cleanup_old_tracks(self):
        """Remove tracks that haven't been seen recently."""
        now = time.time()
        max_age = 10.0
        to_remove = [tid for tid, track in self.tracks.items() if (now - track.last_seen) > max_age]
        for tid in to_remove:
            del self.tracks[tid]

    def reset(self):
        """Reset detector state."""
        self.tracks.clear()
        self.next_track_id = 1
        logger.info("FallDetector reset")

    def get_stats(self) -> Dict[str, Any]:
        """Get detection statistics."""
        return {
            "enabled": self.enabled,
            "active_tracks": len(self.tracks),
            "total_detections": self.detections,
            "false_positives_prevented": self.false_positives_prevented,
            "angle_threshold": self.angle_threshold,
            "duration_threshold": self.duration_threshold
        }
