"""
Test script for improved Fall Detection.

Usage:
    python test_fall_detector.py [--camera CAMERA_INDEX]

Modes:
  1. Webcam mode: python test_fall_detector.py --camera 0
  2. Video file:  python test_fall_detector.py --video path/to/video.mp4

What to test:
  1. Standing → walk around → sit down (should NOT trigger)
  2. Stand still → slowly crouch → kneel (should NOT trigger)
  3. Stand → drop to floor quickly (SHOULD trigger "fall")
  4. Stand → hold onto chair → slide down (may trigger "fall_warning")
  5. After falling, stay on floor (SHOULD trigger after 1.5s)

Visual indicators:
  - GREEN text: standing/sitting/crouching
  - YELLOW text: falling (rapid descent detected)
  - RED text + BEEP: fall confirmed
  - WHITE skeleton: MediaPipe pose landmarks drawn on body

Press 'q' to quit, 'r' to reset the detector state.
"""

import argparse
import math
import time
import sys
import os

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np

from fall_detector import FallDetector, FallState, PoseLandmark


# ─── Colors ────────────────────────────────────────────────────────────────────
C_STANDING   = (80, 200, 80)   # green
C_SITTING    = (80, 220, 220)  # cyan
C_CROUCHING  = (220, 220, 80)  # yellow
C_FALLING    = (50, 150, 255)  # orange
C_LYING      = (0, 80, 255)    # red-blue
C_FALL       = (0, 0, 255)    # red
C_SKELETON   = (200, 200, 200) # gray
C_TEXT_BG    = (40, 40, 40)


def draw_skeleton(frame, landmarks, color=C_SKELETON, thickness=2):
    """Draw pose skeleton on frame for debugging."""
    if landmarks is None:
        return

    try:
        h, w = frame.shape[:2]
        connections = [
            # Upper body
            (PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER),
            (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW),
            (PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
            (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW),
            (PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
            # Torso
            (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_HIP),
            (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_HIP),
            (PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP),
            # Lower body
            (PoseLandmark.LEFT_HIP, PoseLandmark.LEFT_KNEE),
            (PoseLandmark.LEFT_KNEE, PoseLandmark.LEFT_ANKLE),
            (PoseLandmark.RIGHT_HIP, PoseLandmark.RIGHT_KNEE),
            (PoseLandmark.RIGHT_KNEE, PoseLandmark.RIGHT_ANKLE),
            # Head
            (PoseLandmark.LEFT_EAR, PoseLandmark.RIGHT_EAR),
            (PoseLandmark.NOSE, PoseLandmark.LEFT_EYE_INNER),
            (PoseLandmark.NOSE, PoseLandmark.RIGHT_EYE_INNER),
        ]

        def get_pt(idx):
            lm = landmarks[idx]
            return int(lm.x * w), int(lm.y * h)

        for a_idx, b_idx in connections:
            try:
                x1, y1 = get_pt(a_idx)
                x2, y2 = get_pt(b_idx)
                cv2.line(frame, (x1, y1), (x2, y2), color, thickness)
            except Exception:
                pass

        # Draw key points
        for idx in [PoseLandmark.NOSE, PoseLandmark.LEFT_SHOULDER,
                    PoseLandmark.RIGHT_SHOULDER, PoseLandmark.LEFT_HIP,
                    PoseLandmark.RIGHT_HIP, PoseLandmark.LEFT_ANKLE,
                    PoseLandmark.RIGHT_ANKLE]:
            try:
                x, y = get_pt(idx)
                cv2.circle(frame, (x, y), 4, color, -1)
            except Exception:
                pass

    except Exception as e:
        pass


def draw_body_line(frame, track, color):
    """Draw a line from hip to shoulder to visualize body angle."""
    if not track or not track.shoulder_mid or not track.hip_mid:
        return
    h, w = frame.shape[:2]
    try:
        sx = int(track.shoulder_mid[0] * w)
        sy = int(track.shoulder_mid[1] * h)
        hx = int(track.hip_mid[0] * w)
        hy = int(track.hip_mid[1] * h)
        ax = int(track.ankle_mid[0] * w)
        ay = int(track.ankle_mid[1] * h)
        cv2.line(frame, (sx, sy), (hx, hy), color, 3)
        cv2.line(frame, (hx, hy), (ax, ay), (150, 150, 150), 2)
        cv2.circle(frame, (sx, sy), 6, color, -1)  # shoulder
        cv2.circle(frame, (hx, hy), 6, (255, 200, 0), -1)  # hip
        cv2.circle(frame, (ax, ay), 5, (100, 200, 255), -1)  # ankle
    except Exception:
        pass


def get_state_color(state: FallState) -> tuple:
    mapping = {
        FallState.UNKNOWN:   (150, 150, 150),
        FallState.STANDING:   C_STANDING,
        FallState.SITTING:   C_SITTING,
        FallState.CROUCHING: C_CROUCHING,
        FallState.FALLING:   C_FALLING,
        FallState.LYING:     C_LYING,
        FallState.LYING_STILL: C_FALL,
    }
    return mapping.get(state, (200, 200, 200))


def main():
    parser = argparse.ArgumentParser(description="Test improved Fall Detection")
    parser.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0)")
    parser.add_argument("--video", type=str, default=None, help="Video file path")
    parser.add_argument("--no-skeleton", action="store_true", help="Hide pose skeleton")
    parser.add_argument("--debug", action="store_true", help="Show velocity/angle debug info")
    args = parser.parse_args()

    # Initialize detector with tuned parameters
    detector = FallDetector(
        angle_threshold=60.0,
        duration_threshold=1.5,
        confidence_threshold=0.45,
        descent_velocity_threshold=0.08,
        enabled=True,
    )

    # Open video source
    if args.video:
        source = args.video
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video file: {args.video}")
            return
        print(f"Playing video: {args.video}")
    else:
        source = args.camera
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera {args.camera}")
            return
        print(f"Using webcam {args.camera}")

    # Stats
    fps = 0.0
    frame_count = 0
    start_time = time.time()
    last_frame_time = time.time()

    print("\n" + "=" * 60)
    print("FALL DETECTION TEST")
    print("=" * 60)
    print("Press 'q' to quit | 'r' to reset detector")
    print("Try: stand → quickly drop to floor (should alert in ~1.5s)")
    print("=" * 60 + "\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Video ended or frame grab failed.")
            break

        now = time.time()
        dt = now - last_frame_time
        fps = 0.9 * fps + 0.1 / dt if dt > 0 else 0
        last_frame_time = now
        frame_count += 1

        h, w = frame.shape[:2]

        # Run pose detection
        landmarks = detector.detect_pose(frame)
        events = []
        if landmarks is not None:
            events = detector.update(landmarks, h, w, timestamp=now)

        # Get first track for display
        track = next(iter(detector.tracks.values()), None)

        # Determine overlay color
        if track:
            state_color = get_state_color(track.current_state)
            angle = track.body_angle
            vis = track.visibility
            vx, vy = track.compute_velocity()
            angle_vel = track.compute_angle_velocity()
        else:
            state_color = (150, 150, 150)
            angle = 0.0
            vis = 0.0
            vx, vy = 0.0, 0.0
            angle_vel = 0.0

        # Draw skeleton for debugging
        if not args.no_skeleton and landmarks is not None:
            draw_skeleton(frame, landmarks,
                          color=state_color if track else C_SKELETON)
            if track:
                draw_body_line(frame, track, state_color)

        # Process fall events
        alert_triggered = False
        for event in events:
            etype = event.get("type", "")
            if etype == "fall":
                alert_triggered = True
                # Draw large alert overlay
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 200), -1)
                cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
                print(f"  *** FALL DETECTED! angle={event.get('body_angle', 0)}° "
                      f"confidence={event.get('confidence', 0):.2f} "
                      f"duration={event.get('fallen_duration', 0):.1f}s ***")
            elif etype == "fall_warning":
                # Subtle warning
                print(f"  [WARNING] Rapid descent: vel_y={event.get('velocity_y', 0):.4f} "
                      f"angle={event.get('body_angle', 0):.1f}°")

        # Build status overlay
        state_text = track.current_state.value.upper() if track else "NO POSE"
        angle_text = f"angle={angle:.1f}°"
        vel_text = f"vel=({vx:.3f},{vy:.3f})"
        vis_text = f"vis={vis:.2f}"
        fps_text = f"FPS={fps:.1f}"
        det_text = f"detections={detector.detections}"

        status_items = [state_text, angle_text, vel_text, vis_text, fps_text, det_text]
        if args.debug:
            status_items.append(f"angvel={angle_vel:.1f}°/s")
            status_items.append(f"tracks={len(detector.tracks)}")

        # Draw status panel
        panel_h = 25 * len(status_items) + 10
        cv2.rectangle(frame, (5, 5), (310, 5 + panel_h), C_TEXT_BG, -1)
        cv2.rectangle(frame, (5, 5), (310, 5 + panel_h), state_color, 2)

        for i, item in enumerate(status_items):
            y = 28 + i * 25
            color = state_color if i == 0 else (220, 220, 220)
            cv2.putText(frame, item, (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Alert box
        if alert_triggered:
            cv2.putText(frame, "!! FALL DETECTED !!", (w // 2 - 200, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
            cv2.putText(frame, "!! FALL DETECTED !!", (w // 2 - 200, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 2)

        cv2.imshow("Fall Detection Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            detector.reset()
            print("[RESET] Detector state reset.")

    elapsed = time.time() - start_time
    print(f"\n--- Session ended ---")
    print(f"Frames: {frame_count}, Time: {elapsed:.1f}s, Avg FPS: {frame_count/elapsed:.1f}")
    print(f"Total fall detections: {detector.detections}")
    print(f"False positives prevented: {detector.false_positives_prevented}")
    stats = detector.get_stats()
    print(f"Stats: {stats}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
