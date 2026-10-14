"""MediaPipe hand landmark detector running in VIDEO mode with thread safety."""

import os
import threading
import time
from typing import Optional, Tuple

import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

from src.config import (
    DETECTION_CONFIDENCE,
    MAX_HANDS,
    MODEL_PATH,
    TRACKING_CONFIDENCE,
)


class HandDetector:
    """MediaPipe hand landmark detector running in VIDEO mode.

    Can be used synchronously or driven from a background thread via
    ``detect()``. Results are stored thread-safely and tagged with a
    monotonic inference ID so the main render loop can distinguish fresh
    detections from stale ones.
    """

    def __init__(self, model_path: str = MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"MediaPipe hand landmarker model not found at: {model_path}")

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            num_hands=MAX_HANDS,
            min_hand_detection_confidence=DETECTION_CONFIDENCE,
            min_tracking_confidence=TRACKING_CONFIDENCE,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)
        self.frame_timestamp_ms: int = 0

        # Thread-safe latest result
        self._lock = threading.Lock()
        self._latest_result = None
        self._latest_frame = None
        self._inference_id: int = 0
        self._last_consumed_id: int = 0
        self.latest_inference_time_ms: float = 0.0

    def detect(self, frame):
        """Run inference on *frame* and store result thread-safely.

        Returns:
            The raw MediaPipe result.
        """
        if frame is None:
            return None

        t_start = time.perf_counter()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        current_ts = int(time.time() * 1000)
        if current_ts <= self.frame_timestamp_ms:
            current_ts = self.frame_timestamp_ms + 1
        self.frame_timestamp_ms = current_ts

        result = self.landmarker.detect_for_video(mp_image, self.frame_timestamp_ms)
        inf_duration_ms = (time.perf_counter() - t_start) * 1000.0

        with self._lock:
            self._latest_result = result
            self._latest_frame = frame
            self._inference_id += 1
            self.latest_inference_time_ms = inf_duration_ms

        return result

    def get_latest(self) -> Tuple[Optional[object], Optional[object], bool]:
        """Return (result, frame, is_new) from the most recent inference.

        *is_new* is True only if this inference result has not been fetched yet.
        """
        with self._lock:
            is_new = self._inference_id > self._last_consumed_id
            self._last_consumed_id = self._inference_id
            return self._latest_result, self._latest_frame, is_new

    def close(self) -> None:
        """Release underlying MediaPipe landmarker resources."""
        if hasattr(self, "landmarker") and self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass