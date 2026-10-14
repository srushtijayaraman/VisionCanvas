"""Threaded webcam capture that continuously grabs the latest video frame."""

import threading
import time
from typing import Optional

import cv2
import numpy as np

from src.config import (
    CAMERA_BUFFER_SIZE,
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
)


class Webcam:
    """Threaded webcam capture that always provides the most recent frame.

    A background thread continuously reads from the camera so that
    ``get_frame()`` never blocks on I/O and never returns a stale,
    buffered frame.
    """

    def __init__(self, camera_index: int = CAMERA_INDEX):
        self.camera_index = camera_index
        self.cap = cv2.VideoCapture(camera_index)
        self.is_opened = self.cap.isOpened()

        if not self.is_opened:
            # Note: Do not crash immediately so applications or tests can detect failure gracefully
            self._running = False
            self._thread = None
            self._latest_frame = None
            self._lock = threading.Lock()
            return

        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

        # Minimize internal buffer to avoid processing stale frames
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, CAMERA_BUFFER_SIZE)

        # Threaded capture state
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._running = True

        # Read first frame synchronously so we have something immediately
        success, frame = self.cap.read()
        if success:
            self._latest_frame = frame

        # Start background capture thread
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _capture_loop(self):
        """Continuously grab the latest frame from the camera."""
        while self._running:
            success, frame = self.cap.read()
            if success:
                with self._lock:
                    self._latest_frame = frame
            else:
                # Sleep briefly on read failure to avoid busy-wait spinning at 100% CPU
                time.sleep(0.01)

    def get_frame(self) -> Optional[np.ndarray]:
        """Return the most recent frame (never blocks on camera I/O).

        NOTE: The frame is NOT flipped here. Caller handles flip.
        """
        if not self.is_opened:
            return None

        with self._lock:
            if self._latest_frame is None:
                return None
            return self._latest_frame.copy()

    def release(self) -> None:
        """Stop background worker and release OpenCV video capture device."""
        self._running = False
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.is_opened = False