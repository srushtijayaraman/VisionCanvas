"""Virtual cursor for converting hand landmarks to smooth, stable screen coordinates.

Features:
- Dual-axis One Euro Filter smoothing for low-latency jitter reduction.
- Outlier jump rejection to filter transient tracking glitches.
- Persistent-jump recovery so cursor never freezes when moving rapidly.
- Explicit tracking-loss and reacquisition handling.
- Deterministic timestamp injection for automated testing.
- Boundary clamping and landmark sanitization.
"""

import math
import time
from typing import Optional, Tuple

from src.config import (
    CURSOR_OUTLIER_RECOVERY_FRAMES,
    MAX_CURSOR_JUMP,
    ONE_EURO_BETA,
    ONE_EURO_DCUTOFF,
    ONE_EURO_MIN_CUTOFF,
    SMOOTHING_ENABLED,
)
from src.utils.one_euro_filter import OneEuroFilter


class VirtualCursor:
    """Converts normalized hand landmark positions to smooth screen coordinates."""

    def __init__(
        self,
        max_jump: float = MAX_CURSOR_JUMP,
        recovery_frames: int = CURSOR_OUTLIER_RECOVERY_FRAMES,
        min_cutoff: float = ONE_EURO_MIN_CUTOFF,
        beta: float = ONE_EURO_BETA,
        d_cutoff: float = ONE_EURO_DCUTOFF,
        smoothing_enabled: bool = SMOOTHING_ENABLED,
    ):
        self.max_jump = float(max_jump)
        self.recovery_frames = max(1, int(recovery_frames))
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.smoothing_enabled = smoothing_enabled

        self.filter_x: Optional[OneEuroFilter] = None
        self.filter_y: Optional[OneEuroFilter] = None
        self.last_x: int = 0
        self.last_y: int = 0

        # Tracking state
        self.is_tracking: bool = False
        self.consecutive_outliers: int = 0
        self.candidate_x: float = 0.0
        self.candidate_y: float = 0.0

    def reset(self) -> None:
        """Reset all filter states and tracking flags."""
        self.filter_x = None
        self.filter_y = None
        self.last_x = 0
        self.last_y = 0
        self.is_tracking = False
        self.consecutive_outliers = 0
        self.candidate_x = 0.0
        self.candidate_y = 0.0

    def on_tracking_lost(self) -> None:
        """Explicitly notify the cursor that hand tracking was lost.

        Next reacquisition will smoothly reinitialize rather than rejecting
        the new hand position as a jump.
        """
        self.is_tracking = False
        self.consecutive_outliers = 0

    def update(
        self,
        landmark,
        frame_width: int,
        frame_height: int,
        timestamp: Optional[float] = None,
    ) -> Tuple[int, int]:
        """Update virtual cursor position with a landmark observation.

        Args:
            landmark: Landmark object with .x and .y attributes (normalized 0.0-1.0),
                      or None if hand is missing/invalid.
            frame_width: Screen / frame width in pixels.
            frame_height: Screen / frame height in pixels.
            timestamp: Optional explicit time in seconds (defaults to time.time()).

        Returns:
            (cx, cy): Integer pixel coordinates clamped to the screen.
        """
        if timestamp is None:
            t = time.time()
        else:
            t = float(timestamp)

        # Validate landmark
        if landmark is None:
            self.on_tracking_lost()
            return self.last_x, self.last_y

        try:
            lx = float(landmark.x)
            ly = float(landmark.y)
        except (AttributeError, TypeError, ValueError):
            self.on_tracking_lost()
            return self.last_x, self.last_y

        if math.isnan(lx) or math.isnan(ly) or math.isinf(lx) or math.isinf(ly):
            self.on_tracking_lost()
            return self.last_x, self.last_y

        # Clamp raw normalized coordinates to sane bounds
        raw_x = max(0.0, min(float(frame_width), lx * frame_width))
        raw_y = max(0.0, min(float(frame_height), ly * frame_height))

        # First frame or reacquisition after tracking loss: initialize/reset filter
        if not self.is_tracking or self.filter_x is None or self.filter_y is None:
            self.filter_x = OneEuroFilter(
                t,
                raw_x,
                mincutoff=self.min_cutoff,
                beta=self.beta,
                dcutoff=self.d_cutoff,
            )
            self.filter_y = OneEuroFilter(
                t,
                raw_y,
                mincutoff=self.min_cutoff,
                beta=self.beta,
                dcutoff=self.d_cutoff,
            )
            self.last_x = int(round(raw_x))
            self.last_y = int(round(raw_y))
            self.is_tracking = True
            self.consecutive_outliers = 0
            return self.last_x, self.last_y

        # Calculate jump distance relative to previous accepted position
        dist = math.hypot(raw_x - self.last_x, raw_y - self.last_y)

        if dist > self.max_jump:
            # Check if this new position is consistent with a previous candidate outlier
            if self.consecutive_outliers > 0:
                candidate_dist = math.hypot(raw_x - self.candidate_x, raw_y - self.candidate_y)
                if candidate_dist <= self.max_jump:
                    self.consecutive_outliers += 1
                else:
                    # Point jumped again to a completely different location
                    self.candidate_x = raw_x
                    self.candidate_y = raw_y
                    self.consecutive_outliers = 1
            else:
                self.candidate_x = raw_x
                self.candidate_y = raw_y
                self.consecutive_outliers = 1

            # If persistent at the new position for recovery_frames, accept the new position!
            if self.consecutive_outliers >= self.recovery_frames:
                self.filter_x.reset(t, raw_x)
                self.filter_y.reset(t, raw_y)
                self.last_x = int(round(raw_x))
                self.last_y = int(round(raw_y))
                self.consecutive_outliers = 0
                return self.last_x, self.last_y

            # Otherwise, reject this isolated jump and maintain last accepted position
            return self.last_x, self.last_y

        # Jump is within threshold -> legitimate continuous movement
        self.consecutive_outliers = 0

        if self.smoothing_enabled:
            sx = self.filter_x(t, raw_x)
            sy = self.filter_y(t, raw_y)
        else:
            sx, sy = raw_x, raw_y

        self.last_x = int(round(max(0.0, min(float(frame_width), sx))))
        self.last_y = int(round(max(0.0, min(float(frame_height), sy))))
        return self.last_x, self.last_y