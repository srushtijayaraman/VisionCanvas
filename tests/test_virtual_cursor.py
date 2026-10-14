"""Unit tests for VirtualCursor stability, outlier rejection, and recovery.

Scenarios tested:
1. Stationary input with small positional noise.
2. Smooth movement across a canvas.
3. A single large outlier.
4. Persistent movement to a distant position.
5. Temporary tracking loss.
6. Hand reacquisition after tracking loss.
7. Invalid and missing coordinates.
8. Filter reinitialization.
"""

import math
import pytest
from src.gestures.cursor import VirtualCursor


class MockLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_cursor_stationary_input_with_noise():
    """1. Stationary input with small positional noise should be smoothed."""
    cursor = VirtualCursor(max_jump=100.0, recovery_frames=3)
    coords = []
    for i in range(25):
        t = i * 0.033
        # Small noise around (0.5, 0.5)
        noise = 0.002 * (1.0 if i % 2 == 0 else -1.0)
        lm = MockLandmark(0.5 + noise, 0.5 + noise)
        cx, cy = cursor.update(lm, frame_width=1000, frame_height=1000, timestamp=t)
        coords.append((cx, cy))

    # All coordinates should remain tightly grouped around (500, 500)
    for cx, cy in coords[5:]:
        assert abs(cx - 500) < 5
        assert abs(cy - 500) < 5


def test_cursor_smooth_movement():
    """2. Smooth movement across a canvas should track continuously."""
    cursor = VirtualCursor(max_jump=150.0, recovery_frames=3)
    outputs = []
    for i in range(30):
        t = i * 0.033
        norm_pos = 0.1 + (i / 30.0) * 0.6  # 0.1 to 0.7
        lm = MockLandmark(norm_pos, norm_pos)
        cx, cy = cursor.update(lm, frame_width=1000, frame_height=1000, timestamp=t)
        outputs.append((cx, cy))

    # Cursor should monotonically increase across the screen
    for i in range(1, len(outputs)):
        assert outputs[i][0] >= outputs[i - 1][0]
        assert outputs[i][1] >= outputs[i - 1][1]

    # Reaches near destination
    assert outputs[-1][0] > 600
    assert outputs[-1][1] > 600


def test_cursor_single_large_outlier():
    """3. A single large outlier jump should be rejected, preserving previous position."""
    cursor = VirtualCursor(max_jump=80.0, recovery_frames=3)

    # Establish baseline at (0.3, 0.3) -> 300, 300
    for i in range(5):
        cursor.update(MockLandmark(0.3, 0.3), frame_width=1000, frame_height=1000, timestamp=i * 0.033)

    accepted_x, accepted_y = cursor.last_x, cursor.last_y
    assert abs(accepted_x - 300) < 5

    # Sudden glitch to (0.9, 0.9) -> 900, 900 (jump of ~848px)
    cx, cy = cursor.update(MockLandmark(0.9, 0.9), frame_width=1000, frame_height=1000, timestamp=5 * 0.033)
    assert cx == accepted_x
    assert cy == accepted_y

    # Next frame returns to normal (0.31, 0.31)
    cx, cy = cursor.update(MockLandmark(0.31, 0.31), frame_width=1000, frame_height=1000, timestamp=6 * 0.033)
    assert abs(cx - 310) < 15


def test_cursor_persistent_movement_recovery():
    """4. Persistent movement to distant position should be accepted after N frames."""
    cursor = VirtualCursor(max_jump=80.0, recovery_frames=3)

    # Establish baseline at (0.2, 0.2) -> 200, 200
    for i in range(5):
        cursor.update(MockLandmark(0.2, 0.2), frame_width=1000, frame_height=1000, timestamp=i * 0.033)

    # Frame 1 at distant position (0.8, 0.8) -> 800, 800: rejected
    c1 = cursor.update(MockLandmark(0.8, 0.8), frame_width=1000, frame_height=1000, timestamp=0.20)
    assert c1[0] < 300

    # Frame 2 at distant position: still rejected
    c2 = cursor.update(MockLandmark(0.8, 0.8), frame_width=1000, frame_height=1000, timestamp=0.23)
    assert c2[0] < 300

    # Frame 3 at distant position: accepted! Cursor must not stay stuck
    c3 = cursor.update(MockLandmark(0.8, 0.8), frame_width=1000, frame_height=1000, timestamp=0.26)
    assert c3[0] == 800
    assert c3[1] == 800


def test_cursor_temporary_tracking_loss():
    """5. Temporary tracking loss should notify cursor without crashing."""
    cursor = VirtualCursor()
    cursor.update(MockLandmark(0.4, 0.4), 1000, 1000, timestamp=0.0)
    assert cursor.is_tracking is True

    # Tracking lost (landmark=None)
    cx, cy = cursor.update(None, 1000, 1000, timestamp=0.033)
    assert cursor.is_tracking is False
    assert cx == cursor.last_x


def test_cursor_reacquisition_after_tracking_loss():
    """6. Reacquisition after tracking loss reinitializes smoothly without jump rejection."""
    cursor = VirtualCursor(max_jump=50.0)

    # Hand initially at (0.1, 0.1) -> 100, 100
    cursor.update(MockLandmark(0.1, 0.1), 1000, 1000, timestamp=0.0)

    # Tracking lost for several frames
    cursor.update(None, 1000, 1000, timestamp=0.033)
    cursor.update(None, 1000, 1000, timestamp=0.066)

    # Hand reacquired far away at (0.8, 0.8) -> 800, 800
    # Must immediately acquire new position rather than rejecting as a jump!
    cx, cy = cursor.update(MockLandmark(0.8, 0.8), 1000, 1000, timestamp=0.1)
    assert cx == 800
    assert cy == 800
    assert cursor.is_tracking is True


def test_cursor_invalid_and_missing_coordinates():
    """7. Invalid (NaN, Inf, non-numeric) coordinates should be safely handled."""
    cursor = VirtualCursor()
    cursor.update(MockLandmark(0.5, 0.5), 1000, 1000, timestamp=0.0)

    # NaN coordinates
    cx, cy = cursor.update(MockLandmark(float("nan"), 0.5), 1000, 1000, timestamp=0.033)
    assert cursor.is_tracking is False

    # Inf coordinates
    cx, cy = cursor.update(MockLandmark(0.5, float("inf")), 1000, 1000, timestamp=0.066)
    assert cursor.is_tracking is False

    # Object without coordinates
    cx, cy = cursor.update("invalid_object", 1000, 1000, timestamp=0.1)
    assert cursor.is_tracking is False


def test_cursor_reset():
    """8. Explicit reset reinitializes cursor state completely."""
    cursor = VirtualCursor()
    cursor.update(MockLandmark(0.5, 0.5), 1000, 1000, timestamp=0.0)
    assert cursor.filter_x is not None

    cursor.reset()
    assert cursor.filter_x is None
    assert cursor.filter_y is None
    assert cursor.is_tracking is False
    assert cursor.last_x == 0
    assert cursor.last_y == 0
