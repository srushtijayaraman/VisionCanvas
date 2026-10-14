"""Unit tests for FingerDetector and GestureRecognizer."""

import math
import pytest
from src.gestures.gesture_recognizer import GestureRecognizer
from src.hand_tracking.finger_detector import FingerDetector


class Landmark:
    def __init__(self, x: float, y: float, z: float = 0.0):
        self.x = x
        self.y = y
        self.z = z


def build_synthetic_hand(
    wrist=(0.5, 0.8),
    angle_rad=0.0,
    extended_fingers=(False, True, False, False, False),
    thumb_extended=False,
):
    """Generate 21 synthetic hand landmarks at a given rotation angle."""
    wx, wy = wrist
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    def rot(dx, dy):
        return (wx + dx * cos_a - dy * sin_a, wy + dx * sin_a + dy * cos_a)

    # 21 landmarks
    lms = [None] * 21
    lms[0] = Landmark(*wrist)

    # Thumb: 1: CMC, 2: MCP, 3: IP, 4: TIP
    if thumb_extended:
        lms[1] = Landmark(*rot(-0.04, -0.04))
        lms[2] = Landmark(*rot(-0.08, -0.07))
        lms[3] = Landmark(*rot(-0.12, -0.10))
        lms[4] = Landmark(*rot(-0.16, -0.13))
    else:
        lms[1] = Landmark(*rot(-0.03, -0.04))
        lms[2] = Landmark(*rot(-0.05, -0.06))
        lms[3] = Landmark(*rot(-0.04, -0.08))
        lms[4] = Landmark(*rot(-0.02, -0.09))  # Tucked near index MCP

    # Finger MCP bases (Index 5, Middle 9, Ring 13, Pinky 17)
    base_offsets = [(-0.04, -0.12), (0.0, -0.13), (0.04, -0.12), (0.07, -0.10)]
    finger_indices = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]

    for ext, (mcp_id, pip_id, dip_id, tip_id), (bx, by) in zip(
        extended_fingers[1:], finger_indices, base_offsets
    ):
        lms[mcp_id] = Landmark(*rot(bx, by))
        if ext:
            # Extended straight out away from wrist
            lms[pip_id] = Landmark(*rot(bx, by - 0.08))
            lms[dip_id] = Landmark(*rot(bx, by - 0.15))
            lms[tip_id] = Landmark(*rot(bx, by - 0.22))
        else:
            # Curled into palm
            lms[pip_id] = Landmark(*rot(bx, by - 0.06))
            lms[dip_id] = Landmark(*rot(bx, by - 0.03))
            lms[tip_id] = Landmark(*rot(bx, by - 0.01))

    return lms


def test_pointing_gesture_draw():
    """Index finger extended should classify as DRAW."""
    detector = FingerDetector()
    recognizer = GestureRecognizer()

    lms = build_synthetic_hand(extended_fingers=(False, True, False, False, False), thumb_extended=False)
    states = detector.get_finger_states(lms)
    assert states[1] == 1  # Index
    assert states[2] == 0  # Middle
    assert states[3] == 0  # Ring
    assert states[4] == 0  # Pinky

    gesture = recognizer.recognize(states)
    assert gesture == "DRAW"


def test_two_fingers_select():
    """Index + Middle extended should classify as SELECT."""
    detector = FingerDetector()
    recognizer = GestureRecognizer()

    lms = build_synthetic_hand(extended_fingers=(False, True, True, False, False), thumb_extended=False)
    states = detector.get_finger_states(lms)
    assert states[1] == 1
    assert states[2] == 1
    assert states[3] == 0
    assert states[4] == 0

    gesture = recognizer.recognize(states)
    assert gesture == "SELECT"


def test_three_fingers_undo():
    """Three fingers extended should classify as UNDO."""
    detector = FingerDetector()
    recognizer = GestureRecognizer()

    lms = build_synthetic_hand(extended_fingers=(False, True, True, True, False), thumb_extended=False)
    states = detector.get_finger_states(lms)
    gesture = recognizer.recognize(states)
    assert gesture == "UNDO"


def test_four_fingers_redo():
    """Four fingers extended should classify as REDO."""
    detector = FingerDetector()
    recognizer = GestureRecognizer()

    lms = build_synthetic_hand(extended_fingers=(False, True, True, True, True), thumb_extended=False)
    states = detector.get_finger_states(lms)
    gesture = recognizer.recognize(states)
    assert gesture == "REDO"


def test_thumbs_up_save():
    """Thumbs up pose with other fingers curled should classify as SAVE."""
    detector = FingerDetector()
    recognizer = GestureRecognizer()

    lms = build_synthetic_hand(extended_fingers=(False, False, False, False, False), thumb_extended=True)
    states = detector.get_finger_states(lms)
    assert states[0] == 1  # Thumb
    assert states[1:] == [0, 0, 0, 0]

    gesture = recognizer.recognize(states)
    assert gesture == "SAVE"


def test_fist_none():
    """Closed fist should classify as NONE."""
    detector = FingerDetector()
    recognizer = GestureRecognizer()

    lms = build_synthetic_hand(extended_fingers=(False, False, False, False, False), thumb_extended=False)
    states = detector.get_finger_states(lms)
    assert states == [0, 0, 0, 0, 0]

    gesture = recognizer.recognize(states)
    assert gesture == "NONE"


def test_rotation_invariance():
    """Rotation-invariant finger detector should detect pointing when tilted 45 degrees."""
    detector = FingerDetector()
    recognizer = GestureRecognizer()

    # Hand rotated 45 degrees clockwise
    lms = build_synthetic_hand(
        wrist=(0.5, 0.5),
        angle_rad=math.pi / 4.0,
        extended_fingers=(False, True, False, False, False),
        thumb_extended=False,
    )
    states = detector.get_finger_states(lms)
    assert states[1] == 1  # Index still detected as extended
    assert states[2] == 0
    assert states[3] == 0
    assert states[4] == 0
