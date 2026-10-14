"""Unit tests for GestureStateMachine transitions, debouncing, and one-shot latching."""

import pytest
from src.gestures.state_machine import GestureStateMachine


def test_gesture_continuous_activation():
    """Continuous gestures (DRAW, SELECT) should become active after stability frames."""
    sm = GestureStateMachine(stability_frames=2, cooldown_time=0.4)

    # Frame 1: pending
    ev1 = sm.update("DRAW", timestamp=0.0)
    assert ev1.active_gesture == "NONE"

    # Frame 2: stabilized
    ev2 = sm.update("DRAW", timestamp=0.033)
    assert ev2.active_gesture == "DRAW"
    assert ev2.is_continuous is True
    assert ev2.triggered_action is None


def test_gesture_one_shot_does_not_repeat_on_sustained_pose():
    """Holding a one-shot pose (e.g. UNDO, SAVE) for multiple seconds should trigger ONCE."""
    sm = GestureStateMachine(stability_frames=2, cooldown_time=0.4)

    # Stabilize into UNDO
    sm.update("UNDO", timestamp=0.0)
    ev2 = sm.update("UNDO", timestamp=0.033)
    assert ev2.active_gesture == "UNDO"
    assert ev2.triggered_action == "UNDO"  # Fired on rising edge!

    # Sustained for 50 more frames (e.g. 1.5 seconds)
    triggered_count = 0
    for i in range(3, 50):
        t = i * 0.033
        ev = sm.update("UNDO", timestamp=t)
        if ev.triggered_action == "UNDO":
            triggered_count += 1

    # Must NOT have triggered again while sustained
    assert triggered_count == 0


def test_gesture_one_shot_reactivation_after_release():
    """One-shot action can trigger again only after being released to neutral and cooldown elapses."""
    sm = GestureStateMachine(stability_frames=2, cooldown_time=0.3)

    # 1. Trigger UNDO
    sm.update("UNDO", timestamp=0.0)
    ev = sm.update("UNDO", timestamp=0.033)
    assert ev.triggered_action == "UNDO"

    # 2. Release pose (back to NONE)
    sm.update("NONE", timestamp=0.1)
    sm.update("NONE", timestamp=0.15)
    assert sm.current_stable_gesture == "NONE"

    # 3. Enter UNDO again BEFORE cooldown (at 0.2s): should not trigger
    sm.update("UNDO", timestamp=0.20)
    ev_early = sm.update("UNDO", timestamp=0.23)
    assert ev_early.triggered_action is None

    # 4. Release again and enter AFTER cooldown (at 0.5s): triggers!
    sm.update("NONE", timestamp=0.35)
    sm.update("NONE", timestamp=0.40)
    sm.update("UNDO", timestamp=0.50)
    ev_later = sm.update("UNDO", timestamp=0.53)
    assert ev_later.triggered_action == "UNDO"


def test_gesture_noisy_transition_filtering():
    """A single-frame spurious gesture spike should not trigger a transition."""
    sm = GestureStateMachine(stability_frames=3, cooldown_time=0.4)

    # Currently DRAWing
    sm.update("DRAW", timestamp=0.0)
    sm.update("DRAW", timestamp=0.033)
    sm.update("DRAW", timestamp=0.066)
    assert sm.current_stable_gesture == "DRAW"

    # Spurious 1-frame flicker to UNDO
    ev_spike = sm.update("UNDO", timestamp=0.099)
    assert ev_spike.active_gesture == "DRAW"
    assert ev_spike.triggered_action is None

    # Resumes DRAW
    ev_resume = sm.update("DRAW", timestamp=0.132)
    assert ev_resume.active_gesture == "DRAW"


def test_gesture_tracking_loss():
    """Tracking loss should reset stable gesture to NONE and disarm actions."""
    sm = GestureStateMachine(stability_frames=2)

    sm.update("DRAW", timestamp=0.0)
    sm.update("DRAW", timestamp=0.033)
    assert sm.current_stable_gesture == "DRAW"

    sm.on_tracking_lost()
    assert sm.current_stable_gesture == "NONE"
    assert sm.active_one_shot is None
