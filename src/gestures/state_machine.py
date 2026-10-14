"""Gesture state machine for robust action dispatch and temporal stabilization.

Features:
- Distinguishes continuous actions (DRAW, SELECT) from one-shot actions (UNDO, REDO, SAVE).
- Requires configurable temporal stability before accepting gesture changes.
- Requires deliberate gesture release before a one-shot action can re-fire.
- Cooldown timer prevents accidental double-triggers.
- Cleanly handles tracking loss and ambiguous inputs.
"""

import time
from dataclasses import dataclass
from typing import Optional, Set

from src.config import GESTURE_COOLDOWN_TIME, GESTURE_STABILITY_FRAMES


@dataclass
class GestureEvent:
    """Result of processing a frame through the gesture state machine."""
    active_gesture: str
    triggered_action: Optional[str]
    is_continuous: bool
    raw_gesture: str


class GestureStateMachine:
    """Manages gesture transitions and dispatches one-shot vs continuous actions."""

    ONE_SHOT_GESTURES: Set[str] = {"UNDO", "REDO", "SAVE", "CLEAR"}
    CONTINUOUS_GESTURES: Set[str] = {"DRAW", "SELECT", "NONE"}

    def __init__(
        self,
        stability_frames: int = GESTURE_STABILITY_FRAMES,
        cooldown_time: float = GESTURE_COOLDOWN_TIME,
    ):
        self.stability_frames = max(1, int(stability_frames))
        self.cooldown_time = float(cooldown_time)

        # State tracking
        self.current_stable_gesture: str = "NONE"
        self.pending_gesture: str = "NONE"
        self.pending_count: int = 0

        # One-shot latching & cooldowns
        self.active_one_shot: Optional[str] = None
        self.last_action_timestamp: float = -1000.0

    def reset(self) -> None:
        """Reset state machine to initial neutral state."""
        self.current_stable_gesture = "NONE"
        self.pending_gesture = "NONE"
        self.pending_count = 0
        self.active_one_shot = None
        self.last_action_timestamp = -1000.0

    def on_tracking_lost(self) -> None:
        """Notify state machine that hand tracking was lost."""
        self.current_stable_gesture = "NONE"
        self.pending_gesture = "NONE"
        self.pending_count = 0
        self.active_one_shot = None

    def update(
        self,
        raw_gesture: Optional[str],
        timestamp: Optional[float] = None,
    ) -> GestureEvent:
        """Process a raw gesture reading and return the resulting GestureEvent.

        Args:
            raw_gesture: Classified gesture name (e.g. "DRAW", "UNDO", "NONE") or None.
            timestamp: Optional explicit time in seconds (defaults to time.time()).

        Returns:
            GestureEvent detailing the active continuous gesture and any triggered action.
        """
        if timestamp is None:
            now = time.time()
        else:
            now = float(timestamp)

        if not raw_gesture or raw_gesture == "NONE":
            # Hand absent or neutral pose: release any latched one-shot gesture
            self.current_stable_gesture = "NONE"
            self.pending_gesture = "NONE"
            self.pending_count = 0
            self.active_one_shot = None
            return GestureEvent(
                active_gesture="NONE",
                triggered_action=None,
                is_continuous=False,
                raw_gesture="NONE",
            )

        # Temporal stability check: require N consecutive frames of same raw gesture
        if raw_gesture == self.pending_gesture:
            self.pending_count += 1
        else:
            self.pending_gesture = raw_gesture
            self.pending_count = 1

        new_stable = False
        if self.pending_count >= self.stability_frames:
            if self.current_stable_gesture != self.pending_gesture:
                self.current_stable_gesture = self.pending_gesture
                new_stable = True

        triggered_action: Optional[str] = None

        # Check if the stable gesture is a one-shot action
        if self.current_stable_gesture in self.ONE_SHOT_GESTURES:
            # Action only triggers once on entry, after cooldown has elapsed
            if new_stable and self.active_one_shot != self.current_stable_gesture:
                if (now - self.last_action_timestamp) >= self.cooldown_time:
                    triggered_action = self.current_stable_gesture
                    self.active_one_shot = self.current_stable_gesture
                    self.last_action_timestamp = now
        else:
            # Entering a non-one-shot gesture releases the latch
            self.active_one_shot = None

        is_continuous = self.current_stable_gesture in self.CONTINUOUS_GESTURES and self.current_stable_gesture != "NONE"

        return GestureEvent(
            active_gesture=self.current_stable_gesture,
            triggered_action=triggered_action,
            is_continuous=is_continuous,
            raw_gesture=raw_gesture,
        )
