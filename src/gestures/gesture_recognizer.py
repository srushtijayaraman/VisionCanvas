"""Gesture recognizer mapping finger configurations to application commands."""

from typing import List, Optional, Tuple


class GestureRecognizer:
    """Classifies 5-finger states [thumb, index, middle, ring, pinky] into named gestures.

    Supports exact configurations as well as natural thumb-relaxed variations
    to prevent flickering when the user naturally holds a pointing or victory pose.
    """

    # Canonical mappings
    EXACT_MAP = {
        (0, 1, 0, 0, 0): "DRAW",
        (1, 1, 0, 0, 0): "DRAW",    # Pointing with relaxed thumb
        (0, 1, 1, 0, 0): "SELECT",  # Victory / two-finger pointing
        (1, 1, 1, 0, 0): "SELECT",  # Two fingers with relaxed thumb
        (0, 1, 1, 1, 0): "UNDO",    # Three fingers
        (1, 1, 1, 1, 0): "UNDO",    # Three fingers with thumb
        (0, 1, 1, 1, 1): "REDO",    # Four fingers
        (1, 1, 1, 1, 1): "REDO",    # Five fingers / open palm
        (1, 0, 0, 0, 0): "SAVE",    # Thumbs up
    }

    def recognize(self, finger_states: Optional[List[int]]) -> str:
        """Classify finger states into a gesture string.

        Args:
            finger_states: List or tuple of 5 binary integers [thumb, index, middle, ring, pinky].

        Returns:
            Gesture name ("DRAW", "SELECT", "UNDO", "REDO", "SAVE", or "NONE").
        """
        if not finger_states or len(finger_states) != 5:
            return "NONE"

        key = tuple(finger_states)
        return self.EXACT_MAP.get(key, "NONE")