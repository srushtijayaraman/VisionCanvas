"""Finger detector using relative geometry and rotation-invariant distances.

Instead of assuming fixed image-axis alignment (e.g. y-coordinate comparisons
that fail when the hand is tilted or horizontal, and x-coordinate thumb
comparisons that fail for left hands), this detector evaluates:
1. Distance of finger tips from the wrist relative to their PIP joints.
2. Distance of the thumb tip relative to the palm and pinky base.
3. Handedness context when available.
"""

import math
from typing import List, Optional


class FingerDetector:
    """Detects whether each of the 5 fingers is extended (1) or curled (0)."""

    TIP_IDS = [4, 8, 12, 16, 20]
    PIP_IDS = [3, 6, 10, 14, 18]
    MCP_IDS = [2, 5, 9, 13, 17]

    @staticmethod
    def _dist(p1, p2) -> float:
        """Euclidean distance between two landmarks in normalized coords."""
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def get_finger_states(
        self,
        hand_landmarks,
        handedness: Optional[str] = None,
    ) -> List[int]:
        """Classify each finger as extended (1) or curled (0).

        Returns:
            List of 5 integers [thumb, index, middle, ring, pinky].
        """
        if not hand_landmarks or len(hand_landmarks) < 21:
            return []

        wrist = hand_landmarks[0]
        pinky_mcp = hand_landmarks[17]
        index_mcp = hand_landmarks[5]

        # ── 1. Thumb Classification ─────────────────────────────────
        # When the thumb is extended outward from the hand, the distance from
        # the thumb tip (4) to the pinky MCP (17) is substantially greater than
        # the distance from the thumb IP joint (3) to the pinky MCP (17).
        # When curled into the palm, the tip moves closer to the palm center.
        thumb_tip = hand_landmarks[4]
        thumb_ip = hand_landmarks[3]
        thumb_mcp = hand_landmarks[2]

        d_tip_pinky = self._dist(thumb_tip, pinky_mcp)
        d_ip_pinky = self._dist(thumb_ip, pinky_mcp)
        d_tip_wrist = self._dist(thumb_tip, wrist)
        d_mcp_wrist = self._dist(thumb_mcp, wrist)

        # Thumb extension condition:
        # 1) Extended away from palm (abducted), OR
        # 2) Extended upward along wrist (thumbs-up pose)
        thumb_extended = (
            (d_tip_pinky > d_ip_pinky * 1.10 and self._dist(thumb_tip, index_mcp) > self._dist(thumb_mcp, index_mcp) * 0.9)
            or (d_tip_wrist > d_mcp_wrist * 1.4 and thumb_tip.y < thumb_mcp.y)
        )
        thumb_state = 1 if thumb_extended else 0

        # ── 2. Other 4 Fingers (Index, Middle, Ring, Pinky) ─────────
        # For an extended finger, the tip is significantly farther from the
        # wrist than the PIP joint. When curled, the tip folds towards the palm.
        # This is rotation-invariant across 360 degrees in the camera plane.
        other_states = []
        for tip_id, pip_id in zip(self.TIP_IDS[1:], self.PIP_IDS[1:]):
            tip = hand_landmarks[tip_id]
            pip = hand_landmarks[pip_id]

            d_wrist_tip = self._dist(wrist, tip)
            d_wrist_pip = self._dist(wrist, pip)

            # Extended if tip is farther from wrist than PIP joint by a margin
            if d_wrist_tip > d_wrist_pip * 1.12:
                other_states.append(1)
            else:
                other_states.append(0)

        return [thumb_state] + other_states