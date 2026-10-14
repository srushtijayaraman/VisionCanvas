"""Drawing canvas with stroke interpolation, undo/redo history, alpha masking,

and multi-format image export.
"""

import math
import os
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from src.brushes.base_brush import BaseBrush
from src.config import MAX_INTERPOLATION_DISTANCE


class DrawingCanvas:
    """Manages the drawing surface, stroke rendering, undo/redo history,

    alpha composition mask, and export formatting.
    """

    def __init__(
        self,
        width: int,
        height: int,
        brush: BaseBrush,
        max_undo: int = 25,
    ):
        self.width = int(width)
        self.height = int(height)
        self.brush = brush
        self.max_undo = max(1, int(max_undo))

        # Core canvas layers: BGR image and single-channel alpha mask (0-255)
        self.canvas: np.ndarray = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.mask: np.ndarray = np.zeros((self.height, self.width), dtype=np.uint8)

        self.prev_point: Optional[Tuple[int, int]] = None
        self.is_drawing: bool = False
        self.is_eraser: bool = False

        # History stacks store tuples of (canvas_copy, mask_copy)
        self.undo_stack: List[Tuple[np.ndarray, np.ndarray]] = [
            (self.canvas.copy(), self.mask.copy())
        ]
        self.redo_stack: List[Tuple[np.ndarray, np.ndarray]] = []

    def set_brush(self, brush: BaseBrush, is_eraser: bool = False) -> None:
        """Switch current brush."""
        self.brush = brush
        self.is_eraser = is_eraser

    def draw(self, point: Tuple[int, int]) -> None:
        """Draw to the specified point, interpolating from previous point if active."""
        if point is None:
            return

        px, py = int(point[0]), int(point[1])
        # Clamp to canvas boundaries
        px = max(0, min(self.width - 1, px))
        py = max(0, min(self.height - 1, py))
        curr_point = (px, py)

        # Start of a new continuous stroke
        if not self.is_drawing:
            # Push state before stroke begins
            self.undo_stack.append((self.canvas.copy(), self.mask.copy()))
            if len(self.undo_stack) > self.max_undo:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
            self.is_drawing = True

        if self.prev_point is None:
            # Single tap / initial contact point: draw a solid circle dot
            thickness = getattr(self.brush, "thickness", 8)
            radius = max(1, thickness // 2)
            if self.is_eraser:
                cv2.circle(self.canvas, curr_point, radius, (0, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(self.mask, curr_point, radius, 0, -1, cv2.LINE_AA)
            else:
                color = getattr(self.brush, "color", (255, 0, 255))
                cv2.circle(self.canvas, curr_point, radius, color, -1, cv2.LINE_AA)
                cv2.circle(self.mask, curr_point, radius, 255, -1, cv2.LINE_AA)
            self.prev_point = curr_point
            return

        # Continuous movement: interpolate between prev_point and curr_point
        dx = curr_point[0] - self.prev_point[0]
        dy = curr_point[1] - self.prev_point[1]
        dist = math.hypot(dx, dy)
        max_dist = max(1.0, float(MAX_INTERPOLATION_DISTANCE))

        thickness = getattr(self.brush, "thickness", 8)

        if dist > max_dist:
            steps = int(dist / max_dist)
            for i in range(1, steps + 1):
                t_curr = i / steps
                t_prev = (i - 1) / steps

                p_prev = (
                    int(round(self.prev_point[0] + t_prev * dx)),
                    int(round(self.prev_point[1] + t_prev * dy)),
                )
                p_curr = (
                    int(round(self.prev_point[0] + t_curr * dx)),
                    int(round(self.prev_point[1] + t_curr * dy)),
                )

                if self.is_eraser:
                    cv2.line(self.canvas, p_prev, p_curr, (0, 0, 0), thickness, cv2.LINE_AA)
                    cv2.line(self.mask, p_prev, p_curr, 0, thickness, cv2.LINE_AA)
                else:
                    self.brush.stroke(self.canvas, p_prev, p_curr)
                    cv2.line(self.mask, p_prev, p_curr, 255, thickness, cv2.LINE_AA)
        else:
            if self.is_eraser:
                cv2.line(self.canvas, self.prev_point, curr_point, (0, 0, 0), thickness, cv2.LINE_AA)
                cv2.line(self.mask, self.prev_point, curr_point, 0, thickness, cv2.LINE_AA)
            else:
                self.brush.stroke(self.canvas, self.prev_point, curr_point)
                cv2.line(self.mask, self.prev_point, curr_point, 255, thickness, cv2.LINE_AA)

        self.prev_point = curr_point

    def reset_stroke(self) -> None:
        """End the current drawing stroke."""
        self.prev_point = None
        self.is_drawing = False

    def undo(self) -> bool:
        """Undo the most recent completed stroke or clear action.

        Returns:
            True if an action was undone, False if history stack is at base.
        """
        if len(self.undo_stack) > 1:
            self.redo_stack.append((self.canvas.copy(), self.mask.copy()))
            prev_canvas, prev_mask = self.undo_stack.pop()
            self.canvas = prev_canvas.copy()
            self.mask = prev_mask.copy()
            self.prev_point = None
            self.is_drawing = False
            return True
        return False

    def redo(self) -> bool:
        """Redo the most recently undone action.

        Returns:
            True if an action was redone, False if redo stack is empty.
        """
        if self.redo_stack:
            self.undo_stack.append((self.canvas.copy(), self.mask.copy()))
            redo_canvas, redo_mask = self.redo_stack.pop()
            self.canvas = redo_canvas.copy()
            self.mask = redo_mask.copy()
            self.prev_point = None
            self.is_drawing = False
            return True
        return False

    def clear(self) -> None:
        """Clear the entire canvas and push current state to the undo stack."""
        self.undo_stack.append((self.canvas.copy(), self.mask.copy()))
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

        self.canvas = np.zeros_like(self.canvas)
        self.mask = np.zeros_like(self.mask)
    def draw_shape(self, shape_type: str, points: List[Tuple[int, int]]) -> None:
        """Render a clean geometric shape as a single undoable canvas action.

        Args:
            shape_type: 'LINE', 'CIRCLE', 'RECTANGLE', or 'TRIANGLE'.
            points: Vertices / control points defining the shape.
        """
        if not points or len(points) < 2:
            return

        # Snapshot for undo
        self.undo_stack.append((self.canvas.copy(), self.mask.copy()))
        if len(self.undo_stack) > self.max_undo:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

        thickness = getattr(self.brush, "thickness", 8)

        if shape_type == "LINE":
            p1, p2 = points[0], points[1]
            self.brush.stroke(self.canvas, p1, p2)
            cv2.line(self.mask, p1, p2, 255, thickness, cv2.LINE_AA)
        elif shape_type in ("RECTANGLE", "TRIANGLE", "CIRCLE"):
            num_pts = len(points)
            for i in range(num_pts):
                p1 = points[i]
                p2 = points[(i + 1) % num_pts]
                self.brush.stroke(self.canvas, p1, p2)
                cv2.line(self.mask, p1, p2, 255, thickness, cv2.LINE_AA)

        self.prev_point = None
        self.is_drawing = False

    # ── Export methods ──────────────────────────────────────────

    def get_composed_image(self, background: str = "white") -> np.ndarray:
        """Return the drawing composed over a solid background or as RGBA.

        Args:
            background: 'white', 'black', or 'transparent'.
        """
        if background == "transparent":
            # 4-channel BGRA image
            b, g, r = cv2.split(self.canvas)
            return cv2.merge([b, g, r, self.mask])

        # Solid background blend
        bg_val = 255 if background == "white" else 0
        bg = np.full_like(self.canvas, bg_val, dtype=np.uint8)

        alpha = (self.mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        composed = (self.canvas.astype(np.float32) * alpha + bg.astype(np.float32) * (1.0 - alpha))
        return np.clip(composed, 0, 255).astype(np.uint8)

    def export_image(
        self,
        filepath: str,
        background: str = "white",
    ) -> bool:
        """Save the drawing to disk.

        Args:
            filepath: Destination file path (e.g. 'output/canvas_123.png').
            background: 'white', 'black', or 'transparent'.

        Returns:
            True if successfully saved, False on error.
        """
        try:
            target_path = Path(filepath)
            target_path.parent.mkdir(parents=True, exist_ok=True)

            img = self.get_composed_image(background=background)
            success = cv2.imwrite(str(target_path), img)
            return bool(success)
        except Exception:
            return False