"""Explainable rule-based geometric shape recognizer for Smart Ink.

Recognizes hand-drawn strokes as clean geometric primitives:
- LINE
- RECTANGLE
- CIRCLE
- TRIANGLE

Pipeline:
1. Point collection & deduplication.
2. Arc length, bounding box, and closure analysis.
3. Least-squares circle fitting and radial variance check.
4. Perpendicular distance line straightness assessment.
5. Adaptive Douglas-Peucker polygon corner extraction.
6. Convexity, angle, and aspect ratio validation.
7. Graceful fallback when confidence is insufficient.
"""

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class ShapeRecognitionResult:
    """Structured result returned by the GeometryRecognizer."""
    shape_type: str                         # "LINE", "RECTANGLE", "CIRCLE", "TRIANGLE", or "UNKNOWN"
    refined_points: List[Tuple[int, int]]   # Vertices or control points of the clean shape
    confidence: float                       # Score 0.0 - 1.0
    is_recognized: bool                     # True if confidence >= threshold
    rejection_reason: Optional[str]         # Reason for rejection if not recognized
    features: Dict[str, Any] = field(default_factory=dict)
    original_points: List[Tuple[int, int]] = field(default_factory=list)


class GeometryRecognizer:
    """Analyzes 2D stroke polylines and fits clean geometric primitives."""

    def __init__(self, min_confidence: float = 0.65):
        self.min_confidence = float(min_confidence)

    def recognize(self, points: List[Tuple[int, int]]) -> ShapeRecognitionResult:
        """Classify a completed 2D stroke into a geometric shape.

        Args:
            points: Sequence of (x, y) coordinates from a completed stroke.

        Returns:
            ShapeRecognitionResult with classified shape or freehand fallback.
        """
        if not points or len(points) < 5:
            return ShapeRecognitionResult(
                shape_type="UNKNOWN",
                refined_points=list(points or []),
                confidence=0.0,
                is_recognized=False,
                rejection_reason="Too few points (minimum 5 required)",
                original_points=list(points or []),
            )

        # ── 1. Deduplicate consecutive identical/near points ────────
        clean_pts: List[Tuple[float, float]] = []
        for p in points:
            if not clean_pts:
                clean_pts.append((float(p[0]), float(p[1])))
            else:
                dist = math.hypot(p[0] - clean_pts[-1][0], p[1] - clean_pts[-1][1])
                if dist >= 2.0:
                    clean_pts.append((float(p[0]), float(p[1])))

        if len(clean_pts) < 5:
            return ShapeRecognitionResult(
                shape_type="UNKNOWN",
                refined_points=list(points),
                confidence=0.0,
                is_recognized=False,
                rejection_reason="Stroke path too short after deduplication",
                original_points=list(points),
            )

        # ── 2. Compute Basic Geometric Features ─────────────────────
        np_pts = np.array(clean_pts, dtype=np.float32)
        n = len(clean_pts)

        # Arc length
        diffs = np_pts[1:] - np_pts[:-1]
        seg_lengths = np.hypot(diffs[:, 0], diffs[:, 1])
        total_length = float(np.sum(seg_lengths))

        if total_length < 25.0:
            return ShapeRecognitionResult(
                shape_type="UNKNOWN",
                refined_points=list(points),
                confidence=0.0,
                is_recognized=False,
                rejection_reason="Stroke total arc length too small",
                original_points=list(points),
            )

        # Start-to-end distance and closure ratio
        d_start_end = float(math.hypot(clean_pts[-1][0] - clean_pts[0][0], clean_pts[-1][1] - clean_pts[0][1]))
        closure_ratio = d_start_end / total_length

        # Bounding box
        x_min, y_min = float(np.min(np_pts[:, 0])), float(np.min(np_pts[:, 1]))
        x_max, y_max = float(np.max(np_pts[:, 0])), float(np.max(np_pts[:, 1]))
        bb_w = max(1.0, x_max - x_min)
        bb_h = max(1.0, y_max - y_min)
        aspect_ratio = bb_w / bb_h
        bb_diag = math.hypot(bb_w, bb_h)

        features = {
            "total_length": total_length,
            "d_start_end": d_start_end,
            "closure_ratio": closure_ratio,
            "bb_w": bb_w,
            "bb_h": bb_h,
            "aspect_ratio": aspect_ratio,
            "bb_diag": bb_diag,
        }

        # ── 3. Test Candidates ──────────────────────────────────────

        # Candidate A: Straight Line (Open stroke with high straightness)
        line_res = self._fit_line(clean_pts, np_pts, total_length, d_start_end, closure_ratio)
        if line_res and line_res.confidence >= self.min_confidence:
            line_res.features = features
            line_res.original_points = list(points)
            return line_res

        # Closed shape candidates require approximate closure (closure_ratio < 0.40 or start near end)
        if closure_ratio < 0.45 or d_start_end < max(35.0, 0.25 * bb_diag):
            # Candidate B: Circle
            circle_res = self._fit_circle(clean_pts, np_pts, bb_w, bb_h, aspect_ratio, total_length)
            if circle_res and circle_res.confidence >= self.min_confidence:
                circle_res.features = features
                circle_res.original_points = list(points)
                return circle_res

            # Candidate C: Rectangle or Triangle via Polygon Approximation
            poly_res = self._fit_polygon(clean_pts, np_pts, bb_w, bb_h, total_length)
            if poly_res and poly_res.confidence >= self.min_confidence:
                poly_res.features = features
                poly_res.original_points = list(points)
                return poly_res

        # Fallback: Ambiguous / low-confidence stroke
        return ShapeRecognitionResult(
            shape_type="UNKNOWN",
            refined_points=list(points),
            confidence=0.0,
            is_recognized=False,
            rejection_reason="Ambiguous stroke geometry; preserved freehand stroke",
            features=features,
            original_points=list(points),
        )

    # ── Shape Fitting Helpers ───────────────────────────────────

    def _fit_line(
        self,
        clean_pts: List[Tuple[float, float]],
        np_pts: np.ndarray,
        total_length: float,
        d_start_end: float,
        closure_ratio: float,
    ) -> Optional[ShapeRecognitionResult]:
        """Fit straight line segment between endpoints."""
        # Line must be open (closure_ratio must be reasonably high)
        if closure_ratio < 0.65:
            return None

        p_start = np_pts[0]
        p_end = np_pts[-1]
        line_vec = p_end - p_start
        line_len = math.hypot(line_vec[0], line_vec[1])

        if line_len < 10.0:
            return None

        # Perpendicular distance from each point to the line
        # Formula: |(x2-x1)(y1-y0) - (x1-x0)(y2-y1)| / line_len
        cross = np.abs(line_vec[0] * (p_start[1] - np_pts[:, 1]) - (p_start[0] - np_pts[:, 0]) * line_vec[1])
        perp_dists = cross / line_len

        max_dev = float(np.max(perp_dists))
        mean_dev = float(np.mean(perp_dists))

        # Straightness ratio: ratio of chord length to total path length, penalized by max deviation
        chord_ratio = line_len / total_length
        rel_max_dev = max_dev / line_len

        if rel_max_dev < 0.16 and chord_ratio > 0.82 and mean_dev / line_len < 0.08:
            confidence = max(0.0, min(1.0, 0.5 * chord_ratio + 0.5 * (1.0 - rel_max_dev * 4.0)))
            refined = [
                (int(round(p_start[0])), int(round(p_start[1]))),
                (int(round(p_end[0])), int(round(p_end[1]))),
            ]
            return ShapeRecognitionResult(
                shape_type="LINE",
                refined_points=refined,
                confidence=confidence,
                is_recognized=True,
                rejection_reason=None,
            )
        return None

    def _fit_circle(
        self,
        clean_pts: List[Tuple[float, float]],
        np_pts: np.ndarray,
        bb_w: float,
        bb_h: float,
        aspect_ratio: float,
        total_length: float,
    ) -> Optional[ShapeRecognitionResult]:
        """Fit a circle using algebraic centroid and evaluate radial variance."""
        # Circle must have roughly 1:1 bounding box
        if not (0.60 <= aspect_ratio <= 1.65):
            return None

        center_x = float(np.mean(np_pts[:, 0]))
        center_y = float(np.mean(np_pts[:, 1]))

        # Calculate radius for all points
        radii = np.hypot(np_pts[:, 0] - center_x, np_pts[:, 1] - center_y)
        r_mean = float(np.mean(radii))

        if r_mean < 8.0:
            return None

        # Radial variance & standard deviation
        r_std = float(np.std(radii))
        rel_std = r_std / r_mean

        # Expected circumference vs actual arc length
        circ = 2.0 * math.pi * r_mean
        length_ratio = total_length / circ

        # Circles should have low radial variance and arc length near 2*pi*R (between 0.75 and 1.45)
        if rel_std < 0.20 and 0.70 <= length_ratio <= 1.55:
            confidence = max(0.0, min(1.0, 1.0 - 2.8 * rel_std))

            # Discretize clean circle into a 36-point polygon
            num_samples = 36
            refined = []
            for i in range(num_samples):
                theta = 2.0 * math.pi * i / num_samples
                px = int(round(center_x + r_mean * math.cos(theta)))
                py = int(round(center_y + r_mean * math.sin(theta)))
                refined.append((px, py))

            return ShapeRecognitionResult(
                shape_type="CIRCLE",
                refined_points=refined,
                confidence=confidence,
                is_recognized=True,
                rejection_reason=None,
            )
        return None

    def _fit_polygon(
        self,
        clean_pts: List[Tuple[float, float]],
        np_pts: np.ndarray,
        bb_w: float,
        bb_h: float,
        total_length: float,
    ) -> Optional[ShapeRecognitionResult]:
        """Simplify contour and classify triangles or rectangles."""
        int_pts = np.array(clean_pts, dtype=np.int32).reshape((-1, 1, 2))

        # Try multiple Douglas-Peucker epsilon values to find clean 3-gon or 4-gon
        best_candidate: Optional[ShapeRecognitionResult] = None

        for eps_factor in [0.03, 0.045, 0.06, 0.02]:
            epsilon = eps_factor * cv2.arcLength(int_pts, True)
            approx = cv2.approxPolyDP(int_pts, epsilon, True)
            num_vertices = len(approx)

            # ── Triangle Test (3 vertices) ──────────────────────────
            if num_vertices == 3:
                tri_pts = approx.reshape((3, 2))
                area = cv2.contourArea(approx)
                bb_area = max(1.0, bb_w * bb_h)

                if cv2.isContourConvex(approx) and area > 0.12 * bb_area:
                    # Calculate angles
                    angles = self._compute_interior_angles(tri_pts)
                    if all(15.0 <= a <= 150.0 for a in angles):
                        conf = max(0.0, min(1.0, 0.75 + 0.25 * (area / bb_area)))
                        refined = [(int(p[0]), int(p[1])) for p in tri_pts]
                        return ShapeRecognitionResult(
                            shape_type="TRIANGLE",
                            refined_points=refined,
                            confidence=conf,
                            is_recognized=True,
                            rejection_reason=None,
                        )

            # ── Rectangle Test (4 vertices) ─────────────────────────
            elif num_vertices == 4:
                rect_pts = approx.reshape((4, 2))
                area = cv2.contourArea(approx)
                bb_area = max(1.0, bb_w * bb_h)

                if cv2.isContourConvex(approx) and area > 0.45 * bb_area:
                    angles = self._compute_interior_angles(rect_pts)
                    # Rectangles have corner angles near 90 degrees (60-120 deg)
                    if all(55.0 <= a <= 125.0 for a in angles):
                        angle_devs = [abs(a - 90.0) for a in angles]
                        conf = max(0.0, min(1.0, 1.0 - (sum(angle_devs) / 120.0)))

                        # Use minimum area rotated rectangle for crisp geometry
                        rect = cv2.minAreaRect(int_pts)
                        box = cv2.boxPoints(rect)
                        refined = [(int(round(p[0])), int(round(p[1]))) for p in box]

                        return ShapeRecognitionResult(
                            shape_type="RECTANGLE",
                            refined_points=refined,
                            confidence=conf,
                            is_recognized=True,
                            rejection_reason=None,
                        )

        return None

    @staticmethod
    def _compute_interior_angles(vertices: np.ndarray) -> List[float]:
        """Compute interior angles in degrees for each vertex of a polygon."""
        num = len(vertices)
        angles = []
        for i in range(num):
            p_prev = vertices[(i - 1) % num]
            p_curr = vertices[i]
            p_next = vertices[(i + 1) % num]

            v1 = (float(p_prev[0] - p_curr[0]), float(p_prev[1] - p_curr[1]))
            v2 = (float(p_next[0] - p_curr[0]), float(p_next[1] - p_curr[1]))

            norm1 = math.hypot(v1[0], v1[1])
            norm2 = math.hypot(v2[0], v2[1])

            if norm1 < 1e-4 or norm2 < 1e-4:
                angles.append(0.0)
                continue

            dot = (v1[0] * v2[0] + v1[1] * v2[1]) / (norm1 * norm2)
            dot = max(-1.0, min(1.0, dot))
            deg = math.degrees(math.acos(dot))
            angles.append(deg)
        return angles
