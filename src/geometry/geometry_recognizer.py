"""Explainable rule-based geometric shape recognizer for Smart Ink.

Recognizes hand-drawn strokes as clean geometric primitives:
- LINE
- RECTANGLE
- CIRCLE
- TRIANGLE
- ARROW

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

        # Closed shape candidates require approximate closure (closed within 22% perimeter or small endpoint gap)
        is_closed = (closure_ratio <= 0.22) or (d_start_end <= max(35.0, 0.22 * bb_diag) and closure_ratio <= 0.28)
        if is_closed:
            # Candidate B: Circle
            circle_res = self._fit_circle(clean_pts, np_pts, bb_w, bb_h, aspect_ratio, total_length, d_start_end, bb_diag)
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

        # Candidate D: Arrow (Open stroke with straight shaft and acute arrowhead)
        if closure_ratio >= 0.35:
            arrow_res = self._fit_arrow(clean_pts, np_pts, total_length, d_start_end)
            if arrow_res and arrow_res.confidence >= self.min_confidence:
                arrow_res.features = features
                arrow_res.original_points = list(points)
                return arrow_res

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

    def _fit_arrow(
        self,
        clean_pts: List[Tuple[float, float]],
        np_pts: np.ndarray,
        total_length: float,
        d_start_end: float,
    ) -> Optional[ShapeRecognitionResult]:
        """Fit a directional arrow composed of a straight shaft and acute arrowhead."""
        if len(clean_pts) < 6 or total_length < 35.0:
            return None

        int_pts = np.array(clean_pts, dtype=np.int32).reshape((-1, 1, 2))

        # Test polygon approximations to find shaft + arrowhead structure
        for eps_factor in [0.035, 0.05, 0.025]:
            epsilon = eps_factor * cv2.arcLength(int_pts, False)
            approx = cv2.approxPolyDP(int_pts, epsilon, False)
            if len(approx) < 3 or len(approx) > 7:
                continue

            verts = [p[0] for p in approx]
            num_v = len(verts)

            # Calculate edge lengths between consecutive simplified vertices
            edge_lens = [
                math.hypot(verts[i + 1][0] - verts[i][0], verts[i + 1][1] - verts[i][1])
                for i in range(num_v - 1)
            ]
            if not edge_lens:
                continue

            max_idx = int(np.argmax(edge_lens))
            max_edge = edge_lens[max_idx]

            # The shaft must be dominant (at least 45% of total edge sum and >= 25 px)
            sum_edges = sum(edge_lens)
            if max_edge < 25.0 or (max_edge / sum_edges) < 0.45:
                continue

            # Determine tail and tip:
            # Case 1: Shaft is the first edge (verts[0] -> verts[1])
            if max_idx == 0:
                tail = (float(verts[0][0]), float(verts[0][1]))
                tip = (float(verts[1][0]), float(verts[1][1]))
                head_verts = verts[1:]
            # Case 2: Shaft is the last edge (verts[-2] -> verts[-1])
            elif max_idx == num_v - 2:
                tail = (float(verts[-1][0]), float(verts[-1][1]))
                tip = (float(verts[-2][0]), float(verts[-2][1]))
                head_verts = verts[:-1]
            else:
                continue

            # Check that head vertices stay close to the tip (within 50% of shaft length)
            shaft_len = math.hypot(tip[0] - tail[0], tip[1] - tail[1])
            if shaft_len < 20.0:
                continue

            head_dists = [math.hypot(hv[0] - tip[0], hv[1] - tip[1]) for hv in head_verts]
            if any(hd > 0.50 * shaft_len for hd in head_dists):
                continue

            # Check directional alignment of arrowhead: barbs should point back along the shaft
            dx = tip[0] - tail[0]
            dy = tip[1] - tail[1]
            back_dx = -dx / shaft_len
            back_dy = -dy / shaft_len

            # Check if at least one head vector points back towards the tail
            valid_barb = False
            for hv in head_verts:
                h_dx = hv[0] - tip[0]
                h_dy = hv[1] - tip[1]
                h_len = math.hypot(h_dx, h_dy)
                if h_len > 4.0:
                    dot = (h_dx * back_dx + h_dy * back_dy) / h_len
                    if dot > 0.15:  # Angle with reverse shaft is acute (< 81 deg)
                        valid_barb = True
                        break

            if not valid_barb:
                continue

            # Generate clean symmetric arrow geometry
            angle = math.atan2(dy, dx)
            barb_len = max(10.0, min(40.0, shaft_len * 0.25))
            barb1 = (
                int(round(tip[0] - barb_len * math.cos(angle - math.radians(30)))),
                int(round(tip[1] - barb_len * math.sin(angle - math.radians(30)))),
            )
            barb2 = (
                int(round(tip[0] - barb_len * math.cos(angle + math.radians(30)))),
                int(round(tip[1] - barb_len * math.sin(angle + math.radians(30)))),
            )
            refined = [
                (int(round(tail[0])), int(round(tail[1]))),
                (int(round(tip[0])), int(round(tip[1]))),
                barb1,
                barb2,
            ]
            confidence = max(0.65, min(0.95, 0.60 + 0.35 * (max_edge / sum_edges)))
            return ShapeRecognitionResult(
                shape_type="ARROW",
                refined_points=refined,
                confidence=confidence,
                is_recognized=True,
                rejection_reason=None,
            )

        return None

    @staticmethod
    def _resample_polyline(pts: np.ndarray, num_samples: int = 64) -> np.ndarray:
        """Resample a 2D polyline into equidistant points along its arc-length.

        Decouples point density from drawing speed / frame duration variations.
        """
        diffs = pts[1:] - pts[:-1]
        seg_lens = np.hypot(diffs[:, 0], diffs[:, 1])
        cum_lens = np.insert(np.cumsum(seg_lens), 0, 0.0)
        total_len = float(cum_lens[-1])
        if total_len < 1e-4:
            return pts
        sample_dists = np.linspace(0.0, total_len, num_samples)
        resampled = np.zeros((num_samples, 2), dtype=np.float32)
        resampled[:, 0] = np.interp(sample_dists, cum_lens, pts[:, 0])
        resampled[:, 1] = np.interp(sample_dists, cum_lens, pts[:, 1])
        return resampled

    @staticmethod
    def _fit_circle_kasa(pts: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """Fit circle to 2D points using linear least-squares (Kasa algebraic method).

        Solves: (x - xc)^2 + (y - yc)^2 = R^2
        Returns (xc, yc, R) or None if degenerate.
        """
        if len(pts) < 3:
            return None
        x = pts[:, 0]
        y = pts[:, 1]
        A = np.column_stack([2.0 * x, 2.0 * y, np.ones(len(x), dtype=np.float32)])
        b = x**2 + y**2
        try:
            sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            xc = float(sol[0])
            yc = float(sol[1])
            K = float(sol[2])
            val = K + xc**2 + yc**2
            if val <= 0.0:
                return None
            return xc, yc, math.sqrt(val)
        except Exception:
            return None

    def _fit_circle(
        self,
        clean_pts: List[Tuple[float, float]],
        np_pts: np.ndarray,
        bb_w: float,
        bb_h: float,
        aspect_ratio: float,
        total_length: float,
        d_start_end: float,
        bb_diag: float,
    ) -> Optional[ShapeRecognitionResult]:
        """Fit a circle using least-squares algebraic circle fitting and geometric validation.

        Validates:
        1. Aspect ratio: bounding box approximately square (0.65 to 1.55).
        2. Arc-length resampling: uniform spatial sampling independent of drawing speed.
        3. Kasa algebraic circle fit: accurate geometric center and radius.
        4. Closure and circumference consistency: start-to-end distance and perimeter ratio.
        5. Radial consistency: low standard deviation and low peak deviation from mean radius.
        6. Angular sweep & quadrant coverage: continuous angular sweep covering all 4 quadrants.
        """
        # Circle must have roughly 1:1 bounding box
        if not (0.65 <= aspect_ratio <= 1.55):
            return None

        # 1. Resample along arc-length to eliminate hand-velocity clustering
        resampled = self._resample_polyline(np_pts, num_samples=64)

        # 2. Algebraic least-squares circle fitting (Kasa method)
        fit = self._fit_circle_kasa(resampled)
        if not fit:
            return None
        center_x, center_y, fitted_r = fit

        if fitted_r < 8.0:
            return None

        # 3. Radial variance & standard deviation
        radii = np.hypot(resampled[:, 0] - center_x, resampled[:, 1] - center_y)
        r_mean = float(np.mean(radii))
        r_std = float(np.std(radii))
        rel_std = r_std / r_mean
        max_dev = float(np.max(np.abs(radii - r_mean)) / r_mean)

        # 4. Circumference vs actual arc length
        circ = 2.0 * math.pi * r_mean
        length_ratio = total_length / circ

        # 5. Continuous angular sweep around fitted center
        angles = np.arctan2(resampled[:, 1] - center_y, resampled[:, 0] - center_x)
        d_theta = np.diff(angles)
        d_theta = (d_theta + np.pi) % (2.0 * np.pi) - np.pi
        total_sweep = abs(float(np.sum(d_theta)))

        # 6. Quadrant coverage: all 4 quadrants relative to center must have points
        dx = resampled[:, 0] - center_x
        dy = resampled[:, 1] - center_y
        all_quadrants = bool(
            np.any((dx >= 0) & (dy >= 0))
            and np.any((dx < 0) & (dy >= 0))
            and np.any((dx < 0) & (dy < 0))
            and np.any((dx >= 0) & (dy < 0))
        )

        if not all_quadrants:
            return None

        # Angular sweep must cover ~300 deg up to ~520 deg (allows slight closure gap or modest overlap)
        if not (5.20 <= total_sweep <= 9.20):
            return None

        # Circumference ratio must be between 0.72 and 1.50
        if not (0.72 <= length_ratio <= 1.50):
            return None

        # Closure check: start and end points must be reasonably close
        if d_start_end > max(35.0, 0.30 * bb_diag) and (d_start_end / total_length) > 0.28:
            return None

        # Radial variance & peak deviation:
        # Squares have rel_std ~ 0.112 and max_dev ~ 0.218
        # Hand-drawn circles (even noisy) have rel_std <= 0.088 and max_dev <= 0.185
        if rel_std <= 0.088 and max_dev <= 0.185:
            confidence = max(0.65, min(0.98, 1.0 - 3.6 * rel_std))

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
                    # Rectangles have straight sides with circularity <= 0.80; circles have circularity > 0.82
                    perim = max(1.0, cv2.arcLength(approx, True))
                    circularity = 4.0 * math.pi * area / (perim * perim)
                    if circularity > 0.82:
                        continue

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
