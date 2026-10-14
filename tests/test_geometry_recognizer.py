"""Automated test suite for Smart Ink geometric shape recognizer.

Tests positive and negative cases across:
- Clean lines with varying slopes (horizontal, vertical, diagonal).
- Noisy straight lines.
- Rectangles with unequal side lengths and orientations.
- Triangles with various orientations and aspect ratios.
- Circles with small and large radii.
- Imperfectly closed shapes.
- Incomplete strokes / open curves.
- Random scribbles.
- Degenerate paths (too few points, repeated coordinates, empty lists).
- Execution timing benchmarks.
"""

import math
import random
import time
import pytest

from src.geometry.geometry_recognizer import GeometryRecognizer, ShapeRecognitionResult


@pytest.fixture
def recognizer():
    return GeometryRecognizer(min_confidence=0.65)


# ── Synthetic Coordinate Generators ─────────────────────────────

def make_line(p1, p2, num_pts=30, noise_mag=0.0):
    pts = []
    for i in range(num_pts):
        t = i / (num_pts - 1)
        x = p1[0] + t * (p2[0] - p1[0]) + random.uniform(-noise_mag, noise_mag)
        y = p1[1] + t * (p2[1] - p1[1]) + random.uniform(-noise_mag, noise_mag)
        pts.append((int(round(x)), int(round(y))))
    return pts


def make_circle(center, radius, num_pts=45, noise_mag=0.0, closure_gap_deg=10.0):
    pts = []
    max_deg = 360.0 - closure_gap_deg
    for i in range(num_pts):
        deg = (i / num_pts) * max_deg
        rad = math.radians(deg)
        r = radius + random.uniform(-noise_mag, noise_mag)
        x = center[0] + r * math.cos(rad)
        y = center[1] + r * math.sin(rad)
        pts.append((int(round(x)), int(round(y))))
    return pts


def make_rectangle(top_left, width, height, pts_per_side=15, noise_mag=0.0):
    x0, y0 = top_left
    pts = []
    # Top: left -> right
    for i in range(pts_per_side):
        t = i / pts_per_side
        pts.append((int(round(x0 + t * width + random.uniform(-noise_mag, noise_mag))),
                    int(round(y0 + random.uniform(-noise_mag, noise_mag)))))
    # Right: top -> bottom
    for i in range(pts_per_side):
        t = i / pts_per_side
        pts.append((int(round(x0 + width + random.uniform(-noise_mag, noise_mag))),
                    int(round(y0 + t * height + random.uniform(-noise_mag, noise_mag)))))
    # Bottom: right -> left
    for i in range(pts_per_side):
        t = i / pts_per_side
        pts.append((int(round(x0 + (1 - t) * width + random.uniform(-noise_mag, noise_mag))),
                    int(round(y0 + height + random.uniform(-noise_mag, noise_mag)))))
    # Left: bottom -> top
    for i in range(pts_per_side):
        t = i / pts_per_side
        pts.append((int(round(x0 + random.uniform(-noise_mag, noise_mag))),
                    int(round(y0 + (1 - t) * height + random.uniform(-noise_mag, noise_mag)))))
    return pts


def make_triangle(v1, v2, v3, pts_per_side=15, noise_mag=0.0):
    pts = []
    for (p_start, p_end) in [(v1, v2), (v2, v3), (v3, v1)]:
        for i in range(pts_per_side):
            t = i / pts_per_side
            x = p_start[0] + t * (p_end[0] - p_start[0]) + random.uniform(-noise_mag, noise_mag)
            y = p_start[1] + t * (p_end[1] - p_start[1]) + random.uniform(-noise_mag, noise_mag)
            pts.append((int(round(x)), int(round(y))))
    return pts


# ── Positive Tests ──────────────────────────────────────────────

@pytest.mark.parametrize(
    "p1, p2",
    [
        ((50, 100), (450, 100)),   # Horizontal
        ((200, 50), (200, 450)),   # Vertical
        ((50, 50), (350, 350)),    # Diagonal down-right
        ((350, 50), (50, 350)),    # Diagonal down-left
    ],
)
def test_clean_lines_various_slopes(recognizer, p1, p2):
    pts = make_line(p1, p2, num_pts=30, noise_mag=0.0)
    res = recognizer.recognize(pts)
    assert res.is_recognized is True
    assert res.shape_type == "LINE"
    assert res.confidence >= 0.85
    assert len(res.refined_points) == 2


def test_noisy_straight_line(recognizer):
    pts = make_line((100, 100), (400, 250), num_pts=35, noise_mag=3.0)
    res = recognizer.recognize(pts)
    assert res.is_recognized is True
    assert res.shape_type == "LINE"
    assert res.confidence >= 0.65


@pytest.mark.parametrize(
    "center, radius",
    [
        ((200, 200), 40),    # Small circle
        ((320, 240), 120),   # Large circle
    ],
)
def test_circles(recognizer, center, radius):
    pts = make_circle(center, radius, num_pts=40, noise_mag=2.0, closure_gap_deg=15.0)
    res = recognizer.recognize(pts)
    assert res.is_recognized is True
    assert res.shape_type == "CIRCLE"
    assert res.confidence >= 0.65
    assert len(res.refined_points) > 10


def test_rectangle_unequal_sides(recognizer):
    pts = make_rectangle((100, 100), width=240, height=120, pts_per_side=15, noise_mag=2.0)
    res = recognizer.recognize(pts)
    assert res.is_recognized is True
    assert res.shape_type == "RECTANGLE"
    assert res.confidence >= 0.65
    assert len(res.refined_points) == 4


def test_triangle(recognizer):
    pts = make_triangle((200, 100), (350, 300), (100, 300), pts_per_side=18, noise_mag=2.0)
    res = recognizer.recognize(pts)
    assert res.is_recognized is True
    assert res.shape_type == "TRIANGLE"
    assert res.confidence >= 0.65
    assert len(res.refined_points) == 3


# ── Negative and Edge Cases ─────────────────────────────────────

def test_random_scribble_rejected(recognizer):
    """Random scribble should not be recognized as a clean shape and should fallback gracefully."""
    random.seed(42)
    pts = []
    x, y = 200, 200
    for _ in range(50):
        x += random.randint(-20, 20)
        y += random.randint(-20, 20)
        pts.append((x, y))

    res = recognizer.recognize(pts)
    assert res.is_recognized is False
    assert res.shape_type == "UNKNOWN"
    assert res.rejection_reason is not None
    assert res.refined_points == pts  # Preserves user's freehand stroke!


def test_degenerate_and_empty_paths(recognizer):
    """Paths with 0, 1, or 2 points should be safely rejected without exceptions."""
    assert recognizer.recognize([]).is_recognized is False
    assert recognizer.recognize([(10, 10)]).is_recognized is False
    assert recognizer.recognize([(10, 10), (20, 20)]).is_recognized is False


def test_repeated_identical_coordinates(recognizer):
    """List of repeated identical coordinates should be rejected safely."""
    pts = [(100, 100)] * 30
    res = recognizer.recognize(pts)
    assert res.is_recognized is False


def test_incomplete_open_circle_arc(recognizer):
    """A half-circle arc (open stroke) should not be recognized as a closed circle."""
    pts = []
    for deg in range(0, 180, 5):
        rad = math.radians(deg)
        pts.append((int(200 + 80 * math.cos(rad)), int(200 + 80 * math.sin(rad))))

    res = recognizer.recognize(pts)
    # Must not be classified as a circle because closure ratio is large
    assert res.shape_type != "CIRCLE"


# ── Benchmark / Timing Test ─────────────────────────────────────

def test_recognition_performance_timing(recognizer):
    """Shape recognition should complete well under 5ms per stroke for real-time responsiveness."""
    pts = make_rectangle((150, 150), width=200, height=150, pts_per_side=25, noise_mag=2.0)

    # Warmup
    recognizer.recognize(pts)

    times = []
    for _ in range(50):
        t0 = time.perf_counter()
        recognizer.recognize(pts)
        times.append((time.perf_counter() - t0) * 1000.0)

    avg_ms = sum(times) / len(times)
    assert avg_ms < 5.0, f"Average recognition time {avg_ms:.2f}ms exceeds 5ms limit!"
