"""Unit tests for brush classes."""

import numpy as np
import pytest

from src.brushes.hard_brush import HardBrush
from src.brushes.neon_brush import NeonBrush
from src.brushes.pencil_brush import PencilBrush
from src.brushes.soft_brush import SoftBrush
from src.brushes.watercolor_brush import WatercolorBrush


@pytest.mark.parametrize(
    "brush_cls",
    [HardBrush, SoftBrush, NeonBrush, PencilBrush, WatercolorBrush],
)
def test_all_brushes_stroke(brush_cls):
    """Every brush class should successfully draw a stroke on a canvas without error."""
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    brush = brush_cls(color=(120, 50, 240), thickness=8)

    brush.stroke(canvas, (20, 20), (80, 80))

    # Canvas must contain colored pixels
    assert np.count_nonzero(canvas) > 0


def test_brush_thickness_and_color_propagation():
    """Brushes should store their assigned color and thickness."""
    brush = HardBrush(color=(0, 255, 128), thickness=14)
    assert brush.color == (0, 255, 128)
    assert brush.thickness == 14


def test_brush_opacity_propagation():
    """Brushes should store their assigned opacity and defaults."""
    hard = HardBrush()
    pencil = PencilBrush()
    water = WatercolorBrush()
    soft = SoftBrush()
    assert hard.opacity == 1.0
    assert pencil.opacity == 0.6
    assert water.opacity == 0.4
    assert soft.opacity == 0.85
