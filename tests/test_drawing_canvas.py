"""Unit tests for DrawingCanvas rendering, undo/redo, and export."""

import os
import shutil
import tempfile
import numpy as np
import pytest

from src.brushes.hard_brush import HardBrush
from src.canvas.drawing_canvas import DrawingCanvas


@pytest.fixture
def temp_output_dir():
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_canvas_draw_and_single_point_tap():
    """Drawing a single tap should render a solid dot onto canvas and mask."""
    brush = HardBrush(color=(180, 80, 255), thickness=10)
    canvas = DrawingCanvas(width=200, height=200, brush=brush)

    # Initial canvas is empty
    assert np.count_nonzero(canvas.canvas) == 0
    assert np.count_nonzero(canvas.mask) == 0

    # Single tap at (50, 50)
    canvas.draw((50, 50))
    canvas.reset_stroke()

    # Must have rendered a dot at (50, 50)
    assert canvas.canvas[50, 50].tolist() == [180, 80, 255]
    assert canvas.mask[50, 50] == 255


def test_canvas_stroke_interpolation():
    """Moving between two distant points should interpolate intermediate pixels."""
    brush = HardBrush(color=(255, 0, 0), thickness=6)
    canvas = DrawingCanvas(width=200, height=200, brush=brush)

    canvas.draw((20, 20))
    canvas.draw((80, 80))
    canvas.reset_stroke()

    # The midpoint (50, 50) should be colored due to interpolation
    assert canvas.canvas[50, 50].tolist() == [255, 0, 0]
    assert canvas.mask[50, 50] == 255


def test_canvas_undo_redo_history():
    """Undo and redo should restore exact canvas and mask states across strokes."""
    brush = HardBrush(color=(0, 255, 0), thickness=8)
    canvas = DrawingCanvas(width=100, height=100, brush=brush)

    # Stroke 1: at (20, 20)
    canvas.draw((20, 20))
    canvas.reset_stroke()
    assert canvas.canvas[20, 20, 1] == 255

    # Stroke 2: at (80, 80)
    canvas.draw((80, 80))
    canvas.reset_stroke()
    assert canvas.canvas[80, 80, 1] == 255

    # Undo Stroke 2
    undone = canvas.undo()
    assert undone is True
    assert canvas.canvas[80, 80, 1] == 0       # Stroke 2 removed
    assert canvas.canvas[20, 20, 1] == 255     # Stroke 1 preserved

    # Redo Stroke 2
    redone = canvas.redo()
    assert redone is True
    assert canvas.canvas[80, 80, 1] == 255     # Stroke 2 restored

    # Undo both strokes
    canvas.undo()
    canvas.undo()
    assert np.count_nonzero(canvas.canvas) == 0  # Blank canvas


def test_canvas_clear_and_undo_clear():
    """Clearing should blank the canvas, and undoing should restore the drawing."""
    brush = HardBrush(color=(255, 255, 255), thickness=10)
    canvas = DrawingCanvas(width=100, height=100, brush=brush)

    canvas.draw((30, 30))
    canvas.reset_stroke()
    assert np.count_nonzero(canvas.canvas) > 0

    canvas.clear()
    assert np.count_nonzero(canvas.canvas) == 0
    assert np.count_nonzero(canvas.mask) == 0

    # Undo the clear
    canvas.undo()
    assert canvas.canvas[30, 30, 0] == 255
    assert canvas.mask[30, 30] == 255


def test_canvas_dark_marks_preservation():
    """Black strokes (e.g. pencil / eraser / dark charcoal) must have positive alpha mask."""
    dark_brush = HardBrush(color=(0, 0, 0), thickness=8)
    canvas = DrawingCanvas(width=100, height=100, brush=dark_brush)

    canvas.draw((40, 40))
    canvas.reset_stroke()

    # The mask must record the stroke even though RGB is (0, 0, 0)
    assert canvas.mask[40, 40] == 255


def test_canvas_export(temp_output_dir):
    """Exporting to white background and transparent PNG should generate valid image files."""
    brush = HardBrush(color=(180, 80, 255), thickness=12)
    canvas = DrawingCanvas(width=120, height=120, brush=brush)
    canvas.draw((30, 30))
    canvas.draw((60, 60))
    canvas.reset_stroke()

    # 1. White background export
    white_file = os.path.join(temp_output_dir, "test_white.png")
    ok1 = canvas.export_image(white_file, background="white")
    assert ok1 is True
    assert os.path.exists(white_file)
    assert os.path.getsize(white_file) > 100

    # 2. Transparent PNG export
    trans_file = os.path.join(temp_output_dir, "test_trans.png")
    ok2 = canvas.export_image(trans_file, background="transparent")
    assert ok2 is True
    assert os.path.exists(trans_file)
    assert os.path.getsize(trans_file) > 100
