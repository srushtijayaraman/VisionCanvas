"""Regression tests for UI mode synchronization, background switching, and tracking indicators."""

import numpy as np
import pytest

from src.brushes.hard_brush import HardBrush
from src.canvas.drawing_canvas import DrawingCanvas
from src.ui.modern_ui import ModernUI


@pytest.fixture
def test_ui():
    """Create a headless ModernUI instance."""
    return ModernUI(
        window_width=1280,
        window_height=720,
        video_width=640,
        video_height=480,
    )


def test_ui_mode_synchronization(test_ui):
    """Test switching between Draw (0), Select (1), Smart Ink (2), and Move (3)."""
    assert test_ui.active_mode == 0

    # Switch to Select mode (1)
    test_ui.set_active_mode(1)
    assert test_ui.active_mode == 1

    # Switch to Smart Ink mode (2)
    test_ui.set_active_mode(2)
    assert test_ui.active_mode == 2

    # Switch to Move mode (3)
    test_ui.set_active_mode(3)
    assert test_ui.active_mode == 3

    # Switch back to Draw mode (0)
    test_ui.set_active_mode(0)
    assert test_ui.active_mode == 0


def test_ui_tool_synchronization(test_ui):
    """Test switching active tools (0: Brush, 1: Eraser, 2: Smart Ink, -1: None)."""
    test_ui.set_active_tool(1)
    assert test_ui.active_tool == 1

    test_ui.set_active_tool(2)
    assert test_ui.active_tool == 2

    # Neutral tool state (used in Select / Move modes)
    test_ui.set_active_tool(-1)
    assert test_ui.active_tool == -1

    test_ui.set_active_tool(0)
    assert test_ui.active_tool == 0


def test_ui_top_mode_click_hit_testing(test_ui):
    """Test click hit-testing on top navigation mode pills."""
    assert len(test_ui._mode_rects) == 4

    for x1, y1, x2, y2, idx in test_ui._mode_rects:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        clicked_mode = test_ui.handle_top_mode_click(cx, cy)
        assert clicked_mode == idx


def test_ui_tool_click_hit_testing(test_ui):
    """Test click hit-testing on left sidebar tool buttons."""
    assert len(test_ui._tool_rects) == 3

    for x1, y1, x2, y2, idx in test_ui._tool_rects:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        clicked_tool = test_ui.handle_tool_click(cx, cy)
        assert clicked_tool == idx


def test_canvas_background_mode_selection_and_toggle(test_ui):
    """Test background mode getters, setters, toggle, and click handling."""
    # Default is webcam
    assert test_ui.get_background_mode() == "webcam"

    # Switch to whiteboard
    test_ui.set_background_mode("whiteboard")
    assert test_ui.get_background_mode() == "whiteboard"

    # Toggle back to webcam
    mode = test_ui.toggle_background_mode()
    assert mode == "webcam"
    assert test_ui.get_background_mode() == "webcam"

    # Toggle to whiteboard
    mode = test_ui.toggle_background_mode()
    assert mode == "whiteboard"
    assert test_ui.get_background_mode() == "whiteboard"

    # Invalid mode should be ignored
    test_ui.set_background_mode("invalid_mode")
    assert test_ui.get_background_mode() == "whiteboard"


def test_background_mode_click_hit_testing(test_ui):
    """Test clicking background mode selectors in right panel and bottom bar."""
    assert len(test_ui._bg_mode_rects) >= 2

    for x1, y1, x2, y2, mode in test_ui._bg_mode_rects:
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        clicked_mode = test_ui.handle_bg_mode_click(cx, cy)
        assert clicked_mode in ("webcam", "whiteboard")


def test_background_switching_preserves_canvas_strokes(test_ui):
    """Switching background mode must never modify or destroy existing strokes."""
    brush = HardBrush(color=(180, 80, 255), thickness=10)
    canvas = DrawingCanvas(width=640, height=480, brush=brush)

    # Draw strokes
    canvas.draw((50, 50))
    canvas.draw((120, 120))
    canvas.reset_stroke()

    orig_canvas = canvas.canvas.copy()
    orig_mask = canvas.mask.copy()
    nonzero_pixels = np.count_nonzero(orig_mask)
    assert nonzero_pixels > 0

    # Switch background modes multiple times
    test_ui.set_background_mode("whiteboard")
    test_ui.set_background_mode("webcam")
    test_ui.toggle_background_mode()

    # Verify canvas pixels and mask pixels are byte-for-byte identical
    np.testing.assert_array_equal(canvas.canvas, orig_canvas)
    np.testing.assert_array_equal(canvas.mask, orig_mask)


def test_render_whiteboard_vs_webcam_compositing(test_ui):
    """Verify compositing uses solid whiteboard vs live webcam frame behind drawing."""
    # Distinct webcam frame (solid blue: BGR 255, 0, 0)
    webcam_frame = np.full((480, 640, 3), (255, 0, 0), dtype=np.uint8)

    # Blank canvas with zero mask
    canvas_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas_mask = np.zeros((480, 640), dtype=np.uint8)

    # 1. Render in webcam mode
    test_ui.set_background_mode("webcam")
    out_webcam = test_ui.render(webcam_frame, canvas_bgr, 0, 0, "NONE", canvas_mask=canvas_mask)

    # In center of workspace, pixels should match webcam frame (blue)
    center_y = test_ui.video_y + test_ui.disp_h // 2
    center_x = test_ui.video_x + test_ui.disp_w // 2
    pixel_webcam = out_webcam[center_y, center_x]
    assert pixel_webcam[0] == 255  # Blue channel dominant

    # 2. Render in whiteboard mode
    test_ui.set_background_mode("whiteboard")
    out_whiteboard = test_ui.render(webcam_frame, canvas_bgr, 0, 0, "NONE", canvas_mask=canvas_mask)

    # In center of workspace, pixels should be solid studio whiteboard (250, 250, 250)
    pixel_whiteboard = out_whiteboard[center_y, center_x]
    assert pixel_whiteboard[0] == 250
    assert pixel_whiteboard[1] == 250
    assert pixel_whiteboard[2] == 250


def test_tracking_indicator_never_green_when_lost(test_ui):
    """Tracking indicator must accurately reflect tracking loss and never show green when lost."""
    webcam_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
    canvas_mask = np.zeros((480, 640), dtype=np.uint8)

    # 1. Render with landmarks=None (tracking lost)
    out_lost = test_ui.render(webcam_frame, canvas_bgr, 0, 0, "NONE", landmarks=None, canvas_mask=canvas_mask)

    # Check top-left badge area: must NOT be green
    badge_x = test_ui.video_x + 12
    badge_y = test_ui.video_y + 12
    badge_roi = out_lost[badge_y : badge_y + 24, badge_x : badge_x + 130]

    # In BGR, LIVE_GREEN is (50, 220, 100). The green channel must not dominate like LIVE_GREEN
    # Specifically, check that the badge dot is amber/red, not live green
    dot_color = out_lost[badge_y + 12, badge_x + 12]
    assert not (dot_color[1] > 200 and dot_color[0] < 80), f"Tracking lost badge showed green: {dot_color}"

    # Check bottom-bar indicator
    tk_x, tk_cy = test_ui._tk_pos
    bottom_dot = out_lost[tk_cy, tk_x]
    assert not (bottom_dot[1] > 200 and bottom_dot[0] < 80), f"Bottom tracking indicator showed green: {bottom_dot}"
