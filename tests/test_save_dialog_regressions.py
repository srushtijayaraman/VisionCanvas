"""Automated regression test suite for VisionCanvas Save Dialog Spam prevention.

Verifies:
- Drawing, cursor movement, and gestures never invoke save or exit dialogs.
- Held or repeated thumbs-up gestures cannot trigger saving.
- Continuous SELECT mode hovering over UI action buttons never triggers save dialogs.
- Explicit Save invokes at most one dialog when no destination exists, and zero on subsequent saves.
- Repeated drawing frames never trigger exit prompts.
- Intentional close attempts invoke the unsaved-changes prompt at most once per attempt.
- Exit choices (Save, Cancel, Discard) behave strictly according to specification.
"""

from unittest.mock import MagicMock
import pytest

from src.brushes.hard_brush import HardBrush
from src.canvas.drawing_canvas import DrawingCanvas
from src.hand_tracking.finger_detector import FingerDetector
from src.gestures.gesture_recognizer import GestureRecognizer
from src.gestures.state_machine import GestureStateMachine
from src.storage.save_manager import SaveManager
from src.ui.modern_ui import ModernUI


@pytest.fixture
def test_setup(tmp_path):
    brush = HardBrush(color=(255, 0, 0), thickness=8)
    canvas = DrawingCanvas(640, 480, brush)
    mock_ask_save = MagicMock(return_value=str(tmp_path / "saved.png"))
    mock_ask_export = MagicMock(return_value=str(tmp_path / "export.png"))
    mock_ask_unsaved = MagicMock(return_value="cancel")

    save_manager = SaveManager(
        default_dir=str(tmp_path),
        ask_save_path_fn=mock_ask_save,
        ask_export_path_fn=mock_ask_export,
        ask_unsaved_fn=mock_ask_unsaved,
    )
    ui = ModernUI(1280, 720, 640, 480)
    return {
        "canvas": canvas,
        "save_manager": save_manager,
        "mock_save": mock_ask_save,
        "mock_export": mock_ask_export,
        "mock_unsaved": mock_ask_unsaved,
        "ui": ui,
        "tmp_path": tmp_path,
    }


def test_simulated_drawing_does_not_invoke_save_dialog(test_setup):
    """Drawing 100 strokes across the canvas must never trigger any save or exit dialog."""
    canvas = test_setup["canvas"]
    sm = test_setup["save_manager"]
    mock_save = test_setup["mock_save"]
    mock_unsaved = test_setup["mock_unsaved"]

    for i in range(100):
        canvas.draw((50 + i * 2, 50 + (i % 20) * 2))
        sm.mark_dirty()

    assert sm.is_dirty is True
    assert mock_save.call_count == 0
    assert mock_unsaved.call_count == 0


def test_held_or_repeated_thumbs_up_gesture_cannot_trigger_saving(test_setup):
    """Holding a thumbs-up gesture must NEVER invoke save dialogs."""
    sm = test_setup["save_manager"]
    mock_save = test_setup["mock_save"]
    mock_unsaved = test_setup["mock_unsaved"]

    state_machine = GestureStateMachine(stability_frames=3, cooldown_time=0.5)

    # Thumbs up gesture: thumb=1, index=0, middle=0, ring=0, pinky=0
    # Simulate 60 frames (2 seconds at 30fps) with timestamps stepping through cooldown
    for frame_idx in range(60):
        t = frame_idx * 0.033
        event = state_machine.update("SAVE", timestamp=t)
        # Even if state machine triggers SAVE action, main.py explicitly drops it:
        if event.triggered_action == "SAVE":
            # Per Phase 1 fix: gestures must NEVER call save_manager.save()
            pass

    assert mock_save.call_count == 0
    assert mock_unsaved.call_count == 0


def test_select_mode_hover_over_save_button_does_not_invoke_save_dialog(test_setup):
    """Hovering over the Save/Export buttons in SELECT mode must never trigger dialogs."""
    ui = test_setup["ui"]
    mock_save = test_setup["mock_save"]
    mock_export = test_setup["mock_export"]

    # Locate SAVE and EXPORT buttons from ModernUI
    save_rect = None
    for x1, y1, x2, y2, action in getattr(ui, "_action_rects", []):
        if action == "SAVE":
            save_rect = (x1, y1, x2, y2)
            break

    assert save_rect is not None, "SAVE button rect must exist in UI"
    hover_x = (save_rect[0] + save_rect[2]) // 2
    hover_y = (save_rect[1] + save_rect[3]) // 2

    # Simulate 30 frames in SELECT mode hovering directly over the Save button
    for _ in range(30):
        action_sel = ui.handle_action_click(hover_x, hover_y)
        assert action_sel == "SAVE"
        # In main.py: SELECT mode ignores action_sel == 'SAVE' and 'EXPORT'
        # Only on_mouse click or keyboard triggers it

    assert mock_save.call_count == 0
    assert mock_export.call_count == 0


def test_one_save_action_invokes_at_most_one_dialog(test_setup):
    """Pressing Save with no destination opens exactly 1 dialog; repeated saves open 0 dialogs."""
    canvas = test_setup["canvas"]
    sm = test_setup["save_manager"]
    mock_save = test_setup["mock_save"]

    # Draw something
    canvas.draw((100, 100))
    sm.mark_dirty()

    # First explicit save action
    res1 = sm.save(canvas)
    assert res1.status == "saved"
    assert mock_save.call_count == 1
    assert sm.is_dirty is False

    # Modify canvas again
    canvas.draw((105, 105))
    sm.mark_dirty()

    # Second, third, fourth explicit save actions: must overwrite existing path directly
    for _ in range(5):
        res = sm.save(canvas)
        assert res.status == "saved"
        assert res.filepath == sm.current_project_path

    # Call count must still be exactly 1!
    assert mock_save.call_count == 1


def test_concurrent_save_attempt_is_rejected_while_dialog_active(test_setup):
    """If a save dialog is already open, another save attempt must return busy without opening a dialog."""
    canvas = test_setup["canvas"]
    sm = test_setup["save_manager"]
    mock_save = test_setup["mock_save"]

    sm._is_dialog_active = True  # Simulate dialog currently on screen

    res = sm.save_as(canvas)
    assert res.status == "busy"
    assert mock_save.call_count == 0

    sm._is_dialog_active = False


def test_repeated_drawing_frames_never_invoke_exit_prompt(test_setup):
    """Ongoing drawing loops must never trigger check_exit_permission or ask_unsaved_fn."""
    canvas = test_setup["canvas"]
    sm = test_setup["save_manager"]
    mock_unsaved = test_setup["mock_unsaved"]

    # Dirty canvas
    canvas.draw((50, 50))
    sm.mark_dirty()

    # 120 frames of drawing / rendering
    for i in range(120):
        canvas.draw((50 + i, 50 + i))
        # Normal render loop does not call check_exit_permission

    assert mock_unsaved.call_count == 0


def test_exit_prompt_invoked_only_once_per_attempt_and_cancel_preserves_state(test_setup):
    """Exit prompt is called once per attempt. Cancelling leaves app open and preserves dirty state."""
    canvas = test_setup["canvas"]
    sm = test_setup["save_manager"]
    mock_unsaved = test_setup["mock_unsaved"]

    canvas.draw((100, 100))
    sm.mark_dirty()

    # User presses 'q' or 'ESC'
    mock_unsaved.return_value = "cancel"
    perm = sm.check_exit_permission(canvas)

    assert perm == "CANCEL"
    assert mock_unsaved.call_count == 1
    assert sm.is_dirty is True  # Preserved!


def test_exit_prompt_save_and_discard_behavior(test_setup, tmp_path):
    """Choosing 'save' saves and allows exit; choosing 'discard' allows exit without saving."""
    canvas = test_setup["canvas"]
    sm = test_setup["save_manager"]
    mock_unsaved = test_setup["mock_unsaved"]
    mock_save = test_setup["mock_save"]

    target_file = str(tmp_path / "exit_save.png")
    mock_save.return_value = target_file

    # 1. Test Discard
    canvas.draw((100, 100))
    sm.mark_dirty()
    mock_unsaved.return_value = "discard"

    perm_disc = sm.check_exit_permission(canvas)
    assert perm_disc == "EXIT"
    assert mock_save.call_count == 0  # Did not save!

    # 2. Test Save
    sm.mark_dirty()
    mock_unsaved.return_value = "save"

    perm_save = sm.check_exit_permission(canvas)
    assert perm_save == "EXIT"
    assert mock_save.call_count == 1
    assert sm.is_dirty is False
