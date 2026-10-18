"""Automated unit test suite for SaveManager.

Tests explicit Save, Save As, Export, dirty state tracking, and exit prompts.
All dialogs are mocked to enable 100% headless testability without UI popups.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np
import pytest

from src.brushes.hard_brush import HardBrush
from src.canvas.drawing_canvas import DrawingCanvas
from src.storage.save_manager import SaveManager, SaveResult


@pytest.fixture
def mock_canvas():
    """Create a DrawingCanvas with a real HardBrush for testing persistence."""
    brush = HardBrush(color=(255, 0, 0), thickness=8)
    canvas = DrawingCanvas(320, 240, brush)
    return canvas


def test_initial_state():
    """SaveManager should start clean with no active project path."""
    sm = SaveManager(default_dir="/tmp/test_dir")
    assert sm.current_project_path is None
    assert sm.is_dirty is False


def test_mark_dirty_and_clean():
    """Dirty state transitions should be correctly toggled."""
    sm = SaveManager(default_dir="/tmp/test_dir")
    sm.mark_dirty()
    assert sm.is_dirty is True
    sm.mark_clean()
    assert sm.is_dirty is False


def test_save_as_success(mock_canvas, tmp_path):
    """Save As should prompt for path, export image, and set current project path."""
    target_file = str(tmp_path / "my_drawing.png")
    mock_ask_save = MagicMock(return_value=target_file)

    sm = SaveManager(default_dir=str(tmp_path), ask_save_path_fn=mock_ask_save)
    sm.mark_dirty()

    # Draw something so canvas is not blank
    mock_canvas.draw((50, 50))
    mock_canvas.draw((100, 100))

    res = sm.save_as(mock_canvas)

    mock_ask_save.assert_called_once()
    assert res.status == "saved"
    assert res.filepath == target_file
    assert res.is_dirty is False
    assert sm.current_project_path == target_file
    assert sm.is_dirty is False
    assert os.path.exists(target_file)


def test_save_as_cancelled(mock_canvas, tmp_path):
    """If user cancels file selection, status should be cancelled and dirty state preserved."""
    mock_ask_save = MagicMock(return_value=None)

    sm = SaveManager(default_dir=str(tmp_path), ask_save_path_fn=mock_ask_save)
    sm.mark_dirty()

    res = sm.save_as(mock_canvas)

    mock_ask_save.assert_called_once()
    assert res.status == "cancelled"
    assert res.filepath is None
    assert sm.current_project_path is None
    assert sm.is_dirty is True


def test_save_delegates_to_save_as_on_first_save(mock_canvas, tmp_path):
    """Calling save() when no project path is set should delegate to save_as()."""
    target_file = str(tmp_path / "first_save.png")
    mock_ask_save = MagicMock(return_value=target_file)

    sm = SaveManager(default_dir=str(tmp_path), ask_save_path_fn=mock_ask_save)
    sm.mark_dirty()

    res = sm.save(mock_canvas)

    mock_ask_save.assert_called_once()
    assert res.status == "saved"
    assert sm.current_project_path == target_file
    assert sm.is_dirty is False


def test_save_overwrites_existing_path_without_prompt(mock_canvas, tmp_path):
    """Subsequent saves must overwrite existing destination directly without prompting."""
    target_file = str(tmp_path / "project.png")
    mock_ask_save = MagicMock(return_value=target_file)

    sm = SaveManager(default_dir=str(tmp_path), ask_save_path_fn=mock_ask_save)

    # First save
    sm.save(mock_canvas)
    assert sm.current_project_path == target_file
    mock_ask_save.assert_called_once()

    # Modify canvas again
    sm.mark_dirty()
    assert sm.is_dirty is True

    # Second save: mock_ask_save should NOT be called again
    res2 = sm.save(mock_canvas)
    assert res2.status == "saved"
    assert res2.filepath == target_file
    assert sm.is_dirty is False
    assert mock_ask_save.call_count == 1  # Still 1, never re-prompted!


def test_export_image_does_not_change_project_path(mock_canvas, tmp_path):
    """Exporting a derivative image must not overwrite current project file path."""
    project_file = str(tmp_path / "my_project.png")
    export_file = str(tmp_path / "my_export.png")

    mock_ask_export = MagicMock(return_value=export_file)

    sm = SaveManager(default_dir=str(tmp_path), ask_export_path_fn=mock_ask_export)
    sm.current_project_path = project_file
    sm.mark_dirty()

    res = sm.export_image(mock_canvas, background="transparent")

    mock_ask_export.assert_called_once()
    assert res.status == "exported"
    assert res.filepath == export_file
    # Project path and dirty state must remain preserved
    assert sm.current_project_path == project_file
    assert sm.is_dirty is True
    assert os.path.exists(export_file)


def test_export_image_cancelled(mock_canvas, tmp_path):
    """Cancelling export dialog should return cancelled status cleanly."""
    mock_ask_export = MagicMock(return_value=None)

    sm = SaveManager(default_dir=str(tmp_path), ask_export_path_fn=mock_ask_export)
    sm.mark_dirty()

    res = sm.export_image(mock_canvas)
    assert res.status == "cancelled"
    assert sm.is_dirty is True


def test_check_exit_permission_when_clean_or_empty(mock_canvas, tmp_path):
    """If canvas is not dirty or is blank, exit permission should be immediately granted."""
    mock_ask_unsaved = MagicMock()
    sm = SaveManager(default_dir=str(tmp_path), ask_unsaved_fn=mock_ask_unsaved)

    # Clean canvas
    assert sm.check_exit_permission(mock_canvas) == "EXIT"
    mock_ask_unsaved.assert_not_called()

    # Dirty, but canvas is blank (empty)
    sm.mark_dirty()
    assert mock_canvas.is_empty() is True
    assert sm.check_exit_permission(mock_canvas) == "EXIT"
    mock_ask_unsaved.assert_not_called()


def test_check_exit_permission_prompt_save_success(mock_canvas, tmp_path):
    """When dirty, choosing save and completing file selection allows exit."""
    target_file = str(tmp_path / "exit_saved.png")
    mock_ask_save = MagicMock(return_value=target_file)
    mock_ask_unsaved = MagicMock(return_value="save")

    sm = SaveManager(
        default_dir=str(tmp_path),
        ask_save_path_fn=mock_ask_save,
        ask_unsaved_fn=mock_ask_unsaved,
    )

    # Draw content to make canvas non-empty
    mock_canvas.draw((20, 20))
    mock_canvas.draw((40, 40))
    sm.mark_dirty()

    assert sm.check_exit_permission(mock_canvas) == "EXIT"
    mock_ask_unsaved.assert_called_once()
    mock_ask_save.assert_called_once()
    assert sm.is_dirty is False


def test_check_exit_permission_prompt_save_cancelled_aborts_exit(mock_canvas, tmp_path):
    """If user chooses save on exit, but cancels the save-file dialog, exit is cancelled."""
    mock_ask_save = MagicMock(return_value=None)  # User cancels file dialog
    mock_ask_unsaved = MagicMock(return_value="save")

    sm = SaveManager(
        default_dir=str(tmp_path),
        ask_save_path_fn=mock_ask_save,
        ask_unsaved_fn=mock_ask_unsaved,
    )

    mock_canvas.draw((20, 20))
    mock_canvas.draw((40, 40))
    sm.mark_dirty()

    assert sm.check_exit_permission(mock_canvas) == "CANCEL"
    assert sm.is_dirty is True


def test_check_exit_permission_prompt_discard(mock_canvas, tmp_path):
    """Choosing discard allows exit without saving."""
    mock_ask_save = MagicMock()
    mock_ask_unsaved = MagicMock(return_value="discard")

    sm = SaveManager(
        default_dir=str(tmp_path),
        ask_save_path_fn=mock_ask_save,
        ask_unsaved_fn=mock_ask_unsaved,
    )

    mock_canvas.draw((20, 20))
    sm.mark_dirty()

    assert sm.check_exit_permission(mock_canvas) == "EXIT"
    mock_ask_unsaved.assert_called_once()
    mock_ask_save.assert_not_called()


def test_check_exit_permission_prompt_cancel(mock_canvas, tmp_path):
    """Choosing cancel aborts exit and stays in application."""
    mock_ask_save = MagicMock()
    mock_ask_unsaved = MagicMock(return_value="cancel")

    sm = SaveManager(
        default_dir=str(tmp_path),
        ask_save_path_fn=mock_ask_save,
        ask_unsaved_fn=mock_ask_unsaved,
    )

    mock_canvas.draw((20, 20))
    sm.mark_dirty()

    assert sm.check_exit_permission(mock_canvas) == "CANCEL"
    mock_ask_unsaved.assert_called_once()
    mock_ask_save.assert_not_called()
    assert sm.is_dirty is True


def test_project_manifest_serialization(mock_canvas, tmp_path):
    """Project manifest dictionary should contain expected metadata and format."""
    sm = SaveManager(default_dir=str(tmp_path))
    sm.current_project_path = "/path/to/project.png"
    sm.mark_dirty()

    manifest = sm.get_project_manifest(
        mock_canvas,
        background_mode="whiteboard",
        canvas_width=1280,
        canvas_height=720,
    )

    assert manifest["format"] == "VisionCanvasProject"
    assert manifest["version"] == "1.0"
    assert manifest["canvas"]["width"] == 1280
    assert manifest["canvas"]["height"] == 720
    assert manifest["canvas"]["background_mode"] == "whiteboard"
    assert manifest["persistence"]["current_project_path"] == "/path/to/project.png"
    assert manifest["persistence"]["is_dirty"] is True
