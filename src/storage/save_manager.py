"""Explicit save and export management for VisionCanvas.

Features:
- Tracks unsaved modification state (is_dirty).
- Explicit Save vs Save As:
  - First save prompts for destination file path.
  - Subsequent saves overwrite the current destination directly without timestamp duplicates.
  - IMPORTANT: Save persists the canvas as a flattened raster image (PNG/JPG).
    Editable vector project persistence (.vcproj) is not yet implemented;
    saved PNG files are flat raster images, NOT editable project files.
- Explicit Export:
  - Prompts for export destination (PNG/JPEG, White/Transparent).
  - Preserves distinction between working file and flattened raster export.
- Exit verification:
  - Prompts user on exit if unsaved changes exist (Save / Discard / Cancel).
- Decoupled dialog functions for dependency injection and 100% headless testability.
"""

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple


@dataclass
class SaveResult:
    """Outcome of a save or export operation."""
    status: str                         # "saved", "exported", "cancelled", "error"
    filepath: Optional[str]             # Path saved to, or None if cancelled
    message: str                        # Human-readable status message for the UI
    is_dirty: bool                      # Updated dirty state


def default_ask_save_path(
    default_dir: str,
    default_filename: str = "canvas_drawing.png",
    filetypes: Optional[List[Tuple[str, str]]] = None,
    title: str = "Save Canvas Image (Raster PNG/JPG)",
) -> Optional[str]:
    """Native OS file dialog for choosing a save file path."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        types = filetypes or [
            ("PNG Image (*.png)", "*.png"),
            ("JPEG Image (*.jpg)", "*.jpg"),
            ("All Files (*.*)", "*.*"),
        ]

        chosen = filedialog.asksaveasfilename(
            parent=root,
            initialdir=default_dir,
            initialfile=default_filename,
            title=title,
            filetypes=types,
            defaultextension=".png",
            confirmoverwrite=True,
        )
        root.destroy()
        return chosen if chosen else None
    except Exception:
        return None


def default_ask_export_path(
    default_dir: str,
    default_filename: str = "canvas_export.png",
    filetypes: Optional[List[Tuple[str, str]]] = None,
    title: str = "Export Canvas Image",
) -> Optional[str]:
    """Native OS file dialog for choosing an export destination."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        types = filetypes or [
            ("PNG Image (*.png)", "*.png"),
            ("JPEG Image (*.jpg)", "*.jpg"),
            ("All Files (*.*)", "*.*"),
        ]

        chosen = filedialog.asksaveasfilename(
            parent=root,
            initialdir=default_dir,
            initialfile=default_filename,
            title=title,
            filetypes=types,
            defaultextension=".png",
            confirmoverwrite=True,
        )
        root.destroy()
        return chosen if chosen else None
    except Exception:
        return None


def default_ask_unsaved_changes(
    title: str = "Unsaved Changes",
    message: str = "You have unsaved changes in your canvas.\n\nDo you want to save them before exiting?",
) -> str:
    """Prompt user when exiting with unsaved changes.

    Returns:
        "save", "discard", or "cancel".
    """
    try:
        import tkinter as tk
        from tkinter import messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)

        # askyesnocancel returns True (yes/save), False (no/discard), or None (cancel)
        res = messagebox.askyesnocancel(title, message, parent=root)
        root.destroy()

        if res is True:
            return "save"
        elif res is False:
            return "discard"
        else:
            return "cancel"
    except Exception:
        # Fallback for headless / non-interactive environments: discard to avoid deadlocks
        return "discard"


class SaveManager:
    """Coordinates canvas persistence, export, dirty state tracking, and exit prompts."""

    def __init__(
        self,
        default_dir: str,
        ask_save_path_fn: Optional[Callable[..., Optional[str]]] = None,
        ask_export_path_fn: Optional[Callable[..., Optional[str]]] = None,
        ask_unsaved_fn: Optional[Callable[..., str]] = None,
    ):
        """Initialize SaveManager with optional injected dialog functions."""
        self.default_dir = str(default_dir)
        self.ask_save_path_fn = ask_save_path_fn or default_ask_save_path
        self.ask_export_path_fn = ask_export_path_fn or default_ask_export_path
        self.ask_unsaved_fn = ask_unsaved_fn or default_ask_unsaved_changes

        self.current_project_path: Optional[str] = None
        self.is_dirty: bool = False
        self._is_dialog_active: bool = False

    def mark_dirty(self) -> None:
        """Mark that canvas has unsaved changes."""
        self.is_dirty = True

    def mark_clean(self) -> None:
        """Mark that all canvas changes have been persisted."""
        self.is_dirty = False

    def save(self, canvas, background: str = "white") -> SaveResult:
        """Explicit Save.
        
        If a project path already exists, overwrites it directly without prompting.
        If no path exists, delegates to save_as().
        """
        if self.current_project_path is None:
            return self.save_as(canvas, background=background)

        try:
            success = canvas.export_image(self.current_project_path, background=background)
            if success:
                self.is_dirty = False
                filename = Path(self.current_project_path).name
                return SaveResult(
                    status="saved",
                    filepath=self.current_project_path,
                    message=f"Saved Image (Raster): {filename}",
                    is_dirty=False,
                )
            else:
                return SaveResult(
                    status="error",
                    filepath=self.current_project_path,
                    message="Save failed: Could not write file",
                    is_dirty=self.is_dirty,
                )
        except Exception as exc:
            return SaveResult(
                status="error",
                filepath=self.current_project_path,
                message=f"Save error: {exc}",
                is_dirty=self.is_dirty,
            )

    def save_as(
        self,
        canvas,
        background: str = "white",
        default_filename: str = "canvas_drawing.png",
    ) -> SaveResult:
        """Explicit Save As.

        Prompts user for destination file path, overwrites confirmed path, and sets active project path.
        Note: The saved file is a raster image, not an editable vector project.
        """
        if self._is_dialog_active:
            return SaveResult(
                status="busy",
                filepath=None,
                message="Save dialog already active",
                is_dirty=self.is_dirty,
            )

        self._is_dialog_active = True
        try:
            chosen_path = self.ask_save_path_fn(
                self.default_dir,
                default_filename=default_filename,
                title="Save Canvas Image (Raster PNG/JPG)",
            )

            if not chosen_path:
                return SaveResult(
                    status="cancelled",
                    filepath=None,
                    message="Save cancelled",
                    is_dirty=self.is_dirty,
                )

            success = canvas.export_image(chosen_path, background=background)
            if success:
                self.current_project_path = chosen_path
                self.is_dirty = False
                filename = Path(chosen_path).name
                return SaveResult(
                    status="saved",
                    filepath=chosen_path,
                    message=f"Saved Image (Raster): {filename}",
                    is_dirty=False,
                )
            else:
                return SaveResult(
                    status="error",
                    filepath=chosen_path,
                    message="Save failed: Could not write file",
                    is_dirty=self.is_dirty,
                )
        except Exception as exc:
            return SaveResult(
                status="error",
                filepath=None,
                message=f"Save error: {exc}",
                is_dirty=self.is_dirty,
            )
        finally:
            self._is_dialog_active = False

    def export_image(
        self,
        canvas,
        background: str = "transparent",
        default_filename: str = "canvas_export.png",
    ) -> SaveResult:
        """Explicit image export.
        
        Prompts user for export destination. Does NOT change active project file path.
        """
        if self._is_dialog_active:
            return SaveResult(
                status="busy",
                filepath=None,
                message="Export dialog already active",
                is_dirty=self.is_dirty,
            )

        self._is_dialog_active = True
        try:
            chosen_path = self.ask_export_path_fn(
                self.default_dir,
                default_filename=default_filename,
                title="Export Canvas Image",
            )

            if not chosen_path:
                return SaveResult(
                    status="cancelled",
                    filepath=None,
                    message="Export cancelled",
                    is_dirty=self.is_dirty,
                )

            success = canvas.export_image(chosen_path, background=background)
            if success:
                filename = Path(chosen_path).name
                return SaveResult(
                    status="exported",
                    filepath=chosen_path,
                    message=f"Exported: {filename}",
                    is_dirty=self.is_dirty,
                )
            else:
                return SaveResult(
                    status="error",
                    filepath=chosen_path,
                    message="Export failed: Could not write file",
                    is_dirty=self.is_dirty,
                )
        except Exception as exc:
            return SaveResult(
                status="error",
                filepath=None,
                message=f"Export error: {exc}",
                is_dirty=self.is_dirty,
            )
        finally:
            self._is_dialog_active = False

    def check_exit_permission(self, canvas) -> str:
        """Verify whether application can exit safely.

        Returns:
            "EXIT" if safe to exit, or "CANCEL" if exit should be aborted.
        """
        # If no unsaved changes or canvas is blank, exit immediately without prompting
        if not self.is_dirty or canvas.is_empty():
            return "EXIT"

        if self._is_dialog_active:
            return "CANCEL"

        self._is_dialog_active = True
        try:
            choice = self.ask_unsaved_fn(
                title="Unsaved Changes",
                message="You have unsaved changes in your canvas.\n\nDo you want to save them before exiting?",
            )

            if choice == "save":
                # If project path already chosen, save directly
                if self.current_project_path is not None:
                    success = canvas.export_image(self.current_project_path, background="white")
                    if success:
                        self.is_dirty = False
                        return "EXIT"
                    return "CANCEL"
                else:
                    # Prompt user for destination file
                    chosen_path = self.ask_save_path_fn(
                        self.default_dir,
                        default_filename="canvas_drawing.png",
                        title="Save Canvas Image (Raster PNG/JPG)",
                    )
                    if chosen_path and canvas.export_image(chosen_path, background="white"):
                        self.current_project_path = chosen_path
                        self.is_dirty = False
                        return "EXIT"
                    return "CANCEL"
            elif choice == "discard":
                return "EXIT"
            else:  # "cancel" or dismissed
                return "CANCEL"
        finally:
            self._is_dialog_active = False

    def get_project_manifest(
        self,
        canvas,
        background_mode: str = "webcam",
        canvas_width: int = 1280,
        canvas_height: int = 720,
    ) -> dict:
        """Generate structured metadata representing the project state.

        This specification lays the groundwork for full vector-based project files (.vcproj).
        """
        return {
            "format": "VisionCanvasProject",
            "version": "1.0",
            "created_at": int(time.time()),
            "canvas": {
                "width": canvas_width,
                "height": canvas_height,
                "background_mode": background_mode,
                "is_empty": canvas.is_empty(),
            },
            "persistence": {
                "current_project_path": self.current_project_path,
                "is_dirty": self.is_dirty,
            },
        }
