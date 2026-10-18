"""VisionCanvas — Air-Drawing Virtual Canvas Application.

Entry point orchestrating webcam capture, hand tracking inference,
gesture state machine transitions, virtual cursor smoothing, canvas
rendering, and the user interface.
"""

import os
import sys
import threading
import time
from pathlib import Path

import cv2

from src.brushes.hard_brush import HardBrush
from src.brushes.neon_brush import NeonBrush
from src.brushes.pencil_brush import PencilBrush
from src.brushes.soft_brush import SoftBrush
from src.brushes.watercolor_brush import WatercolorBrush
from src.camera.webcam import Webcam
from src.canvas.drawing_canvas import DrawingCanvas
from src.config import (
    CAMERA_HEIGHT,
    CAMERA_INDEX,
    CAMERA_WIDTH,
    CURSOR_OUTLIER_RECOVERY_FRAMES,
    DEBUG_PERFORMANCE,
    GESTURE_COOLDOWN_TIME,
    GESTURE_STABILITY_FRAMES,
    MAX_CURSOR_JUMP,
    OUTPUT_DIR,
    WINDOW_HEIGHT,
    WINDOW_WIDTH,
)
from src.gestures.cursor import VirtualCursor
from src.gestures.gesture_recognizer import GestureRecognizer
from src.gestures.state_machine import GestureStateMachine
from src.hand_tracking.detector import HandDetector
from src.hand_tracking.finger_detector import FingerDetector
from src.geometry.geometry_recognizer import GeometryRecognizer
from src.storage.save_manager import SaveManager
from src.ui.modern_ui import ModernUI

# Brush classes indexed to match UI brush preset order (1-5)
BRUSH_CLASSES = [HardBrush, SoftBrush, NeonBrush, PencilBrush, WatercolorBrush]


def main():
    # ── Verify Camera Availability ─────────────────────────────
    webcam = Webcam(CAMERA_INDEX)
    if not webcam.is_opened:
        print(
            f"[VisionCanvas Error] Camera index {CAMERA_INDEX} could not be opened.\n"
            f"Please verify that your webcam is connected and that system camera permissions are enabled.\n"
            f"If using an external webcam, try setting CAMERA_INDEX = 1 in src/config.py."
        )
        return

    # ── Initialise Core Components ──────────────────────────────
    try:
        detector = HandDetector()
    except FileNotFoundError as err:
        print(f"[VisionCanvas Error] {err}")
        webcam.release()
        return

    finger_detector = FingerDetector()
    gesture_recognizer = GestureRecognizer()
    gesture_state = GestureStateMachine(
        stability_frames=GESTURE_STABILITY_FRAMES,
        cooldown_time=GESTURE_COOLDOWN_TIME,
    )
    cursor = VirtualCursor(
        max_jump=MAX_CURSOR_JUMP,
        recovery_frames=CURSOR_OUTLIER_RECOVERY_FRAMES,
    )
    ui = ModernUI(
        window_width=WINDOW_WIDTH,
        window_height=WINDOW_HEIGHT,
        video_width=CAMERA_WIDTH,
        video_height=CAMERA_HEIGHT,
    )

    # ── Brush & Canvas State ────────────────────────────────────
    brush_size = ui.brush_size
    active_brush_idx = 0
    brush = HardBrush(color=ui.active_color, thickness=brush_size)
    eraser_brush = HardBrush(color=(0, 0, 0), thickness=max(10, brush_size * 2))

    canvas = DrawingCanvas(CAMERA_WIDTH, CAMERA_HEIGHT, brush)
    geometry_recognizer = GeometryRecognizer()
    save_manager = SaveManager(default_dir=OUTPUT_DIR)
    current_stroke_points = []
    debug_mode = DEBUG_PERFORMANCE

    # ── Window & Mouse Setup ────────────────────────────────────
    window_name = "VisionCanvas"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def _finalize_stroke():
        """Finalize the current stroke, executing Smart Ink recognition if active."""
        nonlocal current_stroke_points
        if ui.active_tool == 2 and len(current_stroke_points) >= 5:
            result = geometry_recognizer.recognize(current_stroke_points)
            if result.is_recognized:
                # Remove the rough freehand preview
                canvas.undo()
                # Draw the clean geometric shape
                canvas.draw_shape(result.shape_type, result.refined_points)
                save_manager.mark_dirty()
                ui.set_status_message(f"Smart Ink: {result.shape_type} ({int(result.confidence * 100)}%)")
            else:
                save_manager.mark_dirty()
                ui.set_status_message("Smart Ink: Freehand fallback")
        current_stroke_points = []
        canvas.reset_stroke()

    def _rebuild_brush():
        """Recreate the active brush with current color and size."""
        nonlocal brush, eraser_brush
        cls = BRUSH_CLASSES[active_brush_idx] if active_brush_idx < len(BRUSH_CLASSES) else HardBrush
        brush = cls(color=ui.active_color, thickness=brush_size)
        eraser_brush = HardBrush(color=(0, 0, 0), thickness=max(10, brush_size * 2))
        if ui.active_tool == 0:
            canvas.set_brush(brush, is_eraser=False)
        elif ui.active_tool == 1:
            canvas.set_brush(eraser_brush, is_eraser=True)
        elif ui.active_tool == 2:
            canvas.set_brush(brush, is_eraser=False)

    def _set_mode(idx: int):
        """Switch top navigation mode (0: Draw, 1: Select, 2: Smart Ink, 3: Move)."""
        _finalize_stroke()
        if idx == 0:  # Draw mode
            if ui.active_tool not in (0, 1):
                ui.set_active_tool(0)
            ui.set_active_mode(0)
            tool_name = "Brush" if ui.active_tool == 0 else "Eraser"
            ui.set_status_message(f"Mode: Draw ({tool_name}) — Point index finger to draw")
        elif idx == 1:  # Select mode
            ui.set_active_mode(1)
            ui.set_active_tool(-1)  # No drawing tool active
            ui.set_status_message("Mode: Select — Two fingers to interact without drawing")
        elif idx == 2:  # Smart Ink mode
            ui.set_active_mode(2)
            ui.set_active_tool(2)
            ui.set_status_message("Mode: Smart Ink — Auto-snaps lines, boxes, circles, arrows")
        elif idx == 3:  # Move mode
            ui.set_active_mode(3)
            ui.set_active_tool(-1)  # No drawing tool active
            ui.set_status_message("Mode: Move / Pan — Canvas navigation mode")
        _rebuild_brush()

    def _set_tool(idx: int):
        """Switch active tool (0: Brush, 1: Eraser, 2: Smart Ink)."""
        if idx == 0:
            ui.set_active_mode(0)
            ui.set_active_tool(0)
            ui.set_status_message("Tool: Brush (Draw mode)")
        elif idx == 1:
            ui.set_active_mode(0)
            ui.set_active_tool(1)
            ui.set_status_message("Tool: Eraser (Draw mode)")
        elif idx == 2:
            ui.set_active_mode(2)
            ui.set_active_tool(2)
            ui.set_status_message("Tool: Smart Ink (Auto-shape mode)")
        _rebuild_brush()

    def _set_brush_idx(idx: int):
        """Switch brush preset."""
        nonlocal active_brush_idx
        if 0 <= idx < len(BRUSH_CLASSES):
            active_brush_idx = idx
            ui.set_active_brush(idx)
            _rebuild_brush()
            bname = ui.BRUSH_NAMES[idx] if idx < len(ui.BRUSH_NAMES) else f"Preset {idx+1}"
            ui.set_status_message(f"Brush: {bname}")

    def _change_brush_size(delta: int):
        """Adjust brush size by delta."""
        nonlocal brush_size
        brush_size = max(1, min(100, brush_size + delta))
        ui.set_brush_size(brush_size)
        _rebuild_brush()

    def _handle_save(background: str = "white"):
        """Explicit Save: writes to active project destination or triggers Save As."""
        res = save_manager.save(canvas, background=background)
        ui.set_status_message(res.message)
        if res.filepath:
            print(f"[VisionCanvas] {res.message} -> {res.filepath}")

    def _handle_save_as(background: str = "white"):
        """Explicit Save As: prompts user for destination file path."""
        res = save_manager.save_as(canvas, background=background)
        ui.set_status_message(res.message)
        if res.filepath:
            print(f"[VisionCanvas] {res.message} -> {res.filepath}")

    def _handle_export(background: str = "transparent"):
        """Explicit Export: prompts for user export file path."""
        res = save_manager.export_image(canvas, background=background)
        ui.set_status_message(res.message)
        if res.filepath:
            print(f"[VisionCanvas] {res.message} -> {res.filepath}")

    # ── Mouse Interaction Callback ──────────────────────────────
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Check color palette / wheel
            if ui.handle_selection(x, y):
                _rebuild_brush()
                ui.set_status_message("Color updated")
                return
            # Check brush size slider
            if ui.handle_brush_size_selection(x, y):
                _rebuild_brush()
                return
            # Check top navigation mode pills
            top_mode = ui.handle_top_mode_click(x, y)
            if top_mode is not None:
                _set_mode(top_mode)
                return
            # Check tool buttons
            tool_idx = ui.handle_tool_click(x, y)
            if tool_idx is not None:
                _set_tool(tool_idx)
                return
            # Check canvas background mode selection
            bg_sel = ui.handle_bg_mode_click(x, y)
            if bg_sel is not None:
                ui.set_status_message(f"Canvas Background: {bg_sel.title()}")
                return
            # Check brush presets
            brush_idx = ui.handle_brush_preset_click(x, y)
            if brush_idx is not None:
                _set_brush_idx(brush_idx)
                return
            # Check action buttons (Clear, Undo, Redo, Save, Export)
            action = ui.handle_action_click(x, y)
            if action:
                if action == "CLEAR":
                    canvas.clear()
                    save_manager.mark_dirty()
                    ui.set_status_message("Canvas Cleared")
                elif action == "UNDO":
                    if canvas.undo():
                        save_manager.mark_dirty()
                        ui.set_status_message("Action: Undo")
                elif action == "REDO":
                    if canvas.redo():
                        save_manager.mark_dirty()
                        ui.set_status_message("Action: Redo")
                elif action == "SAVE":
                    _handle_save(background="white")
                elif action == "EXPORT":
                    _handle_export(background="transparent")

    cv2.setMouseCallback(window_name, on_mouse)

    # ── Background Inference Thread ─────────────────────────────
    _inference_running = [True]

    def _inference_loop():
        while _inference_running[0]:
            frame = webcam.get_frame()
            if frame is None:
                time.sleep(0.005)
                continue
            frame = cv2.flip(frame, 1)
            detector.detect(frame)

    inference_thread = threading.Thread(target=_inference_loop, daemon=True)
    inference_thread.start()

    # ── Diagnostics & Performance Counters ──────────────────────
    render_frame_count = 0
    render_fps_timer = time.time()
    render_fps = 0.0

    inf_count_last = 0
    inf_fps_timer = time.time()
    inf_fps = 0.0

    # ── Main Render Loop ────────────────────────────────────────
    _window_close_handled = False
    try:
        while True:
            # Get latest inference result (non-blocking)
            result, inf_frame, is_new = detector.get_latest()

            if inf_frame is not None:
                frame = inf_frame
            else:
                frame = webcam.get_frame()
                if frame is None:
                    time.sleep(0.005)
                    continue
                frame = cv2.flip(frame, 1)

            # ── Process Hand Tracking & Gestures ────────────────────
            cursor_x, cursor_y = 0, 0
            gesture = "NONE"
            landmarks = None

            if result is not None and result.hand_landmarks:
                landmarks = result.hand_landmarks[0]
                fingers = finger_detector.get_finger_states(landmarks)
                raw_gesture = gesture_recognizer.recognize(fingers)
                event = gesture_state.update(raw_gesture)
                gesture = event.active_gesture

                index_tip = landmarks[8]

                # Map x-coordinate across UI workspace margins
                x = index_tip.x
                margin_left, margin_right = 0.10, 0.90
                if x < margin_left:
                    screen_x = (x / margin_left) * ui.video_x
                elif x > margin_right:
                    screen_x = (
                        ui.video_x
                        + ui.disp_w
                        + ((x - margin_right) / (1.0 - margin_right))
                        * (ui.width - (ui.video_x + ui.disp_w))
                    )
                else:
                    screen_x = (
                        ui.video_x
                        + ((x - margin_left) / (margin_right - margin_left))
                        * ui.disp_w
                    )

                # Map y-coordinate
                y = index_tip.y
                margin_top, margin_bottom = 0.10, 0.90
                if y < margin_top:
                    screen_y = (y / margin_top) * ui.video_y
                elif y > margin_bottom:
                    screen_y = (
                        ui.video_y
                        + ui.disp_h
                        + ((y - margin_bottom) / (1.0 - margin_bottom))
                        * (ui.height - (ui.video_y + ui.disp_h))
                    )
                else:
                    screen_y = (
                        ui.video_y
                        + ((y - margin_top) / (margin_bottom - margin_top))
                        * ui.disp_h
                    )

                class MappedLandmark:
                    pass

                mapped = MappedLandmark()
                mapped.x = screen_x / ui.width
                mapped.y = screen_y / ui.height

                cx, cy = cursor.update(mapped, ui.width, ui.height)
                cursor_x, cursor_y = cx, cy

                # ── Action Dispatch ─────────────────────────────────
                # 1. Discrete one-shot actions (Undo, Redo)
                if event.triggered_action == "UNDO":
                    if canvas.undo():
                        save_manager.mark_dirty()
                        ui.set_status_message("Action: Undo")
                elif event.triggered_action == "REDO":
                    if canvas.redo():
                        save_manager.mark_dirty()
                        ui.set_status_message("Action: Redo")
                elif event.triggered_action == "SAVE":
                    # Gestures must never trigger save; explicit click or shortcut required
                    pass

                # 2. Continuous actions (Select, Draw)
                if gesture == "SELECT":
                    if ui.handle_selection(cursor_x, cursor_y):
                        _rebuild_brush()
                    if ui.handle_brush_size_selection(cursor_x, cursor_y):
                        _rebuild_brush()
                    top_mode_sel = ui.handle_top_mode_click(cursor_x, cursor_y)
                    if top_mode_sel is not None:
                        _set_mode(top_mode_sel)
                    tool_sel = ui.handle_tool_click(cursor_x, cursor_y)
                    if tool_sel is not None:
                        _set_tool(tool_sel)
                    bg_sel = ui.handle_bg_mode_click(cursor_x, cursor_y)
                    if bg_sel is not None:
                        ui.set_status_message(f"Canvas Background: {bg_sel.title()}")
                    preset_sel = ui.handle_brush_preset_click(cursor_x, cursor_y)
                    if preset_sel is not None:
                        _set_brush_idx(preset_sel)
                    action_sel = ui.handle_action_click(cursor_x, cursor_y)
                    if action_sel:
                        if action_sel == "CLEAR":
                            canvas.clear()
                            save_manager.mark_dirty()
                            ui.set_status_message("Canvas Cleared")
                        elif action_sel == "UNDO":
                            if canvas.undo():
                                save_manager.mark_dirty()
                                ui.set_status_message("Action: Undo")
                        elif action_sel == "REDO":
                            if canvas.redo():
                                save_manager.mark_dirty()
                                ui.set_status_message("Action: Redo")
                        # Note: Save and Export require explicit mouse click or keyboard shortcut;
                        # hovering in SELECT mode must never trigger file dialogs.
                    _finalize_stroke()

                elif gesture == "DRAW":
                    # Only draw when in Draw (0) or Smart Ink (2) modes
                    if ui.active_mode in (0, 2) and ui.active_tool in (0, 1, 2):
                        if is_new:
                            canv_x, canv_y = ui.ui_to_canvas(cursor_x, cursor_y)
                            if canv_x is not None and canv_y is not None:
                                canvas.draw((canv_x, canv_y))
                                save_manager.mark_dirty()
                                if ui.active_tool == 2:
                                    current_stroke_points.append((canv_x, canv_y))
                            else:
                                _finalize_stroke()
                    else:
                        _finalize_stroke()
                else:
                    _finalize_stroke()
            else:
                # Tracking lost
                gesture_state.on_tracking_lost()
                cursor.on_tracking_lost()
                _finalize_stroke()

            # ── Diagnostics Computation ─────────────────────────────
            render_frame_count += 1
            now = time.time()
            if now - render_fps_timer >= 1.0:
                render_fps = render_frame_count / (now - render_fps_timer)
                render_frame_count = 0
                render_fps_timer = now

            if now - inf_fps_timer >= 1.0:
                current_inf_id = getattr(detector, "_inference_id", 0)
                inf_fps = (current_inf_id - inf_count_last) / (now - inf_fps_timer)
                inf_count_last = current_inf_id
                inf_fps_timer = now

            stats = None
            if debug_mode:
                stats = {
                    "render_fps": render_fps,
                    "inference_fps": inf_fps,
                    "inf_ms": getattr(detector, "latest_inference_time_ms", 0.0),
                    "tracking": "Active" if landmarks else "Lost",
                }

            # ── Render UI & Canvas ──────────────────────────────────
            output = ui.render(
                frame,
                canvas.canvas,
                cursor_x,
                cursor_y,
                gesture,
                landmarks=landmarks,
                canvas_mask=canvas.mask,
                stats=stats,
            )

            cv2.imshow(window_name, output)

            # ── Window Close & Keyboard Controls ────────────────────
            prop_visible = cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE)
            if prop_visible < 1:
                if not _window_close_handled:
                    _window_close_handled = True
                    perm = save_manager.check_exit_permission(canvas)
                    if perm == "EXIT":
                        break
                    else:
                        # User cancelled close: recreate window and mouse callback
                        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                        cv2.setMouseCallback(window_name, on_mouse)
                        cv2.imshow(window_name, output)
                        cv2.waitKey(1)
                        ui.set_status_message("Exit cancelled — unsaved changes preserved")
            else:
                _window_close_handled = False

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # 'q' or ESC
                perm = save_manager.check_exit_permission(canvas)
                if perm == "EXIT":
                    break
                else:
                    ui.set_status_message("Exit cancelled — unsaved changes preserved")
            # Tool switching
            elif key == ord("b"):
                _set_tool(0)  # Brush
            elif key == ord("e"):
                _set_tool(1)  # Eraser
            elif key == ord("i"):
                # Toggle Smart Ink mode
                _set_tool(0 if ui.active_tool == 2 else 2)
            # Background toggle (w: whiteboard / webcam)
            elif key == ord("w"):
                new_bg = ui.toggle_background_mode()
                ui.set_status_message(f"Canvas Background: {new_bg.title()}")
            # Brush presets (1-5)
            elif ord("1") <= key <= ord("5"):
                _set_brush_idx(key - ord("1"))
            # Brush size
            elif key == ord("["):
                _change_brush_size(-4)
            elif key == ord("]"):
                _change_brush_size(4)
            # Canvas operations
            elif key == ord("c"):
                canvas.clear()
                save_manager.mark_dirty()
                ui.set_status_message("Canvas Cleared")
            elif key == ord("z") or key == ord("u"):
                if canvas.undo():
                    save_manager.mark_dirty()
                    ui.set_status_message("Action: Undo")
            elif key == ord("y") or key == ord("r"):
                if canvas.redo():
                    save_manager.mark_dirty()
                    ui.set_status_message("Action: Redo")
            elif key == ord("s"):
                _handle_save(background="white")
            elif key == ord("S"):  # Shift+S: Explicit Save As
                _handle_save_as(background="white")
            elif key == ord("x"):
                _handle_export(background="transparent")
            elif key == ord("d"):
                debug_mode = not debug_mode
                ui.set_status_message("Diagnostics: ON" if debug_mode else "Diagnostics: OFF")

    finally:
        # ── Safe Resource Cleanup ───────────────────────────────────
        _inference_running[0] = False
        if inference_thread.is_alive():
            inference_thread.join(timeout=1.0)
        detector.close()
        webcam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()