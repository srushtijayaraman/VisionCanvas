import cv2
import numpy as np
import math
from PIL import Image, ImageDraw, ImageFont
from src.config import (
    DRAW_HAND_LANDMARKS,
    LANDMARK_COLOR, LANDMARK_CONNECTION_COLOR,
    LANDMARK_DOT_RADIUS, LANDMARK_LINE_THICKNESS,
)


class ModernUI:
    """Premium dark glassmorphic UI matching the Screenshot 5 target.

    Layout:
    ┌─────────────────────────────────────────────────────────┐
    │  TOP NAV: Logo | Draw Select Erase Move | Settings Pro  │
    ├────────┬────────────────────────────────┬───────────────┤
    │  LEFT  │                                │    RIGHT      │
    │  TOOLS │     CAMERA / DRAWING           │    COLORS     │
    │  BRUSH │     WORKSPACE                  │    LAYERS     │
    │  PRES. │                                │               │
    │  SIZE  │     GESTURE GUIDE              │               │
    ├────────┴────────────────────────────────┴───────────────┤
    │  BOTTOM BAR: Canvas | Clear Undo Redo | Track | Save    │
    └─────────────────────────────────────────────────────────┘
    """

    # ── Colour constants ────────────────────────────────────────
    BG = (10, 10, 18)
    PANEL = (18, 18, 30)
    PANEL_BORDER = (35, 35, 55)
    ACCENT = (140, 80, 255)
    ACCENT_GLOW = (180, 120, 255)
    ACCENT_DIM = (80, 50, 160)
    NEON_PINK = (255, 50, 150)
    NEON_CYAN = (0, 220, 255)
    TEXT_PRIMARY = (240, 240, 250)
    TEXT_SECONDARY = (165, 165, 185)
    TEXT_MUTED = (115, 115, 135)
    TEXT_DIM = (85, 85, 105)
    LIVE_GREEN = (50, 220, 100)
    TRACK_LOST = (70, 70, 220)

    # Drawing colour palette (BGR for OpenCV)
    PALETTE_COLORS = [
        ((180, 80, 255), "Purple"),
        ((255, 100, 50), "Blue"),
        ((50, 220, 100), "Green"),
        ((0, 220, 255), "Cyan"),
        ((50, 150, 255), "Orange"),
        ((255, 50, 150), "Pink"),
        ((0, 0, 255), "Red"),
        ((0, 255, 255), "Yellow"),
        ((255, 255, 255), "White"),
        ((128, 128, 128), "Grey"),
    ]

    TOOLS = [
        "Brush",
        "Eraser",
        "Smart Ink",
    ]

    MODES = ["Draw", "Select", "Smart Ink", "Move"]

    GESTURE_GUIDE = [
        ("Draw", "Index Finger"),
        ("Select / Move", "Index + Middle"),
        ("Undo", "3 Fingers"),
        ("Redo", "4 Fingers"),
        ("Save", "Thumbs Up"),
    ]

    BRUSH_NAMES = ["Hard", "Soft", "Neon", "Pencil", "Water"]

    # MediaPipe hand skeleton connections
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),        # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),        # Index
        (5, 9), (9, 10), (10, 11), (11, 12),   # Middle
        (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
        (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (0, 17),                                 # Palm base
    ]

    def __init__(self, window_width=1280, window_height=720,
                 video_width=640, video_height=480):
        self.width = window_width
        self.height = window_height
        self.video_width = video_width
        self.video_height = video_height

        # Layout constants
        self.left_w = 180
        self.right_w = 200
        self.top_h = 52
        self.bottom_h = 60
        self.pad = 10
        self.radius = 12

        # Workspace area
        ws_x = self.left_w + self.pad * 2
        ws_y = self.top_h + self.pad * 2
        ws_w = self.width - self.left_w - self.right_w - self.pad * 6
        ws_h = self.height - self.top_h - self.bottom_h - self.pad * 6

        # Fit video into workspace maintaining aspect ratio
        vid_aspect = self.video_width / self.video_height
        ws_aspect = ws_w / ws_h
        if vid_aspect > ws_aspect:
            self.disp_w = ws_w
            self.disp_h = int(ws_w / vid_aspect)
        else:
            self.disp_h = ws_h
            self.disp_w = int(ws_h * vid_aspect)

        self.video_x = ws_x + (ws_w - self.disp_w) // 2
        self.video_y = ws_y + (ws_h - self.disp_h) // 2

        # State
        self.active_tool = 0
        self.active_mode = 0
        self.active_color_idx = 0
        self.active_color = self.PALETTE_COLORS[0][0]
        self.active_brush = 0
        self.brush_size = 12
        self.tracking_active = True
        self.bg_mode = "webcam"

        # Pre-generate assets
        self._color_wheel, self._wheel_mask = self._generate_color_wheel(130, 14)
        self._sv_square = self._generate_sv_square(270, 56)
        self._hue_bar = self._generate_hue_bar(168, 10)
        self._rounded_mask = self._create_rounded_mask(self.disp_w, self.disp_h, 14)

        # Positions populated by _render_base
        self._color_centers = []
        self._tool_rects = []
        self._mode_rects = []
        self._bg_mode_rects = []
        self._brush_preset_rects = []
        self._action_rects = []
        self._status_msg = ""
        self._status_expires = 0.0

        # Pre-render static parts
        self._base = self._render_base()
        self._build_color_rects()

    # ── Asset generators ────────────────────────────────────────

    def _generate_color_wheel(self, size: int = 130, ring_width: int = 14):
        """Generate smooth HSV color wheel ring using vectorised numpy."""
        center = size // 2
        y_c, x_c = np.ogrid[:size, :size]
        dx = (x_c - center).astype(np.float64)
        dy = (y_c - center).astype(np.float64)
        dist = np.sqrt(dx * dx + dy * dy)
        angle = (np.degrees(np.arctan2(-dy, dx)) + 360) % 360

        outer_r = center - 2
        inner_r = outer_r - ring_width
        ring_mask = (dist >= inner_r) & (dist <= outer_r)

        hsv = np.zeros((size, size, 3), dtype=np.uint8)
        hsv[..., 0] = (angle / 2).astype(np.uint8)  # OpenCV hue 0-179
        hsv[..., 1] = 255
        hsv[..., 2] = 255

        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        bgr[~ring_mask] = 0

        return bgr, ring_mask.astype(np.uint8) * 255

    def _generate_sv_square(self, hue_deg: float, size: int = 56):
        """Generate saturation-value gradient square for a given hue."""
        hue_cv = int((hue_deg % 360) / 2)
        hsv = np.zeros((size, size, 3), dtype=np.uint8)
        hsv[:, :, 0] = hue_cv
        sat = np.linspace(0, 255, size, dtype=np.uint8)
        val = np.linspace(255, 0, size, dtype=np.uint8)
        hsv[:, :, 1] = sat[np.newaxis, :]
        hsv[:, :, 2] = val[:, np.newaxis]
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _generate_hue_bar(self, width: int, height: int = 10):
        """Generate horizontal hue gradient bar."""
        hsv = np.zeros((height, width, 3), dtype=np.uint8)
        hues = np.linspace(0, 179, width, dtype=np.uint8)
        hsv[:, :, 0] = hues[np.newaxis, :]
        hsv[:, :, 1] = 255
        hsv[:, :, 2] = 255
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    def _create_rounded_mask(self, w: int, h: int, radius: int = 14):
        """Pre-compute rounded rectangle mask for camera feed."""
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.rectangle(mask, (radius, 0), (w - radius, h), 255, -1)
        cv2.rectangle(mask, (0, radius), (w, h - radius), 255, -1)
        cv2.circle(mask, (radius, radius), radius, 255, -1)
        cv2.circle(mask, (w - radius, radius), radius, 255, -1)
        cv2.circle(mask, (radius, h - radius), radius, 255, -1)
        cv2.circle(mask, (w - radius, h - radius), radius, 255, -1)
        return mask

    # ── Colour hit-testing ──────────────────────────────────────

    def _build_color_rects(self):
        """Build click regions from positions stored during _render_base."""
        self._color_rects = []
        for (cx, cy) in self._color_centers:
            self._color_rects.append((cx - 12, cy - 12, cx + 12, cy + 12))

    def handle_selection(self, cursor_x, cursor_y):
        """Check if cursor is over a colour button or wheel. Returns True if selected."""
        for i, (x1, y1, x2, y2) in enumerate(self._color_rects):
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                self.active_color_idx = i
                self.active_color = self.PALETTE_COLORS[i][0]
                return True

        wh = self._color_wheel.shape[0]
        hb_h, hb_w = self._hue_bar.shape[:2]
        
        in_wheel = (self._wheel_x <= cursor_x < self._wheel_x + wh and 
                    self._wheel_y <= cursor_y < self._wheel_y + wh)
        in_hue = (self._hue_bar_x <= cursor_x < self._hue_bar_x + hb_w and 
                  self._hue_bar_y <= cursor_y < self._hue_bar_y + hb_h)
                  
        if in_wheel or in_hue:
            if 0 <= cursor_y < self.height and 0 <= cursor_x < self.width:
                b, g, r = self._base[cursor_y, cursor_x]
                new_color = (int(b), int(g), int(r))
                if new_color != self.PANEL[::-1] and new_color != self.BG[::-1] and new_color != self.PANEL_BORDER[::-1]:
                    if new_color != self.active_color:
                        self.active_color = new_color
                        in_sv = (self._sv_x <= cursor_x < self._sv_x + self._sv_square.shape[1] and 
                                 self._sv_y <= cursor_y < self._sv_y + self._sv_square.shape[0])
                        if in_hue or (in_wheel and not in_sv):
                            hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
                            self._sv_square = self._generate_sv_square(int(hsv[0]) * 2, 56)
                        self._base = self._render_base()
                        return True
        return False

    def handle_brush_size_selection(self, cursor_x, cursor_y):
        """Check if cursor is over the brush size slider or presets. Returns True if changed."""
        changed = False
        
        # Check presets
        if hasattr(self, '_size_preset_rects'):
            for x1, y1, x2, y2, size in self._size_preset_rects:
                if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                    if self.brush_size != size:
                        self.set_brush_size(size)
                        changed = True
                        break
                        
        # Check slider
        if not changed and hasattr(self, '_slider_rect'):
            x1, y1, x2, y2 = self._slider_rect
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                pct = (cursor_x - x1) / (x2 - x1)
                pct = max(0.0, min(1.0, pct))
                new_size = int(pct * 100)
                new_size = max(1, new_size)
                if self.brush_size != new_size:
                    self.set_brush_size(new_size)
                    changed = True
                    
        return changed

    def handle_tool_click(self, cursor_x, cursor_y):
        """Check if cursor is over a tool button. Returns tool index or None."""
        for x1, y1, x2, y2, idx in getattr(self, '_tool_rects', []):
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                return idx
        return None

    def handle_top_mode_click(self, cursor_x, cursor_y):
        """Check if cursor is over a top mode pill (Draw, Select, Smart Ink, Move)."""
        for x1, y1, x2, y2, idx in getattr(self, '_mode_rects', []):
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                return idx
        return None

    def set_active_mode(self, idx: int):
        """Set active top navigation mode and refresh UI base."""
        if 0 <= idx < len(self.MODES) and self.active_mode != idx:
            self.active_mode = idx
            self._base = self._render_base()
            self._build_color_rects()

    def handle_brush_preset_click(self, cursor_x, cursor_y):
        """Check if cursor is over a brush preset button. Returns brush index or None."""
        for x1, y1, x2, y2, idx in getattr(self, '_brush_preset_rects', []):
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                return idx
        return None

    def handle_action_click(self, cursor_x, cursor_y):
        """Check if cursor is over an action button (Clear, Undo, Redo, Save, Export)."""
        for x1, y1, x2, y2, action in getattr(self, '_action_rects', []):
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                return action
        return None

    def set_status_message(self, message: str, duration: float = 2.5):
        """Set a user-visible feedback message shown in a banner for *duration* seconds."""
        import time
        self._status_msg = str(message)
        self._status_expires = time.time() + float(duration)

    def ui_to_canvas(self, ui_x, ui_y):
        """Convert a UI screen coordinate to camera/canvas coordinate."""
        if self.video_x <= ui_x <= self.video_x + self.disp_w and self.video_y <= ui_y <= self.video_y + self.disp_h:
            cx = int((ui_x - self.video_x) / self.disp_w * self.video_width)
            cy = int((ui_y - self.video_y) / self.disp_h * self.video_height)
            return cx, cy
        return None, None

    def handle_bg_mode_click(self, cursor_x, cursor_y):
        """Check if cursor is over a background mode button. Returns new mode or None."""
        for x1, y1, x2, y2, mode in getattr(self, '_bg_mode_rects', []):
            if x1 <= cursor_x <= x2 and y1 <= cursor_y <= y2:
                self.set_background_mode(mode)
                return mode
        return None

    def set_background_mode(self, mode: str):
        """Set canvas background mode ('webcam' or 'whiteboard') and refresh UI base."""
        if mode in ("webcam", "whiteboard") and mode != getattr(self, "bg_mode", "webcam"):
            self.bg_mode = mode
            self._base = self._render_base()
            self._build_color_rects()

    def get_background_mode(self) -> str:
        """Return the current background mode."""
        return getattr(self, "bg_mode", "webcam")

    def toggle_background_mode(self) -> str:
        """Toggle background mode between 'webcam' and 'whiteboard'."""
        curr = getattr(self, "bg_mode", "webcam")
        new_mode = "whiteboard" if curr == "webcam" else "webcam"
        self.set_background_mode(new_mode)
        return self.bg_mode

    # ── State changes (re-render base on change) ────────────────

    def set_active_tool(self, idx: int):
        """Set active tool index (-1: None, 0: Brush, 1: Eraser, 2: Smart Ink)."""
        if idx == -1 or 0 <= idx < len(self.TOOLS):
            self.active_tool = idx
            self._base = self._render_base()
            self._build_color_rects()

    def set_active_brush(self, idx: int):
        if 0 <= idx < len(self.BRUSH_NAMES):
            self.active_brush = idx
            self._base = self._render_base()
            self._build_color_rects()

    def set_brush_size(self, size: int):
        self.brush_size = max(1, min(100, size))
        self._base = self._render_base()
        self._build_color_rects()

    # ── Base UI (static panels, rendered via Pillow) ────────────

    def _render_base(self):
        img = Image.new('RGB', (self.width, self.height), self.BG)
        d = ImageDraw.Draw(img)

        # Fonts
        try:
            font_lg = ImageFont.truetype("arial.ttf", 15)
            font_md = ImageFont.truetype("arial.ttf", 13)
            font_sm = ImageFont.truetype("arial.ttf", 11)
            font_xs = ImageFont.truetype("arial.ttf", 9)
            font_brand = ImageFont.truetype("arial.ttf", 18)
            font_icon = ImageFont.truetype("arial.ttf", 14)
        except (IOError, OSError):
            font_lg = ImageFont.load_default()
            font_md = font_sm = font_xs = font_icon = font_lg
            font_brand = font_lg

        r = self.radius

        # ── Top Navigation Bar ──────────────────────────────────
        d.rounded_rectangle(
            [self.pad, self.pad, self.width - self.pad, self.top_h + self.pad],
            radius=r, fill=self.PANEL, outline=self.PANEL_BORDER,
        )
        d.text((self.pad + 16, self.pad + 8), "VisionCanvas", fill=self.ACCENT, font=font_brand)
        d.text((self.pad + 16, self.pad + 30), "Gesture Studio", fill=self.TEXT_MUTED, font=font_sm)

        # Mode pills
        self._mode_rects = []
        pill_w = 94
        pill_h = 32
        pill_gap = 10
        pill_total_w = len(self.MODES) * pill_w + (len(self.MODES) - 1) * pill_gap
        pill_start_x = (self.width - pill_total_w) // 2
        for i, mode in enumerate(self.MODES):
            px = pill_start_x + i * (pill_w + pill_gap)
            py = self.pad + 10
            self._mode_rects.append((px, py, px + pill_w, py + pill_h, i))
            if i == self.active_mode:
                d.rounded_rectangle([px, py, px + pill_w, py + pill_h],
                                    radius=8, fill=self.ACCENT, outline=self.ACCENT_GLOW)
                d.text((px + 14, py + 7), mode, fill=(255, 255, 255), font=font_md)
            else:
                d.rounded_rectangle([px, py, px + pill_w, py + pill_h],
                                    radius=8, fill=(24, 24, 38), outline=self.PANEL_BORDER)
                d.text((px + 14, py + 7), mode, fill=self.TEXT_SECONDARY, font=font_md)

        # Right status tag
        right_x = self.width - self.pad - 180
        d.rounded_rectangle([right_x + 50, self.pad + 12, right_x + 160, self.pad + 38],
                            radius=6, fill=(24, 24, 38), outline=self.PANEL_BORDER)
        d.text((right_x + 64, self.pad + 17), "Studio Studio", fill=self.ACCENT_GLOW, font=font_xs)

        # ── Left Panel ──────────────────────────────────────────
        lp_x1 = self.pad
        lp_y1 = self.top_h + self.pad * 2
        lp_x2 = self.left_w + self.pad
        lp_y2 = self.height - self.bottom_h - self.pad * 2

        d.rounded_rectangle([lp_x1, lp_y1, lp_x2, lp_y2],
                            radius=r, fill=self.PANEL, outline=self.PANEL_BORDER)

        # Tools
        ty = lp_y1 + 12
        d.text((lp_x1 + 16, ty), "TOOLS", fill=self.TEXT_MUTED, font=font_sm)
        ty += 20

        self._tool_rects = []
        for i, tool_name in enumerate(self.TOOLS):
            tool_y = ty + i * 34
            self._tool_rects.append((lp_x1 + 10, tool_y, lp_x2 - 10, tool_y + 28, i))
            
            # Active tool determination
            if self.active_mode == 0:  # Draw mode
                is_active = (i == self.active_tool)
            elif self.active_mode == 2:  # Smart Ink mode
                is_active = (i == 2)
            else:  # Select (1) or Move (3) mode: drawing tools inactive
                is_active = False

            if is_active:
                d.rounded_rectangle([lp_x1 + 10, tool_y, lp_x2 - 10, tool_y + 28],
                                    radius=8, fill=self.ACCENT, outline=self.ACCENT_GLOW)
                icon_color = (255, 255, 255)
                text_color = (255, 255, 255)
            else:
                d.rounded_rectangle([lp_x1 + 10, tool_y, lp_x2 - 10, tool_y + 28],
                                    radius=8, fill=(24, 24, 38), outline=self.PANEL_BORDER)
                icon_color = self.TEXT_MUTED
                text_color = self.TEXT_SECONDARY

            d.text((lp_x1 + 38, tool_y + 6), tool_name, fill=text_color, font=font_md)

            ix, iy = lp_x1 + 18, tool_y + 7
            if tool_name == "Brush":
                d.line([(ix+2, iy+12), (ix+11, iy+3)], fill=icon_color, width=2)
                d.point((ix+1, iy+13), fill=icon_color)
            elif tool_name == "Eraser":
                d.rectangle([(ix, iy+4), (ix+13, iy+11)], outline=icon_color, width=1)
                d.line([(ix+9, iy+4), (ix+9, iy+11)], fill=icon_color, width=1)
            elif tool_name == "Smart Ink":
                d.polygon([(ix+6, iy+1), (ix+13, iy+6), (ix+10, iy+14), (ix+3, iy+14), (ix, iy+6)], outline=icon_color)

        # Mode notice if in non-drawing mode
        badge_y = ty + len(self.TOOLS) * 34 + 4
        if self.active_mode == 1:
            d.rounded_rectangle([lp_x1 + 10, badge_y, lp_x2 - 10, badge_y + 22],
                                radius=6, fill=(18, 36, 44), outline=(0, 180, 210))
            d.text((lp_x1 + 16, badge_y + 5), "Select Mode Active", fill=self.NEON_CYAN, font=font_xs)
            bp_y = badge_y + 28
        elif self.active_mode == 3:
            d.rounded_rectangle([lp_x1 + 10, badge_y, lp_x2 - 10, badge_y + 22],
                                radius=6, fill=(36, 18, 44), outline=(180, 90, 230))
            d.text((lp_x1 + 18, badge_y + 5), "Move Mode Active", fill=self.ACCENT_GLOW, font=font_xs)
            bp_y = badge_y + 28
        else:
            bp_y = ty + len(self.TOOLS) * 34 + 12

        # Brush Presets
        d.text((lp_x1 + 16, bp_y), "BRUSH PRESETS", fill=self.TEXT_MUTED, font=font_sm)
        bp_y += 18

        cols = 3
        cell = 42
        gap = 5
        brush_preview_styles = [
            ((200, 200, 220), 3, "solid"),     # Hard
            ((150, 150, 180), 5, "solid"),     # Soft
            ((255, 120, 255), 2, "solid"),     # Neon
            ((180, 180, 200), 1, "solid"),     # Pencil
            ((130, 160, 200), 4, "wavy"),      # Water
        ]
        self._brush_preset_rects = []
        for i, bname in enumerate(self.BRUSH_NAMES):
            row = i // cols
            col = i % cols
            bx = lp_x1 + 12 + col * (cell + gap)
            by = bp_y + row * (cell + gap)
            self._brush_preset_rects.append((bx, by, bx + cell, by + cell, i))
            outline_c = self.ACCENT if i == self.active_brush else self.PANEL_BORDER
            fill_c = (32, 28, 52) if i == self.active_brush else (22, 22, 36)
            d.rounded_rectangle([bx, by, bx + cell, by + cell], radius=8,
                                fill=fill_c, outline=outline_c)
            # Draw stroke preview
            if i < len(brush_preview_styles):
                color, width, style = brush_preview_styles[i]
                py1 = by + cell // 2
                if style == "wavy":
                    pts = []
                    for sx in range(bx + 8, bx + cell - 8, 3):
                        sy = py1 + int(4 * math.sin((sx - bx) * 0.5))
                        pts.append((sx, sy))
                    if len(pts) >= 2:
                        d.line(pts, fill=color, width=width)
                elif style == "dotted":
                    for sx in range(bx + 10, bx + cell - 8, 6):
                        d.ellipse([sx - 1, py1 - 1, sx + 1, py1 + 1], fill=color)
                else:
                    d.line([(bx + 8, py1 + 4), (bx + cell - 8, py1 - 4)],
                           fill=color, width=width)

        # Brush Size
        bs_y = bp_y + (len(self.BRUSH_NAMES) // cols + 1) * (cell + gap) + 8
        d.text((lp_x1 + 16, bs_y), "BRUSH SIZE", fill=self.TEXT_MUTED, font=font_sm)
        d.text((lp_x2 - 46, bs_y), f"{self.brush_size}px", fill=self.TEXT_PRIMARY, font=font_sm)
        bs_y += 18

        # Slider track
        sl_x1 = lp_x1 + 16
        sl_x2 = lp_x2 - 16
        sl_cy = bs_y + 6
        self._slider_rect = (sl_x1, sl_cy - 12, sl_x2, sl_cy + 12)
        d.rounded_rectangle([sl_x1, sl_cy - 2, sl_x2, sl_cy + 2], radius=2, fill=self.PANEL_BORDER)
        # Slider thumb
        t_frac = min(self.brush_size / 100.0, 1.0)
        thumb_x = int(sl_x1 + t_frac * (sl_x2 - sl_x1))
        d.ellipse([thumb_x - 6, sl_cy - 6, thumb_x + 6, sl_cy + 6], fill=self.ACCENT)

        # Size preset dots
        bs_y += 20
        preset_sizes = [4, 12, 16, 28, 40, 56]
        self._size_preset_rects = []
        for j, ps in enumerate(preset_sizes):
            px = lp_x1 + 14 + j * 26
            pr = max(3, ps // 6)
            
            # Save hit box for gesture selection
            hit_r = max(12, pr + 6)
            self._size_preset_rects.append((px - hit_r, bs_y + 6 - hit_r, px + hit_r, bs_y + 6 + hit_r, ps))
            
            fill_c = self.ACCENT if ps == self.brush_size else self.TEXT_DIM
            d.ellipse([px - pr, bs_y + 6 - pr, px + pr, bs_y + 6 + pr], fill=fill_c)

        # ── Right Panel ─────────────────────────────────────────
        rp_x1 = self.width - self.right_w - self.pad
        rp_y1 = self.top_h + self.pad * 2
        rp_x2 = self.width - self.pad
        rp_y2 = self.height - self.bottom_h - self.pad * 2

        d.rounded_rectangle([rp_x1, rp_y1, rp_x2, rp_y2],
                            radius=r, fill=self.PANEL, outline=self.PANEL_BORDER)

        # Colors header
        cy = rp_y1 + 12
        d.text((rp_x1 + 16, cy), "COLORS", fill=self.TEXT_MUTED, font=font_sm)
        cy += 20

        # ── Positions for color wheel overlay (placed on numpy later) ──
        wheel_size = self._color_wheel.shape[0]  # 130
        self._wheel_x = (rp_x1 + rp_x2) // 2 - wheel_size // 2
        self._wheel_y = cy + 5

        # SV square inside wheel
        sv_size = self._sv_square.shape[0]  # 56
        self._sv_x = self._wheel_x + (wheel_size - sv_size) // 2
        self._sv_y = self._wheel_y + (wheel_size - sv_size) // 2

        # Hue bar below wheel
        hue_bar_w = self._hue_bar.shape[1]
        self._hue_bar_x = rp_x1 + 16
        self._hue_bar_y = self._wheel_y + wheel_size + 10

        # Hex value
        hex_y = self._hue_bar_y + 18
        b, g, rv = self.active_color
        hex_str = f"#{rv:02X}{g:02X}{b:02X}"
        d.rounded_rectangle([rp_x1 + 16, hex_y, rp_x2 - 16, hex_y + 18],
                            radius=6, fill=(22, 22, 36), outline=self.PANEL_BORDER)
        d.text((rp_x1 + 26, hex_y + 2), hex_str, fill=self.TEXT_PRIMARY, font=font_sm)

        # ── Recent Colors (store positions for hit testing) ─────
        rc_y = hex_y + 28
        d.text((rp_x1 + 16, rc_y), "RECENT COLORS", fill=self.TEXT_MUTED, font=font_sm)
        rc_y += 18

        self._color_centers = []
        color_cols = 5
        color_spacing = 28
        for i, (col, _) in enumerate(self.PALETTE_COLORS):
            row = i // color_cols
            coli = i % color_cols
            cx = rp_x1 + 30 + coli * color_spacing
            ccy = rc_y + row * color_spacing
            self._color_centers.append((cx, ccy))
            b_c, g_c, r_c = col
            d.ellipse([cx - 9, ccy - 9, cx + 9, ccy + 9], fill=(r_c, g_c, b_c))
            if i == self.active_color_idx:
                d.ellipse([cx - 11, ccy - 11, cx + 11, ccy + 11],
                          outline=(255, 255, 255), width=2)

        # ── Canvas Background Selector ──────────────────────────
        bg_section_y = rc_y + (len(self.PALETTE_COLORS) // color_cols + 1) * color_spacing + 8
        d.text((rp_x1 + 16, bg_section_y), "CANVAS BACKGROUND", fill=self.TEXT_MUTED, font=font_sm)
        bg_section_y += 20

        self._bg_mode_rects = []
        bg_modes = [
            ("webcam", "Live Webcam", "Camera Feed"),
            ("whiteboard", "Whiteboard", "Solid Canvas"),
        ]
        for i, (bkey, btitle, bdesc) in enumerate(bg_modes):
            by = bg_section_y + i * 36
            is_active_bg = (getattr(self, "bg_mode", "webcam") == bkey)
            self._bg_mode_rects.append((rp_x1 + 8, by, rp_x2 - 8, by + 30, bkey))

            if is_active_bg:
                d.rounded_rectangle([rp_x1 + 8, by, rp_x2 - 8, by + 30],
                                    radius=6, fill=self.ACCENT, outline=self.ACCENT_GLOW)
                d.ellipse([rp_x1 + 16, by + 10, rp_x1 + 26, by + 20], fill=(255, 255, 255))
                d.text((rp_x1 + 34, by + 4), btitle, fill=(255, 255, 255), font=font_sm)
                d.text((rp_x1 + 34, by + 17), bdesc, fill=(220, 200, 255), font=font_xs)
            else:
                d.rounded_rectangle([rp_x1 + 8, by, rp_x2 - 8, by + 30],
                                    radius=6, fill=(24, 24, 38), outline=self.PANEL_BORDER)
                d.ellipse([rp_x1 + 16, by + 10, rp_x1 + 26, by + 20], outline=self.TEXT_MUTED)
                d.text((rp_x1 + 34, by + 4), btitle, fill=self.TEXT_SECONDARY, font=font_sm)
                d.text((rp_x1 + 34, by + 17), bdesc, fill=self.TEXT_DIM, font=font_xs)

        # ── Bottom Bar ──────────────────────────────────────────
        bb_y1 = self.height - self.bottom_h - self.pad
        bb_y2 = self.height - self.pad
        d.rounded_rectangle([self.pad, bb_y1, self.width - self.pad, bb_y2],
                            radius=r, fill=self.PANEL, outline=self.PANEL_BORDER)

        # Canvas size & quick background toggle
        d.text((self.pad + 16, bb_y1 + 10), "CANVAS", fill=self.TEXT_MUTED, font=font_sm)
        d.rounded_rectangle([self.pad + 72, bb_y1 + 8, self.pad + 162, bb_y1 + 28],
                            radius=6, fill=(22, 22, 36), outline=self.PANEL_BORDER)
        d.text((self.pad + 78, bb_y1 + 11), f"{self.video_width}x{self.video_height}",
               fill=self.TEXT_PRIMARY, font=font_sm)

        # Quick background mode toggle button in bottom bar
        bg_btn_x1 = self.pad + 170
        bg_btn_x2 = self.pad + 268
        curr_bg = getattr(self, "bg_mode", "webcam")
        bg_btn_lbl = "BG: Webcam" if curr_bg == "webcam" else "BG: Board"
        target_toggle = "whiteboard" if curr_bg == "webcam" else "webcam"
        self._bg_mode_rects.append((bg_btn_x1, bb_y1 + 8, bg_btn_x2, bb_y1 + 28, target_toggle))
        d.rounded_rectangle([bg_btn_x1, bb_y1 + 8, bg_btn_x2, bb_y1 + 28],
                            radius=6, fill=(26, 26, 42), outline=self.ACCENT_DIM)
        d.text((bg_btn_x1 + 8, bb_y1 + 11), bg_btn_lbl, fill=self.ACCENT_GLOW, font=font_sm)

        # Action buttons: Undo, Redo, Clear
        act_start = self.width // 2 - 170
        self._action_rects = []

        # Undo button
        ax = act_start
        ay = bb_y1 + 6
        self._action_rects.append((ax, ay, ax + 72, ay + 38, "UNDO"))
        d.rounded_rectangle([ax, ay, ax + 72, ay + 38], radius=8,
                            fill=(28, 28, 44), outline=(55, 55, 80))
        d.text((ax + 18, ay + 4), "Undo", fill=self.TEXT_PRIMARY, font=font_md)
        d.text((ax + 17, ay + 21), "Ctrl+Z", fill=self.TEXT_MUTED, font=font_sm)

        # Redo button
        ax += 80
        self._action_rects.append((ax, ay, ax + 72, ay + 38, "REDO"))
        d.rounded_rectangle([ax, ay, ax + 72, ay + 38], radius=8,
                            fill=(28, 28, 44), outline=(55, 55, 80))
        d.text((ax + 18, ay + 4), "Redo", fill=self.TEXT_PRIMARY, font=font_md)
        d.text((ax + 17, ay + 21), "Ctrl+Y", fill=self.TEXT_MUTED, font=font_sm)

        # Clear button (distinct coral red warning styling)
        ax += 80
        self._action_rects.append((ax, ay, ax + 72, ay + 38, "CLEAR"))
        d.rounded_rectangle([ax, ay, ax + 72, ay + 38], radius=8,
                            fill=(38, 22, 28), outline=(95, 45, 55))
        d.text((ax + 18, ay + 4), "Clear", fill=(255, 125, 125), font=font_md)
        d.text((ax + 18, ay + 21), "Key C", fill=(200, 110, 110), font=font_sm)

        # Tracking indicator anchor
        tk_x = act_start + 265
        tk_cy = (bb_y1 + bb_y2) // 2
        self._tk_pos = (tk_x, tk_cy)

        # Save & Export buttons
        ax = tk_x + 130
        # Save button
        self._action_rects.append((ax, ay, ax + 74, ay + 38, "SAVE"))
        d.rounded_rectangle([ax, ay, ax + 74, ay + 38], radius=8,
                            fill=(22, 38, 28), outline=(45, 85, 55))
        d.text((ax + 16, ay + 4), "Save", fill=(130, 240, 160), font=font_md)
        d.text((ax + 8, ay + 21), "White (S)", fill=(100, 180, 120), font=font_sm)

        # Export button
        ax += 82
        self._action_rects.append((ax, ay, ax + 74, ay + 38, "EXPORT"))
        d.rounded_rectangle([ax, ay, ax + 74, ay + 38], radius=8,
                            fill=(24, 32, 48), outline=(50, 70, 105))
        d.text((ax + 12, ay + 4), "Export", fill=(140, 210, 255), font=font_md)
        d.text((ax + 8, ay + 21), "Alpha (X)", fill=(100, 150, 195), font=font_sm)

        # Zoom
        zoom_x = self.width - self.pad - 180
        d.text((zoom_x, bb_y1 + 14), "ZOOM", fill=self.TEXT_MUTED, font=font_sm)
        d.rounded_rectangle([zoom_x + 46, bb_y1 + 14, zoom_x + 110, bb_y1 + 26],
                            radius=4, fill=self.PANEL_BORDER)
        d.ellipse([zoom_x + 72, bb_y1 + 12, zoom_x + 82, bb_y1 + 28], fill=self.ACCENT)
        d.text((zoom_x + 118, bb_y1 + 14), "100%", fill=self.TEXT_PRIMARY, font=font_sm)

        # ── Convert to OpenCV BGR and overlay generated assets ──
        base_np = np.array(img)
        base_bgr = cv2.cvtColor(base_np, cv2.COLOR_RGB2BGR)

        # Overlay color wheel
        wh, ww = self._color_wheel.shape[:2]
        wy, wx = self._wheel_y, self._wheel_x
        if 0 <= wy and wy + wh <= self.height and 0 <= wx and wx + ww <= self.width:
            roi = base_bgr[wy:wy + wh, wx:wx + ww]
            mask = self._wheel_mask
            roi[mask > 0] = self._color_wheel[mask > 0]

        # Overlay SV square
        sv_h, sv_w = self._sv_square.shape[:2]
        sy, sx = self._sv_y, self._sv_x
        if 0 <= sy and sy + sv_h <= self.height and 0 <= sx and sx + sv_w <= self.width:
            base_bgr[sy:sy + sv_h, sx:sx + sv_w] = self._sv_square

        # Overlay hue bar
        hb_h, hb_w = self._hue_bar.shape[:2]
        hy, hx = self._hue_bar_y, self._hue_bar_x
        if 0 <= hy and hy + hb_h <= self.height and 0 <= hx and hx + hb_w <= self.width:
            base_bgr[hy:hy + hb_h, hx:hx + hb_w] = self._hue_bar

        return base_bgr

    # ── Hand landmark drawing ───────────────────────────────────

    def _draw_hand_landmarks(self, frame, landmarks, frame_w, frame_h):
        """Draw neon-coloured hand skeleton on a camera frame."""
        if not DRAW_HAND_LANDMARKS or not landmarks:
            return

        # Convert normalised landmarks to pixel coords
        pts = []
        for lm in landmarks:
            px = int(lm.x * frame_w)
            py = int(lm.y * frame_h)
            pts.append((px, py))

        # Draw connections (glow line)
        for (a, b) in self.HAND_CONNECTIONS:
            if a < len(pts) and b < len(pts):
                cv2.line(frame, pts[a], pts[b],
                         LANDMARK_CONNECTION_COLOR,
                         LANDMARK_LINE_THICKNESS + 2, cv2.LINE_AA)
                cv2.line(frame, pts[a], pts[b],
                         LANDMARK_COLOR,
                         LANDMARK_LINE_THICKNESS, cv2.LINE_AA)

        # Draw landmark dots
        for i, (px, py) in enumerate(pts):
            r = LANDMARK_DOT_RADIUS + 1 if i == 8 else LANDMARK_DOT_RADIUS
            cv2.circle(frame, (px, py), r + 2, LANDMARK_CONNECTION_COLOR, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), r, (255, 255, 255), -1, cv2.LINE_AA)

        # Special highlight on index fingertip (landmark 8)
        if len(pts) > 8:
            cv2.circle(frame, pts[8], LANDMARK_DOT_RADIUS + 5,
                       self.ACCENT, 2, cv2.LINE_AA)

    # ── Dynamic rendering (per frame) ───────────────────────────

    def render(self, frame, canvas_frame, cursor_x, cursor_y, gesture,
              landmarks=None, canvas_mask=None, stats=None):
        import time
        ui = self._base.copy()

        # Resize frames to display size
        disp_frame = cv2.resize(frame, (self.disp_w, self.disp_h))

        # Select background layer (live webcam or solid studio whiteboard)
        if getattr(self, "bg_mode", "webcam") == "whiteboard":
            disp_bg = np.full((self.disp_h, self.disp_w, 3), 250, dtype=np.uint8)
        else:
            disp_bg = disp_frame.copy()

        # Draw hand landmarks on the background layer
        if landmarks:
            self._draw_hand_landmarks(disp_bg, landmarks, self.disp_w, self.disp_h)

        # Resize canvas and merge with background
        if (canvas_frame.shape[1] != self.disp_w or
                canvas_frame.shape[0] != self.disp_h):
            disp_canvas = cv2.resize(canvas_frame, (self.disp_w, self.disp_h))
        else:
            disp_canvas = canvas_frame

        if canvas_mask is not None:
            if (canvas_mask.shape[1] != self.disp_w or canvas_mask.shape[0] != self.disp_h):
                mask = cv2.resize(canvas_mask, (self.disp_w, self.disp_h), interpolation=cv2.INTER_LINEAR)
            else:
                mask = canvas_mask
        else:
            gray = cv2.cvtColor(disp_canvas, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Smooth alpha blend of canvas drawing over background layer
        alpha = (mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        merged = np.clip(
            disp_canvas.astype(np.float32) * alpha + disp_bg.astype(np.float32) * (1.0 - alpha),
            0, 255
        ).astype(np.uint8)

        # Apply rounded corners to fit inside the workspace panel
        vx, vy = self.video_x, self.video_y
        bg_region = ui[vy:vy + self.disp_h, vx:vx + self.disp_w]
        corner_alpha = (self._rounded_mask.astype(np.float32) / 255.0)[:, :, np.newaxis]
        combined = np.clip(
            merged.astype(np.float32) * corner_alpha + bg_region.astype(np.float32) * (1.0 - corner_alpha),
            0, 255
        ).astype(np.uint8)
        ui[vy:vy + self.disp_h, vx:vx + self.disp_w] = combined

        # Thin border around workspace region
        cv2.rectangle(ui, (vx - 1, vy - 1),
                      (vx + self.disp_w, vy + self.disp_h),
                      self.PANEL_BORDER, 1)

        # ── Viewport Tracking Badge (Top-Left) ───────────────────
        badge_x, badge_y = vx + 12, vy + 12
        if landmarks:
            # Active tracking (Emerald Green)
            cv2.rectangle(ui, (badge_x, badge_y), (badge_x + 130, badge_y + 24),
                          (15, 42, 20), cv2.FILLED)
            cv2.rectangle(ui, (badge_x, badge_y), (badge_x + 130, badge_y + 24),
                          (40, 130, 60), 1)
            cv2.circle(ui, (badge_x + 12, badge_y + 12), 4,
                       self.LIVE_GREEN, cv2.FILLED, cv2.LINE_AA)
            cv2.putText(ui, "TRACKING: ON", (badge_x + 22, badge_y + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.LIVE_GREEN, 1, cv2.LINE_AA)
        else:
            # Tracking lost / searching (Muted Amber/Red - NEVER GREEN)
            cv2.rectangle(ui, (badge_x, badge_y), (badge_x + 145, badge_y + 24),
                          (35, 22, 16), cv2.FILLED)
            cv2.rectangle(ui, (badge_x, badge_y), (badge_x + 145, badge_y + 24),
                          (70, 45, 170), 1)
            cv2.circle(ui, (badge_x + 12, badge_y + 12), 4,
                       (70, 70, 220), 1, cv2.LINE_AA)
            cv2.putText(ui, "TRACKING: LOST", (badge_x + 22, badge_y + 17),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80, 140, 240), 1, cv2.LINE_AA)

        # ── Viewport Active Mode Badge (Top-Right) ──────────────
        mode_names = [
            "DRAW [BRUSH]" if self.active_tool == 0 else "DRAW [ERASER]",
            "SELECT",
            "SMART INK",
            "MOVE",
        ]
        active_mode_str = mode_names[self.active_mode] if self.active_mode < len(mode_names) else "DRAW"
        mb_w = len(active_mode_str) * 8 + 24
        mb_x = vx + self.disp_w - mb_w - 12
        mb_y = vy + 12
        cv2.rectangle(ui, (mb_x, mb_y), (mb_x + mb_w, mb_y + 24), (20, 20, 32), cv2.FILLED)
        cv2.rectangle(ui, (mb_x, mb_y), (mb_x + mb_w, mb_y + 24), self.PANEL_BORDER, 1)
        cv2.putText(ui, active_mode_str, (mb_x + 10, mb_y + 17),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.ACCENT_GLOW, 1, cv2.LINE_AA)

        # ── Gesture Guide overlay ───────────────────────────────
        guide_h = 48
        guide_y = vy + self.disp_h - guide_h - 6
        guide_x = vx + 8
        guide_w = self.disp_w - 16

        # Semi-transparent background
        if guide_y >= 0 and guide_y + guide_h <= self.height:
            ov_region = ui[guide_y:guide_y + guide_h, guide_x:guide_x + guide_w]
            dark = np.full_like(ov_region, (12, 12, 20), dtype=np.uint8)
            cv2.addWeighted(dark, 0.78, ov_region, 0.22, 0, dst=ov_region)

            cv2.putText(ui, "GESTURE GUIDE", (guide_x + 8, guide_y + 13),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, self.TEXT_MUTED, 1, cv2.LINE_AA)

            item_w = guide_w // len(self.GESTURE_GUIDE)
            for i, (gname, gdesc) in enumerate(self.GESTURE_GUIDE):
                ix = guide_x + 8 + i * item_w
                # Highlight active gesture
                is_active = (
                    gesture == gname.upper().split("/")[0].strip().split(" ")[0]
                    or (gesture == "SELECT" and "Select" in gname)
                    or (gesture == "SAVE" and "Save" in gname)
                )
                col = self.ACCENT_GLOW if is_active else self.TEXT_PRIMARY
                cv2.putText(ui, gname, (ix, guide_y + 28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, col, 1, cv2.LINE_AA)
                cv2.putText(ui, gdesc, (ix, guide_y + 42),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.28, self.TEXT_MUTED, 1, cv2.LINE_AA)

        # ── Cursor ──────────────────────────────────────────────
        if cursor_x > 0 and cursor_y > 0:
            if gesture == "DRAW":
                b, g, rv = self.active_color
                cv2.circle(ui, (cursor_x, cursor_y), 14,
                           (b // 3, g // 3, rv // 3), 2, cv2.LINE_AA)
                cv2.circle(ui, (cursor_x, cursor_y), 6,
                           self.active_color, cv2.FILLED, cv2.LINE_AA)
            else:
                cv2.circle(ui, (cursor_x, cursor_y), 10,
                           self.NEON_CYAN, 2, cv2.LINE_AA)
                cv2.circle(ui, (cursor_x, cursor_y), 3,
                           (255, 255, 255), cv2.FILLED, cv2.LINE_AA)

        # ── Status Feedback Banner ──────────────────────────────
        if self._status_msg and time.time() < self._status_expires:
            msg_text = self._status_msg
            banner_y = vy + 10
            banner_w = min(self.disp_w - 20, len(msg_text) * 9 + 40)
            banner_x = vx + (self.disp_w - banner_w) // 2
            cv2.rectangle(ui, (banner_x, banner_y), (banner_x + banner_w, banner_y + 28),
                          (20, 20, 32), cv2.FILLED)
            cv2.rectangle(ui, (banner_x, banner_y), (banner_x + banner_w, banner_y + 28),
                          self.ACCENT, 1)
            cv2.putText(ui, msg_text, (banner_x + 16, banner_y + 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.ACCENT_GLOW, 1, cv2.LINE_AA)

        # ── Diagnostics Overlay ─────────────────────────────────
        if stats:
            dy = vy + 38
            dx = vx + 12
            cv2.rectangle(ui, (dx - 4, dy - 14), (dx + 230, dy + 56), (14, 14, 24), cv2.FILLED)
            cv2.rectangle(ui, (dx - 4, dy - 14), (dx + 230, dy + 56), self.PANEL_BORDER, 1)
            cv2.putText(ui, f"Render FPS: {stats.get('render_fps', 0):.1f}", (dx, dy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 255, 120), 1, cv2.LINE_AA)
            cv2.putText(ui, f"Inference FPS: {stats.get('inference_fps', 0):.1f}", (dx, dy + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(ui, f"Inference Latency: {stats.get('inf_ms', 0):.1f}ms", (dx, dy + 32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.TEXT_PRIMARY, 1, cv2.LINE_AA)
            cv2.putText(ui, f"Tracking: {stats.get('tracking', 'Active')}", (dx, dy + 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, self.ACCENT_GLOW, 1, cv2.LINE_AA)

        # ── Dynamic Tracking Indicator on Bottom Bar ───────────
        tk_x, tk_cy = getattr(self, '_tk_pos', (self.width // 2 + 95, self.height - 30))
        if landmarks:
            # Active tracking: Glowing green ring and solid core
            cv2.circle(ui, (tk_x, tk_cy), 12, (20, 50, 25), cv2.FILLED, cv2.LINE_AA)
            cv2.circle(ui, (tk_x, tk_cy), 12, self.LIVE_GREEN, 2, cv2.LINE_AA)
            cv2.circle(ui, (tk_x, tk_cy), 4, self.LIVE_GREEN, cv2.FILLED, cv2.LINE_AA)
            cv2.putText(ui, "Tracking: Active", (tk_x + 18, tk_cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, self.LIVE_GREEN, 1, cv2.LINE_AA)
        else:
            # Tracking lost / searching: Muted amber/red indicator - NEVER GREEN
            cv2.circle(ui, (tk_x, tk_cy), 12, (25, 20, 20), cv2.FILLED, cv2.LINE_AA)
            cv2.circle(ui, (tk_x, tk_cy), 12, (50, 50, 160), 2, cv2.LINE_AA)
            cv2.circle(ui, (tk_x, tk_cy), 3, (60, 60, 180), cv2.FILLED, cv2.LINE_AA)
            cv2.putText(ui, "Tracking: Lost", (tk_x + 18, tk_cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (70, 110, 200), 1, cv2.LINE_AA)

        return ui
