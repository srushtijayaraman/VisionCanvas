# ============================================================
# VisionCanvas — Centralized Configuration
# ============================================================
from pathlib import Path

# --- Project Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = str(PROJECT_ROOT / "src" / "hand_tracking" / "hand_landmarker.task")
OUTPUT_DIR = str(PROJECT_ROOT / "output")

# --- Camera ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
CAMERA_BUFFER_SIZE = 1  # Minimize frame buffering for low latency

# --- Hand Tracking ---
MAX_HANDS = 1
DETECTION_CONFIDENCE = 0.7
TRACKING_CONFIDENCE = 0.7

# --- One Euro Filter (Smoothing) ---
# mincutoff: lower = smoother but laggier. 1.0-5.0 ideal for hand air-drawing
# beta: higher = more responsive to fast movement. 0.05-0.2 typical
# dcutoff: derivative cutoff, usually left at 1.0
SMOOTHING_ENABLED = True
ONE_EURO_MIN_CUTOFF = 2.0
ONE_EURO_BETA = 0.10
ONE_EURO_DCUTOFF = 1.0

# --- Virtual Cursor & Outlier Rejection ---
MAX_CURSOR_JUMP = 150  # Jump rejection threshold in pixels
CURSOR_OUTLIER_RECOVERY_FRAMES = 3  # Accept new persistent position after N consistent frames

# --- Drawing ---
MAX_INTERPOLATION_DISTANCE = 5.0
BRUSH_DEFAULT_SIZE = 12
BRUSH_DEFAULT_COLOR = (180, 80, 255)  # BGR purple

# --- Canvas ---
CANVAS_WIDTH = 640   # Matched to camera for 1:1 mapping
CANVAS_HEIGHT = 480

# --- UI Window ---
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 720

# --- Gesture State Machine ---
GESTURE_STABILITY_FRAMES = 2  # Consecutive frames required to enter a new gesture
GESTURE_COOLDOWN_TIME = 0.4   # Minimum seconds before a released one-shot action can re-fire
GESTURE_DEBOUNCE_TIME = 0.4   # Legacy alias for backward compatibility

# --- Performance / Debug ---
DEBUG_PERFORMANCE = False
TARGET_FPS = 30

# --- Hand Landmark Overlay ---
DRAW_HAND_LANDMARKS = True
LANDMARK_COLOR = (255, 120, 255)            # BGR neon pink
LANDMARK_CONNECTION_COLOR = (180, 80, 255)  # BGR purple
LANDMARK_DOT_RADIUS = 4
LANDMARK_LINE_THICKNESS = 2
