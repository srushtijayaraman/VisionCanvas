# VisionCanvas 2.0 🎨🖐️

VisionCanvas is a real-time, gesture-controlled virtual drawing application powered by **OpenCV**, **MediaPipe**, and **NumPy**. Draw, sketch, erase, select colors, and automatically convert hand-drawn shapes into crisp geometric primitives in the air using natural hand gestures through your webcam.

---

## ✨ Features

- **Real-Time Hand Tracking & Gesture Control:** Powered by MediaPipe's HandLandmarker model running asynchronously in video mode to deliver low-latency tracking.
- **Gesture State Machine & Temporal Debounce:** Strict state-transition logic distinguishes continuous interactions (drawing, selecting) from discrete one-shot commands (`UNDO`, `REDO`, `SAVE`, `CLEAR`), preventing sustained poses from firing repeated commands.
- **Adaptive Virtual Cursor with Outlier Recovery:** Dual-axis **One Euro Filter** smoothing reduces hand jitter at low velocities while remaining responsive during rapid movement. Automatic outlier recovery prevents the cursor from freezing after large hand movements or re-entry.
- **Smart Ink Geometric Shape Recognizer:** Draw rough shapes in the air (lines, circles, rectangles, triangles); VisionCanvas automatically detects the geometry and replaces the rough stroke with a clean mathematical shape, with automatic freehand fallback for low-confidence or arbitrary sketches.
- **5 Expressive Brush Presets:**
  1. **Hard Brush:** Solid, anti-aliased clean lines.
  2. **Soft Brush:** Gaussian-blurred airbrush effect using localized ROI processing.
  3. **Neon Brush:** Multi-layer luminous glow with alpha blending.
  4. **Pencil Brush:** Semi-transparent, fine sketch pencil stroke.
  5. **Watercolor Brush:** Textured, organic wet watercolor stroke with thickness jitter.
- **True Alpha-Masked Canvas & Export:** Separate 8-bit alpha mask preserves dark marks, black pencil sketches, and subtle shading over the camera feed. Exports clean white-background drawings or transparent PNGs (`output/`).
- **Interactive Glassmorphic UI:** Control application state using both in-air gesture pointing (`SELECT` mode) and mouse clicks on tools, brush presets, the color palette, the color wheel, the brush size slider, and bottom action buttons.
- **Diagnostics Mode:** Real-time on-screen telemetry reporting separate Render FPS, Inference FPS, inference latency, and tracking status.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+ (tested on Python 3.13)
- Working webcam (USB or built-in)
- MediaPipe Hand Landmarker model (`src/hand_tracking/hand_landmarker.task`, included in repository)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Shashwat2506/VisionCanvas.git
   cd VisionCanvas
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv

   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1

   # Windows (cmd):
   .\.venv\Scripts\activate.bat

   # Linux/macOS:
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *To run tests, install development dependencies:*
   ```bash
   pip install -r requirements-dev.txt
   ```

4. **Verify environment:**
   ```bash
   python -m pip check
   ```

---

## 🎮 Running the Application

Launch the main application:

```bash
python main.py
```

*(Or explicitly with your virtual environment on Windows: `.\.venv\Scripts\python.exe main.py`)*

---

## 🖐️ Hand Gestures Guide

| Gesture | Hand Pose | Action |
| :--- | :--- | :--- |
| **DRAW** | Index finger pointing up | Draws continuous strokes (or erases if Eraser is active). |
| **SELECT** | Index + Middle fingers pointing up (V-pose) | Hover to select tools, colors, brush presets, sliders, and buttons without drawing. |
| **UNDO** | 3 fingers extended (Index, Middle, Ring) | Triggers one undo step. Pose must be released before re-triggering. |
| **REDO** | 4 or 5 fingers extended | Triggers one redo step. Pose must be released before re-triggering. |
| **SAVE** | Thumbs Up (Thumb up, all 4 fingers curled) | Saves drawing snapshot to `output/canvas_<timestamp>.png`. |
| **NONE** | Closed fist or hand at rest | Neutral state; disarms all drawing and actions. |

---

## ⌨️ Controls & Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| `b` | Switch to **Brush** mode |
| `e` | Switch to **Eraser** mode |
| `i` | Toggle **Smart Ink** (auto shape recognition) |
| `1` – `5` | Switch brush preset (1: Hard, 2: Soft, 3: Neon, 4: Pencil, 5: Water) |
| `[` / `]` | Decrease / Increase brush thickness |
| `c` | Clear canvas (undoable) |
| `z` | Undo stroke |
| `y` | Redo stroke |
| `s` | Save canvas (white background PNG) to `output/` |
| `x` | Export canvas (transparent PNG) to `output/` |
| `d` | Toggle real-time diagnostics overlay (Render FPS, Inference FPS, Latency) |
| `q` / `Esc` | Cleanly exit application |

---

## 📐 Smart Ink (Signature Feature)

Smart Ink brings geometric intelligence to VisionCanvas:
1. Enable Smart Ink by selecting the **Smart Ink** tool in the UI or pressing `i`.
2. Draw a rough geometric shape in the air (a straight line, rectangle, circle, or triangle).
3. When the drawing gesture ends (finger lowered or transitioned to neutral), the `GeometryRecognizer` analyzes the polyline:
   - Evaluates arc length, chord distance, closure ratio, and bounding box dimensions.
   - For open strokes: evaluates straightness and perpendicular deviation to fit a clean line segment.
   - For closed strokes: assesses radial standard deviation to fit a circle, or runs adaptive Douglas-Peucker contour simplification to detect triangles (3 corners) or rectangles (4 perpendicular corners).
4. If confidence is $\ge 65\%$, the rough preview stroke is replaced by a crisp geometric shape.
5. If the stroke is ambiguous (e.g. an organic illustration or random scribble), the original freehand stroke is preserved unmodified.

---

## 🧪 Automated Testing

VisionCanvas includes a comprehensive `pytest` test suite with 50 unit and integration tests covering filters, cursor recovery, state machine latching, rotation invariance, canvas history, export, and shape recognition:

```bash
pytest -v
```

### Test Coverage Highlights:
- **Cursor Stability & Recovery (`test_virtual_cursor.py`):** 8 test scenarios verifying stationary noise smoothing, smooth tracking, single large outlier rejection, 3-frame persistent jump recovery, tracking loss, reacquisition, and invalid (NaN/Inf) coordinate handling.
- **Filter Mathematics (`test_one_euro_filter.py`):** Verified zero/negative delta-time guards, division-by-zero prevention, and filter reinitialization.
- **Gesture State Machine (`test_gesture_state_machine.py`):** Latching tests proving sustained poses never spam one-shot actions, release/cooldown enforcement, and noisy spike rejection.
- **Canvas & Alpha Masking (`test_drawing_canvas.py`):** Single-point tap rendering, stroke interpolation, undo/redo stacks, clearing, dark stroke preservation, and PNG file export.
- **Rotation-Invariant Finger Detector (`test_finger_detector.py`):** Synthetic landmark tests across varying 2D tilt angles and handedness.
- **Smart Ink Recognizer (`test_geometry_recognizer.py`):** Tests for lines of varying slopes, noisy straight paths, rectangles with unequal sides, triangles of various orientations, circles of small/large radii, degenerate paths, scribbles, and micro-benchmark latency ($< 0.2\text{ ms}$).

---

## 📁 Project Structure

```
VisionCanvas/
├── assets/                  # Project static assets
├── docs/
│   └── ARCHITECTURE.md      # Detailed system design, data flow, and benchmarks
├── output/                  # Exported drawing images (.png)
├── src/
│   ├── brushes/             # Brush implementations (Hard, Soft, Neon, Pencil, Water)
│   ├── camera/              # Threaded webcam capture with safe failure handling
│   ├── canvas/              # Drawing canvas, alpha masking, history stacks, and export
│   ├── geometry/            # Smart Ink geometric shape recognition pipeline
│   ├── gestures/            # Virtual cursor, One Euro Filter, and GestureStateMachine
│   ├── hand_tracking/       # MediaPipe HandLandmarker wrapper, finger detector, model task
│   ├── ui/                  # Glassmorphic UI compositor and interaction hit-testing
│   ├── utils/               # 1€ Filter mathematical implementation
│   └── config.py            # Centralized portable configuration and thresholds
├── tests/                   # 50 comprehensive unit and integration tests
├── main.py                  # Application entry point with structured try/finally lifecycle
├── pytest.ini               # Pytest configuration
├── requirements.txt         # Clean runtime dependencies
├── requirements-dev.txt     # Development and testing dependencies
└── README.md
```

---

## 🔧 Troubleshooting

- **Webcam Fails to Open:** Ensure no other application (Zoom, Teams, browser) is locking the webcam. If using an external USB webcam, open `src/config.py` and set `CAMERA_INDEX = 1`.
- **Model File Not Found:** Ensure `src/hand_tracking/hand_landmarker.task` is present in your working tree. `src/config.py` uses portable `pathlib.Path` resolution relative to the project root.
- **OpenCV Version Conflicts:** Ensure you did not manually install both `opencv-python` and `opencv-contrib-python`. `requirements.txt` specifies `opencv-contrib-python>=4.8.0` which is fully compatible with MediaPipe.

---

## 📄 License & Attribution

- VisionCanvas application code is licensed under the MIT License.
- MediaPipe Hand Landmarker model is provided by Google LLC under the Apache 2.0 License.
- One Euro Filter algorithm is based on Casiez et al. (CHI 2012).
