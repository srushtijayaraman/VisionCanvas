# VisionCanvas 2.0 — Architecture and Technical Specification

## 1. System Overview

VisionCanvas is a real-time, desktop computer-vision application that allows users to draw, erase, manipulate tools, and generate clean geometric shapes in the air using hand gestures captured by a webcam.

```
┌──────────────┐     BGR Frame     ┌────────────────────────┐
│    Webcam    │ ────────────────> │      HandDetector      │
│ (Background) │                   │  (MediaPipe Landmarker)│
└──────────────┘                   └───────────┬────────────┘
                                               │ Landmarks
                                               ▼
┌──────────────────┐               ┌────────────────────────┐
│  DrawingCanvas   │               │     FingerDetector     │
│ (BGR + AlphaMask)│ <── Commands  │  (Rotation-Invariant)  │
└────────┬─────────┘               └───────────┬────────────┘
         │ Canvas Frame                        │ Finger States
         │ + Alpha Mask                        ▼
         │                         ┌────────────────────────┐
         │                         │   GestureRecognizer    │
         │                         │    (Exact + Relaxed)   │
         │                         └───────────┬────────────┘
         │                                     │ Raw Gesture
         │                                     ▼
         │                         ┌────────────────────────┐
         │                         │  GestureStateMachine   │
         │                         │  (Debounce + One-Shot) │
         │                         └───────────┬────────────┘
         │                                     │ Stabilized Event
         ▼                                     ▼
┌───────────────────────────────────────────────────────────┐
│                         ModernUI                          │
│          Glassmorphic Compositor & Control Surface        │
└───────────────────────────────────────────────────────────┘
```

---

## 2. Module Responsibilities

| Module | Location | Primary Responsibility |
| :--- | :--- | :--- |
| **Main Orchestrator** | `main.py` | Event loop, thread lifecycle management, keyboard/mouse dispatch, and teardown. |
| **Webcam Capture** | `src/camera/webcam.py` | Threaded capture avoiding I/O blocking, graceful error handling, throttled polling on failure. |
| **Hand Detector** | `src/hand_tracking/detector.py` | Wrapper around MediaPipe HandLandmarker in VIDEO mode; monotonic inference ID and latency tracking. |
| **Finger Detector** | `src/hand_tracking/finger_detector.py` | Rotation-invariant finger classification using wrist-to-tip vs wrist-to-PIP relative distances. |
| **Gesture Recognizer** | `src/gestures/gesture_recognizer.py` | Maps 5-finger state tuples to canonical gesture names (`DRAW`, `SELECT`, `UNDO`, `REDO`, `SAVE`). |
| **Gesture State Machine** | `src/gestures/state_machine.py` | Distinguishes continuous gestures from one-shot actions; latching prevents sustained-pose repetition. |
| **Virtual Cursor** | `src/gestures/cursor.py` | Dual-axis One Euro Filter smoothing with outlier jump rejection and persistent-movement recovery. |
| **One Euro Filter** | `src/utils/one_euro_filter.py` | Low-latency 1st-order adaptive low-pass filter with division-by-zero guards and state reset. |
| **Drawing Canvas** | `src/canvas/drawing_canvas.py` | BGR drawing layer + separate 8-bit alpha mask; stroke interpolation; undo/redo history; PNG export. |
| **Brushes** | `src/brushes/` | `HardBrush`, `SoftBrush`, `NeonBrush`, `PencilBrush`, and `WatercolorBrush`. |
| **Geometry Recognizer** | `src/geometry/geometry_recognizer.py` | Rule-based Smart Ink shape recognizer (`LINE`, `RECTANGLE`, `CIRCLE`, `TRIANGLE`) with freehand fallback. |
| **Modern UI** | `src/ui/modern_ui.py` | Glassmorphic interface compositor, status message banners, diagnostics overlay, and hit-testing. |
| **Configuration** | `src/config.py` | Portable paths (`pathlib.Path`), algorithm thresholds, resolution, and performance settings. |

---

## 3. Data Flow and Processing Pipeline

### 3.1 Frame Capture & Hand Tracking
1. `Webcam` continuously reads BGR frames in a background thread to prevent camera I/O from stalling the render loop.
2. `HandDetector` converts the mirrored frame to RGB and runs MediaPipe's `HandLandmarker` in video mode with monotonically increasing timestamps.
3. Detections are tagged with a monotonic `_inference_id` and timestamp, allowing the render loop to determine whether an inference result is fresh (`is_new`).

### 3.2 Finger State & Gesture Transition
1. `FingerDetector` evaluates landmarks 0 through 20:
   - **Thumb**: Evaluates distance between tip (4) and pinky base (17) relative to IP joint (3) to detect abduction, as well as wrist-to-tip vertical extension for thumbs-up poses.
   - **Fingers (8, 12, 16, 20)**: Compares $d(\text{wrist}, \text{tip})$ vs $d(\text{wrist}, \text{PIP})$. Because Euclidean distances from the wrist are rotationally symmetric, this classification works across 360° of in-plane hand tilt.
2. `GestureRecognizer` maps the 5-finger binary array to gestures, supporting canonical poses and natural relaxed-thumb variations.
3. `GestureStateMachine` enforces:
   - **Stability window**: A candidate gesture must persist for $\ge N$ frames (default 2) before the active gesture changes.
   - **One-shot latching**: Discrete commands (`UNDO`, `REDO`, `SAVE`, `CLEAR`) fire once upon entry and are latched until the user deliberately releases to neutral (`NONE`) and the cooldown timer ($0.4\text{s}$) elapses.

### 3.3 Virtual Cursor & Outlier Rejection
1. Normalized index fingertip coordinates are mapped from the camera viewport across the UI workspace bounds.
2. `VirtualCursor` checks the Euclidean jump distance $\Delta = \sqrt{\Delta x^2 + \Delta y^2}$:
   - **$\Delta \le \text{MAX\_CURSOR\_JUMP}$**: Smooth continuous motion filtered via `OneEuroFilter`.
   - **$\Delta > \text{MAX\_CURSOR\_JUMP}$**:
     - *Isolated glitch (1-2 frames)*: Rejected; cursor holds previous accepted position.
     - *Persistent movement ($\ge 3$ consecutive frames)*: Accepted as deliberate rapid movement or reacquisition; filter state resets cleanly to the new position without freezing.
   - **Tracking loss (`landmark is None`)**: Cursor flags tracking loss; upon reacquisition, filters reinitialize immediately.

### 3.4 Canvas Rendering & Alpha Masking
1. Previous versions used naive grayscale thresholding `cv2.threshold(gray, 10, ...)` to composite drawings over the camera feed, which clipped dark colors, black strokes, and soft pencil lines.
2. `DrawingCanvas` now maintains two parallel layers:
   - `self.canvas`: 3-channel BGR image.
   - `self.mask`: 1-channel uint8 alpha mask ($0 = \text{transparent}, 255 = \text{opaque}$).
3. When drawing, both the BGR canvas and the alpha mask are updated. Black strokes (`0, 0, 0`) retain `mask = 255`, ensuring 100% visibility over camera and white export backgrounds.
4. History stacks (`undo_stack`, `redo_stack`) store lightweight copies of `(canvas, mask)` tuples up to a limit of 25 snapshots.

---

## 4. Smart Ink Geometric Recognition

Smart Ink (`src/geometry/geometry_recognizer.py`) converts rough air-drawn shapes into crisp geometric primitives using explainable geometric algorithms:

```
Raw Stroke Points ──> Deduplication ──> Arc Length & Closure ──> Feature Extraction
                                                                        │
                 ┌──────────────────────────────────────────────────────┴───────────────┐
                 ▼                                                                      ▼
        Open Stroke (Cr > 0.65)                                              Closed Stroke (Cr < 0.45)
                 │                                                                      │
                 ▼                                                      ┌───────────────┴───────────────┐
      Line Fitting (PCA / chord)                                        ▼                               ▼
       Straightness check                                    Circle Fitting (Kåsa)           Douglas-Peucker (DP)
                 │                                           Radial variance < 0.20                     │
                 ▼                                                      │                               ▼
          Shape: LINE                                            Shape: CIRCLE                 Vertex Count N:
                                                                                               N=3: TRIANGLE
                                                                                               N=4: RECTANGLE
```

### 4.1 Recognition Rules
- **Line**:
  - Open stroke ($\text{closure\_ratio} > 0.65$).
  - Perpendicular deviation of all points to the chord segment: $\frac{d_{\max}}{L} < 0.16$ and $\frac{d_{\text{mean}}}{L} < 0.08$.
  - Refined to: `[p_start, p_end]`.
- **Circle**:
  - Closed stroke ($\text{closure\_ratio} < 0.45$).
  - Bounding box aspect ratio $0.60 \le w/h \le 1.65$.
  - Algebraic radial standard deviation $\sigma_R / \bar{R} < 0.20$.
  - Arc length to circumference ratio $0.70 \le L / (2\pi \bar{R}) \le 1.55$.
  - Refined to: 36-point regular circle polygon.
- **Triangle**:
  - Closed contour simplified via adaptive Douglas-Peucker ($\epsilon = [0.02 - 0.06] \times L$) yields 3 vertices.
  - Convex contour with area $> 12\%$ of bounding box area.
  - Interior angles between $15^\circ$ and $150^\circ$.
  - Refined to: 3 corner vertices.
- **Rectangle**:
  - Closed contour simplified via adaptive Douglas-Peucker yields 4 vertices.
  - Convex contour with area $> 45\%$ of bounding box area.
  - Interior corner angles near $90^\circ$ ($55^\circ \le \theta \le 125^\circ$).
  - Refined to: `cv2.boxPoints(cv2.minAreaRect(pts))`.
- **Fallback Guarantee**:
  - If a stroke fails confidence checks ($\text{confidence} < 0.65$), the recognizer returns `UNKNOWN` with the original stroke points preserved, guaranteeing that freehand drawings and complex scribbles are never discarded.

---

## 5. Performance Measurements

Micro-benchmarks executed on the reference environment (Python 3.13, Windows 11 AMD64):

| Component | Benchmark Metric | Measured Time | Real-Time Budget | Headroom |
| :--- | :--- | :--- | :--- | :--- |
| `GeometryRecognizer` | 60-point stroke analysis | **0.107 ms** | $< 5.0\text{ ms}$ | $46\times$ margin |
| `OneEuroFilter` | Per-coordinate filter step | **1.230 µs** | $< 100\text{ µs}$ | $81\times$ margin |
| `DrawingCanvas` | Stroke interpolation & draw | **0.018 ms** | $< 1.0\text{ ms}$ | $55\times$ margin |
| Render Loop (UI + Blend) | Single frame render | **~2.8 ms** | $< 16.6\text{ ms}$ (60 FPS) | $6\times$ margin |
