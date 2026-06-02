# Magic Gesture 3D Controller - Dual Hand

Real-time hand gesture recognition system that controls interactive 3D visuals using webcam input. Supports **dual-hand collaborative interaction** with independent left/right hand tracking and coordinated two-hand gestures.

## Features

### Single-Hand Gestures

| Gesture | Hand | Action |
|---------|------|--------|
| Fist (Left) | Left | Horizontal rotation of 3D model |
| Fist (Right) | Right | Vertical flip/tilt of 3D model |
| Pinch (Right) | Right | Model scale control + particle wave effect |
| Wave (Left) | Left | Color change with pulse ring effect |
| Wave (Right) | Right | Particle explosion effect |
| Pointing | Either | Used for dual-hand zoom (see below) |

### Dual-Hand Gestures

| Gesture | Action |
|---------|--------|
| Both hands pointing (index finger) | Pinch-to-zoom by finger distance |
| Palms close together (< palm width) for 1 second | Reset model to initial position and size |

### Additional Features

- **Digit Recognition**: Real-time count of extended fingers (0-5) displayed for each hand
- **Independent State Machines**: Left and right hands have separate gesture state machines with individual majority-vote filtering and transition cooldowns, preventing cross-hand interference
- **Anti-Jitter**: EMA smoothing, majority voting (5-frame window), and state transition cooldown eliminate hand gesture oscillation
- **Left/Right Confidence Display**: Real-time gesture name, confidence score, and hand detection confidence shown separately for each hand
- **Low Latency**: ~18fps capture rate, optimized JPEG compression, pre-allocated particle buffers

## Tech Stack

- **Backend**: Python, FastAPI, WebSocket, MediaPipe Hand Landmarker
- **Frontend**: Three.js, Canvas 2D overlay, Web Audio API
- **Communication**: WebSocket for real-time bidirectional frame/gesture data

## Requirements

- Python 3.10+
- Webcam
- Modern browser (Chrome/Edge recommended)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download MediaPipe Hand Landmarker Model

The model file `hand_landmarker.task` must be in the project root. Download from:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 3. Run

```bash
python server.py
```

Or use the provided scripts:

- **Windows**: `start.bat`
- **Linux/macOS**: `./start.sh`

Then open the URL shown in the console (default `http://localhost:8000`, auto-increments if port is occupied).

## Startup Behavior

- **Auto port scanning**: If port 8000 is occupied, the server scans 8001-8099 and uses the first available port. The actual URL is printed to console.
- **Model file discovery**: Searches for `hand_landmarker.task` in the project root and common locations. Prints clear error with download link if missing.
- **MediaPipe retry**: Up to 3 initialization attempts with 1s delay between retries.
- **Detailed logging**: All operations print timestamped logs to console for debugging.

## Architecture

```
Browser (webcam) --[WebSocket: JPEG frames]--> FastAPI Server
                                                  |
                                           MediaPipe Detection
                                           (2 hands, landmarks)
                                                  |
                                           DualHandTracker
                                          /                \
                              Left GSM                Right GSM
                          (independent state)     (independent state)
                                          \                /
                                           Dual Coordination
                                           (proximity, zoom)
                                                  |
Browser <--[WebSocket: JSON gesture data]-- Response Builder
   |
   ├── Left/Right Gesture Display (UI)
   ├── Three.js 3D Model Control
   ├── Particle Effects (wave, explosion)
   └── Hand Overlay (Canvas 2D)
```

## Gesture Recognition Pipeline

1. **Frame Capture**: Browser sends 320x240 JPEG frames at ~18fps
2. **Hand Detection**: MediaPipe detects up to 2 hands with landmarks and handedness classification
3. **Hand Assignment**: Hands are classified as Left/Right based on MediaPipe handedness labels
4. **Per-Hand Processing**: Each hand runs through its own `GestureStateMachine`:
   - Finger curl computation (angle-based)
   - Raw gesture classification (fist, pinch, pointing, wave, none)
   - 5-frame majority vote for stability
   - State machine with transition cooldown
   - EMA smoothing on continuous values
5. **Dual-Hand Coordination**: When both hands are detected:
   - Palm distance monitoring for reset gesture
   - Dual-pointing detection for zoom control
6. **Digit Recognition**: Count of extended fingers per hand (thumb uses x-distance heuristic)

## Controls Reference

| Input | Effect |
|-------|--------|
| Left fist + move left/right | Rotate model horizontally |
| Right fist + move up/down | Tilt model vertically |
| Right pinch | Scale model + particle wave |
| Both index fingers + move apart/together | Zoom in/out |
| Left wave (shake hand) | Cycle model color |
| Right wave (shake hand) | Trigger explosion particles |
| Both palms together (hold 1s) | Reset to default view |

## Configuration

Key thresholds can be adjusted in `server.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `FIST_CURL_THRESHOLD` | 0.32 | Minimum weighted curl for fist detection |
| `PINCH_RATIO` | 0.45 | Max thumb-index distance (relative to palm) for pinch |
| `WAVE_SPEED_THRESHOLD` | 0.45 | Minimum speed for wave detection |
| `PROXIMITY_THRESHOLD` | 0.12 | Palm distance threshold for reset gesture |
| `PROXIMITY_HOLD_TIME` | 1.0 | Seconds palms must stay close for reset |
| `EMA_ALPHA` | 0.35 | Smoothing factor (higher = more responsive, less smooth) |

## Troubleshooting

- **Hands not detected**: Ensure good lighting and that hands are clearly visible in the camera frame
- **Left/right confused**: MediaPipe handedness can swap occasionally; keep hands separated for best results
- **High latency**: Reduce browser tab load; the system is designed for ~18fps send rate
- **Model file missing**: Download `hand_landmarker.task` to the project root directory

## License

MIT
