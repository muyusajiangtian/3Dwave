import asyncio
import base64
import json
import math
import time
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.6,
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)

# ============================================================
# Thresholds
# ============================================================
# Fist: weighted curl of all 5 fingers. Half-fist triggers at 0.32.
FIST_CURL_THRESHOLD = 0.32

# Pinch: thumb-index distance relative to palm size.
# Pinch requires: distance < palm_size * PINCH_RATIO AND other 3 fingers NOT all curled.
PINCH_RATIO = 0.45

# Wave: cumulative displacement over 5 frames, speed > threshold
WAVE_FRAMES = 5
WAVE_SPEED_THRESHOLD = 0.45
WAVE_COOLDOWN = 1.0  # seconds between wave triggers

# Smoothing
EMA_ALPHA = 0.35

# MediaPipe handedness confidence filter
MIN_HAND_CONFIDENCE = 0.65

# ============================================================
# Geometry helpers
# ============================================================

def dist3(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)


def angle_at(a, b, c):
    """Angle (radians) at vertex b in triangle a-b-c."""
    ba = (a.x-b.x, a.y-b.y, a.z-b.z)
    bc = (c.x-b.x, c.y-b.y, c.z-b.z)
    dot_val = ba[0]*bc[0] + ba[1]*bc[1] + ba[2]*bc[2]
    m1 = math.sqrt(ba[0]**2+ba[1]**2+ba[2]**2)
    m2 = math.sqrt(bc[0]**2+bc[1]**2+bc[2]**2)
    if m1 < 1e-7 or m2 < 1e-7:
        return math.pi
    return math.acos(max(-1.0, min(1.0, dot_val/(m1*m2))))


# ============================================================
# Per-finger curl computation
# ============================================================
# Landmark indices: [MCP, PIP, DIP, TIP]
FINGERS = {
    'thumb':  [1, 2, 3, 4],
    'index':  [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring':   [13, 14, 15, 16],
    'pinky':  [17, 18, 19, 20],
}

# Weighted importance for fist detection
FIST_WEIGHTS = {'thumb': 0.12, 'index': 0.24, 'middle': 0.24, 'ring': 0.20, 'pinky': 0.20}


def finger_curl(landmarks, indices):
    """Return curl value in [0,1]. 0=straight, 1=fully curled.
    Based on PIP angle and DIP angle (smaller angle = more curled)."""
    mcp, pip, dip, tip = [landmarks[i] for i in indices]
    # For thumb, indices are CMC, MCP, IP, TIP
    ang_pip = angle_at(mcp, pip, dip)
    ang_dip = angle_at(pip, dip, tip)
    # pi = straight (curl=0), small angle = curled (curl->1)
    c_pip = 1.0 - ang_pip / math.pi
    c_dip = 1.0 - ang_dip / math.pi
    return c_pip * 0.55 + c_dip * 0.45


def compute_all_curls(landmarks):
    """Returns dict of finger -> curl value."""
    return {name: finger_curl(landmarks, idx) for name, idx in FINGERS.items()}


def compute_fist_strength(curls):
    """Weighted average of all finger curls."""
    return sum(curls[f] * FIST_WEIGHTS[f] for f in FIST_WEIGHTS)


def palm_size(landmarks):
    """Distance from wrist to middle MCP as palm reference size."""
    return dist3(landmarks[0], landmarks[9])


# ============================================================
# Gesture State Machine
# ============================================================
class GestureStateMachine:
    """
    States: none, fist, pinch, wave
    Transitions require majority vote over N frames.
    Wave has cooldown timer.
    """
    def __init__(self):
        self.state = "none"
        self.history = deque(maxlen=7)  # raw gesture per frame
        self.wave_last_time = 0.0
        self.frame_count = 0
        # Position history for wave detection (x, timestamp)
        self.pos_history = deque(maxlen=20)
        # EMA values
        self.ema_fist = None
        self.ema_pinch_dist = None
        self.ema_rot_x = None
        self.ema_rot_y = None

    def ema(self, prev, new, alpha=EMA_ALPHA):
        if prev is None:
            return new
        return prev * (1.0 - alpha) + new * alpha

    def update(self, landmarks):
        """Process one frame. Returns gesture result dict with all debug info."""
        self.frame_count += 1
        now = time.time()

        wrist = landmarks[0]
        self.pos_history.append((wrist.x, wrist.y, now))

        # --- Compute raw metrics ---
        curls = compute_all_curls(landmarks)
        fist_strength_raw = compute_fist_strength(curls)
        p_size = palm_size(landmarks)

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_dist_raw = dist3(thumb_tip, index_tip)

        # Normalized pinch distance (relative to palm size)
        pinch_norm = pinch_dist_raw / p_size if p_size > 0.01 else 1.0

        # --- EMA smooth continuous values ---
        self.ema_fist = self.ema(self.ema_fist, fist_strength_raw)
        self.ema_pinch_dist = self.ema(self.ema_pinch_dist, pinch_norm)
        fist_strength = self.ema_fist
        pinch_normalized = self.ema_pinch_dist

        # --- Classify raw gesture for this frame ---
        raw_gesture = self._classify_frame(
            curls, fist_strength, pinch_normalized, p_size, now
        )

        # --- Majority vote (last 5 frames) ---
        self.history.append(raw_gesture)
        voted_gesture = self._majority_vote()

        # --- State transition with hysteresis ---
        final_gesture = self._apply_state_machine(voted_gesture, now)

        # --- Compute output parameters ---
        result = self._build_result(
            final_gesture, landmarks, curls, fist_strength,
            pinch_dist_raw, pinch_normalized, p_size, now
        )

        # --- Debug log ---
        self._log_frame(curls, fist_strength_raw, fist_strength,
                        pinch_dist_raw, pinch_normalized,
                        raw_gesture, voted_gesture, final_gesture, result)

        return result

    def _classify_frame(self, curls, fist_strength, pinch_norm, p_size, now):
        """Determine raw gesture for single frame. STRICT disambiguation."""

        # === FIST CHECK ===
        # Fist = ALL 5 fingers simultaneously curling toward palm
        # Key: even half-fist (fist_strength >= threshold) triggers
        # But we need ALL fingers contributing - no single finger can be too open
        is_fist = False
        if fist_strength >= FIST_CURL_THRESHOLD:
            # Additional check: at least 4 of 5 fingers have curl > 0.2
            curled_count = sum(1 for v in curls.values() if v > 0.2)
            if curled_count >= 4:
                is_fist = True

        # === PINCH CHECK ===
        # Pinch = thumb+index tips close, OTHER 3 fingers NOT curled toward palm
        # Key distinction from fist: middle/ring/pinky must be RELAXED (not curled)
        is_pinch = False
        if pinch_norm < PINCH_RATIO:
            # The other 3 fingers must NOT all be curled
            # For pinch: middle, ring, pinky should be relatively extended
            other_avg_curl = (curls['middle'] + curls['ring'] + curls['pinky']) / 3.0
            # Pinch: other fingers average curl < 0.35 (relaxed/extended)
            # Fist: other fingers average curl > 0.35 (all curling)
            if other_avg_curl < 0.35:
                is_pinch = True

        # === DISAMBIGUATION ===
        # If both could trigger, decide based on the clearer signal
        if is_fist and is_pinch:
            # Compare: if other fingers are quite curled, it's a fist
            other_avg = (curls['middle'] + curls['ring'] + curls['pinky']) / 3.0
            if other_avg > 0.25:
                is_pinch = False  # Other fingers curling -> fist, not pinch
            else:
                is_fist = False   # Other fingers open -> pinch, not fist

        if is_pinch:
            return "pinch"
        if is_fist:
            return "fist"

        # === WAVE CHECK ===
        # Cumulative x-displacement over last 5 frames, with speed threshold
        if len(self.pos_history) >= WAVE_FRAMES and fist_strength < 0.25:
            recent = list(self.pos_history)[-WAVE_FRAMES:]
            total_time = recent[-1][2] - recent[0][2]
            if total_time > 0.05:
                # Cumulative absolute x displacement
                cum_disp = sum(abs(recent[i][0] - recent[i-1][0]) for i in range(1, len(recent)))
                avg_speed = cum_disp / total_time

                # Check x-range (must have actual oscillation, not just drift)
                xs = [p[0] for p in recent]
                x_range = max(xs) - min(xs)

                # Direction changes in the window
                dir_changes = 0
                for i in range(2, len(xs)):
                    if (xs[i]-xs[i-1]) * (xs[i-1]-xs[i-2]) < -0.0001:
                        dir_changes += 1

                if avg_speed > WAVE_SPEED_THRESHOLD and x_range > 0.06 and dir_changes >= 1:
                    # Check cooldown
                    if now - self.wave_last_time > WAVE_COOLDOWN:
                        return "wave"

        return "none"

    def _majority_vote(self):
        """5-frame majority vote from history."""
        if len(self.history) < 3:
            return self.history[-1] if self.history else "none"

        # Use last 5 frames (or whatever is available up to 5)
        window = list(self.history)[-5:]
        counts = Counter(window)
        winner, count = counts.most_common(1)[0]

        # Require at least 3 out of 5 (or 2 out of 3 if less history)
        threshold = max(2, len(window) // 2 + 1)
        if count >= threshold:
            return winner
        # If no clear majority, stick with most recent that's not 'none'
        for g in reversed(window):
            if g != "none":
                return g
        return "none"

    def _apply_state_machine(self, voted, now):
        """Apply state transitions with minimum hold time."""
        if voted == self.state:
            return self.state

        # Transition allowed - update state
        if voted == "wave":
            self.wave_last_time = now
        self.state = voted
        return voted

    def _build_result(self, gesture, landmarks, curls, fist_strength,
                      pinch_dist_raw, pinch_norm, p_size, now):
        """Build the output JSON result."""
        wrist = landmarks[0]
        palm_cx = (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 5
        palm_cy = (landmarks[0].y + landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 5
        palm_cz = (landmarks[0].z + landmarks[5].z + landmarks[9].z + landmarks[13].z + landmarks[17].z) / 5

        result = {
            "gesture": gesture,
            "fist_strength": round(fist_strength, 4),
            "pinch_distance": round(pinch_dist_raw, 4),
            "pinch_normalized": round(pinch_norm, 4),
            "confidence": 0.0,
            "palm": {"x": round(palm_cx, 4), "y": round(palm_cy, 4), "z": round(palm_cz, 4)},
            "frame_id": self.frame_count,
            "curls": {k: round(v, 3) for k, v in curls.items()},
        }

        if gesture == "fist":
            rot_x_raw = (wrist.x - 0.5) * math.pi * 2
            rot_y_raw = (wrist.y - 0.5) * math.pi * 2
            self.ema_rot_x = self.ema(self.ema_rot_x, rot_x_raw, 0.5)
            self.ema_rot_y = self.ema(self.ema_rot_y, rot_y_raw, 0.5)
            result["rotation_x"] = round(self.ema_rot_x, 4)
            result["rotation_y"] = round(self.ema_rot_y, 4)
            result["confidence"] = round(min(1.0, (fist_strength - FIST_CURL_THRESHOLD) / 0.4), 3)

        elif gesture == "pinch":
            pinch_intensity = max(0.0, 1.0 - pinch_norm / PINCH_RATIO)
            scale = max(0.3, min(3.0, 1.0 / (pinch_dist_raw * 12 + 0.1)))
            result["pinch_intensity"] = round(pinch_intensity, 3)
            result["scale"] = round(scale, 3)
            result["confidence"] = round(min(1.0, pinch_intensity + 0.3), 3)

        elif gesture == "wave":
            # Compute speed for display
            if len(self.pos_history) >= WAVE_FRAMES:
                recent = list(self.pos_history)[-WAVE_FRAMES:]
                dt = recent[-1][2] - recent[0][2]
                if dt > 0:
                    cum = sum(abs(recent[i][0]-recent[i-1][0]) for i in range(1,len(recent)))
                    result["palm_speed"] = round(cum/dt, 3)
            result["confidence"] = 0.8

        return result

    def _log_frame(self, curls, fist_raw, fist_ema, pinch_raw, pinch_norm,
                   raw_g, voted_g, final_g, result):
        """Print detailed debug info every frame."""
        curl_str = " ".join(f"{k[0].upper()}:{v:.2f}" for k, v in curls.items())
        other_avg = (curls['middle'] + curls['ring'] + curls['pinky']) / 3.0
        print(
            f"[F{self.frame_count:04d}] "
            f"curls=[{curl_str}] "
            f"fist={fist_raw:.3f}->{fist_ema:.3f} "
            f"pinch_d={pinch_raw:.4f} pinch_n={pinch_norm:.3f} "
            f"other_avg={other_avg:.3f} | "
            f"raw={raw_g} vote={voted_g} final={final_g} "
            f"conf={result.get('confidence', 0):.2f}"
        )


# ============================================================
# FastAPI endpoints
# ============================================================
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    gsm = GestureStateMachine()
    palm_trail = deque(maxlen=15)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                img_data = base64.b64decode(msg["data"].split(",")[1])
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({
                        "gesture": "none", "palm": None,
                        "trail": [], "confidence": 0.0, "fist_strength": 0.0
                    })
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = hand_landmarker.detect(mp_image)

                if results.hand_landmarks and len(results.hand_landmarks) > 0:
                    # Filter by handedness confidence
                    if results.handedness and len(results.handedness) > 0:
                        hand_conf = results.handedness[0][0].score
                        if hand_conf < MIN_HAND_CONFIDENCE:
                            print(f"[F{gsm.frame_count+1:04d}] LOW CONFIDENCE hand={hand_conf:.3f}, skipping")
                            await websocket.send_json({
                                "gesture": gsm.state,
                                "palm": None, "trail": [],
                                "confidence": 0.0, "fist_strength": 0.0,
                                "frame_id": gsm.frame_count
                            })
                            continue

                    landmarks = results.hand_landmarks[0]
                    now = time.time()

                    # Update palm trail
                    pcx = (landmarks[0].x + landmarks[5].x + landmarks[17].x) / 3
                    pcy = (landmarks[0].y + landmarks[5].y + landmarks[17].y) / 3
                    pcz = (landmarks[0].z + landmarks[5].z + landmarks[17].z) / 3
                    palm_trail.append({"x": round(pcx,4), "y": round(pcy,4), "z": round(pcz,4), "t": now})
                    trail_points = [p for p in palm_trail if now - p["t"] < 0.5]

                    # Run state machine
                    gesture_data = gsm.update(landmarks)
                    gesture_data["trail"] = trail_points

                    # Send landmarks for hand overlay
                    gesture_data["landmarks"] = [
                        {"x": round(lm.x,4), "y": round(lm.y,4), "z": round(lm.z,4)}
                        for lm in landmarks
                    ]

                    await websocket.send_json(gesture_data)
                else:
                    # No hand detected - reset state machine smoothly
                    gsm.pos_history.clear()
                    gsm.history.append("none")
                    if len(gsm.history) >= 3:
                        # Only transition to none after majority says so
                        recent = list(gsm.history)[-3:]
                        if recent.count("none") >= 2:
                            gsm.state = "none"
                    gsm.ema_fist = None
                    gsm.ema_pinch_dist = None
                    gsm.ema_rot_x = None
                    gsm.ema_rot_y = None
                    gsm.frame_count += 1
                    await websocket.send_json({
                        "gesture": gsm.state,
                        "palm": None, "trail": [],
                        "confidence": 0.0, "fist_strength": 0.0,
                        "frame_id": gsm.frame_count
                    })

    except WebSocketDisconnect:
        print("[WS] Client disconnected")
    except Exception as e:
        print(f"[WS] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
