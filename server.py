import asyncio
import base64
import json
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

# Initialize HandLandmarker
base_options = mp_python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.6,
    min_tracking_confidence=0.5,
)
hand_landmarker = vision.HandLandmarker.create_from_options(options)


PINCH_THRESHOLD = 0.055
FIST_TIP_PALM_RATIO = 0.95


def dist(p1, p2):
    return ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5


def vec(a, b):
    return (b.x - a.x, b.y - a.y, b.z - a.z)


def dot(u, v):
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]


def mag(v):
    return (v[0] ** 2 + v[1] ** 2 + v[2] ** 2) ** 0.5


def is_finger_curled(tip, mcp, palm_center):
    """A finger is curled if:
    1. The fingertip is closer to the palm center than the MCP is (tip folded inward), OR
    2. The tip-to-MCP distance is small relative to palm size (tip right next to its base).
    Both conditions indicate the finger is folded toward the palm."""
    tip_to_palm = dist(tip, palm_center)
    mcp_to_palm = dist(mcp, palm_center)
    tip_to_mcp = dist(tip, mcp)

    if mcp_to_palm < 0.001:
        return False

    # Condition 1: tip is closer to palm center than MCP
    if tip_to_palm < mcp_to_palm * FIST_TIP_PALM_RATIO:
        return True

    # Condition 2: tip is very close to its own MCP (compact fold)
    if tip_to_mcp < mcp_to_palm * 0.6:
        return True

    return False


class PalmCenter:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


def recognize_gesture_raw(landmarks, prev_positions):
    """Classify gesture for a single frame (no smoothing)."""
    wrist = landmarks[0]
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    index_tip = landmarks[8]
    index_mcp = landmarks[5]
    middle_tip = landmarks[12]
    middle_mcp = landmarks[9]
    ring_tip = landmarks[16]
    ring_mcp = landmarks[13]
    pinky_tip = landmarks[20]
    pinky_mcp = landmarks[17]

    palm_center = PalmCenter(
        (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 5,
        (landmarks[0].y + landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 5,
        (landmarks[0].z + landmarks[5].z + landmarks[9].z + landmarks[13].z + landmarks[17].z) / 5,
    )

    # --- Pinch: thumb tip to index tip distance below threshold ---
    # Guard: at least one of middle/ring/pinky must be extended (not a fist)
    pinch_distance = dist(thumb_tip, index_tip)
    if pinch_distance < PINCH_THRESHOLD:
        others_curled = sum([
            is_finger_curled(middle_tip, middle_mcp, palm_center),
            is_finger_curled(ring_tip, ring_mcp, palm_center),
            is_finger_curled(pinky_tip, pinky_mcp, palm_center),
        ])
        if others_curled <= 1:
            scale = max(0.3, min(3.0, 1.0 / (pinch_distance * 15 + 0.1)))
            return {
                "gesture": "pinch",
                "scale": round(scale, 3),
                "palm": {"x": wrist.x, "y": wrist.y, "z": wrist.z},
            }

    # --- Fist: all four fingers curled toward palm ---
    fingers_curled = [
        is_finger_curled(index_tip, index_mcp, palm_center),
        is_finger_curled(middle_tip, middle_mcp, palm_center),
        is_finger_curled(ring_tip, ring_mcp, palm_center),
        is_finger_curled(pinky_tip, pinky_mcp, palm_center),
    ]
    thumb_curled = dist(thumb_tip, thumb_mcp) < dist(palm_center, thumb_mcp) * 0.9

    if sum(fingers_curled) >= 4 and thumb_curled:
        rotation_x = (wrist.x - 0.5) * 3.14159 * 2
        rotation_y = (wrist.y - 0.5) * 3.14159 * 2
        return {
            "gesture": "fist",
            "rotation_x": round(rotation_x, 4),
            "rotation_y": round(rotation_y, 4),
            "palm": {"x": wrist.x, "y": wrist.y, "z": wrist.z},
        }

    # --- Wave: open hand with lateral oscillation ---
    if len(prev_positions) >= 6:
        x_positions = [p[0] for p in prev_positions]
        x_range = max(x_positions) - min(x_positions)
        if x_range > 0.13:
            direction_changes = 0
            for i in range(2, len(x_positions)):
                d1 = x_positions[i - 1] - x_positions[i - 2]
                d2 = x_positions[i] - x_positions[i - 1]
                if d1 * d2 < -0.0001:
                    direction_changes += 1
            all_extended = sum(fingers_curled) <= 1
            if direction_changes >= 2 and all_extended:
                return {
                    "gesture": "wave",
                    "palm": {"x": wrist.x, "y": wrist.y, "z": wrist.z},
                }

    return {
        "gesture": "none",
        "palm": {"x": wrist.x, "y": wrist.y, "z": wrist.z},
    }


def smooth_gesture(raw_result, gesture_history):
    """Apply 3-frame majority voting to reduce jitter.
    For continuous parameters (rotation, scale), use the current frame's values
    when that gesture wins the vote."""
    gesture_history.append(raw_result["gesture"])

    if len(gesture_history) < 3:
        return raw_result

    last3 = [gesture_history[-1], gesture_history[-2], gesture_history[-3]]
    vote = Counter(last3).most_common(1)[0]
    winner = vote[0]

    if winner == raw_result["gesture"]:
        return raw_result

    result = dict(raw_result)
    result["gesture"] = winner
    return result


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    prev_positions = deque(maxlen=12)
    palm_trail = deque(maxlen=15)
    gesture_history = deque(maxlen=5)
    raw_results_history = deque(maxlen=5)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                img_data = base64.b64decode(msg["data"].split(",")[1])
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    await websocket.send_json({"gesture": "none", "palm": None, "trail": []})
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                results = hand_landmarker.detect(mp_image)

                if results.hand_landmarks and len(results.hand_landmarks) > 0:
                    landmarks = results.hand_landmarks[0]

                    wrist = landmarks[0]
                    prev_positions.append((wrist.x, wrist.y, time.time()))

                    palm_center_x = (landmarks[0].x + landmarks[5].x + landmarks[17].x) / 3
                    palm_center_y = (landmarks[0].y + landmarks[5].y + landmarks[17].y) / 3
                    palm_center_z = (landmarks[0].z + landmarks[5].z + landmarks[17].z) / 3
                    palm_trail.append({
                        "x": round(palm_center_x, 4),
                        "y": round(palm_center_y, 4),
                        "z": round(palm_center_z, 4),
                        "t": time.time(),
                    })

                    now = time.time()
                    trail_points = [p for p in palm_trail if now - p["t"] < 0.5]

                    raw_result = recognize_gesture_raw(landmarks, prev_positions)
                    raw_results_history.append(raw_result)
                    gesture_data = smooth_gesture(raw_result, gesture_history)
                    gesture_data["trail"] = trail_points

                    all_points = []
                    for lm in landmarks:
                        all_points.append({
                            "x": round(lm.x, 4),
                            "y": round(lm.y, 4),
                            "z": round(lm.z, 4),
                        })
                    gesture_data["landmarks"] = all_points

                    await websocket.send_json(gesture_data)
                else:
                    prev_positions.clear()
                    gesture_history.append("none")
                    await websocket.send_json({"gesture": "none", "palm": None, "trail": []})

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WebSocket error: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
