import asyncio
import base64
import json
import math
import os
import socket
import sys
import time
import traceback
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# Logging
# ============================================================
_start_time = time.time()

def log(tag, msg):
    elapsed = time.time() - _start_time
    print(f"[{elapsed:8.2f}s][{tag}] {msg}", flush=True)


# ============================================================
# MediaPipe initialization with retry and model discovery
# ============================================================
def find_model_file():
    """Search for hand_landmarker.task in common locations."""
    candidates = [
        Path("hand_landmarker.task"),
        Path(__file__).parent / "hand_landmarker.task",
        Path(__file__).parent / "models" / "hand_landmarker.task",
        Path.home() / "hand_landmarker.task",
    ]
    for p in candidates:
        if p.exists():
            log("MODEL", f"Found model at: {p.resolve()}")
            return str(p.resolve())
    return None


def init_mediapipe(max_retries=3):
    """Initialize MediaPipe HandLandmarker with retries."""
    import mediapipe as mp_lib
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision

    model_path = find_model_file()
    if model_path is None:
        log("ERROR", "=" * 60)
        log("ERROR", "hand_landmarker.task NOT FOUND!")
        log("ERROR", "Download from:")
        log("ERROR", "  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
        log("ERROR", f"Place it in: {Path(__file__).parent.resolve()}")
        log("ERROR", "=" * 60)
        sys.exit(1)

    for attempt in range(1, max_retries + 1):
        try:
            log("INIT", f"MediaPipe init attempt {attempt}/{max_retries} ...")
            base_options = mp_python.BaseOptions(model_asset_path=model_path)
            options = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.IMAGE,
                num_hands=2,
                min_hand_detection_confidence=0.40,
                min_hand_presence_confidence=0.40,
                min_tracking_confidence=0.35,
            )
            landmarker = vision.HandLandmarker.create_from_options(options)
            log("INIT", f"MediaPipe HandLandmarker ready (num_hands=2, model={model_path})")
            return landmarker, mp_lib
        except Exception as e:
            log("ERROR", f"MediaPipe init failed (attempt {attempt}): {e}")
            if attempt < max_retries:
                time.sleep(1)
            else:
                log("ERROR", "All retries exhausted. Cannot initialize MediaPipe.")
                traceback.print_exc()
                sys.exit(1)


hand_landmarker, mp = init_mediapipe()

# ============================================================
# FastAPI setup
# ============================================================
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI()

# ============================================================
# Thresholds
# ============================================================
FIST_CURL_THRESHOLD = 0.30
PINCH_RATIO = 0.50
WAVE_FRAMES = 5
WAVE_SPEED_THRESHOLD = 0.25  # 基于归一化掌心速度
WAVE_COOLDOWN = 0.8
EMA_ALPHA = 0.30  # 稍微平滑一些，减少远距离抖动
MIN_HAND_CONFIDENCE = 0.40  # 降低：允许远距离手部检测

POINTING_INDEX_MAX_CURL = 0.28
POINTING_OTHER_MIN_CURL = 0.32

PROXIMITY_HOLD_TIME = 0.5

# 数字识别阈值 - 使用指尖高于对应MCP关节来判断伸展
DIGIT_EXTENDED_CURL_THRESHOLD = 0.28  # curl < 此值 = 手指伸展
DIGIT_THUMB_DISTANCE_RATIO = 0.5  # 拇指尖到食指MCP的距离/手掌宽度

# ============================================================
# Geometry helpers
# ============================================================

def dist3(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2)


def dist3_xy(p1, p2):
    return math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2)


def angle_at(a, b, c):
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
FINGERS = {
    'thumb':  [1, 2, 3, 4],
    'index':  [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring':   [13, 14, 15, 16],
    'pinky':  [17, 18, 19, 20],
}

FIST_WEIGHTS = {'thumb': 0.12, 'index': 0.24, 'middle': 0.24, 'ring': 0.20, 'pinky': 0.20}


def finger_curl(landmarks, indices):
    mcp, pip, dip, tip = [landmarks[i] for i in indices]
    ang_pip = angle_at(mcp, pip, dip)
    ang_dip = angle_at(pip, dip, tip)
    c_pip = 1.0 - ang_pip / math.pi
    c_dip = 1.0 - ang_dip / math.pi
    return c_pip * 0.55 + c_dip * 0.45


def compute_all_curls(landmarks):
    return {name: finger_curl(landmarks, idx) for name, idx in FINGERS.items()}


def compute_fist_strength(curls):
    return sum(curls[f] * FIST_WEIGHTS[f] for f in FIST_WEIGHTS)


def palm_size(landmarks):
    return dist3(landmarks[0], landmarks[9])


def palm_center(landmarks):
    cx = (landmarks[0].x + landmarks[5].x + landmarks[9].x + landmarks[13].x + landmarks[17].x) / 5
    cy = (landmarks[0].y + landmarks[5].y + landmarks[9].y + landmarks[13].y + landmarks[17].y) / 5
    cz = (landmarks[0].z + landmarks[5].z + landmarks[9].z + landmarks[13].z + landmarks[17].z) / 5
    return cx, cy, cz


# ============================================================
# Digit Recognition（改进版：指尖计数 + 相邻手指间距辅助）
# ============================================================
def is_finger_extended(landmarks, finger_name, curls):
    """判断手指是否伸展，综合 curl 值 + 指尖相对于MCP的y坐标。"""
    indices = FINGERS[finger_name]
    tip = landmarks[indices[3]]
    mcp = landmarks[indices[0]]

    # curl 值判定
    curl_extended = curls[finger_name] < DIGIT_EXTENDED_CURL_THRESHOLD

    # 指尖应在MCP上方（y坐标更小 = 更上方，归一化坐标系）
    tip_above_mcp = tip.y < mcp.y + 0.02

    return curl_extended and tip_above_mcp


def is_thumb_extended(landmarks, p_size):
    """拇指判定：使用拇指尖到手掌的横向距离。"""
    thumb_tip = landmarks[4]
    thumb_mcp = landmarks[2]
    index_mcp = landmarks[5]
    wrist = landmarks[0]

    # 拇指尖到食指MCP的水平距离（相对手掌大小）
    dx = abs(thumb_tip.x - index_mcp.x)
    ratio = dx / p_size if p_size > 0.01 else 0

    # 拇指尖应远离手掌中心
    return ratio > DIGIT_THUMB_DISTANCE_RATIO


def recognize_digit(curls, landmarks):
    """
    精确数字识别：
    - 使用 curl + 指尖位置双重判定
    - 区分2和3：2=食指+中指，3=食指+中指+无名指
    - 返回 0-5 的整数
    """
    p_size = palm_size(landmarks)

    # 判定每根手指
    thumb_ext = is_thumb_extended(landmarks, p_size)
    index_ext = is_finger_extended(landmarks, 'index', curls)
    middle_ext = is_finger_extended(landmarks, 'middle', curls)
    ring_ext = is_finger_extended(landmarks, 'ring', curls)
    pinky_ext = is_finger_extended(landmarks, 'pinky', curls)

    fingers_state = {
        'thumb': thumb_ext,
        'index': index_ext,
        'middle': middle_ext,
        'ring': ring_ext,
        'pinky': pinky_ext,
    }

    # 计算伸展手指数量
    count = sum(1 for v in fingers_state.values() if v)

    return count, fingers_state


# ============================================================
# Gesture State Machine (per hand) - 增强版
# ============================================================
class GestureStateMachine:
    def __init__(self, hand_label="unknown"):
        self.hand_label = hand_label
        self.state = "none"
        self.history = deque(maxlen=9)  # 增加到9帧，投票窗口更大
        self.wave_last_time = 0.0
        self.frame_count = 0
        self.pos_history = deque(maxlen=25)  # 掌心轨迹增加
        self.ema_fist = None
        self.ema_pinch_dist = None
        self.ema_rot_x = None
        self.ema_rot_y = None
        self.transition_cooldown = 0
        self._last_log_time = 0
        # 新增：独立的握拳/捏合帧计数器（多帧确认）
        self.fist_confirm_count = 0
        self.pinch_confirm_count = 0
        self.wave_confirm_count = 0
        # 新增：EMA平滑后的掌心速度
        self.ema_palm_speed = None
        # 新增：上一帧掌心大小用于归一化
        self.last_palm_size = 0.1

    def ema(self, prev, new, alpha=EMA_ALPHA):
        if prev is None:
            return new
        return prev * (1.0 - alpha) + new * alpha

    def update(self, landmarks):
        self.frame_count += 1
        now = time.time()

        wrist = landmarks[0]
        pcx, pcy, pcz = palm_center(landmarks)
        self.pos_history.append((pcx, pcy, now))

        curls = compute_all_curls(landmarks)
        fist_strength_raw = compute_fist_strength(curls)
        p_size = palm_size(landmarks)
        self.last_palm_size = p_size if p_size > 0.01 else self.last_palm_size

        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        pinch_dist_raw = dist3(thumb_tip, index_tip)
        pinch_norm = pinch_dist_raw / p_size if p_size > 0.01 else 1.0

        self.ema_fist = self.ema(self.ema_fist, fist_strength_raw)
        self.ema_pinch_dist = self.ema(self.ema_pinch_dist, pinch_norm)
        fist_strength = self.ema_fist
        pinch_normalized = self.ema_pinch_dist

        # 计算归一化掌心速度（相对于手掌大小）
        palm_speed_norm = self._compute_palm_speed_normalized()

        raw_gesture = self._classify_frame(
            curls, fist_strength, pinch_normalized, p_size, now, landmarks, palm_speed_norm
        )

        self.history.append(raw_gesture)
        voted_gesture = self._majority_vote()
        final_gesture = self._apply_state_machine(voted_gesture, now)

        digit, fingers_state = recognize_digit(curls, landmarks)

        result = self._build_result(
            final_gesture, landmarks, curls, fist_strength,
            pinch_dist_raw, pinch_normalized, p_size, now, digit, fingers_state,
            palm_speed_norm
        )

        # 调试日志：每0.4s输出一次
        if now - self._last_log_time > 0.4:
            self._last_log_time = now
            curl_str = " ".join(f"{k[0].upper()}:{v:.2f}" for k, v in curls.items())
            fingers_str = "".join("1" if fingers_state[f] else "0"
                                  for f in ['thumb','index','middle','ring','pinky'])
            log(f"GSM-{self.hand_label}",
                f"F{self.frame_count:04d} final={final_gesture} "
                f"fist={fist_strength:.3f} pinch_n={pinch_normalized:.3f} "
                f"speed={palm_speed_norm:.3f} digit={digit}({fingers_str}) "
                f"curls=[{curl_str}] raw={raw_gesture} vote={voted_gesture} "
                f"fist_cf={self.fist_confirm_count} pinch_cf={self.pinch_confirm_count}")

        return result

    def _compute_palm_speed_normalized(self):
        """计算掌心移动速度：使用最近5帧的平均速度，归一化到手掌大小。"""
        if len(self.pos_history) < 5:
            return 0.0

        # 取最近5帧计算平均速度
        recent = list(self.pos_history)[-5:]
        total_time = recent[-1][2] - recent[0][2]
        if total_time < 0.02:
            return 0.0

        # 逐帧位移求和
        cum_disp = 0.0
        for i in range(1, len(recent)):
            dx = recent[i][0] - recent[i-1][0]
            dy = recent[i][1] - recent[i-1][1]
            cum_disp += math.sqrt(dx*dx + dy*dy)

        # 平均速度 = 累积位移 / 总时间
        avg_speed = cum_disp / total_time

        # 归一化到手掌大小
        speed_norm = avg_speed / self.last_palm_size if self.last_palm_size > 0.01 else avg_speed

        # EMA平滑（避免单帧跳变）
        self.ema_palm_speed = self.ema(self.ema_palm_speed, speed_norm, 0.35)
        return self.ema_palm_speed

    def _classify_frame(self, curls, fist_strength, pinch_norm, p_size, now, landmarks, palm_speed_norm):
        # === 握拳检测（增强：多帧确认） ===
        is_fist = False
        if fist_strength >= FIST_CURL_THRESHOLD:
            curled_count = sum(1 for v in curls.values() if v > 0.18)
            if curled_count >= 4:
                self.fist_confirm_count = min(self.fist_confirm_count + 1, 10)
                if self.fist_confirm_count >= 2:  # 需连续2帧确认
                    is_fist = True
            else:
                self.fist_confirm_count = max(0, self.fist_confirm_count - 1)
        else:
            self.fist_confirm_count = max(0, self.fist_confirm_count - 1)

        # === 食指指向检测 ===
        is_pointing = False
        if curls['index'] < POINTING_INDEX_MAX_CURL:
            other_curled = (curls['middle'] > POINTING_OTHER_MIN_CURL and
                          curls['ring'] > POINTING_OTHER_MIN_CURL and
                          curls['pinky'] > POINTING_OTHER_MIN_CURL)
            if other_curled and not is_fist:
                is_pointing = True

        # === 捏合检测（增强：多帧确认） ===
        is_pinch = False
        if pinch_norm < PINCH_RATIO:
            other_avg_curl = (curls['middle'] + curls['ring'] + curls['pinky']) / 3.0
            if other_avg_curl < 0.38:
                self.pinch_confirm_count = min(self.pinch_confirm_count + 1, 10)
                if self.pinch_confirm_count >= 2:  # 需连续2帧确认
                    is_pinch = True
            else:
                self.pinch_confirm_count = max(0, self.pinch_confirm_count - 1)
        else:
            self.pinch_confirm_count = max(0, self.pinch_confirm_count - 1)

        # === 消歧 ===
        if is_fist and is_pinch:
            other_avg = (curls['middle'] + curls['ring'] + curls['pinky']) / 3.0
            if other_avg > 0.25:
                is_pinch = False
                self.pinch_confirm_count = 0
            else:
                is_fist = False
                self.fist_confirm_count = 0

        if is_fist and is_pointing:
            is_pointing = False

        if is_pointing:
            return "pointing"
        if is_pinch:
            return "pinch"
        if is_fist:
            return "fist"

        # === 挥手检测（最近5帧平均速度 + 振荡确认 + 冷却0.8s） ===
        if len(self.pos_history) >= WAVE_FRAMES and fist_strength < 0.28:
            recent = list(self.pos_history)[-WAVE_FRAMES:]
            total_time = recent[-1][2] - recent[0][2]
            if total_time > 0.05:
                # x方向逐帧位移累积
                x_cum_disp = 0.0
                for i in range(1, len(recent)):
                    x_cum_disp += abs(recent[i][0] - recent[i-1][0])
                # 5帧平均x速度（归一化到手掌大小）
                x_avg_speed = (x_cum_disp / total_time) / self.last_palm_size if self.last_palm_size > 0.01 else 0

                # x方向振荡检测（要求至少2次方向变化，确保是左右摆动）
                xs = [p[0] for p in recent]
                dir_changes = 0
                for i in range(2, len(xs)):
                    if (xs[i]-xs[i-1]) * (xs[i-1]-xs[i-2]) < -0.0001:
                        dir_changes += 1

                # 判定条件：5帧平均速度+全局EMA速度双重确认 + 振荡
                speed_ok = palm_speed_norm > WAVE_SPEED_THRESHOLD and x_avg_speed > 0.2
                oscillation_ok = dir_changes >= 2

                if speed_ok and oscillation_ok:
                    self.wave_confirm_count += 1
                    if self.wave_confirm_count >= 3 and now - self.wave_last_time > WAVE_COOLDOWN:
                        self.wave_confirm_count = 0
                        return "wave"
                else:
                    self.wave_confirm_count = max(0, self.wave_confirm_count - 1)
        else:
            self.wave_confirm_count = max(0, self.wave_confirm_count - 1)

        return "none"

    def _majority_vote(self):
        if len(self.history) < 3:
            return self.history[-1] if self.history else "none"

        # 使用最近7帧进行投票（增大窗口提高稳定性）
        window = list(self.history)[-7:]
        counts = Counter(window)
        winner, count = counts.most_common(1)[0]

        # 需要至少4/7的多数票
        threshold = max(3, len(window) // 2 + 1)
        if count >= threshold:
            return winner

        # 如果无清晰多数，取最近出现的非none
        for g in reversed(window):
            if g != "none":
                return g
        return "none"

    def _apply_state_machine(self, voted, now):
        if voted == self.state:
            self.transition_cooldown = 0
            return self.state

        if self.transition_cooldown > 0:
            self.transition_cooldown -= 1
            return self.state

        if voted == "wave":
            self.wave_last_time = now

        self.state = voted
        self.transition_cooldown = 2
        return voted

    def _build_result(self, gesture, landmarks, curls, fist_strength,
                      pinch_dist_raw, pinch_norm, p_size, now, digit, fingers_state,
                      palm_speed_norm):
        wrist = landmarks[0]
        pcx, pcy, pcz = palm_center(landmarks)

        result = {
            "gesture": gesture,
            "fist_strength": round(fist_strength, 4),
            "pinch_distance": round(pinch_dist_raw, 4),
            "pinch_normalized": round(pinch_norm, 4),
            "confidence": 0.0,
            "palm": {"x": round(pcx, 4), "y": round(pcy, 4), "z": round(pcz, 4)},
            "frame_id": self.frame_count,
            "curls": {k: round(v, 3) for k, v in curls.items()},
            "digit": digit,
            "fingers": {k: v for k, v in fingers_state.items()},
            "hand": self.hand_label,
            "palm_speed": round(palm_speed_norm, 3),
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

        elif gesture == "pointing":
            index_tip = landmarks[8]
            result["pointing_x"] = round(index_tip.x, 4)
            result["pointing_y"] = round(index_tip.y, 4)
            result["confidence"] = 0.85

        elif gesture == "wave":
            result["palm_speed"] = round(palm_speed_norm, 3)
            result["confidence"] = 0.8

        return result

    def reset(self):
        self.pos_history.clear()
        self.history.append("none")
        if len(self.history) >= 3:
            recent = list(self.history)[-3:]
            if recent.count("none") >= 2:
                self.state = "none"
        self.ema_fist = None
        self.ema_pinch_dist = None
        self.ema_rot_x = None
        self.ema_rot_y = None
        self.ema_palm_speed = None
        self.fist_confirm_count = 0
        self.pinch_confirm_count = 0
        self.wave_confirm_count = 0
        self.frame_count += 1


# ============================================================
# Dual-hand coordination
# ============================================================
class DualHandTracker:
    def __init__(self):
        self.left_gsm = GestureStateMachine("Left")
        self.right_gsm = GestureStateMachine("Right")
        self.single_gsm = GestureStateMachine("Single")
        self.proximity_start = None
        self.last_left_palm = None
        self.last_right_palm = None
        self.frame_count = 0
        self._last_status_log = 0

    def process(self, results):
        self.frame_count += 1
        now = time.time()

        if not results.hand_landmarks or len(results.hand_landmarks) == 0:
            self.left_gsm.reset()
            self.right_gsm.reset()
            self.single_gsm.reset()
            self.proximity_start = None

            if now - self._last_status_log > 2.0:
                self._last_status_log = now
                log("TRACK", "No hands detected")

            return {
                "type": "no_hand",
                "left": None,
                "right": None,
                "dual": None,
                "frame_id": self.frame_count,
                "hand_count": 0,
            }

        # Classify hands
        left_landmarks = None
        right_landmarks = None
        left_conf = 0
        right_conf = 0

        for i, hand_lm in enumerate(results.hand_landmarks):
            if i >= len(results.handedness):
                continue
            handedness = results.handedness[i][0]
            label = handedness.category_name
            conf = handedness.score

            if conf < MIN_HAND_CONFIDENCE:
                log("TRACK", f"Hand {i} rejected: conf={conf:.3f} < {MIN_HAND_CONFIDENCE}")
                continue

            if label == "Right":
                if conf > right_conf:
                    right_landmarks = hand_lm
                    right_conf = conf
            else:
                if conf > left_conf:
                    left_landmarks = hand_lm
                    left_conf = conf

        hand_count = (1 if left_landmarks else 0) + (1 if right_landmarks else 0)

        # === SINGLE HAND MODE ===
        # If only one hand detected, use single_gsm for full control (backward compat)
        if hand_count == 1:
            single_lm = left_landmarks if left_landmarks else right_landmarks
            single_conf = left_conf if left_landmarks else right_conf
            single_label = "Left" if left_landmarks else "Right"

            single_data = self.single_gsm.update(single_lm)
            single_data["hand_confidence"] = round(single_conf, 3)
            single_data["landmarks"] = [
                {"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)}
                for lm in single_lm
            ]

            # Also populate the correct side for UI display
            left_data = single_data if left_landmarks else None
            right_data = single_data if right_landmarks else None

            if now - self._last_status_log > 1.0:
                self._last_status_log = now
                log("TRACK", f"Single hand: {single_label} gesture={single_data['gesture']} "
                    f"conf={single_conf:.2f} digit={single_data['digit']}")

            return {
                "type": "single",
                "left": left_data,
                "right": right_data,
                "dual": None,
                "frame_id": self.frame_count,
                "hand_count": 1,
                "single_hand": single_label,
            }

        # === DUAL HAND MODE ===
        left_data = None
        right_data = None

        if left_landmarks:
            left_data = self.left_gsm.update(left_landmarks)
            left_data["hand_confidence"] = round(left_conf, 3)
            pcx, pcy, pcz = palm_center(left_landmarks)
            self.last_left_palm = (pcx, pcy, pcz)
            left_data["landmarks"] = [
                {"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)}
                for lm in left_landmarks
            ]

        if right_landmarks:
            right_data = self.right_gsm.update(right_landmarks)
            right_data["hand_confidence"] = round(right_conf, 3)
            pcx, pcy, pcz = palm_center(right_landmarks)
            self.last_right_palm = (pcx, pcy, pcz)
            right_data["landmarks"] = [
                {"x": round(lm.x, 4), "y": round(lm.y, 4), "z": round(lm.z, 4)}
                for lm in right_landmarks
            ]

        dual_data = self._compute_dual(left_landmarks, right_landmarks, now)

        if now - self._last_status_log > 1.0:
            self._last_status_log = now
            lg = left_data['gesture'] if left_data else 'N/A'
            rg = right_data['gesture'] if right_data else 'N/A'
            pd = dual_data['palm_distance'] if dual_data else 0
            log("TRACK", f"Dual hands: L={lg} R={rg} palm_dist={pd:.3f}")

        return {
            "type": "dual",
            "left": left_data,
            "right": right_data,
            "dual": dual_data,
            "frame_id": self.frame_count,
            "hand_count": 2,
        }

    def _compute_dual(self, left_lm, right_lm, now):
        left_palm = palm_center(left_lm)
        right_palm = palm_center(right_lm)

        dx = left_palm[0] - right_palm[0]
        dy = left_palm[1] - right_palm[1]
        palm_dist = math.sqrt(dx*dx + dy*dy)

        # 动态阈值：基于双手手掌平均宽度（一个手掌宽度内即触发）
        left_psize = palm_size(left_lm)
        right_psize = palm_size(right_lm)
        avg_palm_size = (left_psize + right_psize) / 2.0
        dynamic_threshold = avg_palm_size * 1.0  # 距离小于一个手掌宽度

        reset_triggered = False
        if palm_dist < dynamic_threshold:
            if self.proximity_start is None:
                self.proximity_start = now
                log("DUAL", f"Proximity detected: dist={palm_dist:.3f} < threshold={dynamic_threshold:.3f} (palm_size={avg_palm_size:.3f})")
            elif now - self.proximity_start >= PROXIMITY_HOLD_TIME:
                reset_triggered = True
                self.proximity_start = None
                log("DUAL", "RESET TRIGGERED!")
        else:
            self.proximity_start = None

        left_gesture = self.left_gsm.state
        right_gesture = self.right_gsm.state
        dual_pointing_zoom = False
        pointing_distance = None
        if left_gesture == "pointing" and right_gesture == "pointing":
            dual_pointing_zoom = True
            left_index = left_lm[8]
            right_index = right_lm[8]
            pointing_distance = dist3_xy(left_index, right_index)

        proximity_progress = 0.0
        if self.proximity_start is not None:
            proximity_progress = min(1.0, (now - self.proximity_start) / PROXIMITY_HOLD_TIME)

        return {
            "palm_distance": round(palm_dist, 4),
            "reset_triggered": reset_triggered,
            "proximity_progress": round(proximity_progress, 3),
            "dual_pointing_zoom": dual_pointing_zoom,
            "pointing_distance": round(pointing_distance, 4) if pointing_distance else None,
        }


# ============================================================
# FastAPI endpoints
# ============================================================
static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(static_dir / "index.html"))


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    log("WS", "Client connected")

    tracker = DualHandTracker()
    frame_times = deque(maxlen=30)

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "frame":
                t0 = time.time()

                raw = msg.get("data", "")
                if "," not in raw:
                    continue
                img_data = base64.b64decode(raw.split(",")[1])
                np_arr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

                if frame is None:
                    log("WS", "Failed to decode frame")
                    await websocket.send_json({
                        "type": "no_hand",
                        "left": None, "right": None, "dual": None,
                        "frame_id": tracker.frame_count, "hand_count": 0,
                    })
                    continue

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

                try:
                    results = hand_landmarker.detect(mp_image)
                except Exception as e:
                    log("MP", f"Detection error: {e}")
                    await websocket.send_json({
                        "type": "no_hand",
                        "left": None, "right": None, "dual": None,
                        "frame_id": tracker.frame_count, "hand_count": 0,
                    })
                    continue

                response = tracker.process(results)
                # 添加服务器时间戳，供前端多模态事件对齐使用
                response["server_ts"] = int(time.time() * 1000)

                t1 = time.time()
                frame_times.append(t1 - t0)

                # Log processing FPS every 30 frames
                if len(frame_times) == 30:
                    avg_ms = (sum(frame_times) / len(frame_times)) * 1000
                    fps = 1000.0 / avg_ms if avg_ms > 0 else 0
                    log("PERF", f"avg={avg_ms:.1f}ms/frame  capacity={fps:.0f}fps")

                await websocket.send_json(response)

    except WebSocketDisconnect:
        log("WS", "Client disconnected")
    except Exception as e:
        log("WS", f"Error: {e}")
        traceback.print_exc()


# ============================================================
# Port scanning and startup
# ============================================================
def find_available_port(start=8000, end=8100):
    """Scan for an available port in range."""
    for port in range(start, end):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", port))
                return port
            except OSError:
                continue
    return None


if __name__ == "__main__":
    import uvicorn

    log("STARTUP", "=" * 60)
    log("STARTUP", "Magic Gesture 3D Controller - Dual Hand")
    log("STARTUP", "=" * 60)

    port = find_available_port(8000, 8100)
    if port is None:
        log("ERROR", "No available port found in range 8000-8100!")
        sys.exit(1)

    if port != 8000:
        log("STARTUP", f"Port 8000 is occupied, using port {port} instead")

    log("STARTUP", f"Server starting on http://0.0.0.0:{port}")
    log("STARTUP", f"Open in browser: http://localhost:{port}")
    log("STARTUP", f"Working directory: {Path.cwd()}")
    log("STARTUP", f"Static files: {static_dir.resolve()}")
    log("STARTUP", "=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
