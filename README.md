# Magic Gesture 3D Controller - Multimodal Edition

实时手势识别控制交互式3D模型系统，支持**双手协同交互**、**连续手势序列识别（DTW）**和**语音-手势多模态融合控制**。

Real-time hand gesture recognition system that controls interactive 3D visuals using webcam input. Supports **dual-hand collaborative interaction**, **continuous gesture sequence recognition (DTW)**, and **voice-gesture multimodal fusion control**.

## 功能特性 / Features

### 基础手势控制 / Basic Gesture Control

| 手势 Gesture | 手 Hand | 动作 Action |
|---------|------|--------|
| 握拳 Fist (Left) | 左手 Left | 水平旋转3D模型 / Horizontal rotation |
| 握拳 Fist (Right) | 右手 Right | 垂直翻转3D模型 / Vertical tilt |
| 捏合 Pinch (Right) | 右手 Right | 缩放控制+粒子波动 / Scale + particle wave |
| 挥手 Wave (Left) | 左手 Left | 颜色切换+脉冲环 / Color change + pulse ring |
| 挥手 Wave (Right) | 右手 Right | 粒子爆炸效果 / Particle explosion |
| 指向 Pointing | 双手 Both | 双手缩放 / Dual-hand zoom |

### 双手协同 / Dual-Hand Coordination

| 手势 Gesture | 动作 Action |
|---------|--------|
| 双手指向 Both pointing | 双指距离缩放 / Pinch-to-zoom |
| 双掌靠近1秒 Palms close 1s | 重置模型 / Reset model |

### 🆕 连续手势序列识别 / Gesture Sequence Recognition

基于动态时间规整（DTW）算法的手势序列分类器，支持复合手势命令：

DTW-based gesture sequence classifier supporting compound gesture commands:

| 序列 Sequence | 命令 Command |
|---------|--------|
| 握拳 → 挥手 / Fist → Wave | 切换材质 / Switch material |
| 捏合 → 握拳 / Pinch → Fist | 重置视图 / Reset view |
| 挥手×3 / Wave × 3 | 显示帮助 / Show help |
| 握拳 → 捏合 → 挥手 / Fist → Pinch → Wave | 粒子爆发 / Particle burst |
| 指向 → 握拳 / Point → Fist | 切换线框 / Toggle wireframe |
| 捏合 → 指向 / Pinch → Point | 放大模型 / Zoom in |
| 指向 → 捏合 / Point → Pinch | 缩小模型 / Zoom out |
| 握拳 → 挥手 → 握拳 / Fist → Wave → Fist | 旋转爆炸 / Explode |

**自定义模板 / Custom Templates**：用户可录制最多5个自定义手势序列模板，阈值可调。
Users can record up to 5 custom gesture sequence templates with adjustable threshold.

### 🆕 语音命令 / Voice Commands

基于Web Speech API的中文语音识别，支持以下命令词：

Chinese voice commands via Web Speech API:

| 命令词 Command | 意图 Intent |
|---------|--------|
| "切换材质" / "换材质" | 切换模型材质 / Switch material |
| "重置" / "复位" | 重置视图 / Reset view |
| "帮助" / "说明" | 显示帮助面板 / Show help |
| "放大" / "大一点" | 放大模型 / Zoom in |
| "缩小" / "小一点" | 缩小模型 / Zoom out |
| "左转" / "向左旋转" | 向左旋转 / Rotate left |
| "右转" / "向右旋转" | 向右旋转 / Rotate right |
| "换色" / "换颜色" | 切换颜色 / Change color |
| "爆炸" / "粒子爆炸" | 粒子爆炸 / Explode |
| "线框" / "切换线框" | 切换线框显示 / Toggle wireframe |

### 🆕 多模态融合 / Multimodal Fusion

手势序列和语音命令通过融合引擎协同工作：

Gesture sequences and voice commands work together through the fusion engine:

- **同意图融合**：手势和语音同时触发相同意图时，置信度加权提升并立即执行
  When both modalities trigger the same intent, confidence is boosted and executed immediately
- **冲突解决**：支持三种优先级策略
  Three priority strategies for conflicts:
  - 最近优先 / Recent wins（默认）
  - 手势优先 / Gesture first
  - 语音优先 / Voice first
- **独立触发**：单模态高置信度事件可独立触发动作
  Single modality events with high confidence trigger independently
- **时间对齐**：500ms时间窗口内对齐事件，响应延迟<200ms
  500ms alignment window, response latency <200ms

### 调试面板 / Debug Panel

实时显示系统内部状态：

- 手势序列流可视化（手势段+持续时间）
- 语音识别文本（中间结果+最终结果）
- 融合决策日志（来源、意图、置信度、策略）
- DTW阈值滑块调整
- 融合优先级模式切换
- 自定义模板录制和管理

## 技术栈 / Tech Stack

- **后端 Backend**: Python, FastAPI, WebSocket, MediaPipe Hand Landmarker
- **前端 Frontend**: Three.js, Canvas 2D overlay, Web Audio API, Web Speech API
- **通信 Communication**: WebSocket双向实时帧/手势数据
- **算法 Algorithms**: DTW动态时间规整（手写实现）、多模态决策融合（手写实现）

## 系统要求 / Requirements

- Python 3.10+
- 摄像头 Webcam
- 麦克风 Microphone（语音功能需要）
- 现代浏览器 Modern browser (Chrome/Edge recommended, 语音识别需要Chrome)

## 安装与运行 / Setup

### 1. 安装依赖 / Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. 下载MediaPipe模型 / Download Model

`hand_landmarker.task` 必须在项目根目录，下载地址:

```
https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
```

### 3. 运行 / Run

```bash
python server.py
```

或使用脚本 / Or use scripts:
- **Windows**: `start.bat`
- **Linux/macOS**: `./start.sh`

打开控制台显示的URL（默认 `http://localhost:8000`）。

## 架构 / Architecture

```
Browser (webcam + mic)
    |
    ├── 摄像头帧 --[WebSocket: JPEG]--> FastAPI Server
    |                                      |
    |                                MediaPipe检测 (2手, 关键点)
    |                                      |
    |                                DualHandTracker
    |                               /              \
    |                     Left GSM              Right GSM
    |                               \              /
    |                                Dual协调 (接近, 缩放)
    |                                      |
    |   <--[WebSocket: JSON手势数据]--  Response + server_ts
    |
    ├── 手势序列识别器 (DTW)
    |       └── 连续匹配预定义+自定义模板
    |
    ├── 语音识别器 (Web Speech API)
    |       └── 命令词→意图映射
    |
    ├── 多模态融合引擎
    |       ├── 时间窗口对齐 (500ms)
    |       ├── 同意图融合 (置信度提升)
    |       ├── 冲突解决 (优先级策略)
    |       └── 独立触发 (高置信单模态)
    |
    ├── 意图执行器 → Three.js 3D模型控制
    |
    └── 调试面板 (手势流 + 语音 + 融合日志)
```

## 模块结构 / Module Structure

```
static/
├── index.html              主页面 + 集成逻辑
├── three.min.js            Three.js r152
└── js/
    ├── gesture-sequence.js  DTW手势序列识别器
    ├── voice-recognition.js 语音命令识别模块
    ├── multimodal-fusion.js 多模态融合决策引擎
    └── debug-panel.js       调试面板UI模块
```

## 配置 / Configuration

### 服务器参数 (server.py)

| 参数 Parameter | 默认值 Default | 说明 Description |
|-----------|---------|-------------|
| `FIST_CURL_THRESHOLD` | 0.30 | 握拳检测最低curl值 |
| `PINCH_RATIO` | 0.50 | 捏合距离阈值（相对手掌） |
| `WAVE_SPEED_THRESHOLD` | 0.25 | 挥手检测最低速度 |
| `PROXIMITY_THRESHOLD` | 0.12 | 重置手势掌心距离阈值 |
| `EMA_ALPHA` | 0.30 | EMA平滑系数 |

### 前端参数 (可通过调试面板调整)

| 参数 Parameter | 默认值 Default | 说明 Description |
|-----------|---------|-------------|
| DTW匹配阈值 | 0.55 | 序列匹配灵敏度(0.2-0.9) |
| 融合时间窗口 | 500ms | 多模态事件对齐时间范围 |
| 融合优先级 | recent | 冲突解决策略 |
| 执行冷却 | 800ms | 同一动作重复触发间隔 |

## 性能 / Performance

- Three.js渲染: ≥30fps（预分配粒子缓冲区，DTW不在渲染循环中）
- 手势识别延迟: ~55ms/帧（摄像头发送间隔）
- DTW匹配检测: 每4帧一次（~220ms），8×25矩阵计算<0.1ms
- 多模态融合延迟: <200ms（高置信度直接执行，低置信度180ms延迟等待）
- 语音识别: Web Speech API异步，不阻塞渲染线程

## 故障排除 / Troubleshooting

- **手部未检测到**: 确保光照良好，手在摄像头画面内清晰可见
- **左右混淆**: 保持双手分开以获得最佳识别效果
- **语音无响应**: 确认使用Chrome浏览器，已授予麦克风权限
- **手势序列不触发**: 降低DTW阈值（调试面板中调整），确保动作连贯
- **帧率低**: 减少浏览器标签页数量，关闭调试面板可略微提升性能

## 许可证 / License

MIT
