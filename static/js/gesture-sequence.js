/**
 * 手势序列识别模块 - 基于动态时间规整(DTW)的连续手势序列匹配
 * 纯手写算法实现，无外部ML库依赖
 */
(function() {
    'use strict';

    // 预定义手势序列模板
    var PREDEFINED_SEQUENCES = [
        { name: "切换材质", sequence: [{g:"fist", d:500}, {g:"wave", d:300}], action: "switch_material" },
        { name: "重置视图", sequence: [{g:"pinch", d:400}, {g:"fist", d:400}], action: "reset_view" },
        { name: "显示帮助", sequence: [{g:"wave", d:200}, {g:"none", d:150}, {g:"wave", d:200}, {g:"none", d:150}, {g:"wave", d:200}], action: "show_help" },
        { name: "粒子爆发", sequence: [{g:"fist", d:300}, {g:"pinch", d:300}, {g:"wave", d:300}], action: "particle_burst" },
        { name: "切换线框", sequence: [{g:"pointing", d:500}, {g:"fist", d:500}], action: "toggle_wireframe" },
        { name: "放大模型", sequence: [{g:"pinch", d:400}, {g:"pointing", d:400}], action: "zoom_in" },
        { name: "缩小模型", sequence: [{g:"pointing", d:400}, {g:"pinch", d:400}], action: "zoom_out" },
        { name: "旋转爆炸", sequence: [{g:"fist", d:400}, {g:"wave", d:200}, {g:"fist", d:300}], action: "explode" }
    ];

    // 最大模板长度（用于预分配DTW矩阵）
    var MAX_TEMPLATE_LEN = 10;
    var MAX_BUFFER_SEGMENTS = 25;
    var MAX_CUSTOM_TEMPLATES = 5;

    /**
     * 手势序列识别器
     * @param {Object} options - 配置项
     * @param {number} options.matchThreshold - 匹配阈值(0-1)，越小越严格
     * @param {number} options.bufferSize - 原始手势缓冲区大小
     * @param {number} options.checkInterval - 每N帧检查一次匹配
     */
    function GestureSequenceRecognizer(options) {
        options = options || {};
        this._threshold = options.matchThreshold || 0.55;
        this._bufferSize = options.bufferSize || 80;
        this._checkInterval = options.checkInterval || 4;

        // 原始手势流缓冲区 [{gesture, timestamp}]
        this._rawBuffer = [];
        // 分段后的序列 [{g, d}]
        this._segments = [];
        // 帧计数器
        this._frameCount = 0;
        // 上次匹配时间（防止重复触发）
        this._lastMatchTime = 0;
        this._lastMatchAction = '';
        this._matchCooldown = 1000; // 同一动作冷却1秒

        // 模板库
        this._templates = [];
        this._customTemplates = [];

        // 录制状态
        this._isRecording = false;
        this._recordBuffer = [];
        this._recordStartTime = 0;

        // 回调
        this._callbacks = [];

        // 预分配DTW矩阵（避免GC压力）
        var matSize = (MAX_BUFFER_SEGMENTS + 1) * (MAX_TEMPLATE_LEN + 1);
        this._dtwMatrix = new Float64Array(matSize);
        this._matWidth = MAX_TEMPLATE_LEN + 1;

        // 注册预定义模板
        this._initPredefined();
        // 加载自定义模板
        this._loadCustomTemplates();
    }

    GestureSequenceRecognizer.prototype = {
        /**
         * 输入新的手势帧数据（每帧调用）
         * @param {string} gesture - 手势名称
         * @param {number} timestamp - 时间戳(ms)
         */
        feedGesture: function(gesture, timestamp) {
            this._rawBuffer.push({ gesture: gesture, timestamp: timestamp });

            // 限制缓冲区大小
            if (this._rawBuffer.length > this._bufferSize) {
                this._rawBuffer.shift();
            }

            // 录制模式
            if (this._isRecording) {
                this._recordBuffer.push({ gesture: gesture, timestamp: timestamp });
            }

            // 更新分段
            this._updateSegments();

            // 定期检查匹配
            this._frameCount++;
            if (this._frameCount % this._checkInterval === 0) {
                this._checkMatches(timestamp);
            }
        },

        /**
         * 注册模板
         */
        registerTemplate: function(name, sequence, action) {
            this._templates.push({ name: name, sequence: sequence, action: action, custom: false });
        },

        /**
         * 开始录制自定义手势序列
         */
        startRecording: function() {
            this._isRecording = true;
            this._recordBuffer = [];
            this._recordStartTime = Date.now();
            return true;
        },

        /**
         * 停止录制并保存为自定义模板
         * @param {string} name - 模板名称
         * @param {string} action - 触发的意图动作
         * @returns {boolean} 是否保存成功
         */
        stopRecording: function(name, action) {
            this._isRecording = false;
            if (this._recordBuffer.length < 4) {
                return false; // 录制太短
            }
            if (this._customTemplates.length >= MAX_CUSTOM_TEMPLATES) {
                return false; // 超过上限
            }

            // 将录制缓冲区转为分段序列
            var segments = this._bufferToSegments(this._recordBuffer);
            if (segments.length < 2 || segments.length > MAX_TEMPLATE_LEN) {
                return false; // 分段数不合法
            }

            var template = { name: name, sequence: segments, action: action, custom: true };
            this._customTemplates.push(template);
            this._templates.push(template);
            this._saveCustomTemplates();
            return true;
        },

        /**
         * 删除自定义模板
         */
        removeCustomTemplate: function(name) {
            this._customTemplates = this._customTemplates.filter(function(t) { return t.name !== name; });
            this._templates = this._templates.filter(function(t) { return !(t.custom && t.name === name); });
            this._saveCustomTemplates();
        },

        /**
         * 获取所有模板
         */
        getTemplates: function() {
            return this._templates.map(function(t) {
                return { name: t.name, action: t.action, custom: t.custom, length: t.sequence.length };
            });
        },

        /**
         * 获取自定义模板数量
         */
        getCustomCount: function() {
            return this._customTemplates.length;
        },

        /**
         * 获取当前分段序列（供调试面板使用）
         */
        getRecentSegments: function() {
            return this._segments.slice(-15);
        },

        /**
         * 设置匹配阈值
         */
        setThreshold: function(value) {
            this._threshold = Math.max(0.1, Math.min(1.0, value));
        },

        /**
         * 获取当前阈值
         */
        getThreshold: function() {
            return this._threshold;
        },

        /**
         * 是否在录制状态
         */
        isRecording: function() {
            return this._isRecording;
        },

        /**
         * 注册匹配回调
         * @param {Function} callback - (name, confidence, action)
         */
        onSequenceRecognized: function(callback) {
            this._callbacks.push(callback);
        },

        // =================== 内部方法 ===================

        /**
         * 初始化预定义模板
         */
        _initPredefined: function() {
            var self = this;
            PREDEFINED_SEQUENCES.forEach(function(t) {
                self._templates.push({ name: t.name, sequence: t.sequence, action: t.action, custom: false });
            });
        },

        /**
         * 将原始手势缓冲区转换为分段序列
         * 连续相同手势合并为一个段，记录持续时间
         */
        _updateSegments: function() {
            this._segments = this._bufferToSegments(this._rawBuffer);
        },

        /**
         * 通用缓冲区→分段转换
         */
        _bufferToSegments: function(buffer) {
            if (buffer.length < 2) return [];

            var segments = [];
            var currentGesture = buffer[0].gesture;
            var startTime = buffer[0].timestamp;

            for (var i = 1; i < buffer.length; i++) {
                if (buffer[i].gesture !== currentGesture) {
                    var duration = buffer[i].timestamp - startTime;
                    // 过滤太短的片段（噪声）
                    if (duration >= 60) {
                        // 对于none段，只保留超过120ms的（表示有意停顿）
                        if (currentGesture !== 'none' || duration >= 120) {
                            segments.push({ g: currentGesture, d: duration });
                        }
                    }
                    currentGesture = buffer[i].gesture;
                    startTime = buffer[i].timestamp;
                }
            }

            // 处理最后一段（正在进行中的段）
            var lastDuration = buffer[buffer.length - 1].timestamp - startTime;
            if (lastDuration >= 60) {
                if (currentGesture !== 'none' || lastDuration >= 120) {
                    segments.push({ g: currentGesture, d: lastDuration });
                }
            }

            return segments;
        },

        /**
         * 检查所有模板是否匹配当前手势流
         */
        _checkMatches: function(currentTime) {
            if (this._segments.length < 2) return;

            var bestMatch = null;
            var bestDistance = Infinity;

            for (var i = 0; i < this._templates.length; i++) {
                var template = this._templates[i];
                var tLen = template.sequence.length;

                // 取缓冲区尾部与模板等长(+2的余量)的片段进行比较
                var windowSize = Math.min(tLen + 3, this._segments.length);
                var window = this._segments.slice(-windowSize);

                // 子序列DTW：在window中寻找最佳匹配位置
                var result = this._subsequenceDTW(window, template.sequence);

                if (result.distance < this._threshold && result.distance < bestDistance) {
                    // 冷却检查：同一动作避免重复触发
                    if (template.action === this._lastMatchAction &&
                        currentTime - this._lastMatchTime < this._matchCooldown) {
                        continue;
                    }
                    bestDistance = result.distance;
                    bestMatch = template;
                }
            }

            if (bestMatch) {
                this._lastMatchTime = currentTime;
                this._lastMatchAction = bestMatch.action;
                var confidence = Math.max(0, Math.min(1, 1.0 - bestDistance / this._threshold));

                // 清除已匹配的缓冲区内容防止重复触发
                var clearCount = Math.min(bestMatch.sequence.length * 4, this._rawBuffer.length - 5);
                if (clearCount > 0) {
                    this._rawBuffer.splice(0, clearCount);
                }

                // 触发回调
                for (var j = 0; j < this._callbacks.length; j++) {
                    this._callbacks[j](bestMatch.name, confidence, bestMatch.action);
                }
            }
        },

        /**
         * 子序列DTW - 在输入序列中寻找与模板最佳匹配的子段
         * @param {Array} seq - 输入序列片段
         * @param {Array} template - 模板序列
         * @returns {Object} {distance, startIdx}
         */
        _subsequenceDTW: function(seq, template) {
            var n = seq.length;
            var m = template.length;

            if (n === 0 || m === 0) return { distance: Infinity, startIdx: 0 };

            var W = this._matWidth;
            var cost = this._dtwMatrix;

            // 初始化第一行为0（子序列DTW：允许从任意位置开始）
            for (var j = 0; j <= m; j++) {
                cost[0 * W + j] = (j === 0) ? 0 : Infinity;
            }
            for (var i2 = 1; i2 <= n; i2++) {
                cost[i2 * W + 0] = 0; // 子序列DTW：起始代价为0
            }

            // 填充代价矩阵
            for (var i = 1; i <= n; i++) {
                for (var jj = 1; jj <= m; jj++) {
                    var d = this._elementDistance(seq[i - 1], template[jj - 1]);
                    var prev = Math.min(
                        cost[(i - 1) * W + jj],
                        cost[i * W + (jj - 1)],
                        cost[(i - 1) * W + (jj - 1)]
                    );
                    cost[i * W + jj] = d + prev;
                }
            }

            // 在最后一列找最小值（子序列结束位置）
            var minDist = Infinity;
            var endIdx = n;
            for (var i3 = m; i3 <= n; i3++) {
                var val = cost[i3 * W + m];
                if (val < minDist) {
                    minDist = val;
                    endIdx = i3;
                }
            }

            // 归一化
            var normalized = minDist / m;
            return { distance: normalized, endIdx: endIdx };
        },

        /**
         * 计算两个手势段之间的距离
         * @param {Object} a - {g: "fist", d: 500}
         * @param {Object} b - {g: "fist", d: 500}
         * @returns {number} 0~1
         */
        _elementDistance: function(a, b) {
            if (a.g === b.g) {
                // 相同手势：比较持续时间差异（归一化）
                var maxD = Math.max(a.d, b.d, 1);
                return Math.abs(a.d - b.d) / maxD * 0.4;
            }
            // 不同手势：最大惩罚
            return 1.0;
        },

        /**
         * 保存自定义模板到localStorage
         */
        _saveCustomTemplates: function() {
            try {
                var data = this._customTemplates.map(function(t) {
                    return { name: t.name, sequence: t.sequence, action: t.action };
                });
                localStorage.setItem('gesture_custom_templates', JSON.stringify(data));
            } catch (e) {}
        },

        /**
         * 从localStorage加载自定义模板
         */
        _loadCustomTemplates: function() {
            try {
                var raw = localStorage.getItem('gesture_custom_templates');
                if (!raw) return;
                var data = JSON.parse(raw);
                if (!Array.isArray(data)) return;
                var self = this;
                data.slice(0, MAX_CUSTOM_TEMPLATES).forEach(function(t) {
                    if (t.name && t.sequence && t.action) {
                        var template = { name: t.name, sequence: t.sequence, action: t.action, custom: true };
                        self._customTemplates.push(template);
                        self._templates.push(template);
                    }
                });
            } catch (e) {}
        }
    };

    // 导出
    window.GestureSequenceRecognizer = GestureSequenceRecognizer;
    window.PREDEFINED_SEQUENCES = PREDEFINED_SEQUENCES;
})();
