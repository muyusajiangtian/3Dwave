/**
 * 多模态融合决策模块
 * 将手势序列事件和语音命令事件融合为统一的动作指令
 * 支持置信度加权、优先级策略和冲突解决
 */
(function() {
    'use strict';

    // 优先级模式
    var PRIORITY_MODES = {
        RECENT: 'recent',           // 最近触发的优先
        GESTURE_FIRST: 'gesture_first', // 手势优先
        VOICE_FIRST: 'voice_first'      // 语音优先
    };

    /**
     * 多模态融合引擎
     * @param {Object} options - 配置项
     * @param {number} options.timeWindow - 对齐时间窗口(ms)
     * @param {string} options.priorityMode - 优先级模式
     * @param {number} options.executeCooldown - 同一意图执行冷却(ms)
     */
    function MultimodalFusion(options) {
        options = options || {};
        this._timeWindow = options.timeWindow || 500;
        this._priorityMode = options.priorityMode || PRIORITY_MODES.RECENT;
        this._executeCooldown = options.executeCooldown || 800;
        this._singleModeDelay = options.singleModeDelay || 180; // 单模态延迟等待

        // 事件缓冲区
        this._gestureEvents = []; // [{intent, confidence, timestamp}]
        this._voiceEvents = [];

        // 执行历史（用于冷却和冲突解决）
        this._executeHistory = []; // [{intent, source, confidence, timestamp}]
        this._lastExecuteTime = {};  // {intent: timestamp}

        // 待处理的延迟事件
        this._pendingTimers = {};

        // 融合决策日志
        this._logs = [];
        this._maxLogs = 50;

        // 回调
        this._actionCallbacks = [];
    }

    MultimodalFusion.prototype = {
        /**
         * 接收手势序列事件
         * @param {string} intent - 意图名称
         * @param {number} confidence - 置信度(0-1)
         * @param {number} timestamp - 时间戳(ms)
         */
        feedGestureEvent: function(intent, confidence, timestamp) {
            var event = { intent: intent, confidence: confidence, timestamp: timestamp, source: 'gesture' };
            this._gestureEvents.push(event);
            this._pruneBuffer(this._gestureEvents);
            this._tryFuse(event);
        },

        /**
         * 接收语音命令事件
         * @param {string} intent - 意图名称
         * @param {number} confidence - 置信度(0-1)
         * @param {number} timestamp - 时间戳(ms)
         */
        feedVoiceEvent: function(intent, confidence, timestamp) {
            var event = { intent: intent, confidence: confidence, timestamp: timestamp, source: 'voice' };
            this._voiceEvents.push(event);
            this._pruneBuffer(this._voiceEvents);
            this._tryFuse(event);
        },

        /**
         * 设置优先级模式
         * @param {string} mode - 'recent' | 'gesture_first' | 'voice_first'
         */
        setPriorityMode: function(mode) {
            if (PRIORITY_MODES[mode.toUpperCase()] || Object.values(PRIORITY_MODES).indexOf(mode) !== -1) {
                this._priorityMode = mode;
            }
        },

        /**
         * 获取当前优先级模式
         */
        getPriorityMode: function() {
            return this._priorityMode;
        },

        /**
         * 设置时间窗口
         */
        setTimeWindow: function(ms) {
            this._timeWindow = Math.max(100, Math.min(2000, ms));
        },

        /**
         * 注册动作执行回调
         * @param {Function} callback - (intent, source, confidence, logEntry)
         */
        onAction: function(callback) {
            this._actionCallbacks.push(callback);
        },

        /**
         * 获取最近的融合决策日志
         * @param {number} count - 返回条数
         */
        getRecentLogs: function(count) {
            return this._logs.slice(-(count || 15));
        },

        /**
         * 获取执行历史
         */
        getExecuteHistory: function() {
            return this._executeHistory.slice(-20);
        },

        // =================== 内部方法 ===================

        /**
         * 尝试融合新事件
         * 策略：
         * 1. 在另一模态的缓冲区中查找时间窗口内的匹配事件
         * 2. 同意图 → 融合（提升置信度）
         * 3. 不同意图 → 冲突解决
         * 4. 无匹配 → 延迟后单独执行
         */
        _tryFuse: function(newEvent) {
            var otherBuffer = (newEvent.source === 'gesture') ? this._voiceEvents : this._gestureEvents;
            var now = newEvent.timestamp;

            // 在另一模态中查找时间窗口内的事件
            var matchingSameIntent = null;
            var matchingDiffIntent = null;

            for (var i = otherBuffer.length - 1; i >= 0; i--) {
                var other = otherBuffer[i];
                var timeDiff = Math.abs(now - other.timestamp);

                if (timeDiff > this._timeWindow) continue;

                if (other.intent === newEvent.intent) {
                    if (!matchingSameIntent || other.timestamp > matchingSameIntent.timestamp) {
                        matchingSameIntent = other;
                    }
                } else {
                    if (!matchingDiffIntent || other.timestamp > matchingDiffIntent.timestamp) {
                        matchingDiffIntent = other;
                    }
                }
            }

            // 情况1：双模态同意图 → 融合
            if (matchingSameIntent) {
                this._cancelPending(newEvent.intent);
                var fusedConfidence = Math.min(1.0,
                    Math.max(newEvent.confidence, matchingSameIntent.confidence) + 0.15);
                this._executeAction(newEvent.intent, 'fused', fusedConfidence,
                    '双模态一致: ' + newEvent.source + '(' + newEvent.confidence.toFixed(2) +
                    ') + ' + matchingSameIntent.source + '(' + matchingSameIntent.confidence.toFixed(2) + ')');
                return;
            }

            // 情况2：双模态不同意图 → 冲突解决
            if (matchingDiffIntent) {
                this._cancelPending(newEvent.intent);
                this._cancelPending(matchingDiffIntent.intent);
                var winner = this._resolveConflict(newEvent, matchingDiffIntent);
                this._executeAction(winner.intent, winner.source, winner.confidence,
                    '冲突解决[' + this._priorityMode + ']: ' +
                    newEvent.source + ':' + newEvent.intent + ' vs ' +
                    matchingDiffIntent.source + ':' + matchingDiffIntent.intent +
                    ' → ' + winner.intent);
                return;
            }

            // 情况3：无匹配 → 高置信度直接执行，否则延迟等待
            if (newEvent.confidence >= 0.8) {
                this._executeAction(newEvent.intent, newEvent.source, newEvent.confidence,
                    '单模态(高置信): ' + newEvent.source + ':' + newEvent.intent);
            } else {
                // 延迟执行，给另一模态时间到达
                this._schedulePending(newEvent);
            }
        },

        /**
         * 冲突解决
         */
        _resolveConflict: function(event1, event2) {
            switch (this._priorityMode) {
                case PRIORITY_MODES.GESTURE_FIRST:
                    return (event1.source === 'gesture') ? event1 : event2;
                case PRIORITY_MODES.VOICE_FIRST:
                    return (event1.source === 'voice') ? event1 : event2;
                case PRIORITY_MODES.RECENT:
                default:
                    return (event1.timestamp >= event2.timestamp) ? event1 : event2;
            }
        },

        /**
         * 延迟执行（等待另一模态可能的事件）
         */
        _schedulePending: function(event) {
            var self = this;
            var key = event.source + ':' + event.intent;

            // 取消之前同意图的延迟
            this._cancelPending(event.intent);

            this._pendingTimers[key] = setTimeout(function() {
                delete self._pendingTimers[key];
                self._executeAction(event.intent, event.source, event.confidence,
                    '单模态(延迟): ' + event.source + ':' + event.intent);
            }, this._singleModeDelay);
        },

        /**
         * 取消待处理的延迟事件
         */
        _cancelPending: function(intent) {
            var keys = Object.keys(this._pendingTimers);
            for (var i = 0; i < keys.length; i++) {
                if (keys[i].indexOf(':' + intent) !== -1) {
                    clearTimeout(this._pendingTimers[keys[i]]);
                    delete this._pendingTimers[keys[i]];
                }
            }
        },

        /**
         * 执行动作（含去重防抖）
         */
        _executeAction: function(intent, source, confidence, reason) {
            var now = Date.now();

            // 冷却检查
            if (this._lastExecuteTime[intent] &&
                now - this._lastExecuteTime[intent] < this._executeCooldown) {
                this._addLog(now, 'BLOCKED', intent, source, confidence, reason + ' [冷却中]');
                return;
            }

            this._lastExecuteTime[intent] = now;

            // 记录日志
            var logEntry = this._addLog(now, 'EXECUTE', intent, source, confidence, reason);

            // 记录执行历史
            this._executeHistory.push({ intent: intent, source: source, confidence: confidence, timestamp: now });
            if (this._executeHistory.length > 30) {
                this._executeHistory.shift();
            }

            // 触发回调
            for (var i = 0; i < this._actionCallbacks.length; i++) {
                this._actionCallbacks[i](intent, source, confidence, logEntry);
            }
        },

        /**
         * 添加日志条目
         */
        _addLog: function(timestamp, type, intent, source, confidence, reason) {
            var entry = {
                timestamp: timestamp,
                time: this._formatTime(timestamp),
                type: type,
                intent: intent,
                source: source,
                confidence: confidence,
                reason: reason
            };
            this._logs.push(entry);
            if (this._logs.length > this._maxLogs) {
                this._logs.shift();
            }
            return entry;
        },

        /**
         * 格式化时间
         */
        _formatTime: function(ts) {
            var d = new Date(ts);
            return d.getHours().toString().padStart(2, '0') + ':' +
                   d.getMinutes().toString().padStart(2, '0') + ':' +
                   d.getSeconds().toString().padStart(2, '0');
        },

        /**
         * 清理过期事件
         */
        _pruneBuffer: function(buffer) {
            var now = Date.now();
            var cutoff = now - 3000; // 保留最近3秒
            while (buffer.length > 0 && buffer[0].timestamp < cutoff) {
                buffer.shift();
            }
            // 最多保留15条
            while (buffer.length > 15) {
                buffer.shift();
            }
        }
    };

    // 导出
    window.MultimodalFusion = MultimodalFusion;
    window.FUSION_PRIORITY_MODES = PRIORITY_MODES;
})();
