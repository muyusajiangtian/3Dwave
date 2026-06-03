/**
 * 调试面板模块 - 显示手势流、语音识别和融合决策信息
 * 提供阈值调整、优先级切换、模板录制等控件
 */
(function() {
    'use strict';

    /**
     * 调试面板
     * @param {string} containerId - 容器元素ID
     */
    function DebugPanel(containerId) {
        this._containerId = containerId;
        this._container = null;
        this._isVisible = false;
        this._isMinimized = false;

        // DOM引用
        this._els = {};

        // 更新节流
        this._lastUpdateTime = 0;
        this._updateInterval = 180; // ms

        // 日志限制
        this._maxSpeechLogs = 12;
        this._maxFusionLogs = 15;
        this._speechLogs = [];
        this._fusionLogs = [];
    }

    DebugPanel.prototype = {
        /**
         * 初始化面板（注入DOM）
         */
        init: function() {
            this._container = document.getElementById(this._containerId);
            if (!this._container) return;
            this._container.innerHTML = this._buildHTML();
            this._cacheElements();
            this._bindEvents();
            this._container.style.display = 'none';
        },

        /**
         * 显示/隐藏面板
         */
        toggle: function() {
            this._isVisible = !this._isVisible;
            this._container.style.display = this._isVisible ? 'block' : 'none';
        },

        /**
         * 显示面板
         */
        show: function() {
            this._isVisible = true;
            this._container.style.display = 'block';
        },

        /**
         * 隐藏面板
         */
        hide: function() {
            this._isVisible = false;
            this._container.style.display = 'none';
        },

        /**
         * 面板是否可见
         */
        isVisible: function() {
            return this._isVisible;
        },

        /**
         * 更新手势流显示
         * @param {Array} segments - [{g, d}]
         */
        updateGestureStream: function(segments) {
            if (!this._isVisible) return;
            var now = Date.now();
            if (now - this._lastUpdateTime < this._updateInterval) return;
            this._lastUpdateTime = now;

            var el = this._els.gestureStream;
            if (!el) return;

            var html = '';
            var colors = { fist: '#f59e0b', pinch: '#8b5cf6', wave: '#10b981', pointing: '#3b82f6', none: '#6b7280' };
            var icons = { fist: '✊', pinch: '🤏', wave: '👋', pointing: '☝', none: '·' };

            var start = Math.max(0, segments.length - 12);
            for (var i = start; i < segments.length; i++) {
                var seg = segments[i];
                var color = colors[seg.g] || '#6b7280';
                var icon = icons[seg.g] || '?';
                html += '<span class="dp-seg" style="color:' + color + '">';
                html += icon + '<small>' + Math.round(seg.d) + 'ms</small>';
                html += '</span>';
                if (i < segments.length - 1) html += '<span class="dp-arrow">→</span>';
            }

            el.innerHTML = html || '<span style="color:#666">等待手势输入...</span>';
        },

        /**
         * 更新当前识别的序列名称
         * @param {string} name - 序列名称
         * @param {number} confidence - 置信度
         */
        updateSequenceName: function(name, confidence) {
            var el = this._els.sequenceName;
            if (el) {
                el.textContent = name + ' (' + (confidence * 100).toFixed(0) + '%)';
                el.style.opacity = '1';
                // 淡出效果
                setTimeout(function() { el.style.opacity = '0.5'; }, 2000);
            }

            // 同时更新外部状态显示
            var extEl = document.getElementById('sequence-status');
            if (extEl) {
                extEl.textContent = '🎯 ' + name;
                extEl.style.opacity = '1';
                extEl.style.display = 'block';
                setTimeout(function() {
                    extEl.style.opacity = '0';
                    setTimeout(function() { extEl.style.display = 'none'; }, 500);
                }, 2500);
            }
        },

        /**
         * 添加语音识别文本
         * @param {string} text - 识别文本
         * @param {boolean} isFinal - 是否为最终结果
         * @param {number} confidence - 置信度
         */
        addSpeechText: function(text, isFinal, confidence) {
            if (!this._isVisible) return;

            if (isFinal) {
                this._speechLogs.push({
                    text: text,
                    confidence: confidence,
                    isFinal: true,
                    time: this._formatTime(Date.now())
                });
                if (this._speechLogs.length > this._maxSpeechLogs) {
                    this._speechLogs.shift();
                }
                this._renderSpeechLogs();
            } else {
                // 中间结果：只更新临时行
                var el = this._els.speechInterim;
                if (el) {
                    el.textContent = '🎙️ ' + text + '...';
                    el.style.display = text ? 'block' : 'none';
                }
            }
        },

        /**
         * 添加融合决策日志
         * @param {Object} entry - {time, type, intent, source, confidence, reason}
         */
        addFusionLog: function(entry) {
            this._fusionLogs.push(entry);
            if (this._fusionLogs.length > this._maxFusionLogs) {
                this._fusionLogs.shift();
            }
            if (this._isVisible) {
                this._renderFusionLogs();
            }
        },

        /**
         * 更新录制状态
         * @param {boolean} isRecording - 是否在录制
         * @param {number} templateCount - 当前自定义模板数
         */
        updateRecordingState: function(isRecording, templateCount) {
            var btn = this._els.recordBtn;
            var status = this._els.templateCount;
            if (btn) {
                btn.textContent = isRecording ? '⏹ 停止录制' : '⏺ 录制手势';
                btn.className = isRecording ? 'dp-btn dp-btn-recording' : 'dp-btn';
            }
            if (status) {
                status.textContent = '自定义模板: ' + templateCount + '/5';
            }
        },

        /**
         * 更新语音状态
         * @param {boolean} active - 是否激活
         */
        updateVoiceState: function(active) {
            var el = this._els.voiceStatus;
            if (el) {
                el.textContent = active ? '🟢 语音识别中' : '🔴 语音已停止';
                el.style.color = active ? '#4ade80' : '#f87171';
            }
        },

        /**
         * 获取当前阈值
         */
        getThreshold: function() {
            var el = this._els.thresholdSlider;
            return el ? parseFloat(el.value) : 0.55;
        },

        /**
         * 获取优先级模式
         */
        getPriorityMode: function() {
            var radios = this._container.querySelectorAll('input[name="dp-priority"]');
            for (var i = 0; i < radios.length; i++) {
                if (radios[i].checked) return radios[i].value;
            }
            return 'recent';
        },

        // =================== 内部方法 ===================

        _buildHTML: function() {
            return '' +
            '<div class="dp-panel">' +
                '<div class="dp-header">' +
                    '<span class="dp-title">🔧 调试面板</span>' +
                    '<button class="dp-close" id="dp-close-btn">×</button>' +
                '</div>' +
                '<div class="dp-body" id="dp-body">' +
                    // 手势序列流
                    '<div class="dp-section">' +
                        '<div class="dp-section-title">手势序列流</div>' +
                        '<div class="dp-gesture-stream" id="dp-gesture-stream"></div>' +
                        '<div class="dp-sequence-name" id="dp-sequence-name">等待匹配...</div>' +
                    '</div>' +

                    // 语音识别
                    '<div class="dp-section">' +
                        '<div class="dp-section-title">语音识别</div>' +
                        '<div class="dp-voice-status" id="dp-voice-status">🔴 语音未启动</div>' +
                        '<div class="dp-speech-interim" id="dp-speech-interim" style="display:none"></div>' +
                        '<div class="dp-speech-logs" id="dp-speech-logs"></div>' +
                    '</div>' +

                    // 融合决策
                    '<div class="dp-section">' +
                        '<div class="dp-section-title">融合决策日志</div>' +
                        '<div class="dp-fusion-logs" id="dp-fusion-logs"></div>' +
                    '</div>' +

                    // 设置
                    '<div class="dp-section">' +
                        '<div class="dp-section-title">设置</div>' +
                        '<div class="dp-setting">' +
                            '<label>DTW匹配阈值: <span id="dp-threshold-val">0.55</span></label>' +
                            '<input type="range" min="0.2" max="0.9" step="0.05" value="0.55" ' +
                                'id="dp-threshold-slider" class="dp-slider">' +
                        '</div>' +
                        '<div class="dp-setting">' +
                            '<label>融合优先级:</label>' +
                            '<div class="dp-radio-group">' +
                                '<label><input type="radio" name="dp-priority" value="recent" checked> 最近优先</label>' +
                                '<label><input type="radio" name="dp-priority" value="gesture_first"> 手势优先</label>' +
                                '<label><input type="radio" name="dp-priority" value="voice_first"> 语音优先</label>' +
                            '</div>' +
                        '</div>' +
                        '<div class="dp-setting">' +
                            '<span id="dp-template-count">自定义模板: 0/5</span>' +
                        '</div>' +
                        '<div class="dp-actions">' +
                            '<button class="dp-btn" id="dp-record-btn">⏺ 录制手势</button>' +
                            '<button class="dp-btn" id="dp-clear-templates-btn">清空自定义</button>' +
                        '</div>' +
                    '</div>' +
                '</div>' +
            '</div>';
        },

        _cacheElements: function() {
            this._els.gestureStream = document.getElementById('dp-gesture-stream');
            this._els.sequenceName = document.getElementById('dp-sequence-name');
            this._els.voiceStatus = document.getElementById('dp-voice-status');
            this._els.speechInterim = document.getElementById('dp-speech-interim');
            this._els.speechLogs = document.getElementById('dp-speech-logs');
            this._els.fusionLogs = document.getElementById('dp-fusion-logs');
            this._els.thresholdSlider = document.getElementById('dp-threshold-slider');
            this._els.thresholdVal = document.getElementById('dp-threshold-val');
            this._els.recordBtn = document.getElementById('dp-record-btn');
            this._els.templateCount = document.getElementById('dp-template-count');
        },

        _bindEvents: function() {
            var self = this;

            // 关闭按钮
            var closeBtn = document.getElementById('dp-close-btn');
            if (closeBtn) {
                closeBtn.addEventListener('click', function() { self.hide(); });
            }

            // 阈值滑块
            var slider = this._els.thresholdSlider;
            if (slider) {
                slider.addEventListener('input', function() {
                    self._els.thresholdVal.textContent = parseFloat(slider.value).toFixed(2);
                    self._fireEvent('thresholdChange', parseFloat(slider.value));
                });
            }

            // 优先级切换
            var radios = this._container.querySelectorAll('input[name="dp-priority"]');
            for (var i = 0; i < radios.length; i++) {
                radios[i].addEventListener('change', function() {
                    self._fireEvent('priorityChange', self.getPriorityMode());
                });
            }

            // 录制按钮
            var recordBtn = this._els.recordBtn;
            if (recordBtn) {
                recordBtn.addEventListener('click', function() {
                    self._fireEvent('recordToggle');
                });
            }

            // 清空模板按钮
            var clearBtn = document.getElementById('dp-clear-templates-btn');
            if (clearBtn) {
                clearBtn.addEventListener('click', function() {
                    self._fireEvent('clearTemplates');
                });
            }
        },

        _renderSpeechLogs: function() {
            var el = this._els.speechLogs;
            if (!el) return;

            var html = '';
            for (var i = this._speechLogs.length - 1; i >= Math.max(0, this._speechLogs.length - 8); i--) {
                var log = this._speechLogs[i];
                html += '<div class="dp-speech-entry">';
                html += '<span class="dp-time">' + log.time + '</span> ';
                html += '<span class="dp-speech-text">"' + log.text + '"</span>';
                if (log.confidence !== null && log.confidence !== undefined) {
                    html += ' <span class="dp-conf">(' + (log.confidence * 100).toFixed(0) + '%)</span>';
                }
                html += '</div>';
            }
            el.innerHTML = html || '<span style="color:#666">等待语音输入...</span>';
        },

        _renderFusionLogs: function() {
            var el = this._els.fusionLogs;
            if (!el) return;

            var html = '';
            var typeColors = { EXECUTE: '#4ade80', BLOCKED: '#f87171', FUSED: '#a78bfa' };

            for (var i = this._fusionLogs.length - 1; i >= Math.max(0, this._fusionLogs.length - 10); i--) {
                var log = this._fusionLogs[i];
                var color = typeColors[log.type] || '#94a3b8';
                var sourceIcon = log.source === 'fused' ? '🔗' : (log.source === 'voice' ? '🎤' : '✋');

                html += '<div class="dp-fusion-entry">';
                html += '<span class="dp-time">' + log.time + '</span> ';
                html += '<span style="color:' + color + '">[' + log.type + ']</span> ';
                html += sourceIcon + ' ';
                html += '<span class="dp-intent">' + log.intent + '</span>';
                html += ' <span class="dp-conf">' + (log.confidence * 100).toFixed(0) + '%</span>';
                html += '</div>';
            }
            el.innerHTML = html || '<span style="color:#666">等待融合事件...</span>';
        },

        _formatTime: function(ts) {
            var d = new Date(ts);
            return d.getHours().toString().padStart(2, '0') + ':' +
                   d.getMinutes().toString().padStart(2, '0') + ':' +
                   d.getSeconds().toString().padStart(2, '0');
        },

        // 简易事件系统
        _eventListeners: {},
        on: function(event, callback) {
            if (!this._eventListeners[event]) this._eventListeners[event] = [];
            this._eventListeners[event].push(callback);
        },
        _fireEvent: function(event, data) {
            var listeners = this._eventListeners[event];
            if (listeners) {
                for (var i = 0; i < listeners.length; i++) {
                    listeners[i](data);
                }
            }
        }
    };

    // 导出
    window.DebugPanel = DebugPanel;
})();
