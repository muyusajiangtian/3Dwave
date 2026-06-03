/**
 * 语音识别模块 - 基于Web Speech API的离线语音命令识别
 * 支持中文连续识别，命令词→意图映射
 */
(function() {
    'use strict';

    // 语音命令映射表（中文关键词 → 意图）
    var VOICE_COMMANDS = {
        "switch_material": ["切换材质", "换材质", "改材质", "材质"],
        "reset_view":      ["重置", "复位", "重置视图", "回到原位", "归位"],
        "show_help":       ["帮助", "显示帮助", "说明", "怎么用"],
        "zoom_in":         ["放大", "大一点", "变大", "拉近"],
        "zoom_out":        ["缩小", "小一点", "变小", "拉远"],
        "rotate_left":     ["向左旋转", "左转", "往左", "左旋"],
        "rotate_right":    ["向右旋转", "右转", "往右", "右旋"],
        "change_color":    ["换色", "变色", "换颜色", "下一个颜色", "改颜色"],
        "explode":         ["爆炸", "粒子爆炸", "爆发", "炸"],
        "toggle_wireframe":["线框", "切换线框", "网格", "显示线框"],
        "particle_burst":  ["粒子", "粒子爆发", "发射粒子"],
        "stop":            ["停止", "停", "暂停"]
    };

    // 最低置信度阈值
    var MIN_CONFIDENCE = 0.4;

    /**
     * 语音命令识别器
     * @param {Object} options - 配置项
     * @param {string} options.lang - 识别语言
     * @param {boolean} options.continuous - 是否连续识别
     */
    function VoiceCommandRecognizer(options) {
        options = options || {};
        this._lang = options.lang || 'zh-CN';
        this._continuous = options.continuous !== false;

        this._recognition = null;
        this._isActive = false;
        this._isSupported = false;
        this._autoRestart = true;
        this._initialized = false;
        this._networkRetryCount = 0;
        this._maxNetworkRetries = 3;
        this._networkRetryDelay = 2000;

        // 命令注册表 {intent: [phrases]}
        this._commands = {};
        // 反向索引 {phrase: intent}
        this._phraseIndex = {};

        // 回调
        this._commandCallbacks = [];
        this._interimCallbacks = [];
        this._stateCallbacks = [];
        this._errorCallbacks = [];

        // 去重：避免同一语句被多次处理
        this._lastResultText = '';
        this._lastResultTime = 0;
        this._resultCooldown = 600;

        // 仅检查浏览器是否支持Speech API（不初始化实例）
        var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        this._isSupported = !!SpeechRecognition;
        this._registerDefaultCommands();
    }

    VoiceCommandRecognizer.prototype = {
        /**
         * 启动语音识别（用户交互触发后才初始化实例）
         */
        start: function() {
            if (!this._isSupported) {
                this._fireError('not_supported', '浏览器不支持Web Speech API，语音功能已禁用');
                return false;
            }
            if (this._isActive) return true;

            // 延迟初始化：仅在用户首次点击时创建SpeechRecognition实例
            if (!this._initialized) {
                this._initSpeechAPI();
                this._initialized = true;
            }

            if (!this._recognition) {
                this._fireError('init_failed', '语音识别初始化失败');
                return false;
            }

            this._networkRetryCount = 0;
            var self = this;
            // 先检查麦克风权限
            if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
                navigator.mediaDevices.getUserMedia({ audio: true }).then(function(stream) {
                    stream.getTracks().forEach(function(t) { t.stop(); });
                    self._doStart();
                }).catch(function(err) {
                    console.warn('[VoiceRecog] 麦克风权限被拒绝:', err.name, err.message);
                    self._fireError('mic_denied', '麦克风权限被拒绝: ' + err.name + '. 请在浏览器设置中允许麦克风访问。');
                });
            } else {
                this._doStart();
            }
            return true;
        },

        /**
         * 实际启动识别
         */
        _doStart: function() {
            try {
                this._autoRestart = true;
                this._recognition.start();
                console.log('[VoiceRecog] 语音识别已启动');
            } catch (e) {
                if (e.name === 'InvalidStateError') {
                    return;
                }
                console.error('[VoiceRecog] 启动失败:', e.name, e.message);
                this._fireError('start_error', '语音识别启动失败: ' + e.message);
            }
        },

        /**
         * 停止语音识别
         */
        stop: function() {
            this._autoRestart = false;
            if (this._recognition) {
                try {
                    this._recognition.stop();
                } catch (e) {}
            }
            this._setActive(false);
        },

        /**
         * 获取当前状态
         */
        isActive: function() {
            return this._isActive;
        },

        /**
         * 是否支持
         */
        isSupported: function() {
            return this._isSupported;
        },

        /**
         * 注册额外命令
         * @param {Array} phrases - 触发短语列表
         * @param {string} intent - 意图名称
         */
        registerCommand: function(phrases, intent) {
            if (!this._commands[intent]) {
                this._commands[intent] = [];
            }
            var self = this;
            phrases.forEach(function(p) {
                var normalized = p.toLowerCase().trim();
                self._commands[intent].push(normalized);
                self._phraseIndex[normalized] = intent;
            });
        },

        /**
         * 注册命令识别回调
         * @param {Function} callback - (intent, confidence, rawText, timestamp)
         */
        onCommand: function(callback) {
            this._commandCallbacks.push(callback);
        },

        /**
         * 注册中间结果回调
         * @param {Function} callback - (partialText)
         */
        onInterim: function(callback) {
            this._interimCallbacks.push(callback);
        },

        /**
         * 注册状态变化回调
         */
        onStateChange: function(callback) {
            this._stateCallbacks.push(callback);
        },

        /**
         * 注册错误回调
         */
        onError: function(callback) {
            this._errorCallbacks.push(callback);
        },

        // =================== 内部方法 ===================

        /**
         * 初始化Web Speech API
         */
        _initSpeechAPI: function() {
            var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            if (!SpeechRecognition) {
                this._isSupported = false;
                return;
            }

            this._isSupported = true;
            var recog = new SpeechRecognition();
            recog.lang = this._lang;
            recog.continuous = this._continuous;
            recog.interimResults = true;
            recog.maxAlternatives = 3;

            var self = this;

            recog.onstart = function() {
                self._networkRetryCount = 0; // 启动成功，重置网络重试计数
                self._setActive(true);
            };

            recog.onresult = function(event) {
                self._handleResult(event);
            };

            recog.onerror = function(event) {
                var errCode = event.error;
                var errMsg = '';
                switch (errCode) {
                    case 'no-speech':
                        console.debug('[VoiceRecog] 未检测到语音输入');
                        return;
                    case 'aborted':
                        console.debug('[VoiceRecog] 识别被中止');
                        return;
                    case 'audio-capture':
                        errMsg = '无法捕获音频，请检查麦克风是否连接';
                        break;
                    case 'not-allowed':
                        errMsg = '麦克风权限被拒绝，请在浏览器地址栏左侧点击允许';
                        self._autoRestart = false;
                        break;
                    case 'network':
                        // 网络错误：超时重试机制
                        self._networkRetryCount++;
                        console.warn('[VoiceRecog] 网络错误 (重试 ' + self._networkRetryCount + '/' + self._maxNetworkRetries + ')');
                        if (self._networkRetryCount <= self._maxNetworkRetries) {
                            errMsg = '网络连接中断，正在重试(' + self._networkRetryCount + '/' + self._maxNetworkRetries + ')...';
                            self._fireError(errCode, errMsg);
                            setTimeout(function() {
                                if (self._autoRestart) {
                                    try { recog.start(); } catch(e) {}
                                }
                            }, self._networkRetryDelay);
                            return;
                        }
                        errMsg = '请检查网络连接或重试';
                        self._autoRestart = false;
                        break;
                    case 'service-not-allowed':
                        errMsg = '语音识别服务不可用，请使用Chrome浏览器';
                        self._autoRestart = false;
                        break;
                    case 'language-not-supported':
                        errMsg = '不支持中文语音识别';
                        break;
                    default:
                        errMsg = '语音识别错误: ' + errCode;
                }
                console.warn('[VoiceRecog] 错误码=' + errCode + ', 信息=' + errMsg);
                self._fireError(errCode, errMsg);
            };

            recog.onend = function() {
                self._setActive(false);
                // 自动重启（网络重试期间由onerror中的setTimeout处理，此处不重复）
                if (self._autoRestart && self._networkRetryCount === 0) {
                    setTimeout(function() {
                        if (self._autoRestart && !self._isActive) {
                            try {
                                recog.start();
                            } catch (e) {}
                        }
                    }, 200);
                }
            };

            this._recognition = recog;
        },

        /**
         * 处理识别结果
         */
        _handleResult: function(event) {
            var now = Date.now();

            for (var i = event.resultIndex; i < event.results.length; i++) {
                var result = event.results[i];
                var transcript = result[0].transcript.trim();

                if (result.isFinal) {
                    // 最终结果 - 尝试匹配命令
                    var confidence = result[0].confidence || 0.7;

                    // 去重检查
                    if (transcript === this._lastResultText &&
                        now - this._lastResultTime < this._resultCooldown) {
                        continue;
                    }
                    this._lastResultText = transcript;
                    this._lastResultTime = now;

                    var match = this._matchCommand(transcript);
                    if (match && confidence >= MIN_CONFIDENCE) {
                        for (var j = 0; j < this._commandCallbacks.length; j++) {
                            this._commandCallbacks[j](match.intent, confidence, transcript, now);
                        }
                    }
                } else {
                    // 中间结果 - 通知调试面板
                    for (var k = 0; k < this._interimCallbacks.length; k++) {
                        this._interimCallbacks[k](transcript);
                    }
                }
            }
        },

        /**
         * 命令匹配：精确匹配 + 包含匹配 + 模糊匹配
         * @param {string} transcript - 识别到的文本
         * @returns {Object|null} {intent, phrase, matchType}
         */
        _matchCommand: function(transcript) {
            var normalized = transcript.toLowerCase().trim();
            // 移除标点和空格
            var clean = normalized.replace(/[，。！？、\s]/g, '');

            // 1. 精确匹配
            if (this._phraseIndex[clean]) {
                return { intent: this._phraseIndex[clean], phrase: clean, matchType: 'exact' };
            }
            if (this._phraseIndex[normalized]) {
                return { intent: this._phraseIndex[normalized], phrase: normalized, matchType: 'exact' };
            }

            // 2. 包含匹配（transcript包含某个命令短语）
            var bestMatch = null;
            var bestLen = 0;
            for (var phrase in this._phraseIndex) {
                if (clean.indexOf(phrase) !== -1 || normalized.indexOf(phrase) !== -1) {
                    if (phrase.length > bestLen) {
                        bestLen = phrase.length;
                        bestMatch = { intent: this._phraseIndex[phrase], phrase: phrase, matchType: 'contains' };
                    }
                }
            }
            if (bestMatch) return bestMatch;

            // 3. 反向包含（某个命令短语包含transcript）
            for (var phrase2 in this._phraseIndex) {
                if (phrase2.indexOf(clean) !== -1 && clean.length >= 2) {
                    return { intent: this._phraseIndex[phrase2], phrase: phrase2, matchType: 'partial' };
                }
            }

            return null;
        },

        /**
         * 注册默认命令集
         */
        _registerDefaultCommands: function() {
            var self = this;
            Object.keys(VOICE_COMMANDS).forEach(function(intent) {
                self.registerCommand(VOICE_COMMANDS[intent], intent);
            });
        },

        /**
         * 设置激活状态
         */
        _setActive: function(active) {
            if (this._isActive !== active) {
                this._isActive = active;
                for (var i = 0; i < this._stateCallbacks.length; i++) {
                    this._stateCallbacks[i](active);
                }
            }
        },

        /**
         * 触发错误回调
         */
        _fireError: function(type, message) {
            for (var i = 0; i < this._errorCallbacks.length; i++) {
                this._errorCallbacks[i](type, message);
            }
        }
    };

    // 导出
    window.VoiceCommandRecognizer = VoiceCommandRecognizer;
    window.VOICE_COMMANDS = VOICE_COMMANDS;
})();
