"""Qt desktop control room and the matching SteamVR conversation panel."""
from __future__ import annotations

import os
import queue
import threading
from dataclasses import replace
from typing import Callable

from PySide6.QtCore import Qt, Signal, QObject, QTimer
from PySide6.QtGui import QCloseEvent, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QProgressBar, QScrollArea, QSpinBox, QTabWidget, QTextEdit,
    QVBoxLayout, QWidget,
)

from .audio import AudioCapture, AudioError, AudioClip, list_devices, transcribe_clip
from .core import ConversationState, DialogueError, GenerationRequest
from .languages import LANGUAGES, language_name
from .providers import ProviderSettings, generate_reply, DEMO_UTTERANCES
from .settings import Settings, load_settings, save_settings


STYLE = """
QWidget { background: #101720; color: #e7edf5; font-family: 'Microsoft YaHei UI', 'Noto Sans CJK SC', 'Arial'; font-size: 14px; }
QMainWindow { background: #101720; }
QFrame#side { background: #151e2a; border: 1px solid #263348; border-radius: 14px; }
QFrame#card { background: #1a2635; border: 1px solid #2a3c52; border-radius: 12px; }
QFrame#pin { background: #16302e; border: 1px solid #397b6c; border-radius: 14px; }
QFrame#card QLabel, QFrame#pin QLabel, QFrame#side QLabel { background: transparent; }
QLabel#title { font-size: 28px; font-weight: 600; }
QLabel#section { font-size: 15px; color: #a4b7cc; font-weight: 600; }
QLabel#subtle { color: #93a8bf; }
QLabel#original { font-size: 19px; }
QLabel#translation { font-size: 22px; color: #f2f6fc; }
QLabel#foreign { font-size: 21px; }
QLabel#pinnedForeign { font-size: 30px; color: #f1fff9; font-weight: 500; }
QLabel#meaning { color: #bccee1; font-size: 16px; }
QLabel#status { background: #192535; padding: 10px 14px; border-radius: 8px; color: #b9cee4; }
QPushButton { background: #233449; color: #e7edf5; border: 1px solid #35516a; border-radius: 8px; padding: 9px 14px; min-height: 24px; }
QPushButton:hover { background: #304963; }
QPushButton:pressed { background: #1c6a64; }
QPushButton:disabled { color: #63748a; background: #182332; border-color: #253346; }
QPushButton#primary { background: #70d4b4; color: #10291f; border: 1px solid #70d4b4; font-weight: 600; }
QPushButton#primary:hover { background: #91e5ca; }
QPushButton#danger { color: #ffc9c6; border-color: #885957; }
QLineEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #0f1823; color: #e7edf5; border: 1px solid #35475d; border-radius: 7px; padding: 7px; selection-background-color: #315d79; }
QComboBox QAbstractItemView { background: #182433; selection-background-color: #315d79; }
QComboBox::drop-down { border: none; width: 24px; }
QCheckBox { spacing: 8px; }
QProgressBar { border: none; border-radius: 3px; background: #273449; max-height: 7px; }
QProgressBar::chunk { border-radius: 3px; background: #70d4b4; }
QScrollArea { border: none; }
QTabWidget::pane { border: 1px solid #263348; }
QTabBar::tab { background: #192636; padding: 10px 18px; border-radius: 4px; margin-right: 4px; }
QTabBar::tab:selected { background: #314c63; }
QToolTip { background: #18283a; color: #e7edf5; border: 1px solid #45627e; padding: 6px; }
"""


def label(text: str = "", name: str = "", wrap: bool = True) -> QLabel:
    result = QLabel(text)
    result.setTextFormat(Qt.TextFormat.PlainText)
    result.setWordWrap(wrap)
    if name:
        result.setObjectName(name)
    return result


def button(text: str, callback: Callable | None = None, *, primary: bool = False,
           action: str | None = None) -> QPushButton:
    result = QPushButton(text)
    if primary:
        result.setObjectName("primary")
    if action:
        result.setProperty("vr_action", action)
    if callback:
        result.clicked.connect(lambda _checked=False: callback())
    return result


class DialoguePanel(QWidget):
    """Only this compact view is shared to VR. Settings and secrets never enter it."""
    action = Signal(str)

    def __init__(self, vr: bool = False):
        super().__init__()
        self.vr = vr
        self.setStyleSheet(STYLE)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)
        top = QHBoxLayout()
        top.addWidget(label("对方刚才说", "section"))
        self.language = label("等待识别语言", "subtle")
        self.language.setAlignment(Qt.AlignmentFlag.AlignRight)
        top.addWidget(self.language)
        layout.addLayout(top)
        self.original = label("从左侧载入一段演示，或开始收听。", "original")
        self.translation = label("中文翻译会出现在这里。", "translation")
        layout.addWidget(self.original)
        layout.addWidget(self.translation)
        self.section = label("选择你想表达的意思", "section")
        layout.addWidget(self.section)
        self.summary = label("每条外语回复都带有对应中文。", "subtle")
        layout.addWidget(self.summary)
        self.candidate_container = QWidget()
        candidates_layout = QVBoxLayout(self.candidate_container)
        candidates_layout.setContentsMargins(0, 0, 0, 0)
        candidates_layout.setSpacing(10)
        self.cards = []
        for i in range(3):
            frame = QFrame()
            frame.setObjectName("card")
            row = QHBoxLayout(frame)
            row.setContentsMargins(16, 12, 14, 12)
            body = QVBoxLayout()
            intent = label(f"{i + 1:02d}  等待回复建议", "section")
            foreign = label("", "foreign")
            meaning = label("", "meaning")
            body.addWidget(intent)
            body.addWidget(foreign)
            body.addWidget(meaning)
            row.addLayout(body, 1)
            choose = button("选这句", lambda i=i: self.action.emit(f"select_{i+1}"),
                            action=f"select_{i+1}")
            choose.setEnabled(False)
            row.addWidget(choose)
            candidates_layout.addWidget(frame)
            self.cards.append((frame, intent, foreign, meaning, choose))
        layout.addWidget(self.candidate_container)
        self.pin = QFrame()
        self.pin.setObjectName("pin")
        pin_layout = QVBoxLayout(self.pin)
        pin_layout.setContentsMargins(22, 20, 22, 20)
        self.pin_heading = label("已固定 · 你可以照着读", "section")
        self.pin_foreign = label("", "pinnedForeign")
        self.pin_meaning = label("", "meaning")
        self.pin_reading = label("", "subtle")
        pin_layout.addWidget(self.pin_heading)
        pin_layout.addWidget(self.pin_foreign)
        pin_layout.addWidget(self.pin_meaning)
        pin_layout.addWidget(self.pin_reading)
        pin_layout.addSpacing(10)
        actions = QHBoxLayout()
        actions.addWidget(button("已完整说完", lambda: self.action.emit("done"),
                                 primary=True, action="done"))
        actions.addWidget(button("取消这句", lambda: self.action.emit("cancel"), action="cancel"))
        pin_layout.addLayout(actions)
        pin_layout.addWidget(label("点“已完整说完”后，这句才会加入对话记录。", "subtle"))
        layout.addWidget(self.pin)
        self.pin.hide()
        layout.addStretch(1)
        controls = QHBoxLayout()
        self.refresh_button = button("重新建议", lambda: self.action.emit("regenerate"), action="regenerate")
        controls.addWidget(self.refresh_button)
        self.dictate = button("按住说中文", action="dictate_start")
        self.dictate.pressed.connect(lambda: self.action.emit("dictate_start"))
        self.dictate.released.connect(lambda: self.action.emit("dictate_stop"))
        controls.addWidget(self.dictate)
        layout.addLayout(controls)
        self.footer = label("录中文前请在游戏内静音，保持 VD 麦克风开启。", "subtle")
        layout.addWidget(self.footer)

    def update_state(self, state: ConversationState, *, busy: bool = False,
                     recording: bool = False, demo: bool = True) -> None:
        other = state.latest_other
        self.language.setText(f"回复：{language_name(state.reply_language)} · " +
                              ("手动" if state.manual_language else "自动"))
        self.original.setText(other.text if other else "载入一段演示，或开始收听。")
        self.translation.setText((other.translation_zh or "正在理解这句话…") if other else "中文翻译会出现在这里。")
        self.summary.setText("正在生成…" if busy else state.summary_zh or "请选择最符合你想法的一条。")
        for i, (frame, intent, foreign, meaning, choose) in enumerate(self.cards):
            visible = i < len(state.candidates)
            frame.setVisible(visible or not state.candidates)
            choose.setEnabled(visible)
            if visible:
                c = state.candidates[i]
                intent.setText(f"{i+1:02d}  {c.intent_zh}")
                foreign.setText(c.text)
                meaning.setText(c.meaning_zh)
            else:
                intent.setText(f"{i+1:02d}  等待回复建议")
                foreign.setText("")
                meaning.setText("")
        self.pin.setVisible(state.pinned is not None)
        self.candidate_container.setVisible(state.pinned is None)
        self.section.setText("你的提词" if state.pinned else "选择你想表达的意思")
        if state.pinned:
            c = state.pinned.candidate
            self.pin_heading.setText(f"已固定 · {language_name(state.pinned.language)} · 尚未计入对话")
            self.pin_foreign.setText(c.text)
            self.pin_meaning.setText(c.meaning_zh)
            self.pin_reading.setText("读音参考：" + c.reading_hint if c.reading_hint else "")
        self.dictate.setText("正在收中文 · 松开生成" if recording else "按住说中文")
        self.refresh_button.setEnabled(other is not None and not busy)
        self.footer.setText("演示：固定样例，无模型调用。可在设置中连接 GPT 或 Codex。" if demo else
                            "录中文前请在游戏内静音；此工具不会替你开麦或发送消息。")


class SettingsDialog(QDialog):
    def __init__(self, settings: Settings, api_key: str, asr_key: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("KotonohaVR · 言叶 · 设置")
        self.resize(680, 650)
        self.result_settings = settings
        self.result_api_key = api_key
        self.result_asr_key = asr_key
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        model_page = QWidget()
        form = QFormLayout(model_page)
        form.setSpacing(14)
        self.provider = QComboBox()
        for name, value in [("离线演示（固定样例）", "demo"), ("GPT · Responses API", "openai"), ("Codex · 本机 CLI", "codex")]:
            self.provider.addItem(name, value)
        self.provider.setCurrentIndex(self.provider.findData(settings.provider))
        form.addRow("回复来源", self.provider)
        self.model = QLineEdit(settings.openai_model)
        self.base_url = QLineEdit(settings.openai_base_url)
        self.api_key = QLineEdit(api_key)
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("仅保留在本次运行；留空读取 OPENAI_API_KEY")
        form.addRow("GPT 模型", self.model)
        form.addRow("GPT API 地址", self.base_url)
        form.addRow("GPT API Key", self.api_key)
        self.codex_command = QLineEdit(settings.codex_command)
        self.codex_model = QLineEdit(settings.codex_model)
        self.codex_model.setPlaceholderText("留空使用 Codex 默认模型；不会加载用户工具配置")
        form.addRow("Codex 可执行文件", self.codex_command)
        form.addRow("Codex 模型", self.codex_model)
        form.addRow(label("Codex 由本机官方 CLI 处理登录；本软件不读取登录令牌。\nGPT API 与 Codex 的可用模型、权限和额度以各自账户为准。", "subtle"))
        self.style = QLineEdit(settings.style)
        form.addRow("回复风格", self.style)
        tabs.addTab(model_page, "模型")
        audio_page = QWidget()
        audio_form = QFormLayout(audio_page)
        audio_form.setSpacing(14)
        self.asr_backend = QComboBox()
        self.asr_backend.addItem("本地 faster-whisper · CPU", "local")
        self.asr_backend.addItem("OpenAI 语音 API", "openai")
        self.asr_backend.setCurrentIndex(self.asr_backend.findData(settings.asr_backend))
        self.local_model = QLineEdit(settings.local_asr_model)
        self.cloud_model = QLineEdit(settings.cloud_asr_model)
        self.asr_base = QLineEdit(settings.asr_base_url)
        self.asr_key = QLineEdit(asr_key)
        self.asr_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.asr_key.setPlaceholderText("留空使用 GPT Key 或 OPENAI_API_KEY")
        audio_form.addRow("语音识别", self.asr_backend)
        audio_form.addRow("本地模型", self.local_model)
        audio_form.addRow("云端语音模型", self.cloud_model)
        audio_form.addRow("语音 API 地址", self.asr_base)
        audio_form.addRow("语音 API Key", self.asr_key)
        self.threshold = QDoubleSpinBox()
        self.threshold.setRange(.001, .5)
        self.threshold.setDecimals(3)
        self.threshold.setSingleStep(.005)
        self.threshold.setValue(settings.energy_threshold)
        self.silence = QSpinBox()
        self.silence.setRange(200, 2500)
        self.silence.setSuffix(" ms")
        self.silence.setValue(settings.silence_ms)
        audio_form.addRow("声音阈值", self.threshold)
        audio_form.addRow("停顿断句", self.silence)
        audio_form.addRow(label("本地模式首次使用会下载模型；需安装 [local-asr] 依赖。\n云端模式会将录到的语音片段发送给所配置的服务。\n首版按短句识别，最多一段 12 秒。", "subtle"))
        tabs.addTab(audio_page, "语音")
        vr_page = QWidget()
        vr_form = QFormLayout(vr_page)
        self.placement = QComboBox()
        for name, value in [("视野前方", "head"), ("左手", "left"), ("右手", "right")]:
            self.placement.addItem(name, value)
        self.placement.setCurrentIndex(self.placement.findData(settings.placement))
        self.width = QDoubleSpinBox()
        self.width.setRange(.4, 2)
        self.width.setSingleStep(.1)
        self.width.setValue(settings.overlay_width)
        self.distance = QDoubleSpinBox()
        self.distance.setRange(.6, 3)
        self.distance.setSingleStep(.1)
        self.distance.setValue(settings.overlay_distance)
        vr_form.addRow("显示位置", self.placement)
        vr_form.addRow("浮窗宽度（米）", self.width)
        vr_form.addRow("前方距离（米）", self.distance)
        vr_form.addRow(label("手柄点选通过 SteamVR 的浮窗射线输入完成。\n开启交互时可能暂时占用游戏手柄输入；\n可取消“手柄点选”，保留只读提词。", "subtle"))
        tabs.addTab(vr_page, "VR 浮窗")
        self.error = label("", "subtle")
        layout.addWidget(self.error)
        controls = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        controls.accepted.connect(self.accept_changes)
        controls.rejected.connect(self.reject)
        layout.addWidget(controls)

    def accept_changes(self):
        try:
            s = replace(self.result_settings,
                        provider=self.provider.currentData(), openai_model=self.model.text().strip(),
                        openai_base_url=self.base_url.text().strip(), codex_command=self.codex_command.text().strip(),
                        codex_model=self.codex_model.text().strip(), style=self.style.text().strip(),
                        asr_backend=self.asr_backend.currentData(), local_asr_model=self.local_model.text().strip(),
                        cloud_asr_model=self.cloud_model.text().strip(), asr_base_url=self.asr_base.text().strip(),
                        energy_threshold=self.threshold.value(), silence_ms=self.silence.value(),
                        placement=self.placement.currentData(), overlay_width=self.width.value(),
                        overlay_distance=self.distance.value())
            s.validate()
            if s.provider == "openai" and not s.openai_model:
                raise ValueError("请填写 GPT 模型名称。")
            save_settings(s)
            self.result_settings = s
            self.result_api_key = self.api_key.text().strip()
            self.result_asr_key = self.asr_key.text().strip()
            self.accept()
        except (ValueError, OSError) as exc:
            self.error.setText(str(exc))


class Bridge(QObject):
    generation = Signal(object, object)
    generation_error = Signal(object, str)
    transcript = Signal(int, str, object)
    audio_error = Signal(int, str)
    level = Signal(int, float)


class MainWindow(QMainWindow):
    def __init__(self, settings: Settings | None = None):
        super().__init__()
        self.settings = settings or load_settings()
        self.state = ConversationState()
        self.api_key = ""
        self.asr_key = ""
        self.closed = False
        self.busy = False
        self.pending: tuple | None = None
        self.recording = False
        self.speaker_capture = None
        self.mic_capture = None
        self.audio_epoch = 0
        self.asr_queue = queue.Queue(maxsize=3)
        self.asr_stopping = threading.Event()
        self.overlay = None
        self.overlay_warning = ""
        self.devices = []
        self.bridge = Bridge()
        self.bridge.generation.connect(self.generation_done)
        self.bridge.generation_error.connect(self.generation_failed)
        self.bridge.transcript.connect(self.transcript_ready)
        self.bridge.audio_error.connect(self.audio_failed)
        self.bridge.level.connect(self.audio_level)
        self.setWindowTitle("KotonohaVR · 言叶 · 多语言对话助手")
        self.resize(1240, 860)
        self.setMinimumSize(920, 640)
        self.setStyleSheet(STYLE)
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(24, 18, 24, 18)
        heading = QHBoxLayout()
        title_group = QVBoxLayout()
        title_group.addWidget(label("言叶 VR", "title"))
        title_group.addWidget(label("听懂对方 · 选择自己的回答 · 照着读", "subtle"))
        heading.addLayout(title_group, 1)
        self.provider_badge = label("离线演示", "section")
        heading.addWidget(self.provider_badge)
        heading.addWidget(button("设置", self.open_settings))
        outer.addLayout(heading)
        content = QHBoxLayout()
        side = QFrame()
        side.setObjectName("side")
        side.setFixedWidth(285)
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(16, 18, 16, 18)
        side_layout.setSpacing(10)
        side_layout.addWidget(label("开始一段对话", "section"))
        self.demo_language = QComboBox()
        for code in ("ja", "en", "ko"):
            self.demo_language.addItem(language_name(code) + " · 邀请同行", code)
        side_layout.addWidget(self.demo_language)
        side_layout.addWidget(button("载入演示对话", self.load_demo, primary=True))
        side_layout.addSpacing(10)
        side_layout.addWidget(label("回复语言", "section"))
        self.target = QComboBox()
        self.target.addItem("自动跟随对方", None)
        for code, name in LANGUAGES.items():
            self.target.addItem(name, code)
        self.target.currentIndexChanged.connect(self.change_language)
        side_layout.addWidget(self.target)
        side_layout.addSpacing(10)
        side_layout.addWidget(label("实时收音 · Windows", "section"))
        self.speaker_device = QComboBox()
        self.speaker_device.addItem("请选择耳机/扬声器回环", -1)
        self.mic_device = QComboBox()
        self.mic_device.addItem("请选择 VD 麦克风", -1)
        side_layout.addWidget(self.speaker_device)
        side_layout.addWidget(self.mic_device)
        side_layout.addWidget(button("刷新音频设备", self.refresh_devices))
        self.listen_button = button("开始收听", self.toggle_listening)
        side_layout.addWidget(self.listen_button)
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setTextVisible(False)
        side_layout.addWidget(self.meter)
        self.auto_suggest = QCheckBox("停顿后自动给出建议")
        self.auto_suggest.setChecked(self.settings.auto_suggest)
        self.auto_suggest.toggled.connect(lambda checked: setattr(self.settings, "auto_suggest", checked))
        side_layout.addWidget(self.auto_suggest)
        side_layout.addSpacing(10)
        side_layout.addWidget(label("SteamVR", "section"))
        self.vr_button = button("显示 VR 浮窗", self.toggle_vr)
        side_layout.addWidget(self.vr_button)
        self.interactive = QCheckBox("手柄点选（占用浮窗射线）")
        self.interactive.setChecked(True)
        self.interactive.toggled.connect(self.set_interactive)
        side_layout.addWidget(self.interactive)
        side_layout.addWidget(button("桌面提词预览", self.preview_overlay))
        side_layout.addStretch(1)
        side_layout.addWidget(button("查看实际对话", self.show_history))
        side_layout.addWidget(button("清空本次对话", self.reset_dialogue))
        content.addWidget(side)
        right = QVBoxLayout()
        self.panel = DialoguePanel()
        self.panel.action.connect(self.handle_action)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.panel)
        right.addWidget(scroll, 1)
        input_row = QHBoxLayout()
        self.input_mode = QComboBox()
        self.input_mode.addItem("对方说的话", "suggest")
        self.input_mode.addItem("我想说的中文", "custom")
        self.text_input = QLineEdit()
        self.text_input.setPlaceholderText("也可以粘贴一句话，测试真实模型…")
        self.text_input.returnPressed.connect(self.submit_text)
        input_row.addWidget(self.input_mode)
        input_row.addWidget(self.text_input, 1)
        input_row.addWidget(button("生成", self.submit_text, primary=True))
        right.addLayout(input_row)
        content.addLayout(right, 1)
        outer.addLayout(content, 1)
        self.status = label("就绪。演示模式不联网；真实收音需先选择设备和模型。", "status")
        outer.addWidget(self.status)
        self.vr_panel = DialoguePanel(vr=True)
        self.vr_panel.setWindowTitle("KotonohaVR · 言叶 · 提词预览")
        self.vr_panel.resize(940, 800)
        self.vr_panel.action.connect(self.handle_action)
        self.vr_timer = QTimer(self)
        self.vr_timer.setInterval(80)
        self.vr_timer.timeout.connect(self.tick_vr)
        self.shortcuts = []
        for key, action in [("Alt+1", "select_1"), ("Alt+2", "select_2"), ("Alt+3", "select_3"),
                            ("Alt+Return", "done"), ("Escape", "cancel")]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(lambda action=action: self.handle_action(action))
            self.shortcuts.append(shortcut)
        self.asr_thread = threading.Thread(target=self.asr_loop, name="kotonoha-asr", daemon=True)
        self.asr_thread.start()
        self.render()

    def say_status(self, text: str):
        self.status.setText(text)

    def render(self):
        for panel in (self.panel, self.vr_panel):
            panel.update_state(self.state, busy=self.busy, recording=self.recording,
                               demo=self.settings.provider == "demo")
        self.provider_badge.setText({"demo": "离线演示", "openai": "GPT · API", "codex": "Codex · CLI"}[self.settings.provider])

    def provider_settings(self) -> ProviderSettings:
        s = self.settings
        return ProviderSettings(provider=s.provider,
                                model=s.codex_model if s.provider == "codex" else s.openai_model,
                                api_key=self.api_key or os.environ.get("OPENAI_API_KEY", ""),
                                base_url=s.openai_base_url, codex_command=s.codex_command)

    def open_settings(self):
        if self.speaker_capture or self.mic_capture:
            self.stop_audio()
        dialog = SettingsDialog(self.settings, self.api_key, self.asr_key, self)
        if dialog.exec():
            demo_mode_changed = (self.settings.provider == "demo") != (dialog.result_settings.provider == "demo")
            self.settings = dialog.result_settings
            self.api_key = dialog.result_api_key
            self.asr_key = dialog.result_asr_key
            self.pending = None
            if demo_mode_changed:
                self.state.reset()
            else:
                self.state.revision += 1
                self.state.candidates.clear()
                self.state.summary_zh = ""
            if self.overlay:
                self.stop_vr()
            self.say_status(("切换演示／真实模式，已清空旧对话。" if demo_mode_changed else "设置已保存。") +
                            "API Key 仅在内存中，关闭程序后需重新输入或使用环境变量。")
            self.render()

    def load_demo(self):
        if self.settings.provider != "demo":
            self.say_status("这是离线演示。请在设置里切换到“离线演示”；真实模型请使用下方文本输入。")
            return
        self.reset_dialogue()
        code = self.demo_language.currentData()
        self.state.receive(DEMO_UTTERANCES[code], code, 1.0)
        self.generate()

    def change_language(self):
        self.state.set_manual_language(self.target.currentData())
        self.pending = None
        self.render()

    def submit_text(self):
        text = self.text_input.text().strip()
        if not text:
            self.say_status("先输入一句话。")
            return
        try:
            if self.input_mode.currentData() == "custom":
                if not self.generate("custom", text):
                    return
            else:
                self.state.receive(text)
                self.generate()
            self.text_input.clear()
        except DialogueError as exc:
            self.say_status(str(exc))

    def generate(self, mode="suggest", custom_text=""):
        try:
            request = self.state.request(mode, custom_text, self.settings.style)
            settings = self.provider_settings()
        except DialogueError as exc:
            self.say_status(str(exc))
            return False
        if self.busy:
            self.pending = (request, settings)
            self.say_status("已更新待处理内容；上一条完成后只处理最新请求。")
            self.render()
            return True
        self.start_generation(request, settings)
        return True

    def start_generation(self, request: GenerationRequest, settings: ProviderSettings):
        self.busy = True
        self.say_status("正在理解上下文并生成双语选项…")
        self.render()

        def run():
            try:
                data = generate_reply(request.payload, settings)
                if not self.closed:
                    self.bridge.generation.emit(request, data)
            except Exception as exc:
                if not self.closed:
                    self.bridge.generation_error.emit(request, str(exc))
        threading.Thread(target=run, name="kotonoha-reply", daemon=True).start()

    def finish_generation(self):
        if self.closed:
            self.busy = False
            self.pending = None
            return
        self.busy = False
        if self.pending:
            request, settings = self.pending
            self.pending = None
            self.start_generation(request, settings)
        self.render()

    def generation_done(self, request, data):
        if self.closed:
            return
        try:
            applied = self.state.apply(request, data)
            self.say_status("双语选项已就绪。选一条后会固定显示。" if applied else "已丢弃过期回复，保留当前对话。")
        except DialogueError as exc:
            self.say_status(str(exc))
        self.finish_generation()

    def generation_failed(self, request, message):
        if self.closed:
            return
        if request.revision == self.state.revision:
            self.say_status(message)
        self.finish_generation()

    def handle_action(self, action: str):
        try:
            if action.startswith("select_"):
                self.state.select(int(action[-1]) - 1)
                self.say_status("提词已固定。读完后点击“已完整说完”；取消不会加入对话。")
            elif action in {"done", "confirm"}:
                self.state.mark_spoken()
                self.say_status("已按你的确认将这句加入实际对话。")
            elif action == "cancel":
                self.state.cancel_pinned()
                self.say_status("已取消提词。")
            elif action == "regenerate":
                self.generate()
            elif action == "dictate_start":
                self.start_dictation()
            elif action == "dictate_stop":
                self.stop_dictation()
            elif action == "toggle_visibility":
                self.toggle_vr()
        except (DialogueError, ValueError) as exc:
            self.say_status(str(exc))
        self.render()

    def refresh_devices(self):
        if self.speaker_capture or self.mic_capture:
            self.stop_audio()
        try:
            self.devices = list_devices()
            self.speaker_device.clear()
            self.mic_device.clear()
            self.speaker_device.addItem("选择耳机/扬声器回环", -1)
            self.mic_device.addItem("选择 VD 麦克风", -1)
            for device in self.devices:
                box = self.speaker_device if device.is_loopback else self.mic_device
                box.addItem(device.name, device.index)
            for box, wanted in [(self.speaker_device, self.settings.speaker_device),
                                (self.mic_device, self.settings.microphone_device)]:
                found = box.findData(wanted)
                if found > 0:
                    box.setCurrentIndex(found)
            self.say_status("设备已刷新。选择你实际听到 VRChat 的播放回环，以及 VD 麦克风。")
        except Exception as exc:
            self.say_status(str(exc))

    def make_capture(self, index, source):
        epoch = self.audio_epoch
        return AudioCapture(index, source,
                            on_clip=lambda clip: self.queue_clip(epoch, clip),
                            on_level=lambda level: self.bridge.level.emit(epoch, level),
                            on_error=lambda message: self.bridge.audio_error.emit(epoch, message),
                            threshold=self.settings.energy_threshold,
                            pause_ms=self.settings.silence_ms)

    def toggle_listening(self):
        if self.speaker_capture:
            self.stop_audio()
            return
        if self.settings.provider == "demo":
            self.say_status("实时对话请先在设置里连接 GPT 或 Codex；演示按钮使用固定样例。")
            return
        index = self.speaker_device.currentData()
        if index is None or index < 0:
            self.say_status("请刷新设备并选择耳机/扬声器回环。")
            return
        try:
            capture = self.make_capture(index, "speaker")
            capture.start()
            self.speaker_capture = capture
            self.settings.speaker_device = index
            self.speaker_device.setEnabled(False)
            self.listen_button.setText("停止收听")
            self.say_status("正在收听。语音会在停顿后识别；背景音乐和多人叠话可能影响结果。")
        except Exception as exc:
            self.say_status(str(exc))

    def start_dictation(self):
        if self.recording:
            return
        if not self.state.reply_language:
            self.say_status("先确定对方语言，或在左侧手动选择。")
            return
        if self.settings.provider == "demo":
            self.input_mode.setCurrentIndex(1)
            self.text_input.setText("我想在这里再待一会儿")
            self.say_status("演示已填入一句中文。点击下方“生成”查看自拟回复。")
            return
        index = self.mic_device.currentData()
        if index is None or index < 0:
            self.say_status("请先选择 VD 麦克风，并在游戏内静音。")
            return
        try:
            if self.mic_capture is None:
                self.mic_capture = self.make_capture(index, "mic")
                self.mic_capture.start()
                self.mic_device.setEnabled(False)
            self.mic_capture.set_enabled(True)
            self.recording = True
            self.settings.microphone_device = index
            self.say_status("正在录你的中文；请保持游戏内静音。松开后生成对方语言的提词。")
        except Exception as exc:
            self.say_status(str(exc))
            if self.mic_capture:
                self.mic_capture.stop()
            self.mic_capture = None
            self.mic_device.setEnabled(True)
        self.render()

    def stop_dictation(self):
        if self.mic_capture and self.recording:
            self.mic_capture.set_enabled(False)
            self.recording = False
            self.say_status("中文录音结束，正在识别…")
        self.render()

    def stop_audio(self):
        self.audio_epoch += 1
        for capture in (self.speaker_capture, self.mic_capture):
            if capture:
                capture.stop()
        self.speaker_capture = self.mic_capture = None
        self.speaker_device.setEnabled(True)
        self.mic_device.setEnabled(True)
        self.recording = False
        self.listen_button.setText("开始收听")
        self.meter.setValue(0)
        self.say_status("已停止收音。")
        self.render()

    def queue_clip(self, epoch: int, clip: AudioClip):
        if self.closed or epoch != self.audio_epoch:
            return
        try:
            self.asr_queue.put_nowait((epoch, clip, replace(self.settings),
                                      self.asr_key or self.api_key or os.environ.get("OPENAI_API_KEY", "")))
        except queue.Full:
            self.bridge.audio_error.emit(epoch, "识别暂时跟不上语速，已丢弃新片段。可换更小的本地模型或云端识别。")

    def asr_loop(self):
        while not self.asr_stopping.is_set():
            try:
                epoch, clip, settings, key = self.asr_queue.get(timeout=.25)
            except queue.Empty:
                continue
            if epoch != self.audio_epoch:
                continue
            try:
                transcript = transcribe_clip(clip, backend=settings.asr_backend,
                                             model=settings.local_asr_model if settings.asr_backend == "local" else settings.cloud_asr_model,
                                             api_key=key, base_url=settings.asr_base_url,
                                             language_hint="zh" if clip.source == "mic" else None)
                if not self.closed and epoch == self.audio_epoch:
                    self.bridge.transcript.emit(epoch, clip.source, transcript)
            except Exception as exc:
                if not self.closed:
                    self.bridge.audio_error.emit(epoch, str(exc))

    def transcript_ready(self, epoch, source, transcript):
        if self.closed or epoch != self.audio_epoch or not transcript.text.strip():
            return
        try:
            if source == "mic":
                self.generate("custom", transcript.text)
            else:
                self.state.receive(transcript.text, transcript.language, transcript.confidence)
                if self.settings.auto_suggest:
                    self.generate()
                else:
                    self.say_status("已识别对方语音。点击“重新建议”生成翻译和回复。")
                    self.render()
        except DialogueError as exc:
            self.say_status(str(exc))

    def audio_failed(self, epoch, message):
        if epoch == self.audio_epoch:
            self.say_status(message)

    def audio_level(self, epoch, level):
        if epoch == self.audio_epoch:
            self.meter.setValue(min(100, int(float(level) * 400)))

    def preview_overlay(self):
        self.vr_panel.show()
        self.vr_panel.raise_()

    def toggle_vr(self):
        if self.overlay:
            self.stop_vr()
            return
        try:
            from .overlay import SteamVROverlay
            self.overlay = SteamVROverlay(self.vr_panel, self.handle_action,
                                          width_m=self.settings.overlay_width,
                                          distance_m=self.settings.overlay_distance)
            self.overlay.start()
            self.overlay.set_placement(self.settings.placement)
            self.overlay.set_interactive(self.interactive.isChecked())
            self.vr_timer.start()
            self.vr_button.setText("关闭 VR 浮窗")
            self.say_status("VR 浮窗已开启。手柄点选开启时，SteamVR 可能暂时接管游戏输入。")
        except Exception as exc:
            self.stop_vr()
            self.say_status(str(exc))

    def set_interactive(self, enabled):
        if self.overlay:
            try:
                self.overlay.set_interactive(enabled)
            except Exception as exc:
                self.say_status(str(exc))

    def tick_vr(self):
        if not self.overlay:
            return
        try:
            self.overlay.update_frame()
            self.overlay.poll_input()
            warning = self.overlay.last_warning
            if warning and warning != self.overlay_warning:
                self.say_status(warning)
            self.overlay_warning = warning
        except Exception as exc:
            self.stop_vr()
            self.say_status(str(exc))

    def stop_vr(self):
        self.vr_timer.stop()
        if self.overlay:
            self.overlay.stop()
            self.overlay = None
        self.overlay_warning = ""
        self.vr_button.setText("显示 VR 浮窗")

    def show_history(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("本次实际对话 · 仅内存")
        dialog.resize(640, 450)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText("\n\n".join(f"{'对方' if t.role == 'other' else '我（已确认说完）'} · {language_name(t.language)}\n{t.text}" for t in self.state.history)
                          or "还没有对话。未说出的候选不会进入这里。")
        layout.addWidget(text)
        layout.addWidget(button("关闭", dialog.accept))
        dialog.exec()

    def reset_dialogue(self):
        # In-flight recognizers may finish later; advance the audio epoch before
        # clearing context so an old clip cannot resurrect the cleared session.
        self.stop_audio()
        self.state.reset()
        self.pending = None
        self.say_status("本次对话已清空。")
        self.render()

    def closeEvent(self, event: QCloseEvent):
        self.closed = True
        self.busy = False
        self.state.revision += 1
        self.asr_stopping.set()
        self.pending = None
        self.stop_audio()
        self.stop_vr()
        self.vr_panel.close()
        event.accept()
