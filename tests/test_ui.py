"""Real Qt UI integration, with fixed offline replies and no hardware/API calls."""

from __future__ import annotations

import json
import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtCore import QCoreApplication, QEvent, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QLabel, QLineEdit, QPushButton

from kotonoha_vr import ui
from kotonoha_vr.audio import AudioError, Transcription
from kotonoha_vr.overlay import OverlayError, SteamVROverlay
from kotonoha_vr.providers import DEMO_SAMPLES, DEMO_UTTERANCES, generate_reply
from kotonoha_vr.settings import Settings, settings_path


def spin_until(app, predicate, *, timeout=2.0):
    """Pump queued cross-thread Qt signals without blocking the main thread."""
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(5)
    app.processEvents()
    assert predicate(), "UI condition did not settle within two seconds"


@pytest.fixture(scope="session")
def qt_app():
    application = QApplication.instance() or QApplication([])
    application.setQuitOnLastWindowClosed(False)
    return application


@pytest.fixture
def window(qt_app, monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))

    def no_hardware_probe():
        raise AudioError("测试环境没有音频设备")

    monkeypatch.setattr(ui, "list_devices", no_hardware_probe)
    instance = ui.MainWindow(Settings(provider="demo"))
    yield instance
    instance.close()
    spin_until(qt_app, lambda: not instance.asr_thread.is_alive())
    instance.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qt_app.processEvents()


def wait_ready(qt_app, window, count=3):
    spin_until(qt_app, lambda: not window.busy and len(window.state.candidates) == count)


def click_action(panel, action):
    controls = [b for b in panel.findChildren(QPushButton) if b.property("vr_action") == action]
    assert len(controls) == 1
    controls[0].click()


def test_start_without_devices_and_failed_vr_start_are_recoverable(window, monkeypatch):
    assert not window.busy
    assert window.speaker_capture is None and window.mic_capture is None
    assert window.overlay is None
    assert window.state.history == []
    assert not window.panel.cards[0][4].isEnabled()
    window.refresh_devices()
    assert "没有音频设备" in window.status.text()
    window.toggle_listening()
    assert "演示" in window.status.text()
    closed = []

    class UnavailableOverlay:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise OverlayError("测试环境没有 SteamVR 头显")

        def stop(self):
            closed.append(True)

    monkeypatch.setattr("kotonoha_vr.overlay.SteamVROverlay", UnavailableOverlay)
    window.toggle_vr()
    assert window.overlay is None
    assert not window.vr_timer.isActive()
    assert "没有 SteamVR 头显" in window.status.text()
    assert closed == [True]


@pytest.mark.parametrize("language", ["ja", "en", "ko"])
def test_offline_demo_has_three_bilingual_candidates(qt_app, window, language):
    window.demo_language.setCurrentIndex(window.demo_language.findData(language))
    window.load_demo()
    wait_ready(qt_app, window)
    assert window.state.reply_language == language
    assert window.panel.original.text() == DEMO_UTTERANCES[language]
    assert "一起" in window.panel.translation.text()
    for panel in (window.panel, window.vr_panel):
        for (_, intent, foreign, meaning, choose), candidate in zip(panel.cards, window.state.candidates):
            assert choose.isEnabled()
            assert candidate.intent_zh in intent.text()
            assert foreign.text() == candidate.text
            assert meaning.text() == candidate.meaning_zh
            assert foreign.text() and meaning.text()
    assert all(turn.role == "other" for turn in window.state.history)


def test_pinned_reply_survives_new_speech_and_only_done_commits_history(qt_app, window):
    window.load_demo()
    wait_ready(qt_app, window)
    window.panel.cards[1][4].click()
    pinned = window.state.pinned
    assert pinned is not None
    assert window.panel.pin_foreign.text() == pinned.candidate.text
    assert window.panel.pin_meaning.text() == pinned.candidate.meaning_zh
    assert len(window.state.history) == 1
    assert all(turn.role == "other" for turn in window.state.history)

    window.transcript_ready(window.audio_epoch, "speaker", Transcription(
        DEMO_SAMPLES["en"]["greeting"], "en", .99
    ))
    wait_ready(qt_app, window)
    assert window.state.reply_language == "en"
    assert window.state.pinned == pinned
    assert len(window.state.history) == 2
    for panel in (window.panel, window.vr_panel):
        assert panel.pin_foreign.text() == pinned.candidate.text
        assert panel.pin_meaning.text() == pinned.candidate.meaning_zh
        assert "日语" in panel.pin_heading.text()
        assert not panel.pin.isHidden()
        assert panel.candidate_container.isHidden()

    click_action(window.panel, "done")
    assert window.state.pinned is None
    spoken = [turn for turn in window.state.history if turn.role == "self"]
    assert len(spoken) == 1
    assert spoken[0].text == pinned.candidate.text
    assert spoken[0].language == pinned.language == "ja"
    assert window.panel.pin.isHidden()
    assert window.vr_panel.pin.isHidden()


def test_cancel_and_unselected_candidates_never_enter_history(qt_app, window):
    window.load_demo()
    wait_ready(qt_app, window)
    history_before = list(window.state.history)
    window.panel.cards[0][4].click()
    click_action(window.panel, "cancel")
    assert window.state.pinned is None
    assert window.state.history == history_before
    assert all(turn.role == "other" for turn in window.state.history)


def test_own_chinese_uses_other_language_without_fabricating_a_turn(qt_app, window):
    window.load_demo()
    wait_ready(qt_app, window)
    window.input_mode.setCurrentIndex(window.input_mode.findData("custom"))
    window.text_input.setText("我想在这里再待一会儿")
    window.submit_text()
    wait_ready(qt_app, window, count=1)
    assert window.state.reply_language == "ja"
    assert len(window.state.history) == 1
    assert window.state.candidates[0].meaning_zh == "我想在这里再待一会儿"
    assert window.state.candidates[0].text == "ここにもう少しいたいです。"
    window.panel.cards[0][4].click()
    assert window.state.pinned.language == "ja"
    assert window.state.history[0].role == "other"


def test_api_keys_are_masked_memory_only_and_absent_from_vr_panel(qt_app, window, monkeypatch):
    text_key = "test-secret-model-key-73142"
    speech_key = "test-secret-speech-key-92553"

    def accept_test_settings(dialog):
        dialog.api_key.setText(text_key)
        dialog.asr_key.setText(speech_key)
        assert dialog.api_key.echoMode() == QLineEdit.EchoMode.Password
        assert dialog.asr_key.echoMode() == QLineEdit.EchoMode.Password
        dialog.accept_changes()
        return dialog.result()

    monkeypatch.setattr(ui.SettingsDialog, "exec", accept_test_settings)
    window.open_settings()
    assert window.api_key == text_key
    assert window.asr_key == speech_key
    assert window.provider_settings().api_key == text_key
    serialized = settings_path().read_text(encoding="utf-8")
    data = json.loads(serialized)
    assert "api_key" not in data and "asr_key" not in data
    assert text_key not in serialized and speech_key not in serialized
    window.load_demo()
    wait_ready(qt_app, window)
    vr_text = "\n".join(
        widget.text() for widget in window.vr_panel.findChildren(QLabel)
    )
    assert text_key not in vr_text and speech_key not in vr_text
    assert window.vr_panel.findChildren(QLineEdit) == []
    assert text_key not in repr(window.provider_settings())


def test_latest_pending_utterance_replaces_older_pending_work(qt_app, window, monkeypatch):
    gate = threading.Event()
    entered = threading.Event()
    requests = []

    def slow_first(payload, settings):
        requests.append(dict(payload))
        if len(requests) == 1:
            entered.set()
            assert gate.wait(1.5), "test must release the first offline generation"
        return generate_reply(payload, settings)

    monkeypatch.setattr(ui, "generate_reply", slow_first)
    try:
        window.load_demo()
        spin_until(qt_app, entered.is_set)
        window.transcript_ready(window.audio_epoch, "speaker", Transcription(DEMO_UTTERANCES["en"], "en", 1.0))
        window.transcript_ready(window.audio_epoch, "speaker", Transcription(DEMO_UTTERANCES["ko"], "ko", 1.0))
        assert window.pending is not None
        gate.set()
        wait_ready(qt_app, window)
        assert [r["latest_text"] for r in requests] == [DEMO_UTTERANCES["ja"], DEMO_UTTERANCES["ko"]]
        assert window.state.reply_language == "ko"
        assert window.state.latest_other.text == DEMO_UTTERANCES["ko"]
        assert window.pending is None
        assert "같이" in window.state.candidates[0].text
    finally:
        gate.set()


def test_language_change_invalidates_running_and_pending_result(qt_app, window, monkeypatch):
    gate = threading.Event()
    entered = threading.Event()
    requests = []

    def slow_first(payload, settings):
        requests.append(dict(payload))
        if len(requests) == 1:
            entered.set()
            assert gate.wait(1.5)
        return generate_reply(payload, settings)

    monkeypatch.setattr(ui, "generate_reply", slow_first)
    try:
        window.load_demo()
        spin_until(qt_app, entered.is_set)
        window.transcript_ready(window.audio_epoch, "speaker", Transcription(DEMO_UTTERANCES["en"], "en", 1.0))
        assert window.pending is not None
        window.target.setCurrentIndex(window.target.findData("ko"))
        assert window.pending is None
        assert window.state.candidates == []
        window.generate()
        gate.set()
        wait_ready(qt_app, window)
        assert len(requests) == 2
        assert requests[-1]["source_language"] == "en"
        assert requests[-1]["reply_language"] == "ko"
        assert window.state.reply_language == "ko"
        assert window.state._candidate_language == "ko"
        assert window.panel.original.text() == DEMO_UTTERANCES["en"]
        assert "韩语" in window.panel.language.text()
    finally:
        gate.set()


def test_reset_and_late_audio_do_not_restore_discarded_dialogue(qt_app, window):
    window.load_demo()
    wait_ready(qt_app, window)
    old_epoch = window.audio_epoch
    window.reset_dialogue()
    window.transcript_ready(old_epoch, "speaker", Transcription(DEMO_UTTERANCES["en"], "en", .99))
    assert window.state.history == []
    assert window.state.candidates == []
    assert window.state.pinned is None


def test_close_stops_idle_audio_worker_timer_and_preview(qt_app, window):
    window.load_demo()
    wait_ready(qt_app, window)
    window.preview_overlay()
    assert window.vr_panel.isVisible()
    window.vr_timer.start()
    window.close()
    spin_until(qt_app, lambda: not window.asr_thread.is_alive())
    assert window.closed
    assert window.asr_stopping.is_set()
    assert not window.vr_timer.isActive()
    assert not window.vr_panel.isVisible()
    assert window.speaker_capture is None and window.mic_capture is None
    assert not window.recording and window.pending is None


def test_close_during_generation_suppresses_late_completion(qt_app, window, monkeypatch):
    gate = threading.Event()
    entered = threading.Event()
    finished = threading.Event()
    delivered = []

    def slow_reply(payload, settings):
        entered.set()
        try:
            assert gate.wait(1.5)
            return generate_reply(payload, settings)
        finally:
            finished.set()

    monkeypatch.setattr(ui, "generate_reply", slow_reply)
    window.bridge.generation.connect(lambda *_: delivered.append(True))
    try:
        window.load_demo()
        spin_until(qt_app, entered.is_set)
        window.close()
        gate.set()
        spin_until(qt_app, lambda: finished.is_set() and not window.asr_thread.is_alive())
        qt_app.processEvents()
        assert delivered == []
        assert window.state.candidates == []
    finally:
        gate.set()


def test_completion_already_queued_before_close_cannot_mutate_closed_window(qt_app, window, monkeypatch):
    gate = threading.Event()
    entered = threading.Event()
    emitted = threading.Event()

    def slow_reply(payload, settings):
        entered.set()
        assert gate.wait(1.5)
        return generate_reply(payload, settings)

    monkeypatch.setattr(ui, "generate_reply", slow_reply)
    # Observe emission from the worker, while deliberately leaving the UI's
    # normal queued delivery pending until after closeEvent has run.
    window.bridge.generation.connect(
        lambda *_: emitted.set(), Qt.ConnectionType.DirectConnection
    )
    try:
        window.load_demo()
        spin_until(qt_app, entered.is_set)
        gate.set()
        assert emitted.wait(.75)
        window.close()
        qt_app.processEvents()
        assert window.closed
        assert window.state.candidates == [], "queued completion changed an already closed window"
        assert window.pending is None
    finally:
        gate.set()


def test_vr_panel_controls_are_all_reachable_by_overlay_pointer(qt_app, window):
    window.load_demo()
    wait_ready(qt_app, window)
    panel = window.vr_panel
    overlay = SteamVROverlay(panel, window.handle_action)
    panel.show()
    qt_app.processEvents()
    for pinned in (False, True):
        if pinned:
            window.panel.cards[0][4].click()
            qt_app.processEvents()
        panel.grab()  # activate the actual Qt layouts used by overlay rendering
        overlay._mouse_size = (panel.width(), panel.height())
        for control in panel.findChildren(QPushButton):
            action = control.property("vr_action")
            if not action or not control.isVisibleTo(panel) or not control.isEnabled():
                continue
            center = control.mapTo(panel, control.rect().center())
            hit = overlay._target_at(center.x(), panel.height() - center.y())
            assert hit is not None, f"VR control {action!r} cannot be reached by the pointer"
            assert hit[1] == action


def test_overlong_typed_turn_is_reported_without_uncaught_exception(window):
    window.text_input.setText("x" * 12001)
    window.submit_text()
    assert "过长" in window.status.text()
    assert window.state.history == []
    assert not window.busy
def test_switching_from_demo_to_real_model_clears_simulated_context(qt_app, window, monkeypatch):
    window.load_demo()
    wait_ready(qt_app, window)
    window.panel.cards[0][4].click()

    def connect_real_provider(dialog):
        dialog.provider.setCurrentIndex(dialog.provider.findData("openai"))
        dialog.accept_changes()
        return dialog.result()

    monkeypatch.setattr(ui.SettingsDialog, "exec", connect_real_provider)
    window.open_settings()
    assert window.settings.provider == "openai"
    assert not window.state.history
    assert not window.state.candidates
    assert window.state.pinned is None
    assert window.state.reply_language is None
    assert "已清空旧对话" in window.status.text()


def test_own_text_is_preserved_when_target_language_is_missing(window):
    window.input_mode.setCurrentIndex(window.input_mode.findData("custom"))
    window.text_input.setText("我想在这里再待一会儿")
    window.submit_text()
    assert window.text_input.text() == "我想在这里再待一会儿"
    assert not window.busy
    assert "语言" in window.status.text()
