"""OpenVR boundary tests using real Qt rendering, without a VR runtime/headset."""

from __future__ import annotations

import ctypes
import os
import sys
from types import SimpleNamespace

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QWidget

from kotonoha_vr.overlay import OverlayError, SteamVROverlay


class Matrix34(ctypes.Structure):
    _fields_ = [("m", (ctypes.c_float * 4) * 3)]


class Vector2(ctypes.Structure):
    _fields_ = [("v", ctypes.c_float * 2)]


def event(event_type=0, x=0.0, y=0.0, button=1):
    return SimpleNamespace(
        eventType=event_type,
        data=SimpleNamespace(mouse=SimpleNamespace(x=x, y=y, button=button)),
    )


class FakeOverlayAPI:
    def __init__(self):
        self.events = []
        self.created = []
        self.destroyed = []
        self.hidden = []
        self.shown = []
        self.raw = []
        self.transforms = []
        self.widths = []
        self.flags = []
        self.scales = []
        self.input_methods = []
        self.fail_raw = False
        self.fail_create = False
        self.fail_destroy = False
        self.poll_count = 0

    def createOverlay(self, key, title):
        if self.fail_create:
            raise RuntimeError("create failed")
        handle = 100 + len(self.created)
        self.created.append((handle, key, title))
        return handle

    def destroyOverlay(self, handle):
        self.destroyed.append(handle)
        if self.fail_destroy:
            raise RuntimeError("runtime gone")

    def hideOverlay(self, handle):
        self.hidden.append(handle)

    def showOverlay(self, handle):
        self.shown.append(handle)

    def setOverlayFlag(self, handle, flag, enabled):
        self.flags.append((handle, flag, enabled))

    def setOverlayInputMethod(self, handle, method):
        self.input_methods.append((handle, method))

    def setOverlayMouseScale(self, handle, scale):
        self.scales.append((handle, tuple(scale.v)))

    def setOverlayWidthInMeters(self, handle, width):
        self.widths.append((handle, width))

    def setOverlayTransformTrackedDeviceRelative(self, handle, device, transform):
        self.transforms.append(
            (handle, device, tuple(tuple(row) for row in transform.m))
        )

    def setOverlayRaw(self, handle, buffer, width, height, bytes_per_pixel):
        if self.fail_raw:
            raise RuntimeError("raw failed")
        assert isinstance(buffer, ctypes.Array), "pyopenvr adds byref itself"
        assert bytes_per_pixel == 4
        data = ctypes.string_at(ctypes.byref(buffer), width * height * bytes_per_pixel)
        self.raw.append((handle, data, width, height, bytes_per_pixel))

    def pollNextOverlayEvent(self, handle, current_event):
        self.poll_count += 1
        if not self.events:
            # This false-but-nonempty tuple is a critical pyopenvr quirk.
            return False, current_event
        return True, self.events.pop(0)


class FakeOpenVR:
    VRApplication_Overlay = 2
    VROverlayInputMethod_None = 0
    VROverlayInputMethod_Mouse = 1
    VROverlayFlags_MakeOverlaysInteractiveIfVisible = 65536
    VROverlayFlags_VisibleInDashboard = 32768
    k_unTrackedDeviceIndex_Hmd = 0
    k_unTrackedDeviceIndexInvalid = 0xFFFFFFFF
    TrackedControllerRole_LeftHand = 1
    TrackedControllerRole_RightHand = 2
    VREvent_MouseMove = 300
    VREvent_MouseButtonDown = 301
    VREvent_MouseButtonUp = 302
    VREvent_FocusLeave = 304
    VREvent_OverlayHidden = 501
    VREvent_Quit = 700
    VRMouseButton_Left = 1
    HmdMatrix34_t = Matrix34
    HmdVector2_t = Vector2
    VREvent_t = staticmethod(event)

    def __init__(self):
        self.api = FakeOverlayAPI()
        self.init_calls = []
        self.shutdown_calls = 0
        self.roles = {1: 4, 2: 7}  # Deliberately not device IDs 1 and 2.
        self.fail_init = False
        self.installed = True

    def isRuntimeInstalled(self):
        return self.installed

    def init(self, application_type):
        self.init_calls.append(application_type)
        if self.fail_init:
            raise RuntimeError("HmdNotFound")
        return self

    def shutdown(self):
        self.shutdown_calls += 1

    def VROverlay(self):
        return self.api

    def getTrackedDeviceIndexForControllerRole(self, role):
        return self.roles.get(role, self.k_unTrackedDeviceIndexInvalid)


@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance() or QApplication([])
    return app


@pytest.fixture
def runtime(monkeypatch):
    module = FakeOpenVR()
    monkeypatch.setitem(sys.modules, "openvr", module)
    return module


@pytest.fixture
def panel(qapp):
    widget = QWidget()
    widget.resize(400, 200)
    widget.setStyleSheet("QWidget {background:#203040;} QPushButton {background:#607080;}")
    button = QPushButton("中文 / Foreign reply", widget)
    button.setGeometry(20, 10, 180, 50)
    button.setProperty("vr_action", "select_1")
    child_label = QLabel("text child", button)
    child_label.setGeometry(5, 5, 60, 20)
    dictate = QPushButton("hold to dictate", widget)
    dictate.setGeometry(220, 120, 160, 60)
    dictate.setProperty("vr_action", "dictate_start")
    widget.ensurePolished()
    yield widget, button, dictate
    widget.close()
    widget.deleteLater()
    qapp.processEvents()


@pytest.fixture
def overlay(panel, runtime):
    actions = []
    instance = SteamVROverlay(panel[0], actions.append)
    yield instance, actions
    instance.stop()


def queue_click(runtime, x, y):
    runtime.api.events.extend([
        event(runtime.VREvent_MouseButtonDown, x, y),
        event(runtime.VREvent_MouseButtonUp, x, y),
    ])


def test_hidden_qt_widget_renders_correct_rgba_and_head_transform(overlay, panel, runtime):
    instance, _ = overlay
    assert not panel[0].isVisible()  # off-screen render need not expose a window
    instance.start()
    assert instance.is_running and instance.is_visible
    assert runtime.init_calls == [runtime.VRApplication_Overlay]
    _, data, width, height, depth = runtime.api.raw[-1]
    assert (width, height, depth) == (400, 200, 4)
    assert len(data) == 400 * 200 * 4
    # A corner is plain background, so this also catches BGRA/RGBA confusion.
    assert data[0:4] == bytes((0x20, 0x30, 0x40, 255))
    assert bytes(instance._pixel_buffer) == data
    assert runtime.api.scales[-1][1] == (400.0, 200.0)
    _, device, matrix = runtime.api.transforms[-1]
    assert device == 0
    assert matrix[0][0] == matrix[1][1] == matrix[2][2] == 1.0
    assert matrix[2][3] == pytest.approx(-1.4)
    assert matrix[1][3] < 0


def test_upload_skips_unchanged_frame_and_handles_resize(overlay, panel, runtime):
    instance, _ = overlay
    instance.start()
    count = len(runtime.api.raw)
    instance.update_frame()
    assert len(runtime.api.raw) == count
    panel[0].resize(500, 220)
    instance.update_frame()
    assert runtime.api.raw[-1][2:4] == (500, 220)
    assert runtime.api.scales[-1][1] == (500.0, 220.0)


def test_large_widget_texture_is_bounded_but_mouse_scale_stays_logical(overlay, panel, runtime):
    instance, _ = overlay
    panel[0].resize(2560, 1440)
    instance.start()
    assert runtime.api.raw[-1][2:4] == (1280, 720)
    assert runtime.api.scales[-1][1] == (2560.0, 1440.0)


def test_pointer_click_flips_y_finds_nested_button_and_dispatches_once(overlay, panel, runtime):
    instance, actions = overlay
    qt_clicks = []
    panel[1].clicked.connect(lambda: qt_clicks.append("clicked"))
    instance.start()
    # Child label is at Qt (25,15); VR Y is 200 - 20.
    queue_click(runtime, 35, 180)
    instance.poll_input()
    assert actions == ["select_1"]
    assert qt_clicks == []
    assert not panel[1].isDown()
    assert runtime.api.poll_count == 3  # exit on (False, event), not on tuple truthiness


def test_click_is_cancelled_outside_or_when_disabled(overlay, panel, runtime):
    instance, actions = overlay
    instance.start()
    runtime.api.events.extend([
        event(runtime.VREvent_MouseButtonDown, 100, 170),
        event(runtime.VREvent_MouseButtonUp, 399, 195),
    ])
    instance.poll_input()
    assert actions == []
    panel[1].setEnabled(False)
    queue_click(runtime, 100, 170)
    instance.poll_input()
    assert actions == []


@pytest.mark.parametrize("x,y", [(-1, 170), (401, 170), (20, 201), (float("nan"), 10)])
def test_invalid_pointer_coordinates_never_trigger_action(overlay, runtime, x, y):
    instance, actions = overlay
    instance.start()
    queue_click(runtime, x, y)
    instance.poll_input()
    assert actions == []


def test_dictate_hold_stops_on_release_outside_and_on_shutdown(overlay, runtime):
    instance, actions = overlay
    instance.start()
    runtime.api.events.append(event(runtime.VREvent_MouseButtonDown, 300, 50))
    instance.poll_input()
    assert actions == ["dictate_start"]
    runtime.api.events.append(event(runtime.VREvent_MouseButtonUp, 399, 195))
    instance.poll_input()
    assert actions == ["dictate_start", "dictate_stop"]
    runtime.api.events.append(event(runtime.VREvent_MouseButtonDown, 300, 50))
    instance.poll_input()
    instance.stop()
    instance.stop()
    assert actions[-2:] == ["dictate_start", "dictate_stop"]
    assert runtime.shutdown_calls == 1


def test_passive_panel_does_not_capture_input(overlay, runtime):
    instance, actions = overlay
    instance.set_interactive(False)
    instance.start()
    assert runtime.api.input_methods[-1][1] == runtime.VROverlayInputMethod_None
    queue_click(runtime, 100, 170)
    instance.poll_input()
    assert actions == []
    instance.set_interactive(True)
    queue_click(runtime, 100, 170)
    instance.poll_input()
    assert actions == ["select_1"]
    instance.toggle_visibility()
    assert not instance.is_visible
    instance.toggle_visibility()
    assert instance.is_visible


def test_left_right_roles_and_hand_reconnect(overlay, runtime):
    instance, _ = overlay
    instance.start()
    instance.set_placement("left")
    assert instance.placement == "left"
    assert runtime.api.transforms[-1][1] == 4
    assert runtime.api.widths[-1][1] == 0.55
    matrix = runtime.api.transforms[-1][2]
    assert matrix[1][2] > 0 and matrix[2][1] < 0
    instance.set_placement("right")
    assert runtime.api.transforms[-1][1] == 7
    del runtime.roles[2]
    instance._next_anchor_check = 0
    instance.poll_input()
    assert not instance.is_visible
    assert "右手" in instance.last_warning
    runtime.roles[2] = 9
    instance._next_anchor_check = 0
    instance.poll_input()
    assert runtime.api.transforms[-1][1] == 9
    assert instance.is_visible and not instance.last_warning


def test_unavailable_hand_does_not_change_working_placement(overlay, runtime):
    instance, _ = overlay
    instance.start()
    del runtime.roles[1]
    with pytest.raises(OverlayError, match="左手"):
        instance.set_placement("left")
    assert instance.placement == "head"
    assert instance.is_running


def test_two_panels_share_runtime_until_last_stop(panel, runtime):
    first = SteamVROverlay(panel[0], lambda _: None)
    second = SteamVROverlay(panel[0], lambda _: None)
    try:
        first.start()
        first.start()
        second.start()
        assert len(runtime.init_calls) == 1
        assert runtime.api.created[0][1] != runtime.api.created[1][1]
        first.stop()
        assert runtime.shutdown_calls == 0
        assert second.is_running
        second.stop()
        assert runtime.shutdown_calls == 1
    finally:
        first.stop()
        second.stop()


@pytest.mark.parametrize("failure", ["create", "raw"])
def test_partial_start_failure_releases_runtime(overlay, runtime, failure):
    instance, _ = overlay
    setattr(runtime.api, f"fail_{failure}", True)
    with pytest.raises(OverlayError):
        instance.start()
    assert not instance.is_running
    assert runtime.shutdown_calls == 1
    if failure == "raw":
        assert runtime.api.destroyed == [100]
    # Failure must not poison a subsequent startup.
    setattr(runtime.api, f"fail_{failure}", False)
    instance.start()
    assert instance.is_running


def test_init_failure_and_missing_runtime_are_clear_and_retryable(overlay, runtime):
    instance, _ = overlay
    runtime.fail_init = True
    with pytest.raises(OverlayError, match="连接头显"):
        instance.start()
    assert not instance.is_running
    assert runtime.shutdown_calls == 1
    runtime.fail_init = False
    runtime.installed = False
    with pytest.raises(OverlayError, match="未找到 SteamVR"):
        instance.start()
    assert runtime.shutdown_calls == 1
    runtime.installed = True
    instance.start()
    assert instance.is_running


def test_missing_openvr_dependency_is_lazy_and_explained(panel, monkeypatch):
    instance = SteamVROverlay(panel[0], lambda _: None)
    monkeypatch.setitem(sys.modules, "openvr", None)
    with pytest.raises(OverlayError, match="未能加载 OpenVR"):
        instance.start()
    assert not instance.is_running


def test_runtime_quit_and_teardown_failure_still_cleanup(overlay, runtime):
    instance, _ = overlay
    instance.start()
    runtime.api.fail_destroy = True
    runtime.api.events.append(event(runtime.VREvent_Quit))
    with pytest.raises(OverlayError, match="SteamVR 已退出"):
        instance.poll_input()
    assert not instance.is_running
    assert runtime.shutdown_calls == 1


def test_startup_failure_of_second_panel_does_not_shutdown_first(panel, runtime):
    first = SteamVROverlay(panel[0], lambda _: None)
    second = SteamVROverlay(panel[0], lambda _: None)
    try:
        first.start()
        runtime.api.fail_create = True
        with pytest.raises(OverlayError):
            second.start()
        assert first.is_running
        assert runtime.shutdown_calls == 0
    finally:
        second.stop()
        first.stop()
    assert runtime.shutdown_calls == 1
