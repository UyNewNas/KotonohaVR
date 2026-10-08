"""Render a small Qt panel as a regular, in-scene SteamVR overlay.

OpenVR is imported only when the user starts the VR view.  No game process is
injected or modified.  All methods must be called on the widget's Qt thread;
the application owns the timers and must call ``stop`` before destroying Qt.

Interface details were checked against pyopenvr 2.12.1401 and Valve's header:
https://github.com/cmbruns/pyopenvr
https://github.com/ValveSoftware/openvr/blob/master/headers/openvr.h

In that binding ``setOverlayRaw`` applies ctypes.byref itself, so it needs a
ctypes *array*, not an already-created void pointer.  ``pollNextOverlayEvent``
returns ``(available, event)``.  OpenVR mouse Y coordinates start at the bottom
of the texture; Qt's start at the top.
"""

from __future__ import annotations

import ctypes
import importlib
import logging
import math
import os
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from PySide6.QtCore import QPoint, QThread, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QAbstractButton, QWidget


LOG = logging.getLogger(__name__)
Placement = Literal["head", "left", "right"]
_ACTIONS = frozenset(
    {
        "select_1", "select_2", "select_3", "next", "previous", "confirm",
        "done", "cancel", "regenerate", "dictate_start", "dictate_stop", "toggle_visibility",
    }
)


class OverlayError(RuntimeError):
    """A recoverable VR error that can be shown in the desktop status area."""


@dataclass
class _RuntimeSession:
    module: Any
    system: Any
    users: int = 1


_runtime_lock = threading.RLock()
_runtime_session: _RuntimeSession | None = None


def _acquire_runtime(module: Any) -> _RuntimeSession:
    """Share one init/shutdown pair across this application's overlay objects.

    This module is the application's OpenVR lifetime owner.  Other components
    must not call openvr.init/shutdown independently while it is active.
    """
    global _runtime_session
    with _runtime_lock:
        if _runtime_session is not None:
            if _runtime_session.module is not module:
                raise OverlayError("此进程已有另一个 OpenVR 会话，请先关闭现有浮窗。")
            _runtime_session.users += 1
            return _runtime_session
        if hasattr(module, "isRuntimeInstalled") and not module.isRuntimeInstalled():
            raise OverlayError("未找到 SteamVR。请安装并启动 SteamVR，再打开 VR 浮窗。")
        try:
            system = module.init(module.VRApplication_Overlay)
        except Exception as exc:
            # init can fail after its native init step, while acquiring the
            # IVRSystem interface. No other session owned by us exists here.
            try:
                module.shutdown()
            except Exception:
                LOG.debug("OpenVR cleanup after failed init", exc_info=True)
            raise OverlayError(
                f"无法连接 SteamVR。请先连接头显并启动 SteamVR，再重试。详情：{exc}"
            ) from exc
        _runtime_session = _RuntimeSession(module=module, system=system)
        return _runtime_session


def _release_runtime(session: _RuntimeSession) -> None:
    global _runtime_session
    with _runtime_lock:
        if _runtime_session is not session:
            return
        session.users -= 1
        if session.users:
            return
        _runtime_session = None
        try:
            session.module.shutdown()
        except Exception:
            LOG.debug("SteamVR was already unavailable during shutdown", exc_info=True)


class SteamVROverlay:
    """A head/hand-attached Qt panel with SteamVR's native laser pointer.

    Give actionable widgets a ``vr_action`` property from ``_ACTIONS``. A
    pointer press and release on the same enabled control invokes ``on_action``
    exactly once; it does not also emit the Qt button's ``clicked`` signal.
    ``dictate_start`` is a hold action: press starts and release anywhere stops.

    SteamVR's interactive-overlay mode can take controller focus from the
    game. Call ``set_interactive(False)`` for passive subtitles / a pinned
    reading card and re-enable it when the user wants to choose a reply. The
    desktop UI and its keyboard shortcuts remain available as a fallback.

    SetOverlayRaw is intended for modest UI textures, not high-rate video.
    Uploads are bounded to 1280 x 720 and unchanged frames are skipped. The
    caller may render at 10-15 Hz and poll input at 30-60 Hz independently.
    """

    MAX_TEXTURE_WIDTH = 1280
    MAX_TEXTURE_HEIGHT = 720

    def __init__(
        self,
        widget: QWidget,
        on_action: Callable[[str], None],
        width_m: float = 1.1,
        distance_m: float = 1.4,
    ) -> None:
        if not math.isfinite(width_m) or width_m <= 0:
            raise ValueError("width_m must be a finite positive number")
        if not math.isfinite(distance_m) or distance_m <= 0:
            raise ValueError("distance_m must be a finite positive number")
        self.widget = widget
        self.on_action = on_action
        self.width_m = float(width_m)
        self.distance_m = float(distance_m)
        self._placement: Placement = "head"
        self._module: Any = None
        self._session: _RuntimeSession | None = None
        self._api: Any = None
        self._handle: int | None = None
        self._visible = False
        self._interactive = True
        self._pixel_buffer: Any = None
        self._last_pixels: bytes | None = None
        self._texture_size: tuple[int, int] = (0, 0)
        self._mouse_size: tuple[int, int] = (0, 0)
        self._pressed: QWidget | None = None
        self._pressed_action: str | None = None
        self._device: int | None = None
        self._next_anchor_check = 0.0
        self._tracking_available = True
        self.last_warning = ""

    @property
    def is_running(self) -> bool:
        return self._handle is not None and self._session is not None

    @property
    def is_visible(self) -> bool:
        return self.is_running and self._visible and self._tracking_available

    @property
    def is_interactive(self) -> bool:
        return self._interactive

    @property
    def placement(self) -> Placement:
        return self._placement

    def _assert_widget_thread(self) -> None:
        if QThread.currentThread() != self.widget.thread():
            raise OverlayError("VR 浮窗的渲染和操作必须在 Qt 界面线程执行。")

    def start(self) -> None:
        self._assert_widget_thread()
        if self.is_running:
            return
        try:
            module = importlib.import_module("openvr")
        except (ImportError, OSError) as exc:
            raise OverlayError(
                "未能加载 OpenVR。请使用项目安装脚本安装 openvr，"
                "并使用与 SteamVR 匹配的 64 位 Python。"
            ) from exc
        self._module = module
        try:
            self._session = _acquire_runtime(module)
            self._api = module.VROverlay()
            # Distinct handles allow multiple panels without duplicate keys.
            key = f"kotonoha-vr.panel.{os.getpid()}.{uuid.uuid4().hex}"
            self._handle = self._api.createOverlay(key, "VR Dialogue Assistant")
            if not self._handle:
                raise OverlayError("SteamVR 未返回有效的浮窗句柄。")
            self._api.setOverlayInputMethod(
                self._handle,
                module.VROverlayInputMethod_Mouse if self._interactive
                else module.VROverlayInputMethod_None,
            )
            self._api.setOverlayFlag(
                self._handle,
                module.VROverlayFlags_MakeOverlaysInteractiveIfVisible,
                self._interactive,
            )
            self._api.setOverlayFlag(
                self._handle, module.VROverlayFlags_VisibleInDashboard, True
            )
            # QImage RGBA8888 is straight alpha. Top-down image storage agrees
            # with OpenVR's default UV bounds (upper-left 0,0 -> lower-right 1,1).
            self.widget.ensurePolished()
            if self.widget.layout() is not None:
                self.widget.layout().activate()
            self._apply_placement(self._placement)
            self._visible = True
            self.update_frame()
            self._api.showOverlay(self._handle)
        except Exception as exc:
            self.stop()
            if isinstance(exc, OverlayError):
                raise
            raise OverlayError(f"SteamVR 浮窗启动失败：{exc}") from exc

    def stop(self) -> None:
        """Idempotent cleanup, including after a partly completed start."""
        try:
            self._cancel_press()
        except Exception:
            LOG.debug("Action callback failed during overlay cleanup", exc_info=True)
        handle, api, session = self._handle, self._api, self._session
        self._handle = None
        self._api = None
        self._session = None
        self._visible = False
        if handle is not None and api is not None:
            try:
                api.hideOverlay(handle)
            except Exception:
                LOG.debug("Could not hide closing overlay", exc_info=True)
            try:
                api.destroyOverlay(handle)
            except Exception:
                LOG.debug("Could not destroy closing overlay", exc_info=True)
        if session is not None:
            _release_runtime(session)
        self._module = None
        self._pixel_buffer = None
        self._last_pixels = None
        self._texture_size = (0, 0)
        self._mouse_size = (0, 0)
        self._device = None
        self._tracking_available = True

    def set_visible(self, visible: bool) -> None:
        self._assert_widget_thread()
        if not self.is_running:
            return
        if not visible:
            self._cancel_press()
            if not self.is_running:
                return
        try:
            if visible and self._tracking_available:
                self._api.showOverlay(self._handle)
            else:
                self._api.hideOverlay(self._handle)
            self._visible = bool(visible)
        except Exception as exc:
            self.stop()
            raise OverlayError(f"无法改变 VR 浮窗的可见性：{exc}") from exc

    def toggle_visibility(self) -> None:
        self.set_visible(not self._visible)

    def set_interactive(self, interactive: bool) -> None:
        self._assert_widget_thread()
        if not interactive:
            self._cancel_press()
        if self.is_running:
            try:
                self._api.setOverlayFlag(
                    self._handle,
                    self._module.VROverlayFlags_MakeOverlaysInteractiveIfVisible,
                    bool(interactive),
                )
                # Disabled panels should not intercept another overlay's
                # global laser mouse either.
                self._api.setOverlayInputMethod(
                    self._handle,
                    self._module.VROverlayInputMethod_Mouse if interactive
                    else self._module.VROverlayInputMethod_None,
                )
            except Exception as exc:
                self.stop()
                raise OverlayError(f"无法切换 VR 浮窗的交互模式：{exc}") from exc
        self._interactive = bool(interactive)

    def set_placement(self, placement: Placement) -> None:
        self._assert_widget_thread()
        if placement not in ("head", "left", "right"):
            raise ValueError("placement must be 'head', 'left', or 'right'")
        if self.is_running:
            try:
                self._apply_placement(placement)
            except OverlayError:
                raise
            except Exception as exc:
                raise OverlayError(f"无法设置浮窗位置：{exc}") from exc
        self._placement = placement

    def _device_for(self, placement: Placement) -> int:
        if placement == "head":
            return self._module.k_unTrackedDeviceIndex_Hmd
        role = (
            self._module.TrackedControllerRole_LeftHand if placement == "left"
            else self._module.TrackedControllerRole_RightHand
        )
        # Role lookup remains available in pyopenvr and avoids assuming device
        # 1/2 are specific hands. No legacy button-number bindings are used.
        device = self._session.system.getTrackedDeviceIndexForControllerRole(role)
        if device == self._module.k_unTrackedDeviceIndexInvalid:
            hand = "左" if placement == "left" else "右"
            raise OverlayError(f"未发现{hand}手手柄；请唤醒手柄，或选择头部跟随。")
        return int(device)

    def _apply_placement(self, placement: Placement) -> None:
        device = self._device_for(placement)
        transform = self._module.HmdMatrix34_t()
        for row in range(3):
            for column in range(4):
                transform.m[row][column] = 1.0 if row == column else 0.0
        width = self.width_m
        if placement == "head":
            transform.m[1][3] = -0.18
            transform.m[2][3] = -self.distance_m
        else:
            # A readable panel just above the controller. The driver's grip
            # pose differs by device, so the wrist angle needs on-device QA.
            angle = math.radians(-65.0)
            c, s = math.cos(angle), math.sin(angle)
            transform.m[1][1], transform.m[1][2] = c, -s
            transform.m[2][1], transform.m[2][2] = s, c
            transform.m[0][3] = 0.02 if placement == "left" else -0.02
            transform.m[1][3] = 0.14
            transform.m[2][3] = -0.04
            width = min(self.width_m, 0.55)
        self._api.setOverlayTransformTrackedDeviceRelative(
            self._handle, device, transform
        )
        self._api.setOverlayWidthInMeters(self._handle, width)
        was_untracked = not self._tracking_available
        self._device = device
        self._tracking_available = True
        self.last_warning = ""
        if was_untracked and self._visible:
            self._api.showOverlay(self._handle)

    def _refresh_hand_attachment(self) -> None:
        if self._placement == "head" or time.monotonic() < self._next_anchor_check:
            return
        self._next_anchor_check = time.monotonic() + 1.0
        try:
            device = self._device_for(self._placement)
        except OverlayError as exc:
            self.last_warning = str(exc)
            if self._tracking_available:
                self._api.hideOverlay(self._handle)
                self._tracking_available = False
                self._cancel_press()
            return
        if device != self._device or not self._tracking_available:
            self._apply_placement(self._placement)

    def update_frame(self) -> None:
        self._assert_widget_thread()
        if not self.is_running or not self._visible:
            return
        try:
            image = self.widget.grab().toImage()
            if image.isNull():
                raise OverlayError("浮窗界面未能渲染，请保持提词面板为有效的非零尺寸。")
            if (image.width() > self.MAX_TEXTURE_WIDTH
                    or image.height() > self.MAX_TEXTURE_HEIGHT):
                image = image.scaled(
                    self.MAX_TEXTURE_WIDTH, self.MAX_TEXTURE_HEIGHT,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            image = image.convertToFormat(QImage.Format.Format_RGBA8888)
            row_bytes = image.width() * 4
            data = bytes(image.constBits())
            if image.bytesPerLine() != row_bytes:
                stride = image.bytesPerLine()
                data = b"".join(
                    data[y * stride:y * stride + row_bytes]
                    for y in range(image.height())
                )
            logical_size = (self.widget.width(), self.widget.height())
            if logical_size != self._mouse_size:
                scale = self._module.HmdVector2_t()
                scale.v[0], scale.v[1] = map(float, logical_size)
                self._api.setOverlayMouseScale(self._handle, scale)
                self._mouse_size = logical_size
            texture_size = (image.width(), image.height())
            if data == self._last_pixels and texture_size == self._texture_size:
                return
            # Keep the exact contiguous array alive on the instance. pyopenvr
            # calls byref(array); passing c_void_p would add a pointer level.
            buffer = (ctypes.c_ubyte * len(data)).from_buffer_copy(data)
            self._api.setOverlayRaw(
                self._handle, buffer, image.width(), image.height(), 4
            )
            self._pixel_buffer = buffer
            self._last_pixels = data
            self._texture_size = texture_size
        except Exception as exc:
            self.stop()
            if isinstance(exc, OverlayError):
                raise
            raise OverlayError(f"VR 浮窗画面更新失败：{exc}") from exc

    def _target_at(self, x: float, y: float) -> tuple[QWidget, str] | None:
        width, height = self._mouse_size
        if (not math.isfinite(x) or not math.isfinite(y)
                or not (0 <= x <= width and 0 <= y <= height)
                or width <= 0 or height <= 0):
            return None
        point = QPoint(min(width - 1, int(x)), min(height - 1, int(height - y)))
        target = self.widget.childAt(point)
        while target is not None:
            action = target.property("vr_action")
            if isinstance(action, str) and action in _ACTIONS:
                if target.isEnabled():
                    return target, action
                return None
            if target is self.widget:
                break
            target = target.parentWidget()
        return None

    def _cancel_press(self) -> None:
        target, action = self._pressed, self._pressed_action
        self._pressed = None
        self._pressed_action = None
        if isinstance(target, QAbstractButton):
            try:
                target.setDown(False)
            except RuntimeError:  # Qt may already have deleted the child.
                pass
        if action == "dictate_start":
            self.on_action("dictate_stop")

    def _handle_mouse(self, event: Any) -> None:
        event_type = event.eventType
        module = self._module
        mouse = event.data.mouse
        hit = self._target_at(float(mouse.x), float(mouse.y))
        if event_type == module.VREvent_MouseMove:
            if isinstance(self._pressed, QAbstractButton):
                self._pressed.setDown(bool(hit and hit[0] is self._pressed))
            return
        if mouse.button != module.VRMouseButton_Left:
            return
        if event_type == module.VREvent_MouseButtonDown:
            self._cancel_press()
            if hit is not None:
                self._pressed, self._pressed_action = hit
                if isinstance(self._pressed, QAbstractButton):
                    self._pressed.setDown(True)
                if self._pressed_action == "dictate_start":
                    self.on_action("dictate_start")
        elif event_type == module.VREvent_MouseButtonUp:
            target, action = self._pressed, self._pressed_action
            self._cancel_press()
            if (action and action != "dictate_start" and hit
                    and hit[0] is target and hit[1] == action):
                self.on_action(action)

    def poll_input(self) -> None:
        self._assert_widget_thread()
        if not self.is_running:
            return
        try:
            self._refresh_hand_attachment()
            if not self.is_running:
                return
            module = self._module
            event = module.VREvent_t()
            # Bounded to keep a burst of VR events from starving the Qt loop.
            for _ in range(128):
                result = self._api.pollNextOverlayEvent(self._handle, event)
                if isinstance(result, tuple):
                    available, event = result
                else:  # Older pyopenvr releases mutate event and return bool.
                    available = result
                if not available:
                    break
                if event.eventType == module.VREvent_Quit:
                    self.stop()
                    raise OverlayError("SteamVR 已退出，VR 浮窗已停止。可以稍后重新打开。")
                if event.eventType in (
                    module.VREvent_FocusLeave, module.VREvent_OverlayHidden,
                ):
                    self._cancel_press()
                if not self._interactive or not self.is_visible:
                    continue
                if event.eventType in (
                    module.VREvent_MouseMove, module.VREvent_MouseButtonDown,
                    module.VREvent_MouseButtonUp,
                ):
                    self._handle_mouse(event)
                # A callback may close the overlay, e.g. its hide/done button.
                if not self.is_running:
                    break
        except Exception as exc:
            self.stop()
            if isinstance(exc, OverlayError):
                raise
            raise OverlayError(f"VR 浮窗输入处理失败：{exc}") from exc
