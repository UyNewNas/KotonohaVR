from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="言叶 VR · 多语言对话助手")
    parser.add_argument("--demo", action="store_true", help="强制离线演示，不使用保存的模型配置")
    parser.add_argument("--screenshot", help="离线渲染一张演示界面截图后退出（开发检查）")
    args = parser.parse_args()
    if args.screenshot:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QMessageBox
    from .settings import Settings, load_settings
    from .ui import MainWindow
    application = QApplication(sys.argv[:1])
    application.setApplicationName("KotonohaVR")
    application.setOrganizationName("KotonohaVR")
    try:
        settings = Settings() if args.demo or args.screenshot else load_settings()
        window = MainWindow(settings)
    except Exception as exc:
        if args.screenshot:
            print(f"Startup failed: {exc}", file=sys.stderr)
        else:
            QMessageBox.critical(None, "启动失败", str(exc))
        return 1
    window.show()
    if args.demo or args.screenshot:
        QTimer.singleShot(100, window.load_demo)
    if args.screenshot:
        deadline = time.monotonic() + 8
        def capture():
            if window.busy or len(window.state.candidates) != 3:
                if time.monotonic() < deadline:
                    QTimer.singleShot(50, capture)
                    return
                print("Offline demo did not become ready.", file=sys.stderr)
                window.close()
                application.exit(1)
                return
            destination = Path(args.screenshot).expanduser()
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
                saved = window.grab().save(str(destination))
            except OSError:
                saved = False
            if not saved:
                print("Could not save the demo screenshot.", file=sys.stderr)
            window.close()
            application.exit(0 if saved else 1)
        QTimer.singleShot(250, capture)
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
