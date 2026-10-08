"""Start the real GUI process and check its fixed, account-free demo screenshot.

Use --source for an installed source checkout, or --executable for a frozen app.
A Python parent waits for windowed Windows executables, so this check cannot pass
merely because PowerShell launched the app. No microphone, headset, model weights
or API credentials are used. The child must exit successfully and save a decoded,
nonblank image after its three demo replies become ready.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def stop_process(process: subprocess.Popen, log) -> None:
    """Do not leave a hanging GUI or its children on the CI runner."""
    if process.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                check=False, timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired):
            pass
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    if process.poll() is None:
        process.kill()
    process.wait(timeout=10)


def check_image(path: Path) -> dict:
    from PySide6.QtGui import QImage

    if not path.is_file() or path.stat().st_size < 1024:
        raise RuntimeError("The app did not write a demo screenshot.")
    image = QImage(str(path))
    if image.isNull() or image.width() < 640 or image.height() < 480:
        raise RuntimeError("The demo screenshot is invalid or unexpectedly small.")
    colors = {
        image.pixelColor(x, y).rgba()
        for y in range(0, image.height(), max(1, image.height() // 50))
        for x in range(0, image.width(), max(1, image.width() // 50))
    }
    if len(colors) < 8:
        raise RuntimeError("The demo screenshot is blank or lacks the expected UI.")
    return {
        "width": image.width(), "height": image.height(),
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def run_child(command: list[str], *, cwd: Path | str, env: dict, log,
              timeout: float) -> int:
    process = subprocess.Popen(
        command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT,
        start_new_session=sys.platform != "win32",
    )
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        stop_process(process, log)
        raise RuntimeError(f"The app did not exit within {timeout:g} seconds.") from None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--source", action="store_true")
    mode.add_argument("--executable", type=Path)
    parser.add_argument("--screenshot", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=45,
                        help="maximum app runtime in seconds (default: 45)")
    args = parser.parse_args()
    if not 1 <= args.timeout <= 300:
        parser.error("--timeout must be between 1 and 300 seconds")
    root = Path(__file__).resolve().parents[1]
    screenshot = args.screenshot.resolve()
    if screenshot.suffix.lower() != ".png":
        parser.error("--screenshot must end in .png")
    screenshot.parent.mkdir(parents=True, exist_ok=True)
    report_path = screenshot.with_suffix(".json")
    log_path = screenshot.with_suffix(".log")
    # A previous successful run must not satisfy this run's output checks.
    screenshot.unlink(missing_ok=True)
    command = ([sys.executable, "-m", "kotonoha_vr"] if args.source
               else [str(args.executable.resolve())])
    command += ["--demo", "--screenshot", str(screenshot)]
    report: dict = {"passed": False, "mode": "source" if args.source else "packaged",
                    "command": command, "hardware_tested": False}
    started = time.monotonic()
    try:
        with tempfile.TemporaryDirectory(prefix="KotonohaVR-smoke-") as run_dir:
            env = os.environ.copy()
            env.update(QT_QPA_PLATFORM="offscreen", LOCALAPPDATA=run_dir,
                       HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
            env.pop("OPENAI_API_KEY", None)
            if not args.source:
                # A frozen launch must not borrow Python modules or Qt plugins
                # from the checkout or the build environment.
                for name in ("PYTHONPATH", "PYTHONHOME", "QT_PLUGIN_PATH",
                             "QT_QPA_PLATFORM_PLUGIN_PATH", "QML2_IMPORT_PATH"):
                    env.pop(name, None)
            with log_path.open("wb") as log:
                if not args.source:
                    runtime_report = screenshot.with_suffix(".runtime.json")
                    runtime_report.unlink(missing_ok=True)
                    probe = [command[0], "--check-runtime", str(runtime_report)]
                    runtime_code = run_child(probe, cwd=run_dir, env=env, log=log,
                                             timeout=args.timeout)
                    report["runtime_exit_code"] = runtime_code
                    if runtime_report.is_file():
                        report["runtime"] = json.loads(runtime_report.read_text(encoding="utf-8"))
                    if runtime_code or not report.get("runtime", {}).get("passed"):
                        raise RuntimeError(f"Packaged dependency import check failed; see {runtime_report.name}.")
                    if not report["runtime"].get("frozen"):
                        raise RuntimeError("The dependency check did not run inside a frozen application.")
                returncode = run_child(command, cwd=root if args.source else run_dir,
                                       env=env, log=log, timeout=args.timeout)
                report["exit_code"] = returncode
                if returncode:
                    raise RuntimeError(f"The app exited with code {returncode}.")
            report["image"] = check_image(screenshot)
            report["passed"] = True
    except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
        report["error"] = str(exc)
        print(f"Demo launch check failed: {exc}", file=sys.stderr)
        if log_path.is_file():
            tail = log_path.read_bytes()[-8000:].decode("utf-8", errors="replace")
            if tail.strip():
                print(tail, file=sys.stderr)
    finally:
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["passed"]:
        print(f"Demo launch check passed: {screenshot} ({report['image']['width']} x {report['image']['height']})")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
