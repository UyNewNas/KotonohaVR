"""Build an onedir Windows package from an already prepared virtual environment.

Run: python -m pip install -e ".[dev,windows,local-asr]"
     python scripts/build_windows.py
CI also runs build_smoke.py against the resulting EXE. Actual audio capture and
SteamVR operation still need testing on the intended headset and machine.
"""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from entrypoint import RUNTIME_MODULES

RUNTIME_DISTRIBUTIONS = (
    "kotonoha-vr", "PySide6", "httpx", "numpy", "openvr", "PyAudioWPatch",
    "faster-whisper", "ctranslate2", "onnxruntime", "av", "tokenizers",
    "huggingface-hub", "PyInstaller", "pyinstaller-hooks-contrib",
)


def build_command(root: Path) -> list[str]:
    """Keep the collection rules explicit, including the app's lazy imports."""
    return [
        sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean",
        "--onedir", "--windowed", "--name", "KotonohaVR",
        "--paths", str(root / "src"),
        # Qt's bundled hooks collect the platform and image plugins needed by
        # QtWidgets/QtGui. Collecting *all* of PySide6 would also bundle unused
        # WebEngine, multimedia and QML components.
        "--collect-all", "openvr", "--collect-all", "pyaudiowpatch",
        "--collect-all", "faster_whisper",
        "--hidden-import", "ctranslate2",
        "--collect-binaries", "ctranslate2",
        "--copy-metadata", "ctranslate2",
        "--hidden-import", "onnxruntime",
        # huggingface_hub exposes snapshot_download through __getattr__, which
        # static import analysis cannot infer from faster_whisper.utils.
        "--hidden-import", "huggingface_hub._snapshot_download",
        "--copy-metadata", "huggingface-hub",
        "--collect-submodules", "tokenizers",
        "--copy-metadata", "tokenizers",
        "--collect-data", "kotonoha_vr",
        "--distpath", str(root / "dist"),
        "--workpath", str(root / "build" / "pyinstaller"),
        "--specpath", str(root / "build"),
        str(root / "scripts" / "entrypoint.py"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--print-command", action="store_true",
                        help="print the build arguments without building (any OS)")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    command = build_command(root)
    if args.print_command:
        print(json.dumps(command, indent=2))
        return 0
    if sys.platform != "win32" or sys.maxsize <= 2**32:
        print("Build this application with 64-bit Python on Windows.", file=sys.stderr)
        return 1

    # Missing collect targets otherwise produce warnings and an apparently
    # successful but incomplete build. Imports do not initialize any devices,
    # load model weights, or contact a model service.
    try:
        for module in RUNTIME_MODULES:
            importlib.import_module(module)
        versions = {name: importlib.metadata.version(name) for name in RUNTIME_DISTRIBUTIONS}
    except (ImportError, OSError, importlib.metadata.PackageNotFoundError) as exc:
        print(f"Runtime dependency check failed: {exc}", file=sys.stderr)
        print('Install the .[dev,windows,local-asr] extras first.', file=sys.stderr)
        return 1

    (root / "build").mkdir(exist_ok=True)
    result = subprocess.run(command, cwd=root, check=False)
    if result.returncode:
        return result.returncode
    destination = root / "dist" / "KotonohaVR"
    if not (destination / "KotonohaVR.exe").is_file():
        print("PyInstaller did not create the expected EXE.", file=sys.stderr)
        return 1
    for name in ("README.md", "LICENSE"):
        if (root / name).is_file():
            shutil.copy2(root / name, destination / name)
    if (root / "docs").is_dir():
        shutil.copytree(root / "docs", destination / "docs", dirs_exist_ok=True)
    # The source archive's launcher prepares a Python environment. A packaged
    # app already has its runtime and needs a separate, EXE-only launcher.
    (destination / "START_DEMO.cmd").write_text(
        '@echo off\n'
        'setlocal\n'
        'cd /d "%~dp0"\n'
        'start "" /wait "%~dp0KotonohaVR.exe" --demo\n'
        'set "KOTONOHA_EXIT=%ERRORLEVEL%"\n'
        'if not "%KOTONOHA_EXIT%"=="0" (\n'
        '    echo KotonohaVR could not start. Exit code: %KOTONOHA_EXIT%\n'
        '    pause\n'
        ')\n'
        'exit /b %KOTONOHA_EXIT%\n',
        encoding="ascii", newline="\r\n",
    )
    info = {
        "application": "KotonohaVR",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": versions,
        "source_runtime_imports_passed": list(RUNTIME_MODULES),
        "hardware_tested": False,
        "note": "See ci-demo.json for the separate packaged application launch check.",
    }
    (destination / "BUILD-INFO.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    print(f"Built {destination / 'KotonohaVR.exe'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
