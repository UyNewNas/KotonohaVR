"""PyInstaller entrypoint, including an account-free dependency import probe."""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import sys


RUNTIME_MODULES = (
    "PySide6.QtWidgets", "httpx", "numpy", "openvr", "pyaudiowpatch",
    "faster_whisper", "ctranslate2._ext", "onnxruntime", "av", "tokenizers",
    "huggingface_hub._snapshot_download",
)


def check_runtime(destination: Path) -> int:
    """Import binary/lazy dependencies without constructing devices or models."""
    checks = []
    for name in RUNTIME_MODULES:
        try:
            importlib.import_module(name)
            checks.append({"module": name, "passed": True})
        except Exception as exc:
            checks.append({"module": name, "passed": False,
                           "error": f"{type(exc).__name__}: {exc}"})
    try:
        from faster_whisper.utils import get_assets_path
        assets = sorted(Path(get_assets_path()).glob("*.onnx"))
        if not assets or any(path.stat().st_size == 0 for path in assets):
            raise RuntimeError("The bundled voice-activity model asset is missing.")
        checks.append({"module": "faster_whisper.assets", "passed": True,
                       "files": [path.name for path in assets]})
    except Exception as exc:
        checks.append({"module": "faster_whisper.assets", "passed": False,
                       "error": f"{type(exc).__name__}: {exc}"})
    result = {"passed": all(check["passed"] for check in checks),
              "frozen": bool(getattr(sys, "frozen", False)),
              "hardware_tested": False, "checks": checks}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


def main() -> int:
    if "--check-runtime" in sys.argv[1:]:
        parser = argparse.ArgumentParser(description="Check packaged runtime imports without devices or accounts.")
        parser.add_argument("--check-runtime", type=Path, required=True, metavar="REPORT_JSON")
        return check_runtime(parser.parse_args().check_runtime)
    from kotonoha_vr.__main__ import main as start_application
    return start_application()

if __name__ == "__main__":
    raise SystemExit(main())
