"""Non-secret user preferences. API keys stay in memory / environment variables."""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path


@dataclass
class Settings:
    provider: str = "demo"
    openai_model: str = "gpt-4.1-mini"
    openai_base_url: str = "https://api.openai.com/v1"
    codex_model: str = ""
    codex_command: str = "codex"
    asr_backend: str = "local"
    local_asr_model: str = "small"
    cloud_asr_model: str = "whisper-1"
    asr_base_url: str = "https://api.openai.com/v1"
    speaker_device: int = -1
    microphone_device: int = -1
    auto_suggest: bool = True
    energy_threshold: float = 0.015
    silence_ms: int = 650
    placement: str = "head"
    overlay_width: float = 1.15
    overlay_distance: float = 1.4
    style: str = "自然、友好、简短，适合口头聊天"

    def validate(self) -> None:
        if not isinstance(self.style, str) or not self.style.strip() or len(self.style) > 300:
            raise ValueError("回复风格须为 1 至 300 个字符。")
        if self.provider not in {"demo", "openai", "codex"}:
            raise ValueError("未知模型来源。")
        if self.asr_backend not in {"local", "openai"}:
            raise ValueError("未知语音识别来源。")
        if self.placement not in {"head", "left", "right"}:
            raise ValueError("未知浮窗位置。")
        if not 0.001 <= float(self.energy_threshold) <= 0.5:
            raise ValueError("声音阈值应在 0.001 至 0.5 之间。")
        if not 200 <= int(self.silence_ms) <= 2500:
            raise ValueError("断句停顿应在 200 至 2500 毫秒之间。")
        if not 0.4 <= float(self.overlay_width) <= 2.0:
            raise ValueError("浮窗宽度应在 0.4 至 2 米之间。")
        if not 0.6 <= float(self.overlay_distance) <= 3.0:
            raise ValueError("浮窗距离应在 0.6 至 3 米之间。")


def settings_path() -> Path:
    folder = Path(os.environ.get("LOCALAPPDATA", Path.home() / ".local" / "share"))
    return folder / "KotonohaVR" / "settings.json"


def load_settings(path: Path | None = None) -> Settings:
    path = path or settings_path()
    if not path.exists():
        return Settings()
    data = json.loads(path.read_text(encoding="utf-8"))
    known = {f.name for f in fields(Settings)}
    result = Settings(**{k: v for k, v in data.items() if k in known})
    result.validate()
    return result


def save_settings(settings: Settings, path: Path | None = None) -> None:
    settings.validate()
    path = path or settings_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(asdict(settings), ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
