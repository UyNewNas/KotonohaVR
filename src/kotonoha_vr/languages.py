"""Language labels are presentation only; detection comes from ASR/model output."""
from __future__ import annotations

import re

LANGUAGES = {
    "en": "英语", "ja": "日语", "ko": "韩语", "zh": "中文", "yue": "粤语",
    "fr": "法语", "de": "德语", "es": "西班牙语", "pt": "葡萄牙语",
    "ru": "俄语", "it": "意大利语", "ar": "阿拉伯语", "hi": "印地语",
    "th": "泰语", "vi": "越南语", "id": "印尼语", "nl": "荷兰语",
    "pl": "波兰语", "tr": "土耳其语", "uk": "乌克兰语", "sv": "瑞典语",
}
ALIASES = {
    "english": "en", "japanese": "ja", "korean": "ko", "chinese": "zh",
    "mandarin": "zh", "cmn": "zh", "cantonese": "yue", "french": "fr",
    "german": "de", "spanish": "es", "russian": "ru", "portuguese": "pt",
    "eng": "en", "jpn": "ja", "kor": "ko", "zho": "zh", "fra": "fr",
    "deu": "de", "spa": "es", "rus": "ru", "por": "pt",
}


def normalize_language(value: str | None) -> str | None:
    if not value or not isinstance(value, str):
        return None
    code = value.strip().lower().replace("_", "-")
    if code in {"auto", "unknown", "und", "mixed", "none"}:
        return None
    code = ALIASES.get(code, code)
    if not re.fullmatch(r"[a-z]{2,3}(?:-[a-z0-9]{2,8})*", code):
        return None
    return code.split("-")[0]


def language_name(code: str | None) -> str:
    code = normalize_language(code)
    return LANGUAGES.get(code, code or "等待识别")


def is_short_acknowledgment(text: str) -> bool:
    text = text.strip().strip(".!?。！？… ").lower()
    return text in {"ok", "okay", "yes", "no", "hi", "hello", "thanks", "yeah", "sure",
                    "はい", "うん", "いいえ", "네", "응", "好", "好的", "嗯", "哦"}
