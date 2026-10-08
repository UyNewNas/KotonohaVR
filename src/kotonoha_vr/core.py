"""UI-independent dialogue state. Drafts never silently become spoken messages."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .languages import is_short_acknowledgment, normalize_language

MAX_TURN_CHARACTERS = 4000


class DialogueError(ValueError):
    pass


@dataclass(frozen=True)
class Candidate:
    id: str
    intent_zh: str
    text: str
    meaning_zh: str
    reading_hint: str = ""


@dataclass
class Turn:
    id: int
    role: str
    text: str
    language: str | None
    translation_zh: str = ""


@dataclass(frozen=True)
class GenerationRequest:
    sequence: int
    revision: int
    turn_id: int | None
    payload: dict[str, Any]


@dataclass(frozen=True)
class PinnedReply:
    candidate: Candidate
    language: str
    based_on_turn: int | None


@dataclass
class ConversationState:
    context_limit: int = 20
    history: list[Turn] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    pinned: PinnedReply | None = None
    detected_language: str | None = None
    manual_language: str | None = None
    summary_zh: str = ""
    revision: int = 0
    _sequence: int = 0
    _turn_serial: int = 0
    _candidate_language: str | None = None
    _candidate_turn: int | None = None

    @property
    def reply_language(self) -> str | None:
        return self.manual_language or self.detected_language

    @property
    def latest_other(self) -> Turn | None:
        return next((t for t in reversed(self.history) if t.role == "other"), None)

    def _append(self, role: str, text: str, language: str | None) -> Turn:
        if not text.strip():
            raise DialogueError("内容为空。")
        if len(text) > MAX_TURN_CHARACTERS:
            raise DialogueError("这段内容过长，请分成短句。")
        self._turn_serial += 1
        turn = Turn(self._turn_serial, role, text.strip(), normalize_language(language))
        self.history.append(turn)
        del self.history[:-self.context_limit]
        self.revision += 1
        return turn

    def receive(self, text: str, language: str | None = None,
                confidence: float | None = None) -> Turn:
        """Only the other audio/text channel may update the detected reply language."""
        code = normalize_language(language)
        trustworthy = code and (confidence is None or confidence >= 0.70)
        turn = self._append("other", text, code if trustworthy else None)
        if trustworthy and (self.detected_language is None or not is_short_acknowledgment(text)):
            self.detected_language = code
        self.candidates.clear()
        self.summary_zh = ""
        return turn

    def set_manual_language(self, language: str | None) -> None:
        if language and not normalize_language(language):
            raise DialogueError("语言代码无效。")
        self.manual_language = normalize_language(language)
        self.revision += 1
        self.candidates.clear()

    def request(self, mode: str = "suggest", custom_text: str = "",
                style: str = "自然、友好、简短，适合口头聊天") -> GenerationRequest:
        other = self.latest_other
        if mode not in {"suggest", "custom"}:
            raise DialogueError("未知生成模式。")
        if mode == "suggest" and other is None:
            raise DialogueError("先听对方说一句，或在文本输入里试一句。")
        if mode == "custom" and not self.reply_language:
            raise DialogueError("请先识别对方语言，或手动选择回复语言。")
        if mode == "custom" and not custom_text.strip():
            raise DialogueError("先输入或录下你想说的中文。")
        latest_text = custom_text.strip() if mode == "custom" else other.text
        if len(latest_text) > MAX_TURN_CHARACTERS:
            raise DialogueError("这段内容过长，请分成短句。")
        self._sequence += 1
        payload = {
            "mode": mode,
            "latest_text": latest_text,
            "source_language": "zh" if mode == "custom" else other.language,
            "reply_language": (self.reply_language if mode == "custom" or self.manual_language
                               or other.language or is_short_acknowledgment(other.text) else None),
            "context": [{"role": t.role, "text": t.text, "language": t.language}
                        for t in self.history],
            "style": style,
            "max_words": 24,
        }
        return GenerationRequest(self._sequence, self.revision,
                                 other.id if other else None, payload)

    def apply(self, request: GenerationRequest, data: dict[str, Any]) -> bool:
        """Return False for an obsolete completion; never replace a pinned reply."""
        if request.revision != self.revision or request.sequence != self._sequence:
            return False
        if not isinstance(data, dict):
            raise DialogueError("模型结果不是有效对象。")
        target = normalize_language(data.get("reply_language"))
        source = normalize_language(data.get("source_language"))
        expected = normalize_language(request.payload.get("reply_language"))
        if not target:
            raise DialogueError("模型没有确定回复语言，请手动选择后重试。")
        if not expected and source and target != source:
            raise DialogueError("回复没有使用对方所说的语言，已拒绝显示。")
        if expected and target != expected:
            raise DialogueError("模型返回的语言与当前回复语言不一致，已拒绝显示。")
        values = data.get("candidates")
        wanted = 1 if request.payload["mode"] == "custom" else 3
        if not isinstance(values, list) or len(values) != wanted:
            raise DialogueError(f"模型应返回 {wanted} 条回复，本次结果格式不完整。")
        parsed: list[Candidate] = []
        for i, item in enumerate(values):
            if not isinstance(item, dict):
                raise DialogueError("候选回复格式错误。")
            required = ["intent_zh", "text", "meaning_zh"]
            if any(not isinstance(item.get(k), str) or not item[k].strip() for k in required):
                raise DialogueError("候选回复缺少外语正文或对应中文。")
            if any(len(item[k]) > 1500 for k in required):
                raise DialogueError("候选回复过长，请使用短句重新生成。")
            reading = item.get("reading_hint", "")
            if not isinstance(reading, str) or len(reading) > 1500:
                raise DialogueError("读音提示格式错误。")
            parsed.append(Candidate(str(i + 1), item["intent_zh"].strip(), item["text"].strip(),
                                    item["meaning_zh"].strip(), reading.strip()))
        if len({c.text for c in parsed}) != len(parsed):
            raise DialogueError("模型给出了重复选项，请重新生成。")
        for field_name in ("source_zh", "summary_zh"):
            if not isinstance(data.get(field_name), str) or len(data[field_name]) > 12000:
                raise DialogueError("翻译结果格式错误。")
        if request.payload["mode"] == "suggest":
            if source and (self.detected_language is None or
                           not is_short_acknowledgment(request.payload["latest_text"])):
                self.detected_language = source
            for turn in self.history:
                if turn.id == request.turn_id:
                    turn.translation_zh = data["source_zh"]
        self.candidates = parsed
        self._candidate_language = target
        self._candidate_turn = request.turn_id
        self.summary_zh = data["summary_zh"]
        return True

    def select(self, index: int) -> PinnedReply:
        if not 0 <= index < len(self.candidates):
            raise DialogueError("这条回复已过期，请选择当前显示的候选。")
        self.pinned = PinnedReply(self.candidates[index], self._candidate_language or "und",
                                  self._candidate_turn)
        return self.pinned

    def cancel_pinned(self) -> None:
        self.pinned = None

    def mark_spoken(self, actual_text: str | None = None) -> Turn:
        if not self.pinned:
            raise DialogueError("请先选择要读的回复。")
        text = self.pinned.candidate.text if actual_text is None else actual_text
        turn = self._append("self", text, self.pinned.language)
        self.pinned = None
        self.candidates.clear()
        return turn

    def reset(self) -> None:
        self.history.clear()
        self.candidates.clear()
        self.pinned = None
        self.detected_language = None
        self.summary_zh = ""
        self.revision += 1
        self._sequence += 1
