import pytest

from kotonoha_vr.core import ConversationState, DialogueError


def response(lang="ja", mode="suggest"):
    return {"source_language": lang, "source_zh": "要一起去下一个世界吗？",
            "reply_language": lang, "summary_zh": "邀请同行",
            "candidates": [{"id": str(i), "intent_zh": "选择", "text": f"reply {i}",
                            "meaning_zh": f"中文 {i}", "reading_hint": ""}
                           for i in range(1, 2 if mode == "custom" else 4)]}


def ready():
    state = ConversationState()
    state.receive("次のワールド、一緒に行きませんか？", "ja", .96)
    req = state.request()
    state.apply(req, response())
    return state


def test_own_chinese_does_not_change_target_and_is_not_history():
    state = ready()
    before = len(state.history)
    req = state.request("custom", "我想在这里再待一会儿")
    assert req.payload["source_language"] == "zh"
    assert req.payload["reply_language"] == "ja"
    data = response("ja", "custom")
    data["source_language"] = "zh"
    state.apply(req, data)
    assert state.reply_language == "ja"
    assert len(state.history) == before


def test_pinned_text_survives_new_speech_and_generation():
    state = ready()
    pin = state.select(0)
    assert all(t.role == "other" for t in state.history)
    state.receive("Would you prefer a quiet world?", "en", .95)
    req = state.request()
    state.apply(req, response("en"))
    assert state.pinned == pin
    assert state.reply_language == "en"
    spoken = state.mark_spoken("I changed my mind.")
    assert spoken.text == "I changed my mind."
    assert spoken.role == "self"
    assert state.pinned is None


def test_obsolete_generation_is_discarded():
    state = ready()
    old = state.request()
    state.receive("We are leaving now.", "en")
    assert not state.apply(old, response())
    assert not state.candidates


def test_newest_request_wins_without_a_new_turn():
    state = ready()
    old = state.request()
    new = state.request()
    assert not state.apply(old, response())
    assert state.apply(new, response())


def test_short_ack_does_not_switch_and_manual_language_wins():
    state = ready()
    state.receive("OK", "en", .99)
    assert state.reply_language == "ja"
    state.set_manual_language("ko")
    state.receive("Let's go to another world.", "en", .99)
    assert state.reply_language == "ko"


def test_language_mismatch_and_incomplete_bilingual_pair_rejected():
    state = ready()
    req = state.request()
    with pytest.raises(DialogueError, match="不一致"):
        state.apply(req, response("en"))
    data = response()
    data["candidates"][0]["meaning_zh"] = ""
    with pytest.raises(DialogueError, match="缺少"):
        state.apply(req, data)


def test_reset_invalidates_pending_response():
    state = ready()
    req = state.request()
    state.reset()
    assert not state.apply(req, response())
    assert not state.history


def test_custom_requires_a_language():
    with pytest.raises(DialogueError, match="语言"):
        ConversationState().request("custom", "你好")


def test_cancel_does_not_fabricate_a_spoken_reply():
    state = ready()
    state.select(1)
    state.cancel_pinned()
    assert len(state.history) == 1


def test_model_can_identify_language_for_typed_foreign_text():
    state = ConversationState()
    state.receive("次のワールド、一緒に行きませんか？")
    req = state.request()
    assert state.apply(req, response())
    assert state.reply_language == "ja"

def test_rejected_long_turn_does_not_poison_context_or_change_language():
    state = ConversationState()
    state.receive("こんにちは", "ja", 1.0)
    revision = state.revision
    with pytest.raises(DialogueError):
        state.receive("x" * 4001, "en", 1.0)
    assert state.reply_language == "ja"
    assert state.revision == revision
    assert len(state.history) == 1
    request = state.request()
    with pytest.raises(DialogueError):
        state.request("custom", "中" * 4001)
    assert state._sequence == request.sequence
