# -*- coding: utf-8 -*-
"""reflection 模块单元测试：make_reflection / inject_into_system / 防无限。"""
import pytest

from app.reflection import (
    REFLECTION_MAX_LEN,
    REFLECTION_RETRY_LIMIT,
    ReflectionState,
    ReflectionTrigger,
    format_reflection_for_prompt,
    inject_into_system,
    should_inject_reflection,
)


class TestInjectIntoSystem:
    """测试注入逻辑：把反思塞到 system 之后。"""

    def test_empty_note_returns_unchanged(self):
        msgs = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "hi"},
        ]
        out = inject_into_system(msgs, "")
        assert out == msgs

    def test_inject_after_system(self):
        msgs = [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ]
        out = inject_into_system(msgs, "反思：上次用了错的工具")
        assert len(out) == 4
        # 找到反射位置
        refl_idx = next(
            i for i, m in enumerate(out) if m.get("content", "").startswith("[反思笔记]")
        )
        assert refl_idx == 1
        # 后面原消息顺序保留
        assert out[2:] == msgs[1:]

    def test_inject_no_system_inserts_at_start(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = inject_into_system(msgs, "反思")
        assert out[0]["content"].startswith("[反思笔记]")
        assert out[1] == msgs[0]

    def test_original_not_mutated(self):
        msgs = [{"role": "system", "content": "you are X"}, {"role": "user", "content": "q"}]
        original_len = len(msgs)
        inject_into_system(msgs, "test")
        assert len(msgs) == original_len


class TestShouldInject:
    def test_inject_after_first_reflection(self):
        state = ReflectionState(session_id="s1", count=1, last_note="反思内容")
        assert should_inject_reflection(state) is True

    def test_skip_when_count_zero(self):
        state = ReflectionState(session_id="s1", count=0, last_note="")
        assert should_inject_reflection(state) is False

    def test_skip_over_limit(self):
        state = ReflectionState(session_id="s1", count=REFLECTION_RETRY_LIMIT + 1, last_note="x")
        assert should_inject_reflection(state) is False

    def test_skip_empty_note(self):
        state = ReflectionState(session_id="s1", count=1, last_note="")
        assert should_inject_reflection(state) is False


class TestFormatReflection:
    def test_format_includes_prefix(self):
        text = format_reflection_for_prompt("工具选错了")
        assert text.startswith("[反思笔记] ")

    def test_format_with_empty(self):
        text = format_reflection_for_prompt("")
        assert text == "[反思笔记] "


class TestTrigger:
    def test_trigger_holds_data(self):
        trigger = ReflectionTrigger(
            user_query="请假三天",
            failed_calls=[
                {"name": "submit_leave", "args": {}, "observation": "参数校验失败"},
            ],
            round_idx=2,
        )
        assert trigger.user_query == "请假三天"
        assert trigger.round_idx == 2
        assert len(trigger.failed_calls) == 1
