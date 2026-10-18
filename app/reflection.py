# -*- coding: utf-8 -*-
"""反思机制：工具失败时让 LLM 写一段反思笔记，下次决策时塞进 prompt 头部。

📌 面试考点：实战⑨ 反思机制（不是「失败止损」，是「让 LLM 写反思文本调整策略」）
                 → 见 make_reflection() / format_reflection_for_prompt()

⚠️ 诚实边界：
  · 真反思（Reflexion / Self-Refine）需要让 LLM 自由生成，**不能**用 prompt 模板限制，
    否则就成了「另一种 prompt」；本实现用结构化提示 + max_tokens 限制 token 量级。
  · 反思笔记是「passive context」——塞到下一轮 prompt 但不强制生效，模型可以忽略。
  · 防无限反思：单 session 反思次数上限（默认 3 次），超限直接放弃反思。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import List, Optional

_log = logging.getLogger(__name__)

REFLECTION_MAX_LEN = 200  # 反思笔记最大字符数（避免污染 prompt）
REFLECTION_RETRY_LIMIT = 3  # 单 session 反思次数上限（防无限反思）


# ---------------------------------------------------------------------------
# 失败记录（每轮一组观察 + 是否全失败）
# ---------------------------------------------------------------------------
@dataclass
class ReflectionTrigger:
    """触发反思的输入：用户原 query + 失败的工具调用列表。"""
    user_query: str
    failed_calls: List[dict]  # [{name, args, observation}, ...]
    round_idx: int  # 第几轮触发（用于诊断）


@dataclass
class ReflectionState:
    """单 session 的反思状态：避免无限反思。"""
    session_id: str
    count: int = 0  # 已反思次数
    last_note: str = ""  # 最近一次的反思文本（用于诊断/展示）
    history: List[str] = field(default_factory=list)  # 历次反思记录


# ---------------------------------------------------------------------------
# 反思生成（结构化 prompt，让 LLM 写最多 200 字反思笔记）
# ---------------------------------------------------------------------------
REFLECTION_SYSTEM = (
    "你是一个教学助手的内部反思模块。每次工具调用失败时，请用 1-3 句话总结：\n"
    "1. 失败的根因（参数错误？权限不足？工具不存在？业务约束？）\n"
    "2. 下次要怎么调整（换工具？修正参数？直接告诉用户无法完成？）\n"
    "不要重复用户原话，不要寒暄，直入主题。"
)


async def make_reflection(llm, trigger: ReflectionTrigger) -> Optional[str]:
    """调 LLM 生成反思笔记。

    Args:
        llm: LLMClient 实例（用 qwen-turbo，省钱+短答案友好）
        trigger: 触发条件（用户 query + 失败工具调用列表）

    Returns:
        反思文本（≤ 200 字符），失败返回 None
    """
    from app.llm import Message  # 局域 import 避免循环

    if not trigger.failed_calls:
        return None

    # 失败列表压缩成结构化文本
    failures = []
    for c in trigger.failed_calls[:3]:  # 最多取前 3 个失败，避免 prompt 膨胀
        obs = (c.get("observation") or "")[:200]  # 单条 observation 也限长
        failures.append(f"- 工具 {c.get('name', '?')} 参数={c.get('args', {})} 观察={obs}")
    fail_text = "\n".join(failures)

    prompt = (
        f"用户问：{trigger.user_query[:300]}\n\n"
        f"第 {trigger.round_idx} 轮工具调用全部失败：\n{fail_text}\n\n"
        f"请用 ≤ {REFLECTION_MAX_LEN} 字符写出反思笔记。"
    )

    try:
        resp = await llm.chat(
            messages=[
                Message(role="system", content=REFLECTION_SYSTEM),
                Message(role="user", content=prompt),
            ],
            temperature=0.2,  # 反思要稳定不要发散
        )
        text = (resp.content or "").strip()
        if not text:
            return None
        # 截断到 REFLECTION_MAX_LEN 字符（防御 LLM 输出失控）
        return text[:REFLECTION_MAX_LEN]
    except Exception as e:  # noqa: BLE001 - 反思失败不致命
        _log.warning("make_reflection 调 LLM 失败 (%s)", e)
        return None


# ---------------------------------------------------------------------------
# 提示注入（把反思作为 system 消息插到 prompt 头部）
# ---------------------------------------------------------------------------
_REFLECTION_HEADER_PREFIX = "[反思笔记] "


def format_reflection_for_prompt(note: str) -> str:
    """把反思文本格式化成 system 消息。

    设计：反思放在 system 而不是 user —— 防止下游误把反思当用户输入处理。
    """
    return f"{_REFLECTION_HEADER_PREFIX}{note}"


def should_inject_reflection(state: ReflectionState) -> bool:
    """是否要注入反思到下一轮 prompt：
    · 已反思次数 < 上限
    · 上一次反思非空
    """
    return state.count > 0 and state.count <= REFLECTION_RETRY_LIMIT and bool(state.last_note)


def inject_into_system(messages: list, note: str) -> list:
    """把反思作为 system 消息注入 prompt 头部。

    Args:
        messages: 现有消息列表（OpenAI 格式 [{role, content}, ...]）
        note: 反思笔记

    Returns:
        新消息列表（在 system 之后、最近 user 之前插入反思）
    """
    if not note:
        return messages
    reflection_msg = {"role": "system", "content": format_reflection_for_prompt(note)}
    # 找到第一个 system 之后插入（保持系统指令最先）
    new_msgs = list(messages)
    inserted = False
    for i, m in enumerate(new_msgs):
        if m.get("role") == "system":
            new_msgs.insert(i + 1, reflection_msg)
            inserted = True
            break
    if not inserted:
        new_msgs.insert(0, reflection_msg)
    return new_msgs
