# -*- coding: utf-8 -*-
"""备考库口径统一（2026-09-12）：pytest 数字 112 → 122，并区分确定性与真 LLM 用例。

2026-09-12 全量实测：122 passed / 0 failed
  · 118 个确定性单测（mock LLM）—— 任何一次跑都应全过
  ·   4 个真 LLM e2e（打 llm 标记）—— 依赖模型，有随机性，已加有限重试
"""
from pathlib import Path

BANK = Path(r"D:\简历\技术点深挖备考库.md")

REPLACES = [
    # 开篇速查表
    (
        "pytest **112 全过**（本地全量，不含 4 个 e2e；2026-09-11 新增多轮对话类型还原回归 2 个、工具 Schema 加固后全量回归）",
        "pytest **122 全过**（2026-09-12 全量：118 个确定性单测 + 4 个真 LLM e2e；新增 Token 预算 P0 回归 6 个）",
    ),
    # 2.3 Schema 章节
    (
        "全量回归 **110 个单测通过**（不含 4 个 e2e）",
        "全量回归 **118 个确定性单测通过**（不含 4 个真 LLM e2e）",
    ),
    # 3.13 评测章节
    (
        "pytest **112 全过**（RAG 相关 + 治理 + P0 修复 + 记忆反思 + Redis 三件套全栈）",
        "pytest **122 全过**（RAG 相关 + 治理 + P0 修复 + 记忆反思 + Redis 三件套全栈；其中 118 个确定性单测 + 4 个真 LLM e2e）",
    ),
    # 4.4 Redis 章节
    (
        "⚠️ 当前 pytest **112 passed**（本地全量，不含 4 个 e2e）——加评测不影响通过率,皆因每次 P0 修复必配单测。",
        "⚠️ 当前 pytest **122 passed**（2026-09-12 全量，118 确定性 + 4 真 LLM e2e）——加评测不影响通过率,皆因每次 P0 修复必配单测；**e2e 已打 `llm` 标记,CI 跑确定性全量用 `pytest -m \"not llm\"` 跳过**。",
    ),
    # Checklist
    (
        "- [ ] **真实数字 pytest 112 全过**(2026-09-11 更新：本地全量不含 4 个 e2e，含新增的多轮对话类型还原回归 2 个)——开篇速查表过一遍",
        "- [ ] **真实数字 pytest 122 全过**(2026-09-12 更新：118 个确定性单测 + 4 个真 LLM e2e；新增 Token 预算 P0 回归 6 个)——开篇速查表过一遍",
    ),
    # 30 秒版本
    (
        "**pytest 112 全过、评测全绿、工具集+MCP 可起",
        "**pytest 122 全过、评测全绿、工具集+MCP 可起",
    ),
    # 附录命令
    (
        "pytest                       # 112 项全过（本地全量，不含 4 个 e2e）",
        "pytest                       # 122 项全过（118 确定性单测 + 4 个真 LLM e2e）\n"
        "pytest -m \"not llm\"          # 只跑 118 个确定性单测（CI 用，不依赖模型）",
    ),
    # 验证结论
    (
        "验证结论：pytest **112 全过**；",
        "验证结论：pytest **122 全过**（118 确定性 + 4 真 LLM e2e）；",
    ),
]


def main():
    text = BANK.read_text(encoding="utf-8")
    for old, new in REPLACES:
        if old not in text:
            print(f"[警告] 未找到：{old[:45]}…")
            continue
        text = text.replace(old, new, 1)
        print(f"[已替换] {old[:38]}…")

    BANK.write_text(text, encoding="utf-8")
    print("\n剩余旧口径 112/110 出现次数：",
          text.count("112 全过") + text.count("**112 passed**") + text.count("110 个单测"))
    print("总行数：", len(text.splitlines()))


if __name__ == "__main__":
    main()
