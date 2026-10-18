# -*- coding: utf-8 -*-
"""验证阿里百炼接入：真实 LLM 规划 + tool_calls + 工具执行 + 综合回答 全闭环。

用法（在 ai-service 根目录）：
  python scripts/verify_dashscope.py

📌 验证点：
  1. get_llm() 返回 OpenAICompatLLM（非 Mock）
  2. LLMPlanner 让 qwen-plus 输出真 tool_calls（而非规则路由）
  3. AgentEngine.run() 端到端：工具真执行 + Synthesizer 真调用 LLM 综合
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.agent import AgentEngine, LLMPlanner, Synthesizer
from app.config import ENABLE_REFLECTION, LLM_MODE, MAX_STEPS
from app.llm import get_llm
from app.memory import MemoryStore
from app.models import RequestContext, Role
from app.tools import register_tools
from app.tools_registry import registry

register_tools()


async def main():
    llm = get_llm()
    planner = LLMPlanner(llm)
    synthesizer = Synthesizer(llm)
    agent = AgentEngine(llm, planner, synthesizer, registry, MemoryStore(), MAX_STEPS, ENABLE_REFLECTION)
    ctx = RequestContext(tenant_id="capital-medical", user_id="teacher-001", role=Role.TEACHER)

    print(f"LLM_MODE={LLM_MODE}  planner={type(planner).__name__}  llm={type(llm).__name__}")

    # 用例1：Text2SQL 数据查询（SQLite 真执行，验证 qwen tool_calls）
    q = "查询一下2026年春季轮转学员张三的去向"
    print("\n" + "=" * 60)
    print(f"提问：{q}")
    answer, steps = await agent.run(q, ctx, "verify-session-1")
    print("\n--- 工具步骤 ---")
    for s in steps:
        print(f"[{s.tool}] args={s.args}")
        print(f"  -> {s.observation[:300]}")
    print("\n--- 最终回答 ---")
    print(answer)

    # 用例2：双知识库（临床库，走真 RAG + bge-m3；加载模型较慢）
    q2 = "DKA 如何补液？"
    print("\n" + "=" * 60)
    print(f"提问：{q2}")
    answer2, steps2 = await agent.run(q2, ctx, "verify-session-2")
    print("\n--- 工具步骤 ---")
    for s in steps2:
        print(f"[{s.tool}] args={s.args}")
        print(f"  -> {s.observation[:300]}")
    print("\n--- 最终回答 ---")
    print(answer2)


if __name__ == "__main__":
    asyncio.run(main())
