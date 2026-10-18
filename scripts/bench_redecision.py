# -*- coding: utf-8 -*-
"""再决策（redecision）步多工具编排对比实验。

背景：2026-09-11 全量测试发现 test_multi_tool_alert_orchestration 失败——
模型首轮选对了 query_teaching_data，但第二轮换快模型（qwen-turbo）后直接 finish，
没继续调 send_notification，多工具编排被截断。

根因假设：「模型路由」只做了首轮大 / 再决策小的模型分层，
但「再决策步要判断任务是否完成、未完成必须继续调工具」这个意图从未写进提示词。

本实验四组对照，验证修复是否有效：
  A 修前（REDECISION_HINT 置空） + fast=qwen-turbo
  B 修后（注入 REDECISION_HINT） + fast=qwen-turbo
  C 修前 + fast=qwen-plus
  D 修后 + fast=qwen-plus

判据：一次会话内是否调齐了场景要求的所有工具（编排完整率）。
每个场景每组跑 REPEAT 次（LLM 有随机性，单次结论不可信）。
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("OMP_NUM_THREADS", "1")

import app.agent as A  # noqa: E402
from app.config import OPENAI_MODEL  # noqa: E402
from app.llm import get_llm  # noqa: E402
from app.models import RequestContext, Role  # noqa: E402
from app.tools import register_tools  # noqa: E402
from app.tools_registry import registry  # noqa: E402
from app.memory import MemoryStore  # noqa: E402

REPEAT = int(os.getenv("REPEAT", "3"))

# 场景：(用户问句, 期望出现的全部工具集合)
# ⚠️ 选场景的硬要求：**第一个工具必须能真实返回数据**。
#   曾用 query_absent_times 做场景，但它未配真实平台时返回"[演示] 后端未配置"，
#   模型把它当失败、反复重调同一个工具（实测出现 ['query_absent_times','query_absent_times']），
#   测出来的"编排失败"其实是环境未配置，不是编排能力问题。故只保留能真返回的工具。
SCENARIOS = [
    (
        "张三内科成绩低于80分，请预警并通知带教老师",
        {"query_teaching_data", "send_notification"},
    ),
    (
        "查一下糖尿病酮症酸中毒的处理原则，并把要点通知带教老师",
        {"query_knowledge_base", "send_notification"},
    ),
    (
        "检查下当前轮转排班计划是否可行，并把结论通知教学秘书",
        {"check_rotation_feasibility", "send_notification"},
    ),
]

# 只对比弱模型下的「修前 / 修后」——这才是验证 REDECISION_HINT 是否有效的核心格。
# （强模型格作为参考，可用 FAST_MODELS 环境变量加回来。）
FAST_MODELS = os.getenv("FAST_MODELS", "qwen-turbo").split(",")

register_tools()


def build_engine(fast_model: str):
    """构造一个 AgentEngine，可指定再决策步的快模型。"""
    llm = get_llm(OPENAI_MODEL)
    llm_fast = get_llm(fast_model)
    planner = A.LLMPlanner(llm, llm_fast)
    synth = A.Synthesizer(llm)
    memory = MemoryStore(backend="memory", summarizer=None)
    return A.AgentEngine(
        llm=llm,
        planner=planner,
        synthesizer=synth,
        registry=registry,
        memory=memory,
        max_steps=6,
        enable_reflection=False,
    )


def new_ctx(tag: str) -> RequestContext:
    """每组每次用不同的 user_id，避免共享记忆串味。"""
    return RequestContext(
        tenant_id="capital-medical",
        user_id=f"bench-{tag}",
        role=Role.TEACHER,
    )


async def run_once(engine, question: str, tag: str):
    ctx = new_ctx(tag)
    _answer, steps = await engine.run(question, ctx, session_id=tag)
    return [s.tool for s in steps]


async def main():
    llm = get_llm(OPENAI_MODEL)
    print("主模型 =", OPENAI_MODEL)
    print("每组每场景重复 =", REPEAT)
    print()

    out = {}
    for fixed, label in ((False, "修前_无提示"), (True, "修后_有提示")):
        orig = A.REDECISION_HINT
        A.REDECISION_HINT = orig if fixed else ""   # 置空 = 复现修复前行为
        for fast in [m.strip() for m in FAST_MODELS if m.strip()]:
            key = f"{label}__fast={fast}"
            engine = build_engine(fast)
            rows = []
            for question, expect in SCENARIOS:
                ok = 0
                detail = []
                errors = 0
                for i in range(REPEAT):
                    tag = f"{fast}-{fixed}-{i}-{abs(hash(question)) % 997}"
                    tools = None
                    # ⚠️ 429 限流/网络抖动属于基础设施失败，不是编排能力问题。
                    #    若把它算进分母，会把"被限流"误判成"编排失败"（实测已踩过一次坑），
                    #    故这里退避重试，仍失败则记为 error 且不计入分母。
                    for attempt in range(3):
                        try:
                            tools = await run_once(engine, question, tag)
                            break
                        except Exception as e:
                            if attempt == 2:
                                print(f"  [异常·不计入] {question[:12]}… {e}")
                                errors += 1
                            else:
                                await asyncio.sleep(5 * (attempt + 1))
                    if tools is None:
                        continue
                    await asyncio.sleep(1)  # 串行降速，避免触发 QPS/Token 预算限流
                    hit = expect.issubset(set(tools))
                    ok += int(hit)
                    detail.append({"tools": tools, "ok": hit})
                valid = max(1, REPEAT - errors)
                rows.append({
                    "question": question,
                    "expect": sorted(expect),
                    "full_rate": ok / valid,
                    "ok": ok,
                    "valid_runs": valid,
                    "errors": errors,
                    "runs": detail,
                })
                print(f"  {key} | {ok}/{valid} (异常{errors}) | {question[:20]}… | {detail[-1]['tools'] if detail else 'N/A'}")
            total_ok = sum(r["full_rate"] for r in rows) / len(rows)
            out[key] = {"overall_full_rate": round(total_ok, 4), "scenarios": rows}
        A.REDECISION_HINT = orig  # 还原

    outdir = Path(__file__).resolve().parents[1] / "bench_out"
    outdir.mkdir(exist_ok=True)
    (outdir / "redecision_bench.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n=== 汇总 ===")
    for k, v in out.items():
        print(f"{k:45s} 编排完整率 = {v['overall_full_rate']:.2%}")
    print("\n结果已写入 bench_out/redecision_bench.json")


if __name__ == "__main__":
    asyncio.run(main())
