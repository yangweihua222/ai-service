# -*- coding: utf-8 -*-
"""Agent 端到端（真实大模型调用，非 mock）。

⚠️ 本文件与其余单测性质不同，面试前务必知道：
  · 其余 112 个单测是**确定性**的（mock 掉 LLM），任何一次跑都应全过；
  · 本文件用例会**真实调用 LLM**，结果受模型随机性影响、存在偶发失败，
    故统一打上 `llm` 标记；CI 跑确定性全量用 `pytest -m "not llm"`。
    （原文件头写的是「mock 模式，零依赖」，与事实不符，2026-09-11 已更正。）
  · 处理非确定性的正确姿势是「有限重试 + 断言不放松」，不是删测试或放宽断言：
    见 chat_until()——只有不满足预期才重试，次数用尽后仍按原断言报错。

📌 工具口径提醒（别讲混）：
  对外（简历 / 备考库）口径是「4 类 7 动作」：
    ① 临床 RAG  ② 系统使用说明  ③ 轮转可行性
    ④ OA ×4：submit_leave / query_duty_schedule / query_absent_times / query_teaching_event
  下面出现的 query_teaching_data（Text2SQL）与 send_notification 属于
  **代码内保留的技术验证工具**，已从简历与备考库对外口径下线，面试不主动讲；
  此处仅用于验证「Agent 多步编排机制本身是否还通」。
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tools import register_tools

pytestmark = pytest.mark.llm

register_tools()
client = TestClient(app)


def chat_until(predicate, message, tries: int = 3, **extra):
    """发一次对话，不满足 predicate 就重试（LLM 有随机性），用尽后返回最后一次结果。

    只有「模型这一把没发挥好」才重试；断言本身不做任何放松。
    """
    last = None
    for _ in range(tries):
        resp = client.post("/api/v1/chat", json={"message": message, **extra})
        assert resp.status_code == 200, f"HTTP {resp.status_code}: {resp.text[:300]}"
        last = resp.json()
        if predicate(last):
            return last
    return last


def _tools_of(data):
    return [s["tool"] for s in data["steps"]]


def test_kb_single_tool():
    data = chat_until(lambda d: "query_knowledge_base" in _tools_of(d),
                      "糖尿病酮症酸中毒怎么处理？")
    assert "query_knowledge_base" in _tools_of(data)


def test_multi_tool_alert_orchestration():
    """多步编排机制验证：查数据 + 发通知（用两个技术验证工具，不对外讲）。

    📌 这里守住的是「Agent 能不能连续调两个工具」，不是这两个工具本身。
    2026-09-11 曾挂：首轮选对 query_teaching_data 后，再决策步换快模型 qwen-turbo
    直接收尾、没继续调 send_notification。根因是「再决策要判断任务是否完成」
    这个意图只体现在换模型上、从未写进提示词 → 已在 agent.py 补 REDECISION_HINT。
    ⚠️ 关于效果：做过「修前 vs 修后 × 弱/强模型」对照实验（scripts/bench_redecision.py），
    但每格只跑 3 次，两次独立重复的结果互相矛盾（同一格一次 100%、一次 50%），
    说明样本量不足、结论不可量化。**目前只能定性说"提示词确实缺了这块引导"**，
    不要向面试官报任何提升百分比。
    """
    data = chat_until(
        lambda d: {"query_teaching_data", "send_notification"}.issubset(set(_tools_of(d))),
        "张三内科成绩低于80分，请预警并通知带教老师",
        tries=3,
        stream=False,
    )
    tools = _tools_of(data)
    assert "query_teaching_data" in tools
    assert "send_notification" in tools
    assert data["tenant_id"]


def test_scheduling_feasibility_tool():
    data = chat_until(lambda d: "check_rotation_feasibility" in _tools_of(d),
                      "检查下当前轮转排班计划是否可行")
    assert "check_rotation_feasibility" in _tools_of(data)


def test_health():
    assert client.get("/health").json()["status"] == "ok"
