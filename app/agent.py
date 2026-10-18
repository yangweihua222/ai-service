# -*- coding: utf-8 -*-
"""Agent 核心：ReAct 循环 + 可插拔 Planner / Synthesizer + 防死循环 + 反思。
对应简历「Agent 编排（ReAct）+ 工具调用」「反思机制」「成本控制/防死循环」。

📌 面试考点（对应《技术点深挖备考库》）：
  · 技术点4 Function Calling / Tool Use（四步闭环 + ReAct） → 见 AgentEngine.run() / Planner
  · 技术点6 多 Agent 协同（延展，本项目是单 Agent 多工具）  → 见模块说明（被追问能讲清为何不用多 Agent）
  · 实战④ 防死循环（max_steps + 重复路径检测 + 反思熔断）   → 见 AgentEngine.run() 的循环与 recent 检测
  · 实战⑨ 反思机制（工具失败/检索不佳时修正，按需开启）       → 见 run() 末尾 reflection 分支
  · 实战⑫ 自研 ReAct Loop 而非 LangGraph（致命红线）          → 见 LLMPlanner / RulePlanner（调研过 LangGraph，生产自研）
"""
import json
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Optional, Tuple

from app.llm import LLMClient, LLMResponse, Message, ToolCall
from app.governance import RateLimitError, CircuitOpenError
from app.memory import MemoryStore
from app.reflection import (
    REFLECTION_RETRY_LIMIT,
    ReflectionState,
    ReflectionTrigger,
    inject_into_system,
    make_reflection,
)
from app.models import RequestContext, ToolStep
from app.tools_registry import ToolRegistry, validate_args
from app import observability
from app import rbac

# 📌 降级兜底话术：LLM 层限流/熔断触发时快速失败给用户，不崩整条链路（fail-fast + 可解释）
GOV_DEGRADE_MSG = (
    "当前 AI 服务繁忙（模型调用已限流/熔断保护），请稍后再试；"
    "紧急业务请直接在平台对应模块操作。"
)

SYS_PROMPT = (
    "你是临床实习教学平台的 AI 教学助手（小萱同款能力），使用提供的工具回答医学教学与管理工作。"
    "知识检索用 query_knowledge_base；学员数据用 query_teaching_data；轮转可行性用 check_rotation_feasibility；"
    "发送通知用 send_notification；提交请假用 submit_leave；值班查询用 query_duty_schedule；"
    "缺勤/缺卡统计用 query_absent_times；教学活动查询用 query_teaching_event；"
    "系统使用说明查询用 query_usage_guide。"
    "若无需工具即可回答，直接给出最终答案。"
)

# 📌 再决策提示（2026-09-11 实测发现后补）：
#    「模型路由」只做了首轮大/再决策小的模型分层，但**再决策步的意图从未写进提示词**——
#    模型拿到工具结果后没有任何"任务是否完成"的判断引导，弱模型（qwen-turbo）会直接收尾，
#    导致多工具编排被截断（实测：'查成绩+发通知'只调了 query_teaching_data 就 finish）。
#    补上显式核对清单后，再决策步才知道"要逐条核对、未完成必须继续调工具"。
REDECISION_HINT = (
    "【再决策·必读】你已经拿到若干工具返回的结果。请逐条核对用户的原始请求：\n"
    "1) 把用户请求拆成若干个动作（例如：①查成绩 ②发通知），逐个检查是否已有对应的工具结果；\n"
    "2) 只要还有动作没完成，**必须继续调用对应的工具**，不允许直接回答，也不允许声称已完成；\n"
    "3) 只有所有动作都完成（或工具明确失败且重试无意义）时，才直接给出最终答案。\n"
    "注意：不要用相同参数重复调用已经成功返回过的工具，避免死循环。"
)


def _sys_prompt(redecision: bool = False) -> str:
    """系统提示词 + 动态上下文（当前日期 / 实习管理系统工具组）。

    📌 为什么必须注入当前日期：LLM 训练数据有时序盲区，实测 qwen3.8-max 回答"我现在在哪个科室轮转"
    时把"今天"幻觉成 2026-07-24，导致把未来轮转说成"已结束/未开始"。注入真实日期后不再误判。
    """
    _w = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][date.today().weekday()]
    ctx = (
        f"\n【当前日期】{date.today().isoformat()}（{_w}）。所有涉及“今天/本周/本月/最近”的判断必须以该日期为准，禁止凭空猜测日期。"
        "\n【实习管理系统工具组】这些工具直连本平台的真实业务数据，按登录用户身份自动过滤（学生只能查自己的）："
        " internship_my_rotation 查我的轮转科室与起止；internship_my_duty 查我的值班；"
        " internship_my_attendance 查我的打卡明细；internship_attendance_stats 查考勤统计；"
        " internship_my_leaves 查我的请假单；internship_submit_leave 代提交请假；"
        " internship_teaching_events 查教学活动；internship_notices 查通知公告；"
        " internship_my_evaluations 查我的考核成绩；internship_my_cases 查我的病历记录；"
        " internship_tasks_summary 查我的待办汇总。"
        "\n【工具调用原则】只调用回答所必需的工具，不要顺带查无关数据；参数能省则省。"
    )
    return SYS_PROMPT + ctx + ("\n" + REDECISION_HINT if redecision else "")


def _trim_for_decision(history: List[Message], max_tool_chars: int = 2000) -> List[Message]:
    """📌 A' ② 决策步上下文裁剪：工具返回（RAG 原文 / 大表）可能很长，再决策只需判断
    『够不够 / 要不要继续调工具』，截断到上限即可；Synthesizer 仍用完整 history，最终答案质量不受影响。"""
    out: List[Message] = []
    for m in history:
        if m.role == "tool" and m.content and len(m.content) > max_tool_chars:
            out.append(Message(
                role=m.role,
                content=m.content[:max_tool_chars] + f"\n…[已截断，原文 {len(m.content)} 字]",
                tool_call_id=m.tool_call_id,
                name=m.name,
                tool_calls=m.tool_calls,
            ))
        else:
            out.append(m)
    return out


@dataclass
class Action:
    finish: bool
    thought: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)


class Planner(ABC):
    @abstractmethod
    async def decide(self, user_input: str, history: List[Message], tools, redecision: bool = False) -> Action:
        ...


class RulePlanner(Planner):
    """Mock 模式决策：关键词路由（不依赖 LLM），复用 agent_demo 逻辑。
    历史感知：已执行的工具不再重复触发，避免每轮 ReAct 都重新规划同一工具。
    📌 技术点4：Mock 用关键词路由模拟 Planner；真实模式由 LLMPlanner 接管(LLM 输出 tool_calls)，二者实现同一 decide() 接口。"""

    async def decide(self, user_input, history, tools, redecision: bool = False) -> Action:
        def uid() -> str:
            return "call_" + uuid.uuid4().hex[:8]

        # 基于关键词规划「应调用的工具集合」（去重）
        planned = []
        # 📌 实习管理系统（Laravel 工具网关）路由：
        #    命中这些关键词时，优先调 internship_* 工具（真实业务数据），
        #    并抑制同义的老医院后端工具（/dify/tools/* 未配置时只会返回演示文案）。
        is_i_rotation = re.search(r"轮转|哪个科室|在哪个科室|科室安排|我的科室", user_input)
        is_i_duty = re.search(r"值班", user_input)
        is_i_attend = re.search(r"考勤|打卡|缺勤|缺卡|出勤|迟到|早退", user_input)
        is_i_leave = re.search(r"请假|休假|告假|准假", user_input)
        is_i_tasks = re.search(r"待办|待审批|待处理|有什么要做", user_input)
        is_i_event = re.search(r"教学活动|讲座|培训|小讲课|教学查房", user_input)
        is_i_notice = re.search(r"通知|公告", user_input)
        is_i_eval = re.search(r"考核|成绩|评价|评教", user_input)
        is_i_case = re.search(r"病历|病案|病程", user_input)
        # 只要命中实习类意图，就走实习工具，不再触发老后端/教学库兜底
        hit_internship = any([is_i_rotation, is_i_duty, is_i_attend, is_i_leave, is_i_tasks,
                              is_i_event, is_i_notice, is_i_eval, is_i_case])

        # 📌 真实后端工具路由（小萱同源）
        is_leave = (not hit_internship) and bool(re.search(r"请假|请.{0,3}?假|休假|告假|准假", user_input))
        is_duty = (not hit_internship) and bool(re.search(r"值班", user_input))
        is_absent = (not hit_internship) and bool(re.search(r"缺勤|缺卡|考勤|打卡", user_input))
        is_event = (not hit_internship) and bool(re.search(r"教学活动|教学安排|活动安排", user_input))
        # 📌 使用说明 RAG 路由（双知识库之一）：操作 FAQ 类意图
        is_usage = re.search(
            r"提交请假|查值班|值班安排|轮转计划|补卡|缺卡|操作说明|使用说明|怎么用|如何查|在哪看|在哪查",
            user_input,
        )
        # 📌 原有工具路由
        is_kb = (not hit_internship) and bool(re.search(r"知识|指南|要点|怎么处理|治疗|综合征|处理", user_input))
        is_data = (not hit_internship) and bool(re.search(r"成绩|档案|考核|数据|查询", user_input))
        is_sched = (not hit_internship) and bool(re.search(r"排班|冲突|计划可行|可行性|轮转", user_input))
        is_alert = (not hit_internship) and bool(re.search(r"预警|提醒|通知|低于|不及格", user_input))

        # ---- 实习管理系统工具（真实业务数据，走 Laravel 网关）----
        if is_i_rotation:
            planned.append("internship_my_rotation")
        if is_i_duty:
            planned.append("internship_my_duty")
        if is_i_attend:
            planned.append("internship_attendance_stats")
            planned.append("internship_my_attendance")
        if is_i_tasks:
            planned.append("internship_tasks_summary")
        if is_i_event:
            planned.append("internship_teaching_events")
        if is_i_notice:
            planned.append("internship_notices")
        if is_i_eval:
            planned.append("internship_my_evaluations")
        if is_i_case:
            planned.append("internship_my_cases")
        if is_i_leave:
            planned.append("internship_my_leaves")
        if is_leave:
            planned.append("submit_leave")
        if is_duty:
            planned.append("query_duty_schedule")
        if is_absent:
            planned.append("query_absent_times")
        if is_event:
            planned.append("query_teaching_event")
        if is_usage:
            planned.append("query_usage_guide")
        if is_kb:
            planned.append("query_knowledge_base")
        if is_data:
            planned.append("query_teaching_data")
        if is_sched and not is_data:
            planned.append("check_rotation_feasibility")
        if is_alert:
            if not is_data:
                planned.append("query_teaching_data")
            planned.append("send_notification")
        if not planned:
            planned.append("query_knowledge_base")

        # 历史中已执行的工具（role=tool 的 name 字段）
        called = {m.name for m in history if m.role == "tool"}
        remaining = [t for t in planned if t not in called]
        if not remaining:
            return Action(finish=True, thought="所需工具已全部执行，生成最终回答")

        # 默认日期窗口（mock 联调用；真实模式由 LLM 填真实日期）
        # 改为动态取「今天 / 本周」，避免写死日期导致查错月份
        _d = date.today()
        _today = _d.isoformat()
        _monday = _d - timedelta(days=_d.weekday())
        _week = (_monday.isoformat(), (_monday + timedelta(days=6)).isoformat())
        calls: List[ToolCall] = []
        for t in remaining:
            if t == "query_knowledge_base":
                args = {"query": user_input}
            elif t == "query_usage_guide":
                args = {"query": user_input}
            elif t == "query_teaching_data":
                args = {"question": user_input}
            elif t == "check_rotation_feasibility":
                args = {}
            elif t == "submit_leave":
                # 简单抽取事由（mock）；真实模式 LLM 会填 reason/起止时间
                m = re.search(r"因为([^，。,]+)", user_input)
                args = {"reason": m.group(1) if m else user_input, "start_time": _today, "end_time": _today}
            elif t == "query_duty_schedule":
                args = {"start_time": _week[0], "end_time": _week[1]}
            elif t == "query_absent_times":
                args = {"start_date": _week[0], "end_date": _week[1]}
            elif t == "query_teaching_event":
                args = {"start_date": _week[0], "end_date": _week[1]}
            elif t.startswith("internship_"):
                # 实习管理系统工具网关：身份由 token 判定，多数工具无需参数
                args = {}
                if t == "internship_attendance_stats":
                    args = {"month": _today[:7]}
                elif t == "internship_submit_leave":
                    m = re.search(r"因为([^，。,]+)", user_input)
                    args = {"reason": m.group(1) if m else user_input,
                            "start_date": _today, "end_date": _today}
            else:  # send_notification
                args = {"target": "带教老师", "message": "有学员成绩需关注，请查看教学数据"}
            calls.append(ToolCall(uid(), t, args))
        return Action(finish=False, tool_calls=calls)


class LLMPlanner(Planner):
    """真实模式决策：LLM 输出 tool_calls。

    📌 A' ① 模型路由：首轮决策（工具选择，质量敏感）用主模型 self.llm；
    后续再决策（判断『够不够/要不要继续调工具』，较简单）用快模型 self.llm_fast，
    兼顾路由质量与延迟/成本。验证发现 qwen-turbo 直接做首轮选型会误路由
    （如把医学问答错派给 send_notification、漏掉多工具编排），故仅再决策步走快模型。
    """

    def __init__(self, llm: LLMClient, llm_fast: LLMClient = None):
        self.llm = llm                       # 主模型（首轮决策 / 工具选择）
        self.llm_fast = llm_fast or llm      # 快模型（再决策步）

    async def decide(self, user_input, history, tools, redecision: bool = False) -> Action:
        # history 已含本轮 user 消息（run() 入口追加），此处直接复用，避免重复拼接
        llm = self.llm_fast if redecision else self.llm
        system_msg = _sys_prompt(redecision)
        msgs = [Message(role="system", content=system_msg)] + list(history)
        try:
            resp: LLMResponse = await llm.chat(msgs, tools=tools)
        except (RateLimitError, CircuitOpenError) as e:
            # 📌 降级：限流/熔断触发 → 快速失败给兜底话术，不重试（重试会加剧拥塞）
            observability.log.warning("llm_governance_degrade", extra={"error": str(e)})
            return Action(finish=True, thought=GOV_DEGRADE_MSG)
        if resp.tool_calls:
            return Action(finish=False, thought=resp.content, tool_calls=resp.tool_calls)
        return Action(finish=True, thought=resp.content)

    async def decide_stream(self, user_input, history, tools, redecision: bool = False):
        """流式决策：逐 token yield 正文（str），最后 yield 一个 Action 表示决策结果。

        - 纯对话（模型不调工具）：正文随生成逐步 yield，最后 yield Action(finish=True, thought=全文)。
        - 需调工具：正文（通常是"我来查一下"之类前言）同样逐步 yield，最后 yield Action(finish=False, tool_calls=...)。
        """
        llm = self.llm_fast if redecision else self.llm
        system_msg = _sys_prompt(redecision)
        msgs = [Message(role="system", content=system_msg)] + list(history)
        parts: List[str] = []
        tool_calls: List[ToolCall] = []
        try:
            async for chunk in llm.stream_chat(msgs, tools=tools):
                if chunk.delta:
                    parts.append(chunk.delta)
                    yield chunk.delta
                if chunk.tool_calls:
                    tool_calls = chunk.tool_calls
        except (RateLimitError, CircuitOpenError) as e:
            # 📌 降级：流式链路上触发限流/熔断 → 吐兜底话术后正常收尾（SSE 不中断）
            observability.log.warning("llm_governance_degrade", extra={"error": str(e)})
            yield GOV_DEGRADE_MSG
            yield Action(finish=True, thought=GOV_DEGRADE_MSG)
            return
        if tool_calls:
            yield Action(finish=False, thought="".join(parts), tool_calls=tool_calls)
        else:
            yield Action(finish=True, thought="".join(parts))


class Synthesizer:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def synthesize(self, user_input, observations, history) -> str:
        if not observations:
            return "（暂无工具结果）"
        # history 已含本轮 user 消息与全部 tool 结果，无需重复拼接
        msgs = list(history)
        try:
            resp = await self.llm.chat(msgs)
        except (RateLimitError, CircuitOpenError) as e:
            observability.log.warning("llm_governance_degrade", extra={"error": str(e)})
            return GOV_DEGRADE_MSG
        return resp.content

    async def synthesize_stream(self, user_input, observations, history):
        """流式综合：工具结果齐全后，边生成边逐 token yield 正文。"""
        if not observations:
            yield "（暂无工具结果）"
            return
        msgs = list(history)
        try:
            async for chunk in self.llm.stream_chat(msgs):
                if chunk.delta:
                    yield chunk.delta
        except (RateLimitError, CircuitOpenError) as e:
            observability.log.warning("llm_governance_degrade", extra={"error": str(e)})
            yield GOV_DEGRADE_MSG


class AgentEngine:
    def __init__(self, llm, planner, synthesizer, registry: ToolRegistry, memory: MemoryStore,
                 max_steps: int, enable_reflection: bool,
                 reflection_llm: Optional[object] = None):
        """
        Args:
            llm: 主 LLM 客户端（生成最终答案）
            reflection_llm: 用于反思的 LLM（建议 qwen-turbo，省钱）；默认复用 llm
        """
        self.llm = llm
        self.reflection_llm = reflection_llm or llm
        self.planner = planner
        self.synthesizer = synthesizer
        self.registry = registry
        self.memory = memory
        self.max_steps = max_steps
        self.enable_reflection = enable_reflection
        # 📌 反思计数（进程内 Dict，多 worker 各算各；够用，因为反思本身就是"放弃"语义）
        self._reflection_counts: Dict[str, int] = {}

    @staticmethod
    def _reflection_key(ctx: RequestContext, sid: str) -> str:
        return f"{ctx.tenant_id}:{ctx.user_id}:{sid}"

    def _reflection_count(self, ctx: RequestContext, sid: str) -> int:
        return self._reflection_counts.get(self._reflection_key(ctx, sid), 0)

    def _bump_reflection_count(self, ctx: RequestContext, sid: str) -> None:
        key = self._reflection_key(ctx, sid)
        self._reflection_counts[key] = self._reflection_counts.get(key, 0) + 1

    def _finalize(self, user_input: str, ctx: RequestContext, steps: List[ToolStep],
                  start: float, answer: str = "", error: str = "") -> None:
        """一次请求收尾：写 JSONL 审计 + 记 Prometheus 指标（token/耗时/工具）。"""
        dur = time.perf_counter() - start
        usage = observability.get_usage()
        tools = [s.tool for s in steps]
        observability.audit({
            "user": ctx.user_id,
            "tenant": ctx.tenant_id,
            "role": ctx.role.value,
            "question": user_input,
            "answer_len": len(answer),
            "tools": tools,
            "usage": usage,
            "duration_ms": round(dur * 1000, 2),
            "status": "error" if error else "ok",
            "error": error,
        })
        observability.record_request(
            ctx.tenant_id, ctx.role.value, "error" if error else "ok", dur, usage, tools
        )
        observability.log.info(
            "request_done",
            extra={"duration_ms": round(dur * 1000, 2), "tools": ",".join(tools) or "-",
                   "tokens": usage["total"]},
        )

    async def run(self, user_input: str, ctx: RequestContext, session_id: str = "") -> Tuple[str, List[ToolStep]]:
        observability.reset_usage()
        start = time.perf_counter()
        sid = session_id or ctx.user_id
        # 📌 多租户隔离（P0 修复）：key 含 tenant_id/user_id/session_id
        history = self.memory.load(ctx.tenant_id, ctx.user_id, sid)
        history.append(Message(role="user", content=user_input))
        steps: List[ToolStep] = []
        observations: List[str] = []
        recent: List[str] = []
        direct_answer: str = ""  # 📌 无需工具时 Planner 直接给出的答案（普通对话路径）
        reflection_note: str = ""  # 本轮从 memory 取出的反思笔记，下轮 planner 决策前注入
        last_reflection = self.memory.load_reflection(ctx.tenant_id, ctx.user_id, sid) or ""
        if last_reflection:
            reflection_note = last_reflection

        for round_idx, _ in enumerate(range(self.max_steps)):  # 📌 实战④ 防死循环：max_steps 硬上限，ReAct 不会无限转
            # 📌 实战⑨ 反思注入：失败 ≥1 次后，每次 planner 决策前把反思塞进 history 头部
            decision_history = _trim_for_decision(history)
            if reflection_note:
                decision_history = [Message(role="system", content=f"[反思笔记] {reflection_note}")] + decision_history
            action = await self.planner.decide(
                user_input, decision_history, self.registry.for_role(ctx.role),
                redecision=bool(observations),
            )
            if action.finish:
                # 无任何工具观察且 Planner 已给出内容 → 普通对话，直接采用 LLM 回答
                if not observations and action.thought:
                    direct_answer = action.thought
                break
            # 真实模式：把 assistant(tool_calls) 写入历史，保证多轮一致
            if action.tool_calls:
                history.append(Message(role="assistant", content=action.thought, tool_calls=action.tool_calls))
            broke = False
            failed_calls: List[dict] = []  # 本轮失败集合（用于触发反思）
            for tc in action.tool_calls:
                if not rbac.can_use(ctx.role, tc.name):  # 📌 技术点8/7 RBAC 工具级校验：学员只能调知识库，写操作/数据查询仅 admin/teacher
                    obs = "⛔ 当前角色无权限调用该工具"
                else:
                    spec = self.registry.get(tc.name)
                    if not spec or not spec.execute:
                        obs = f"⚠️ 未找到工具 {tc.name}"
                    else:
                        # 📌 参数校验（第二道防线）：Schema 只约束模型输出格式，执行前必须本地再校验一次
                        arg_err = validate_args(spec, tc.arguments)
                        if arg_err:
                            obs = f"⚠️ 参数校验失败：{arg_err}。请按工具参数要求修正后重新调用。"
                        else:
                            t0 = time.perf_counter()
                            try:
                                with observability.span("tool.execute", tool=tc.name):
                                    obs = await spec.execute(tc.arguments, ctx)
                            except Exception as e:  # 工具异常兜底，避免整体崩溃
                                observability.capture_exception(e)
                                obs = f"工具执行异常：{e}"
                            observability.record_tool_duration(tc.name, time.perf_counter() - t0)
                observations.append(obs)
                steps.append(ToolStep(tool=tc.name, args=tc.arguments, observation=obs))
                history.append(Message(role="tool", content=obs, tool_call_id=tc.id, name=tc.name))
                sig = (tc.name, json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False))
                if sig in recent:  # 📌 实战④ 防死循环：重复相同(工具+参数)调用则终止，防 Agent 空转
                    observations.append("⚠️ 检测到重复工具调用，提前终止以避免死循环。")
                    broke = True
                    break
                recent.append(sig)
                # 收集失败调用（用于反思）
                if obs.startswith("⛔") or obs.startswith("⚠️"):
                    failed_calls.append({"name": tc.name, "args": tc.arguments, "observation": obs})
            if broke:
                break
            # 📌 实战⑨ 反思：本轮所有工具调用都失败 → 调 LLM 写反思笔记，存到 memory 供下次注入
            all_failed = (
                self.enable_reflection
                and action.tool_calls
                and len(failed_calls) == len(action.tool_calls)
            )
            if all_failed and failed_calls:
                # 检查反思次数上限（防无限反思）
                state_count = self._reflection_count(ctx, sid)
                if state_count < REFLECTION_RETRY_LIMIT:
                    trigger = ReflectionTrigger(
                        user_query=user_input,
                        failed_calls=failed_calls,
                        round_idx=round_idx,
                    )
                    note = await make_reflection(self.reflection_llm, trigger)
                    if note:
                        self.memory.save_reflection(ctx.tenant_id, ctx.user_id, sid, note)
                        reflection_note = note
                        self._bump_reflection_count(ctx, sid)
                # 即使反思失败也给用户兜底：终止空转
                break

        if direct_answer:
            # 普通对话：无需 Synthesizer，Planner 已给出完整回答
            answer = direct_answer
        else:
            answer = await self.synthesizer.synthesize(user_input, observations, history)
        history.append(Message(role="assistant", content=answer))
        self.memory.save(ctx.tenant_id, ctx.user_id, sid, history)
        self.memory.compact(ctx.tenant_id, ctx.user_id, sid)
        self._finalize(user_input, ctx, steps, start, answer=answer)
        return answer, steps

    async def run_stream(self, user_input: str, ctx: RequestContext, session_id: str = ""):
        """流式版 run()：ReAct 决策与工具执行同 run()，但最终回答边生成边逐 token yield。

        yield 事件字典：{"type": "token", "text": ...} 逐字产出正文；
        结束时 yield {"type": "steps", "steps": [...]} 附带工具轨迹（供前端渲染 trace）。
        与 run() 语义等价，仅生成方式从『一次性』变为『逐 token 流式』。
        """
        observability.reset_usage()
        start = time.perf_counter()
        sid = session_id or ctx.user_id
        # 📌 多租户隔离（P0 修复）
        history = self.memory.load(ctx.tenant_id, ctx.user_id, sid)
        history.append(Message(role="user", content=user_input))
        steps: List[ToolStep] = []
        observations: List[str] = []
        recent: List[str] = []
        action: Action = None
        first_round = True  # 仅第一轮（尚无工具结果）透传决策正文，后续轮仅用于决策、不重复流出
        reflection_note: str = self.memory.load_reflection(ctx.tenant_id, ctx.user_id, sid) or ""

        for round_idx, _ in enumerate(range(self.max_steps)):  # 📌 实战④ 防死循环：max_steps 硬上限
            decision_history = _trim_for_decision(history)
            if reflection_note:
                decision_history = [Message(role="system", content=f"[反思笔记] {reflection_note}")] + decision_history
            # 决策：真模式走流式决策（纯对话时正文随生成逐 token 产出），mock 模式非流式
            if isinstance(self.planner, LLMPlanner):
                gen = self.planner.decide_stream(
                    user_input, decision_history, self.registry.for_role(ctx.role),
                    redecision=bool(observations),
                )
                async for item in gen:
                    if isinstance(item, Action):  # 决策完成（最后一个事件）
                        action = item
                        break
                    if first_round:  # 纯对话正文 / 首轮前言才透传，避免与综合回答重复
                        yield {"type": "token", "text": item}
            else:
                action = await self.planner.decide(user_input, decision_history, self.registry.for_role(ctx.role), redecision=bool(observations))
            first_round = False

            if action.finish:
                break

            if action.tool_calls:
                history.append(Message(role="assistant", content=action.thought, tool_calls=action.tool_calls))
            broke = False
            failed_calls: List[dict] = []  # 本轮失败集合（用于触发反思）
            for tc in action.tool_calls:
                if not rbac.can_use(ctx.role, tc.name):  # 📌 RBAC 工具级校验
                    obs = "⛔ 当前角色无权限调用该工具"
                else:
                    spec = self.registry.get(tc.name)
                    if not spec or not spec.execute:
                        obs = f"⚠️ 未找到工具 {tc.name}"
                    else:
                        # 📌 参数校验（第二道防线）：Schema 只约束模型输出格式，执行前必须本地再校验一次
                        arg_err = validate_args(spec, tc.arguments)
                        if arg_err:
                            obs = f"⚠️ 参数校验失败：{arg_err}。请按工具参数要求修正后重新调用。"
                        else:
                            t0 = time.perf_counter()
                            try:
                                with observability.span("tool.execute", tool=tc.name):
                                    obs = await spec.execute(tc.arguments, ctx)
                            except Exception as e:
                                observability.capture_exception(e)
                                obs = f"工具执行异常：{e}"
                            finally:
                                observability.record_tool_duration(tc.name, time.perf_counter() - t0)
                observations.append(obs)
                steps.append(ToolStep(tool=tc.name, args=tc.arguments, observation=obs))
                history.append(Message(role="tool", content=obs, tool_call_id=tc.id, name=tc.name))
                sig = (tc.name, json.dumps(tc.arguments, sort_keys=True, ensure_ascii=False))
                if sig in recent:  # 📌 防死循环：重复调用则终止
                    observations.append("⚠️ 检测到重复工具调用，提前终止以避免死循环。")
                    broke = True
                    break
                recent.append(sig)
                # 收集失败（用于反思）
                if obs.startswith("⛔") or obs.startswith("⚠️"):
                    failed_calls.append({"name": tc.name, "args": tc.arguments, "observation": obs})
            if broke:
                break
            # 📌 实战⑨ 反思：本轮所有工具调用都失败 → 写反思笔记 + 终止空转
            all_failed = (
                self.enable_reflection
                and action.tool_calls
                and len(failed_calls) == len(action.tool_calls)
            )
            if all_failed and failed_calls:
                state_count = self._reflection_count(ctx, sid)
                if state_count < REFLECTION_RETRY_LIMIT:
                    trigger = ReflectionTrigger(
                        user_query=user_input,
                        failed_calls=failed_calls,
                        round_idx=round_idx,
                    )
                    note = await make_reflection(self.reflection_llm, trigger)
                    if note:
                        self.memory.save_reflection(ctx.tenant_id, ctx.user_id, sid, note)
                        reflection_note = note
                        self._bump_reflection_count(ctx, sid)
                break

        # 先发工具轨迹（供前端渲染 trace），再流式产出最终回答正文
        yield {"type": "steps", "steps": steps}

        # 最终回答：工具场景流式综合；纯对话正文已在决策阶段流式产出
        if observations:
            parts: List[str] = []
            async for tok in self.synthesizer.synthesize_stream(user_input, observations, history):
                parts.append(tok)
                yield {"type": "token", "text": tok}
            answer = "".join(parts)
        elif action and action.thought:
            answer = action.thought
        else:
            answer = "（暂无工具结果）"

        history.append(Message(role="assistant", content=answer))
        self.memory.save(ctx.tenant_id, ctx.user_id, sid, history)
        self.memory.compact(ctx.tenant_id, ctx.user_id, sid)
        self._finalize(user_input, ctx, steps, start, answer=answer)
