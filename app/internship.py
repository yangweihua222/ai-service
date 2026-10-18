# -*- coding: utf-8 -*-
"""实习管理系统（Laravel）工具集：AI 服务通过 Laravel 的 AI 工具网关查/写实习业务数据。

📌 为什么单独一个文件（对接 D:\\实习系统\\internship-server）：
   · 实习管理系统的工具网关是「单一入口 + 白名单」：POST {base}/ai/tools/call
     body {"tool": <工具名>, "arguments": {...}}，Header 带用户 Sanctum token；
   · 权限不由 AI 服务判定，而是由 token 所属用户在 Laravel 侧判定（学生调老师工具 → 403），
     AI 服务只负责「选工具 + 传参 + 把结果说人话」，与本项目「AI 当脑子、业务建单走 Laravel」的分层一致；
   · 零侵入：本文件只被 app/tools.py 的 register_tools() 调用一行，不改动原有 RAG / Text2SQL 工具。

工具清单与 Laravel 侧 AiToolController::TOOLS 白名单一一对应。
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import httpx

from app.config import INTERNSHIP_API_BASE, INTERNSHIP_TIMEOUT
from app.models import RequestContext, Role
from app.observability import get_trace_id
from app.tools_registry import registry, ToolSpec

TOOL_CALL_PATH = "/ai/tools/call"


class InternshipGateway:
    """Laravel AI 工具网关客户端（server-to-server，透传用户 token）。"""

    def __init__(self, base: str, timeout: float):
        self.base = (base or "").rstrip("/")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.base)

    async def call(self, tool: str, arguments: Dict[str, Any], token: str) -> Dict[str, Any]:
        """调网关，返回 {"ok": bool, "data"/"error": ...}。

        📌 失败不抛异常给 Agent：统一转成 ok=False 的结构，由调用方决定降级话术，
           避免单个工具抖动让整轮对话崩掉（对应治理里的「工具级容错」）。
        """
        if not self.configured:
            return {"ok": False, "error": "实习系统网关未配置（INTERNSHIP_API_BASE 为空）"}
        if not token:
            return {"ok": False, "error": "缺少用户登录凭证，无法查询实习系统数据"}
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
            "X-Trace-Id": get_trace_id() or "",
        }
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base}{TOOL_CALL_PATH}",
                headers=headers,
                content=json.dumps({"tool": tool, "arguments": arguments}, ensure_ascii=False).encode("utf-8"),
            )
        if resp.status_code != 200:
            return {"ok": False, "error": f"网关返回 HTTP {resp.status_code}"}
        try:
            return resp.json()
        except Exception:
            return {"ok": False, "error": "网关返回非 JSON"}


gateway = InternshipGateway(INTERNSHIP_API_BASE, INTERNSHIP_TIMEOUT)


# ============ 结果渲染：把 Laravel 的结构化数据压成 LLM 好读的短文本 ============

def _fmt_lines(items: List[str], limit: int = 15) -> str:
    if not items:
        return "（无记录）"
    shown = items[:limit]
    tail = f"\n…共 {len(items)} 条，已列出前 {limit} 条" if len(items) > limit else ""
    return "\n".join(shown) + tail


async def _call(tool: str, args: Dict[str, Any], ctx: RequestContext) -> tuple[bool, Any]:
    """统一调用 + 解包：(ok, data|error)"""
    res = await gateway.call(tool, args, getattr(ctx, "internship_token", "") or "")
    if not res.get("ok"):
        return False, res.get("error", "未知错误")
    return True, res.get("data")


async def tool_my_rotation(args: Dict, ctx: RequestContext) -> str:
    """我的轮转安排（哪个时间段在哪个科室）。"""
    ok, data = await _call("get_my_rotation", {}, ctx)
    if not ok:
        return f"[轮转查询失败] {data}"
    lines = [
        f"{it['start_date']} ~ {it['end_date']}：{it.get('department') or '未分配'}"
        + ("（当前）" if it.get("is_current") else "")
        for it in data.get("list", [])
    ]
    return f"[我的轮转安排]\n{_fmt_lines(lines)}"


async def tool_my_duty(args: Dict, ctx: RequestContext) -> str:
    """我的值班表。"""
    ok, data = await _call("get_my_duty", {"month": args.get("month")} if args.get("month") else {}, ctx)
    if not ok:
        return f"[值班查询失败] {data}"
    lines = [f"{it['date']} {it.get('shift')}（{it.get('department') or '-'}）" for it in data.get("list", [])]
    return f"[我的值班 · {data.get('month')}]\n{_fmt_lines(lines)}"


async def tool_my_attendance(args: Dict, ctx: RequestContext) -> str:
    """我的逐日打卡记录。"""
    ok, data = await _call("get_my_attendance", {"month": args.get("month")} if args.get("month") else {}, ctx)
    if not ok:
        return f"[打卡记录查询失败] {data}"
    status_cn = {"normal": "正常", "late": "迟到", "early": "早退", "absent": "缺勤", "leave": "请假", "unknown": "未判定"}
    lines = [
        f"{it['date']}：{status_cn.get(it.get('status'), it.get('status'))}"
        + (f"（上班 {it['check_in_at']}）" if it.get("check_in_at") else "")
        + (f"（下班 {it['check_out_at']}）" if it.get("check_out_at") else "")
        for it in data.get("list", [])
    ]
    return f"[我的打卡 · {data.get('month')}]\n{_fmt_lines(lines)}"


async def tool_attendance_stats(args: Dict, ctx: RequestContext) -> str:
    """我的月度考勤统计（正常/迟到/早退/缺勤/请假天数）。"""
    ok, data = await _call("get_attendance_stats", {"month": args.get("month")} if args.get("month") else {}, ctx)
    if not ok:
        return f"[考勤统计查询失败] {data}"
    s = data.get("stat", {})
    return (
        f"[考勤统计 · {data.get('month')}] 应出勤 {data.get('scheduled_days')} 天："
        f"正常 {s.get('normal', 0)}，迟到 {s.get('late', 0)}，早退 {s.get('early', 0)}，"
        f"缺勤 {s.get('absent', 0)}，请假 {s.get('leave', 0)}"
    )


async def tool_my_leaves(args: Dict, ctx: RequestContext) -> str:
    """我的 OA 申请（请假/换班/补卡）及审批状态。"""
    ok, data = await _call("get_my_leaves", {}, ctx)
    if not ok:
        return f"[我的申请查询失败] {data}"
    status_cn = {"pending": "审批中", "approved": "已通过", "rejected": "已驳回", "canceled": "已撤销"}
    lines = [
        f"#{it['id']} {it.get('title')} → {status_cn.get(it.get('status'), it.get('status'))}"
        for it in data.get("list", [])
    ]
    return f"[我的申请]\n{_fmt_lines(lines)}"


async def tool_submit_leave(args: Dict, ctx: RequestContext) -> str:
    """提交请假（写操作）：审批人由 Laravel 侧自动分配，仍需走 OA 审批流。"""
    ok, data = await _call(
        "submit_leave",
        {
            "leave_type": args.get("leave_type") or "事假",
            "start_date": args.get("start_date", ""),
            "end_date": args.get("end_date", ""),
            "reason": args.get("reason", ""),
        },
        ctx,
    )
    if not ok:
        return f"[请假提交失败] {data}"
    return f"[请假已提交] 单号 #{data.get('process_id')}，{data.get('message')}"


async def tool_teaching_events(args: Dict, ctx: RequestContext) -> str:
    """教学活动列表。"""
    ok, data = await _call("list_teaching_events", {"status": args["status"]} if args.get("status") else {}, ctx)
    if not ok:
        return f"[教学活动查询失败] {data}"
    lines = [
        f"《{it.get('title')}》{it.get('start_at')} @ {it.get('location') or '地点待定'}（{it.get('department') or '-'}）"
        for it in data.get("list", [])
    ]
    return f"[教学活动]\n{_fmt_lines(lines)}"


async def tool_notices(args: Dict, ctx: RequestContext) -> str:
    """通知公告。"""
    ok, data = await _call("list_notices", {}, ctx)
    if not ok:
        return f"[通知查询失败] {data}"
    lines = [f"《{it.get('title')}》{it.get('published_at') or ''}" for it in data.get("list", [])]
    return f"[通知公告]\n{_fmt_lines(lines)}"


async def tool_my_evaluations(args: Dict, ctx: RequestContext) -> str:
    """我的考核成绩与评语。"""
    ok, data = await _call("get_my_evaluations", {}, ctx)
    if not ok:
        return f"[考核成绩查询失败] {data}"
    lines = [
        f"{it.get('template') or it.get('period') or '考核'}：{it.get('total_score')} 分"
        + (f"，评语：{it['comment']}" if it.get("comment") else "")
        for it in data.get("list", [])
    ]
    return f"[我的考核]\n{_fmt_lines(lines)}"


async def tool_my_cases(args: Dict, ctx: RequestContext) -> str:
    """我的病历记录及老师批改。"""
    ok, data = await _call("get_my_patient_cases", {}, ctx)
    if not ok:
        return f"[病历查询失败] {data}"
    lines = [
        f"《{it.get('title')}》（{it.get('department') or '-'}，{it.get('status')}"
        + (f"，{it['score']} 分" if it.get("score") is not None else "")
        + (f"，评语：{it['comment']}" if it.get("comment") else "")
        + "）"
        for it in data.get("list", [])
    ]
    return f"[我的病历]\n{_fmt_lines(lines)}"


async def tool_tasks_summary(args: Dict, ctx: RequestContext) -> str:
    """待办汇总。"""
    ok, data = await _call("get_my_tasks_summary", {}, ctx)
    if not ok:
        return f"[待办查询失败] {data}"
    parts = [
        f"待评审病历 {data.get('unreviewed_patient_cases', 0)} 份",
        f"进行中的申请 {data.get('my_pending_processes', 0)} 个",
        f"本月剩余值班 {data.get('duty_left_this_month', 0)} 次",
    ]
    if data.get("pending_approvals") is not None:
        parts.append(f"待我审批 {data['pending_approvals']} 单")
    return "[我的待办] " + "，".join(parts)


async def tool_department_students(args: Dict, ctx: RequestContext) -> str:
    """某科室当天在科学生（老师/管理员）。"""
    ok, data = await _call(
        "get_department_students",
        {"department_id": args["department_id"], **({"date": args["date"]} if args.get("date") else {})},
        ctx,
    )
    if not ok:
        return f"[在科学生查询失败] {data}"
    lines = [f"{it.get('student')}（{it.get('username')}）{it['start_date']} ~ {it['end_date']}" for it in data.get("list", [])]
    return f"[在科学生 · {data.get('date')}]\n{_fmt_lines(lines)}"


async def tool_pending_approvals(args: Dict, ctx: RequestContext) -> str:
    """待我审批的单子（老师/管理员）。"""
    ok, data = await _call("get_pending_approvals", {}, ctx)
    if not ok:
        return f"[待审批查询失败] {data}"
    lines = [f"#{it['process_id']} {it.get('title')}（申请人：{it.get('applicant')}）" for it in data.get("list", [])]
    return f"[待我审批]\n{_fmt_lines(lines)}"


# 工具名清单（供 RBAC 授权用）
INTERNSHIP_TOOLS = [
    "internship_my_rotation",
    "internship_my_duty",
    "internship_my_attendance",
    "internship_attendance_stats",
    "internship_my_leaves",
    "internship_submit_leave",
    "internship_teaching_events",
    "internship_notices",
    "internship_my_evaluations",
    "internship_my_cases",
    "internship_tasks_summary",
    "internship_department_students",
    "internship_pending_approvals",
]

_MONTH_PARAM = {
    "type": "object",
    "properties": {"month": {"type": "string", "description": "月份 YYYY-MM，不传为当月"}},
    "required": [],
}


def register_internship_tools() -> None:
    """把实习管理系统的工具注册进全局 registry。

    📌 命名一律 internship_ 前缀：与本项目原有工具（query_knowledge_base / submit_leave 等）区分，
       避免和老医院 Laravel 的 /dify/tools/* 工具语义混淆。
    """
    specs = [
        ToolSpec(
            name="internship_my_rotation",
            description="查询当前登录用户在实习管理系统里的轮转安排：每个时间段被分配到哪个科室",
            parameters={"type": "object", "properties": {}, "required": []},
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_my_rotation,
        ),
        ToolSpec(
            name="internship_my_duty",
            description="查询当前登录用户的值班安排（哪天哪个班次、在哪个科室）",
            parameters=_MONTH_PARAM,
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_my_duty,
        ),
        ToolSpec(
            name="internship_my_attendance",
            description="查询当前登录用户的逐日打卡记录（正常/迟到/早退/缺勤/请假及打卡时间）",
            parameters=_MONTH_PARAM,
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_my_attendance,
        ),
        ToolSpec(
            name="internship_attendance_stats",
            description="查询当前登录用户某个月的考勤统计：应出勤天数、正常/迟到/早退/缺勤/请假各多少天",
            parameters=_MONTH_PARAM,
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_attendance_stats,
        ),
        ToolSpec(
            name="internship_my_leaves",
            description="查询当前登录用户提交的 OA 申请（请假/换班/补卡）及其审批状态",
            parameters={"type": "object", "properties": {}, "required": []},
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_my_leaves,
        ),
        ToolSpec(
            name="internship_submit_leave",
            description="为当前登录用户在实习管理系统里提交请假申请（写操作，审批人由系统自动分配，仍需走审批流）。提交前需向用户确认请假类型、起止日期和事由",
            parameters={
                "type": "object",
                "properties": {
                    "leave_type": {"type": "string", "description": "事假 / 病假 / 其他，默认事假"},
                    "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD"},
                    "reason": {"type": "string", "description": "请假事由"},
                },
                "required": ["start_date", "end_date", "reason"],
            },
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_submit_leave,
        ),
        ToolSpec(
            name="internship_teaching_events",
            description="查询实习管理系统里的教学活动（讲座/培训）时间、地点与所属科室",
            parameters={"type": "object", "properties": {"status": {"type": "string"}}, "required": []},
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_teaching_events,
        ),
        ToolSpec(
            name="internship_notices",
            description="查询实习管理系统里的最新通知公告",
            parameters={"type": "object", "properties": {}, "required": []},
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_notices,
        ),
        ToolSpec(
            name="internship_my_evaluations",
            description="查询当前登录用户的出科考核成绩与带教评语",
            parameters={"type": "object", "properties": {}, "required": []},
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_my_evaluations,
        ),
        ToolSpec(
            name="internship_my_cases",
            description="查询当前登录用户（学生）书写的病历记录、评分与老师批改意见",
            parameters={"type": "object", "properties": {}, "required": []},
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_my_cases,
        ),
        ToolSpec(
            name="internship_tasks_summary",
            description="查询当前登录用户的待办汇总：待评审病历、进行中的申请、剩余值班次数、待审批单数",
            parameters={"type": "object", "properties": {}, "required": []},
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_tasks_summary,
        ),
        ToolSpec(
            name="internship_department_students",
            description="查询某科室在指定日期有哪些学生在科轮转（仅老师/管理员可用，学生调用会被拒绝）",
            parameters={
                "type": "object",
                "properties": {
                    "department_id": {"type": "integer", "description": "科室 ID"},
                    "date": {"type": "string", "description": "日期 YYYY-MM-DD，默认今天"},
                },
                "required": ["department_id"],
            },
            allowed_roles=[Role.ADMIN, Role.TEACHER],
            execute=tool_department_students,
        ),
        ToolSpec(
            name="internship_pending_approvals",
            description="查询待当前登录用户审批的 OA 单子（仅老师/管理员可用）",
            parameters={"type": "object", "properties": {}, "required": []},
            allowed_roles=[Role.ADMIN, Role.TEACHER],
            execute=tool_pending_approvals,
        ),
    ]
    for spec in specs:
        registry.register(spec)
