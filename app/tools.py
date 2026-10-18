# -*- coding: utf-8 -*-
"""四个业务工具实现，统一签名 async execute(args, ctx) -> str。
工具③严格对齐语义红线：轮转=学员分配到科室，排班=考勤打卡（单独管理）。
对应简历「AI 教学助手 · 工具集（Function Calling / Tool Use）」。

📌 面试考点（对应《技术点深挖备考库》）：
  · 技术点4 工具集（Function Calling 全流程：定义/校验/执行/回填） → 见 register_tools()（每个工具一份 JSON Schema）
  · 实战③ 工具调用异常（JSON错乱/幻觉工具/参数错/超时/超长）      → 见 Agent 层 tool_calls 兜底（agent.py），本文件为工具实现
  · 实战⑦ 幂等（通知是写操作，必须幂等键防重复）                  → 见 tool_notification()（真实场景带唯一业务 ID）
"""
from typing import Dict, List

from app import data, rag
from app.backend import (
    BackendClient,
    XuanwuBackend,
    TOOL_ABSENT_TIMES,
    TOOL_DUTY_SCHEDULE,
    TOOL_SUBMIT_LEAVE,
    TOOL_TEACHING_EVENT,
)
from app.models import RequestContext, Role
from app.tools_registry import ToolSpec, registry
from app.internship import register_internship_tools

# 📌 小萱同源：真实后端客户端（调用 Laravel /api/dify/tools/*）。
#    未配置 BACKEND_BASE_URL 时，真实工具返回友好提示而非报错（演示/联调友好）。
backend = BackendClient()

# 📌 真实平台直连（宣武实习管理平台，AES 加密协议，见 app/xuanwu_client.py）：
#    配齐 XUANWU_API_BASE/NUM/PASSWORD 三项时启用，值班等工具优先走真实平台数据。
#    测试里通过 monkeypatch 把本属性置 None 以保持用例封闭（不发真实网络请求）。
xuanwu = XuanwuBackend()


class NotificationStore:
    """通知队列（内存）。真实场景写消息表 / 推送网关。"""

    _queue: List[Dict] = []

    @classmethod
    def push(cls, tenant_id: str, target: str, message: str, sender: str):
        cls._queue.append(
            {"tenant_id": tenant_id, "target": target, "message": message, "sender": sender}
        )

    @classmethod
    def all(cls) -> List[Dict]:
        return cls._queue


async def tool_rag(args: Dict, ctx: RequestContext) -> str:
    """工具①：临床知识库 RAG 检索（带引用）。"""
    return await rag.rag_search(args.get("query", ""), ctx.tenant_id)


async def tool_usage_guide(args: Dict, ctx: RequestContext) -> str:
    """工具①-b：实习系统使用说明检索（双知识库之一，与临床知识库对称）。"""
    return await rag.usage_search(args.get("query", ""), ctx.tenant_id)


async def tool_text2sql(args: Dict, ctx: RequestContext) -> str:
    """工具②：学员数据 Text2SQL 查询 + 档案摘要。
    ⚙️ 真执行链路：Schema 注入 → 生成 SQL（带租户条件）→ 安全闸门校验 → 本地库真跑 → 摘要。
    📌 安全：执行前 assert_sql_safe 强制"只读 + 必须带 tenant_id"，不信任模型输出。
    📌 语义红线：考勤类问题（考勤/打卡/出勤/缺卡/补卡）走 attendance 摘要器，
       其余轮转成绩问题走 rotation 摘要器——二者职责不同，绝不混用。"""
    q = args.get("question", "")
    try:
        sql = data.generate_sql(q, ctx.tenant_id)
        rows = data.execute_sql(sql, ctx.tenant_id)
    except ValueError as e:  # 安全拦截（如生成了写操作/缺租户过滤）
        return f"查询被安全策略拦截：{e}"
    except Exception as e:
        return f"查询执行失败：{e}"
    if not rows:
        return f"生成SQL：{sql}\n未查询到相关记录。"
    # 路由摘要器（与 data.generate_sql 的表路由保持一致）
    is_attendance = any(k in q for k in ("考勤", "打卡", "出勤", "缺卡", "补卡"))
    summary = data.summarize_attendance(rows) if is_attendance else data.summarize_rotation(rows)
    return f"生成SQL：{sql}\n{summary}"


async def tool_scheduling(args: Dict, ctx: RequestContext) -> str:
    """工具③：轮转/排班可行性检测（贪心+局部搜索引擎）。"""
    return data.check_rotation_feasibility()


async def tool_notification(args: Dict, ctx: RequestContext) -> str:
    """工具④：通知推送（站内信/提醒）。
    📌 实战⑦ 幂等：通知是写操作，真实场景带唯一业务 ID(如 notify_id) 执行前查重，防 Agent 重试重复推送。"""
    target = args.get("target", "")
    message = args.get("message", "")
    NotificationStore.push(ctx.tenant_id, target, message, ctx.user_id)
    return f"[通知已提交] → {target}：{message}"


# ===================== 真实平台数据格式化（宣武实习管理平台字段）=====================
_SHIFT_LABEL = {0: "综合", 1: "白班", 2: "夜班"}
# 打卡详情里的 check_in_status 语义（逆向平台前端确认）：4=缺卡，6=正常（其余见值）
_CHECK_STATUS_LABEL = {1: "正常", 2: "迟到", 3: "早退", 4: "缺卡", 6: "正常"}


def _count_check_status(item: Dict) -> Dict[int, int]:
    """统计某条打卡记录里各 check_in_status 的次数。"""
    counts: Dict[int, int] = {}
    for grp in (item.get("list") or {}).values():
        if isinstance(grp, dict):
            for v in grp.values():
                if isinstance(v, dict) and "check_in_status" in v:
                    st = v["check_in_status"]
                    counts[st] = counts.get(st, 0) + 1
    return counts


def _fmt_absent_line(item: Dict) -> str:
    """缺勤/打卡统计 → 一行可读文本（学员×日期×班次粒度）。"""
    name = item.get("name", "?")
    date = item.get("date", "?")
    shift = _SHIFT_LABEL.get(item.get("shift_type"), "?")
    dept = item.get("department") or "-"
    rate = item.get("attendanceRate")
    rate_s = f"{rate}%" if isinstance(rate, (int, float)) else str(rate)
    line = f"{name} | {date} {shift} | {dept} | 出勤率 {rate_s}"
    counts = _count_check_status(item)
    if counts:
        detail = " ".join(
            f"{_CHECK_STATUS_LABEL.get(k, f'状态{k}')}{v}次" for k, v in sorted(counts.items())
        )
        line += f" | {detail}"
    return line


def _fmt_teaching_event_line(item: Dict) -> str:
    """教学活动 → 一行可读文本。"""
    type_label = (item.get("type_struct") or {}).get("label", "活动")
    summary = item.get("summary") or "-"
    dept = (item.get("department") or {}).get("title") or "-"
    teacher = (item.get("teacher") or {}).get("name") or "-"
    venue = item.get("venue") or "-"
    status = (item.get("status_struct") or {}).get("label") or item.get("status") or "-"
    t = (item.get("start_time") or item.get("estimated_start_time") or "")[:10]
    return f"[{type_label}] {summary} | {dept} | 教师:{teacher} | 地点:{venue} | {t} | {status}"


# ===================== 真实后端工具（小萱同源，调用 Laravel /api/dify/tools/*）=====================
# 📌 实战③ 工具调用异常：网络/超时统一在 backend.call_tool 抛错，此处 try/except 转成可读文本，
#    不会因后端抖动让整个对话崩溃（Agent 层 also 有兜底）。
_NOT_CONFIGURED = "后端未配置：请设置环境变量 BACKEND_BASE_URL 指向你的 Laravel 平台（含 /api），即可连接真实业务。"


async def tool_submit_leave(args: Dict, ctx: RequestContext) -> str:
    """真实工具：提交请假（写操作）→ Laravel /dify/tools/submit_leave。
    📌 与 DifyController::submit_leave 同源：建 oa_leaves + oa_processes + 分配审批人 + 站内信。
    user_id 传 ctx.user_num（学号/工号），生产由前端经 X-User-Num 下发。"""
    if not backend.configured:
        return f"[提交请假·演示] {_NOT_CONFIGURED}"
    payload = {
        "user_id": ctx.user_num,
        "reason": args.get("reason", ""),
        "start_time": args.get("start_time", ""),
        "end_time": args.get("end_time", ""),
    }
    try:
        resp = await backend.call_tool(TOOL_SUBMIT_LEAVE, payload)
        return f"[请假已提交] {backend.extract_data(resp)}"
    except Exception as e:
        return f"[请假提交失败] {e}"


async def tool_query_duty(args: Dict, ctx: RequestContext) -> str:
    """真实工具：值班查询。
    数据源优先级：真实平台（XuanwuBackend，AES 加密协议）> Laravel /dify/tools/get_duty_schedule。
    真实平台的排班接口无「按人查询」参数 → 分页拉全量后客户端按姓名/日期过滤。"""
    name = (args.get("name") or "").strip()
    start_time = args.get("start_time", "")
    end_time = args.get("end_time", "")
    # ---- 优先走真实平台直连 ----
    if xuanwu is not None and xuanwu.configured:
        try:
            rows, total = await xuanwu.query_duty(
                name=name, start_time=start_time, end_time=end_time
            )
        except Exception as e:
            return f"[值班查询失败] 真实平台调用异常：{e}"
        if not rows:
            scope = f"「{name}」" if name else "该时间段"
            return f"[值班安排] 平台共 {total} 条排班记录，未匹配到 {scope} 的值班记录。"
        lines = [
            "{} 周{} | 值班医生:{} | 值班学生:{} | 考勤组:{}".format(
                it.get("date", "?"),
                it.get("week", "?"),
                it.get("doctor_name", "?"),
                (it.get("student") or {}).get("name", "-"),
                it.get("attendance_configs_name", "-"),
            )
            for it in rows[:20]
        ]
        tail = f"\n(共匹配 {len(rows)} 条，显示前 20 条)" if len(rows) > 20 else ""
        return "[值班安排·真实平台数据]\n" + "\n".join(lines) + tail
    # ---- 回退：Laravel 明文契约（/dify/tools/*） ----
    if not backend.configured:
        return f"[值班查询·演示] {_NOT_CONFIGURED}"
    payload = {
        "user_id": ctx.user_num,
        "start_time": start_time,
        "end_time": end_time,
    }
    try:
        resp = await backend.call_tool(TOOL_DUTY_SCHEDULE, payload)
        return f"[值班安排] {backend.extract_data(resp)}"
    except Exception as e:
        return f"[值班查询失败] {e}"


async def tool_query_absent(args: Dict, ctx: RequestContext) -> str:
    """真实工具：缺勤/打卡统计查询。
    数据源优先级：真实平台（XuanwuBackend，attendance_statistics）> Laravel /dify/tools/get_absent_times。
    真实平台接口原生支持 name/start/end 过滤，返回「学员×日期×班次」粒度打卡统计（含出勤率与缺卡次数）。"""
    name = (args.get("name") or "").strip()
    start_date = args.get("start_date", "")
    end_date = args.get("end_date", "")
    # ---- 优先走真实平台直连 ----
    if xuanwu is not None and xuanwu.configured:
        try:
            rows, total = await xuanwu.query_absent(
                name=name, start_date=start_date, end_date=end_date
            )
        except Exception as e:
            return f"[缺勤查询失败] 真实平台调用异常：{e}"
        if not rows:
            scope = f"「{name}」" if name else "该时间段"
            return f"[打卡统计] 平台共 {total} 条记录，未匹配到 {scope} 的打卡/缺勤记录。"
        lines = [_fmt_absent_line(it) for it in rows[:30]]
        tail = f"\n(共匹配 {len(rows)} 条，显示前 30 条)" if len(rows) > 30 else ""
        return "[打卡统计·真实平台数据]\n" + "\n".join(lines) + tail
    # ---- 回退：Laravel 明文契约（/dify/tools/*） ----
    if not backend.configured:
        return f"[缺勤查询·演示] {_NOT_CONFIGURED}"
    payload = {
        "user_id": ctx.user_num,
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        resp = await backend.call_tool(TOOL_ABSENT_TIMES, payload)
        return f"[缺勤统计] {backend.extract_data(resp)}"
    except Exception as e:
        return f"[缺勤查询失败] {e}"


async def tool_query_teaching_event(args: Dict, ctx: RequestContext) -> str:
    """真实工具：教学活动查询（带教老师视角）。
    数据源优先级：真实平台（XuanwuBackend，teaching_events）> Laravel /dify/tools/get_teaching_event。"""
    start_date = args.get("start_date", "")
    end_date = args.get("end_date", "")
    # ---- 优先走真实平台直连 ----
    if xuanwu is not None and xuanwu.configured:
        try:
            rows, total = await xuanwu.query_teaching_events(
                start_date=start_date, end_date=end_date
            )
        except Exception as e:
            return f"[教学活动查询失败] 真实平台调用异常：{e}"
        if not rows:
            return f"[教学活动] 平台共 {total} 条活动，未匹配到该时间段的教学活动。"
        lines = [_fmt_teaching_event_line(it) for it in rows[:30]]
        tail = f"\n(共匹配 {len(rows)} 条，显示前 30 条)" if len(rows) > 30 else ""
        return "[教学活动·真实平台数据]\n" + "\n".join(lines) + tail
    # ---- 回退：Laravel 明文契约（/dify/tools/*） ----
    if not backend.configured:
        return f"[教学活动查询·演示] {_NOT_CONFIGURED}"
    payload = {
        "user_id": ctx.user_num,
        "start_date": start_date,
        "end_date": end_date,
    }
    try:
        resp = await backend.call_tool(TOOL_TEACHING_EVENT, payload)
        return f"[教学活动] {backend.extract_data(resp)}"
    except Exception as e:
        return f"[教学活动查询失败] {e}"


def register_tools():
    """把四个工具注册进全局 registry（每个工具一份 JSON Schema + 权限）。
    📌 技术点4 工具集落地：每个工具一份 JSON Schema(name/description/parameters) + allowed_roles(RBAC 预筛)。"""
    registry.register(
        ToolSpec(
            name="query_knowledge_base",
            description="基于临床知识库(RAG)检索教材/指南/病例模板，回答医学知识类问题，返回带引用来源的答案",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "学员的自然语言医学提问"}},
                "required": ["query"],
            },
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_rag,
        )
    )
    registry.register(
        ToolSpec(
            name="query_usage_guide",
            description="检索实习系统操作说明(FAQ)，如'怎么提交请假''怎么查值班''缺卡了怎么办'，返回带引用来源的答案",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string", "description": "学员的自然语言操作问题"}},
                "required": ["query"],
            },
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_usage_guide,
        )
    )
    registry.register(
        ToolSpec(
            name="query_teaching_data",
            description="基于教学数据库(Text2SQL)查询实习生轮转/考勤/考核数据，返回 SQL 与结果摘要",
            parameters={
                "type": "object",
                "properties": {"question": {"type": "string", "description": "自然语言数据查询，如'张三内科成绩'"}},
                "required": ["question"],
            },
            allowed_roles=[Role.ADMIN, Role.TEACHER],
            execute=tool_text2sql,
        )
    )
    registry.register(
        ToolSpec(
            name="check_rotation_feasibility",
            description="调用轮转分配引擎(贪心+局部搜索)校验轮转(学员→科室)计划可行性，返回缺口与调整建议；排班(考勤)单独管理",
            parameters={"type": "object", "properties": {}, "required": []},
            allowed_roles=[Role.ADMIN, Role.TEACHER],
            execute=tool_scheduling,
        )
    )
    registry.register(
        ToolSpec(
            name="send_notification",
            description="向学生/带教发送站内信或提醒，如成绩预警、轮转安排通知",
            parameters={
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "接收者角色或姓名"},
                    "message": {"type": "string", "description": "提醒内容"},
                },
                "required": ["target", "message"],
            },
            allowed_roles=[Role.ADMIN, Role.TEACHER],
            execute=tool_notification,
        )
    )
    # ---- 真实后端工具（小萱同源：业务建单走 Laravel，AI 服务只编排）----
    registry.register(
        ToolSpec(
            name="submit_leave",
            description="为学生提交请假申请（写操作），调用真实后端建单并分配审批人、发站内信；需要 reason/起止时间",
            parameters={
                "type": "object",
                "properties": {
                    "reason": {"type": "string", "description": "请假事由，如'感冒发烧'"},
                    "start_time": {"type": "string", "description": "请假开始日期，格式 YYYY-MM-DD"},
                    "end_time": {"type": "string", "description": "请假结束日期，格式 YYYY-MM-DD"},
                },
                "required": ["reason", "start_time", "end_time"],
            },
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_submit_leave,
        )
    )
    registry.register(
        ToolSpec(
            name="query_duty_schedule",
            description="查询值班排班安排（真实平台数据），可按值班医生或值班学生的姓名查询，也可按时间段过滤",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "要查询的姓名（值班医生或值班学生的全名），如'张三'；不传则查全部"},
                    "start_time": {"type": "string", "description": "开始日期 YYYY-MM-DD（可选）"},
                    "end_time": {"type": "string", "description": "结束日期 YYYY-MM-DD（可选）"},
                },
                "required": [],
            },
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_query_duty,
        )
    )
    registry.register(
        ToolSpec(
            name="query_absent_times",
            description="查询学员的缺勤/缺卡打卡统计（真实平台数据），可按姓名或时间段过滤，返回出勤率与缺卡次数",
            parameters={
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "要查询的学员姓名，如'陈钰卓'；不传则查全部"},
                    "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD（可选）"},
                    "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD（可选）"},
                },
                "required": [],
            },
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_query_absent,
        )
    )
    registry.register(
        ToolSpec(
            name="query_teaching_event",
            description="查询教学活动安排（真实平台数据，如小讲课/教学查房），可按时间段过滤",
            parameters={
                "type": "object",
                "properties": {
                    "start_date": {"type": "string", "description": "开始日期 YYYY-MM-DD（可选）"},
                    "end_date": {"type": "string", "description": "结束日期 YYYY-MM-DD（可选）"},
                },
                "required": [],
            },
            allowed_roles=[Role.ADMIN, Role.TEACHER, Role.STUDENT],
            execute=tool_query_teaching_event,
        )
    )
    # ---- 实习管理系统工具（D:\实习系统\internship-server 的 AI 工具网关）----
    # 📌 单独一个文件 app/internship.py：不侵入原有 RAG / Text2SQL / 老医院后端工具。
    register_internship_tools()
