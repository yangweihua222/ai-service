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
    """真实工具：值班查询 → Laravel /dify/tools/get_duty_schedule。"""
    start_time = args.get("start_time", "")
    end_time = args.get("end_time", "")
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
    """真实工具：缺勤/打卡统计查询 → Laravel /dify/tools/get_absent_times。"""
    start_date = args.get("start_date", "")
    end_date = args.get("end_date", "")
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
    """真实工具：教学活动查询（带教老师视角）→ Laravel /dify/tools/get_teaching_event。"""
    start_date = args.get("start_date", "")
    end_date = args.get("end_date", "")
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
