# -*- coding: utf-8 -*-
"""多租户 + RBAC：租户隔离、角色权限、鉴权依赖（FastAPI Depends）。

- 租户隔离：每个请求携带 X-Tenant-Id，工具执行时可据此隔离数据（如 RAG 语料、通知队列）。
- 角色权限：admin / teacher / student 三级，工具级细粒度校验。
- 鉴权：dev 模式用固定上下文；jwt 模式预留解码接口。
对应简历「多租户 RBAC」「平台经 API 调用 + 权限约束」。

📌 面试考点（对应《技术点深挖备考库》）：
  · 技术点7 多租户架构（AI 系统服务多院校/机构，租户隔离）   → 见 get_request_context()（注入 tenant_id）
  · 技术点8 RBAC 多角色权限（学员/带教/管理，数据权限）       → 见 ROLE_PERMISSIONS / can_use()
  · 技术点9 多租户数据隔离（行级 + 向量库 namespace）          → 见 get_request_context() + tools 里按 ctx.tenant_id 隔离
  · 实战⑪ 多租户 Agent 隔离（数据/权限/工具三层）              → 见 can_use()（工具级）+ 中间件注入 tenant
"""
import jwt
from fastapi import Header, HTTPException

from app.config import AUTH_MODE, DEV_ROLE, DEV_TENANT, DEV_USER, JWT_SECRET
from app.internship import INTERNSHIP_TOOLS
from app.models import RequestContext, Role

# JWT 解析算法白名单（绝不允许 alg=none / 非对称算法混淆）
JWT_ALGORITHMS = ["HS256"]

# 角色 -> 允许调用的工具名（"*" 表示全部）
# 📌 技术点8 RBAC 多角色权限：学员/带教/管理三级，工具级细粒度；学员仅能查知识库，写操作仅 admin/teacher
# 新增真实后端工具（小萱同源）：学生可提交请假/查值班/查缺勤，老师/管理覆盖全部
REAL_TOOLS = [
    "submit_leave",            # 提交请假（写操作，学生也可，对应 OA 学生端）
    "query_duty_schedule",     # 值班查询
    "query_absent_times",      # 缺勤次数
    "query_teaching_event",    # 教学活动查询
]
ROLE_PERMISSIONS = {
    Role.ADMIN: ["*"],
    Role.TEACHER: [
        "query_knowledge_base",
        "query_usage_guide",
        "query_teaching_data",
        "check_rotation_feasibility",
        "send_notification",
    ] + REAL_TOOLS + INTERNSHIP_TOOLS,
    # 📌 学生可查自己的实习数据（轮转/值班/考勤/成绩/病历）+ 提交请假；
    #    老师专属工具（查在科学生/待审批）在 Laravel 网关侧按 token 权限兜底拒绝（403）
    Role.STUDENT: ["query_knowledge_base", "query_usage_guide"] + REAL_TOOLS + INTERNSHIP_TOOLS,
}

def can_use(role: Role, tool_name: str) -> bool:
    """RBAC 工具级权限校验（权威来源）。
    📌 技术点8：工具调用前统一过 can_use()；技术点7/9 多租户隔离在 get_request_context 注入。"""
    perms = ROLE_PERMISSIONS.get(role, [])
    return "*" in perms or tool_name in perms


async def get_request_context(
    x_tenant_id: str = Header(None),
    x_user_id: str = Header(None),
    x_user_num: str = Header(None, alias="X-User-Num"),
    x_role: str = Header(None),
    x_authorization: str = Header(None, alias="X-Authorization"),
    # 实习管理系统（Laravel）的用户 token：前端下发，工具调用时透传给网关做数据权限判定
    x_internship_token: str = Header(None, alias="X-Internship-Token"),
) -> RequestContext:
    """鉴权依赖：解析租户/用户/角色，注入到每个请求。

    两种模式（AUTH_MODE 环境变量控制）：
      · jwt ：生产模式，解析 X-Authorization 头的 Bearer JWT，验签后取身份（**身份不可伪造**）
      · dev ：本地联调，用固定默认上下文，可用 X-Role / X-Tenant-Id / X-User-Num 头模拟
    无论哪种模式，只要带了 token 就优先走 JWT 解析——保证"生产用 JWT、联调不挡路"。
    📌 技术点7/9 多租户：X-Tenant-Id 一路透传到工具执行(如 rag_search(ctx.tenant_id))，保证数据不串。
    📌 X-User-Num：真实后端的 user_id 字段要「学号/工号」，生产由前端下发，联调可用头模拟。
    """
    # ── ① JWT 优先：有 token 就验签解析（生产身份来源，签名防伪）──
    token = x_authorization
    if token and token.lower().startswith("bearer "):
        token = token[7:].strip()
    if token:
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=JWT_ALGORITHMS)
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="登录凭证已过期，请重新登录")
        except jwt.InvalidTokenError as e:
            raise HTTPException(status_code=401, detail=f"无效的身份凭证：{e}")
        # 角色非法值 → 降级到最小权限 STUDENT（不因脏数据放行高权限）
        raw_role = payload.get("role")
        try:
            role = Role(raw_role) if raw_role else Role.STUDENT
        except ValueError:
            role = Role.STUDENT
        tenant_id = payload.get("tenant_id") or x_tenant_id or DEV_TENANT
        user_id = payload.get("user_id") or payload.get("sub") or x_user_id or DEV_USER or "unknown"
        user_num = payload.get("user_num") or x_user_num or user_id
        return RequestContext(
            tenant_id=tenant_id, user_id=user_id, role=role, user_num=user_num,
            internship_token=x_internship_token or "",
        )

    # ── ② 强制 jwt 模式却没带 token → 直接 401（生产防漏）──
    if AUTH_MODE == "jwt":
        raise HTTPException(status_code=401, detail="缺少身份凭证（请携带 X-Authorization: Bearer <token>）")

    # ── ③ dev 模式：头模拟 / 固定默认上下文（本地联调用，不对外）──
    role = Role(DEV_ROLE)  # dev 默认角色（可被 X-Role 头覆盖）
    if x_role:
        try:
            role = Role(x_role)
        except ValueError:
            role = Role.STUDENT
    # 缺省时回退到 dev 默认上下文，保证 tenant_id / user_id 始终非空
    tenant_id = x_tenant_id or DEV_TENANT
    user_id = x_user_id or DEV_USER or "unknown"
    user_num = x_user_num or user_id
    return RequestContext(
        tenant_id=tenant_id, user_id=user_id, role=role, user_num=user_num,
        internship_token=x_internship_token or "",
    )
