# -*- coding: utf-8 -*-
"""请求 / 响应 与领域模型

📌 面试考点（对应《技术点深挖备考库》）：
  · 技术点7/9 多租户（RequestContext 携带 tenant_id，全程透传） → 见 RequestContext
  · 实战⑧ 可观测（每轮 ToolStep 落盘，badcase 可回溯）          → 见 ToolStep（tool/args/observation）
"""
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Role(str, Enum):
    ADMIN = "admin"
    TEACHER = "teacher"
    STUDENT = "student"


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    stream: bool = False


class ToolStep(BaseModel):
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    observation: str = ""


class ChatResponse(BaseModel):
    answer: str
    steps: List[ToolStep] = Field(default_factory=list)
    tenant_id: str = ""
    session_id: str = ""
    model: str = "mock"


class RequestContext:
    """运行期上下文，由鉴权中间件填充（非 pydantic，便于承载 Enum）。
    📌 技术点7/9 多租户：tenant_id 贯穿全链路(鉴权→工具执行)，保证数据按租户隔离。
    📌 user_num：真实后端的 user_id 字段要的是「学号/工号(id_num)」，不是 AI 服务内部 user_id；
       由 X-User-Num 头注入（生产前端下发），缺省回退 user_id。"""

    def __init__(self, tenant_id: str, user_id: str, role: Role, user_num: str = "", internship_token: str = ""):
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.role = role
        self.user_num = user_num or user_id
        # 实习管理系统（Laravel）的用户 Sanctum token，由前端经 X-Internship-Token 下发，
        # 工具调用时原样透传给 Laravel 网关做数据权限判定（谁的 token 就查谁的数据）。
        self.internship_token = internship_token

    def __repr__(self):
        return f"RequestContext(tenant={self.tenant_id}, user={self.user_id}, role={self.role.value})"
