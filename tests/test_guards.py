# -*- coding: utf-8 -*-
"""执行前两道防线单测：① 工具参数 JSON Schema 校验 ② JWT 身份解析。

对应备考库 2.2「参数校验三层」——Schema 协议约束之外，本地必须再校验一次（不信任模型输出）。
"""
import time

import jwt
import pytest
from fastapi import HTTPException

from app import rbac
from app.config import DEV_ROLE, DEV_TENANT, JWT_SECRET
from app.models import Role
from app.tools_registry import ToolSpec, validate_args


def _spec(parameters: dict) -> ToolSpec:
    return ToolSpec(name="t", description="test", parameters=parameters)


SCHEMA = {
    "type": "object",
    "properties": {
        "query": {"type": "string", "description": "提问"},
        "limit": {"type": "integer", "minimum": 1},
    },
    "required": ["query"],
}


# ---------- 参数校验 ----------
def test_validate_args_ok():
    assert validate_args(_spec(SCHEMA), {"query": "发烧怎么办"}) is None


def test_validate_args_missing_required():
    err = validate_args(_spec(SCHEMA), {})
    assert err and "query" in err  # 缺必填字段应报出字段名


def test_validate_args_wrong_type():
    err = validate_args(_spec(SCHEMA), {"query": 123})
    assert err is not None  # 类型不符


def test_validate_args_constraint_violation():
    err = validate_args(_spec(SCHEMA), {"query": "x", "limit": 0})  # minimum=1
    assert err is not None


def test_validate_args_none_arguments():
    """模型偶发不传 arguments（None）→ 按缺必填处理，不应抛异常。"""
    err = validate_args(_spec(SCHEMA), None)
    assert err is not None


def test_validate_args_schema_error_degrades_open():
    """工具自身 schema 写错 → 降级放行（不因代码 bug 阻塞全部调用）。"""
    assert validate_args(_spec({"type": "not-a-valid-type"}), {"a": 1}) is None


# ---------- JWT 解析 ----------
def _mk_token(payload: dict, secret: str = JWT_SECRET, exp_delta: int = 3600) -> str:
    body = {"exp": int(time.time()) + exp_delta, **payload}
    return jwt.encode(body, secret, algorithm="HS256")


@pytest.mark.asyncio
async def test_jwt_parses_identity():
    """带合法 token → 身份从 JWT 取，优先级高于头。"""
    token = _mk_token({"tenant_id": "hospital-a", "user_id": "s-001", "role": "student"})
    ctx = await rbac.get_request_context(
        x_tenant_id="ignored", x_user_id="ignored", x_user_num=None,
        x_role=None, x_authorization=f"Bearer {token}",
    )
    assert ctx.tenant_id == "hospital-a"
    assert ctx.user_id == "s-001"
    assert ctx.role == Role.STUDENT


@pytest.mark.asyncio
async def test_jwt_invalid_role_falls_back_to_student():
    """JWT 里角色是脏值 → 降级最小权限（不因脏数据放行高权限）。"""
    token = _mk_token({"tenant_id": "hospital-a", "user_id": "s-001", "role": "superadmin"})
    ctx = await rbac.get_request_context(
        x_tenant_id=None, x_user_id=None, x_user_num=None,
        x_role=None, x_authorization=f"Bearer {token}",
    )
    assert ctx.role == Role.STUDENT


@pytest.mark.asyncio
async def test_jwt_expired_rejects_401():
    token = _mk_token({"user_id": "s-001", "role": "student"}, exp_delta=-10)
    with pytest.raises(HTTPException) as ei:
        await rbac.get_request_context(
            x_tenant_id=None, x_user_id=None, x_user_num=None,
            x_role=None, x_authorization=f"Bearer {token}",
        )
    assert ei.value.status_code == 401


@pytest.mark.asyncio
async def test_jwt_wrong_signature_rejects_401():
    """错误签名的 token → 401（防伪核心）。"""
    token = _mk_token({"user_id": "s-001", "role": "admin"}, secret="attacker-secret")
    with pytest.raises(HTTPException) as ei:
        await rbac.get_request_context(
            x_tenant_id=None, x_user_id=None, x_user_num=None,
            x_role=None, x_authorization=f"Bearer {token}",
        )
    assert ei.value.status_code == 401


@pytest.mark.asyncio
async def test_jwt_mode_without_token_rejects_401(monkeypatch):
    """AUTH_MODE=jwt 却不带 token → 401（生产防漏）。"""
    monkeypatch.setattr(rbac, "AUTH_MODE", "jwt")
    with pytest.raises(HTTPException) as ei:
        await rbac.get_request_context(
            x_tenant_id=None, x_user_id=None, x_user_num=None,
            x_role=None, x_authorization=None,
        )
    assert ei.value.status_code == 401


@pytest.mark.asyncio
async def test_dev_mode_header_simulation_still_works(monkeypatch):
    """dev 模式向后兼容：头模拟仍可用（本地联调不挡路）。"""
    monkeypatch.setattr(rbac, "AUTH_MODE", "dev")
    ctx = await rbac.get_request_context(
        x_tenant_id="dev-tenant", x_user_id="u-1", x_user_num="2021001",
        x_role="teacher", x_authorization=None,
    )
    assert ctx.tenant_id == "dev-tenant"
    assert ctx.role == Role.TEACHER


@pytest.mark.asyncio
async def test_dev_mode_defaults_when_no_headers(monkeypatch):
    """dev 模式无任何头 → 回退固定默认上下文，tenant_id/user_id 始终非空。"""
    monkeypatch.setattr(rbac, "AUTH_MODE", "dev")
    ctx = await rbac.get_request_context(
        x_tenant_id=None, x_user_id=None, x_user_num=None,
        x_role=None, x_authorization=None,
    )
    assert ctx.tenant_id == DEV_TENANT
    assert ctx.role == Role(DEV_ROLE)
    assert ctx.user_id
