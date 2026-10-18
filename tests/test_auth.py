# -*- coding: utf-8 -*-
"""多租户 RBAC 鉴权测试（dev 模式用请求头模拟角色）"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.tools import register_tools

register_tools()
client = TestClient(app)


def test_student_tools_limited():
    resp = client.get("/api/v1/tools", headers={"X-Role": "student"})
    names = [t["name"] for t in resp.json()["tools"]]
    assert "query_knowledge_base" in names
    assert "query_teaching_data" not in names
    assert "send_notification" not in names


def test_student_blocked_from_teaching_data():
    resp = client.post(
        "/api/v1/chat",
        json={"message": "查询张三成绩"},
        headers={"X-Role": "student"},
    )
    data = resp.json()
    for s in data["steps"]:
        if s["tool"] == "query_teaching_data":
            assert "无权限" in s["observation"]


def test_teacher_can_use_all_tools():
    resp = client.get("/api/v1/tools", headers={"X-Role": "teacher"})
    names = [t["name"] for t in resp.json()["tools"]]
    assert {"query_knowledge_base", "query_teaching_data",
            "check_rotation_feasibility", "send_notification"} <= set(names)


def test_tenant_isolation_header():
    resp = client.get("/api/v1/tools", headers={"X-Tenant-Id": "hospital-B", "X-Role": "teacher"})
    assert resp.json()["tenant_id"] == "hospital-B"


# ============ 2026-09-08 code review P0-1 修复：validate_security_config 启动期校验 ============
class TestSecurityConfig:
    """验证 AUTH_MODE 校验 + JWT_SECRET 长度校验。"""

    def test_jwt模式密钥太短_抛错(self, monkeypatch):
        from app import config
        monkeypatch.setattr(config, "AUTH_MODE", "jwt")
        monkeypatch.setattr(config, "JWT_SECRET", "short")
        with pytest.raises(RuntimeError, match="JWT_SECRET 长度"):
            config.validate_security_config()

    def test_jwt模式密钥够长_通过(self, monkeypatch):
        from app import config
        monkeypatch.setattr(config, "AUTH_MODE", "jwt")
        monkeypatch.setattr(config, "JWT_SECRET", "a" * 32)
        config.validate_security_config()  # 不抛

    def test_非法auth_mode_抛错(self, monkeypatch):
        from app import config
        monkeypatch.setattr(config, "AUTH_MODE", "production")  # 非法
        with pytest.raises(RuntimeError, match="非法"):
            config.validate_security_config()

    def test_dev模式_不抛错(self, monkeypatch):
        from app import config
        monkeypatch.setattr(config, "AUTH_MODE", "dev")
        # 不会抛错（仅 WARNING）
        config.validate_security_config()
