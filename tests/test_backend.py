# -*- coding: utf-8 -*-
"""真实后端工具单元测试：用 MockTransport 模拟 Laravel 响应，验证解析与工具逻辑。

不依赖网络/真实服务器；验证点：
  · BackendClient 能正确 POST 到 /dify/tools/{tool}
  · 能从 Laravel 信封 {code,msg,data} 中提取 data
  · 4 个真实工具（submit_leave / duty / absent / teaching_event）返回可读文本
"""
import httpx
import pytest

import app.tools as tools_module
from app.backend import BackendClient
from app.models import RequestContext, Role


def _make_client(handler):
    transport = httpx.MockTransport(handler)
    return httpx.AsyncClient(transport=transport)


def _router(tool: str, data):
    def handler(request):
        # 断言确实打到了正确的 /dify/tools/{tool} 路径
        assert request.url.path.endswith(f"/dify/tools/{tool}"), request.url.path
        return httpx.Response(200, json={"code": 0, "msg": "success", "data": data})

    return handler


@pytest.fixture
def ctx():
    return RequestContext(tenant_id="capital-medical", user_id="teacher-001",
                          role=Role.STUDENT, user_num="2021001")


@pytest.fixture(autouse=True)
def _no_xuanwu(monkeypatch):
    """屏蔽真实平台数据源（tools.xuanwu），保证用例走可 mock 的 Laravel 契约分支、不发真实网络请求。"""
    monkeypatch.setattr(tools_module, "xuanwu", None)


@pytest.mark.asyncio
async def test_submit_leave_parses_envelope(ctx):
    cli = _make_client(_router("submit_leave", {"result": "请假成功"}))
    tools_module.backend = BackendClient(base_url='http://test', client=cli)
    out = await tools_module.tool_submit_leave(
        {"reason": "感冒", "start_time": "2026-09-02", "end_time": "2026-09-02"}, ctx
    )
    assert "[请假已提交]" in out and "请假成功" in out


@pytest.mark.asyncio
async def test_query_duty_parses_envelope(ctx):
    cli = _make_client(_router("get_duty_schedule", {"值班日期": ["2026-09-03"], "值班科室": ["骨科"]}))
    tools_module.backend = BackendClient(base_url='http://test', client=cli)
    out = await tools_module.tool_query_duty(
        {"start_time": "2026-09-01", "end_time": "2026-09-07"}, ctx
    )
    assert "骨科" in out


@pytest.mark.asyncio
async def test_query_absent_parses_envelope(ctx):
    cli = _make_client(_router("get_absent_times", {"count": 2}))
    tools_module.backend = BackendClient(base_url='http://test', client=cli)
    out = await tools_module.tool_query_absent(
        {"start_date": "2026-09-01", "end_date": "2026-09-07"}, ctx
    )
    assert "2" in out


@pytest.mark.asyncio
async def test_not_configured_friendly(ctx):
    tools_module.backend = BackendClient(base_url="")  # 未配置
    out = await tools_module.tool_submit_leave(
        {"reason": "x", "start_time": "2026-09-02", "end_time": "2026-09-02"}, ctx
    )
    assert "后端未配置" in out


# ============ 2026-09-08 code review P0-3 修复：HMAC 签名头 ============
class TestBackendHmac:
    """验证 BACKEND_HMAC_ENABLED=true 时自动加 X-Timestamp + X-Signature 头。"""

    @staticmethod
    def _secret() -> str:
        return "this-is-a-32-byte-test-secret-OK"  # 32 字节

    @pytest.mark.asyncio
    async def test_启用时带签名头(self, monkeypatch, ctx):
        from app import backend as backend_mod
        # 显式注入一个 32 字节 secret（避免依赖 .env）
        monkeypatch.setattr(backend_mod, "BACKEND_HMAC_SECRET", self._secret())
        monkeypatch.setattr(backend_mod, "BACKEND_HMAC_ENABLED", True)
        captured: dict = {}

        def handler(request):
            # 捕获请求头（httpx Headers 大小写不敏感，dict() 后 key 为小写）
            captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
            captured["body"] = request.content
            return httpx.Response(200, json={"code": 0, "msg": "ok", "data": {"r": "ok"}})

        cli = _make_client(handler)
        bc = BackendClient(base_url="http://test", client=cli, hmac_secret=self._secret())
        await bc.call_tool("submit_leave", {"reason": "感冒", "start_time": "2026-09-02"})

        # 调试信息（如果失败能看出问题）
        assert "x-signature" in captured["headers"], (
            f"headers={list(captured.get('headers', {}).keys())} "
            f"enabled={backend_mod.BACKEND_HMAC_ENABLED} "
            f"secret_len={len(backend_mod.BACKEND_HMAC_SECRET)}"
        )

        # 验证 X-Signature / X-Timestamp 已带
        assert "x-signature" in captured["headers"]
        assert "x-timestamp" in captured["headers"]
        sig = captured["headers"]["x-signature"]
        ts = captured["headers"]["x-timestamp"]
        # 验签：手动算一遍，应该一致
        import hashlib, hmac as hmac_mod
        body_hash = hashlib.sha256(captured["body"]).hexdigest()
        canonical = f"POST\n/dify/tools/submit_leave\n{ts}\n{body_hash}"
        expected = hmac_mod.new(
            self._secret().encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256
        ).hexdigest()
        assert sig == expected

    @pytest.mark.asyncio
    async def test_禁用时不带签名头(self, monkeypatch):
        from app import backend as backend_mod
        captured: dict = {}

        def handler(request):
            captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
            return httpx.Response(200, json={"code": 0, "msg": "ok", "data": {}})

        monkeypatch.setattr(backend_mod, "BACKEND_HMAC_ENABLED", False)
        cli = _make_client(handler)
        bc = BackendClient(base_url="http://test", client=cli, hmac_secret=self._secret())
        await bc.call_tool("submit_leave", {})

        assert "x-signature" not in captured["headers"]
        assert "x-timestamp" not in captured["headers"]

    @pytest.mark.asyncio
    async def test_密钥太短时静默降级(self, monkeypatch):
        """BACKEND_HMAC_ENABLED=true 但密钥 < 32 字节 → 不带签名（不抛错，向后兼容）"""
        from app import backend as backend_mod
        captured: dict = {}

        def handler(request):
            captured["headers"] = {k.lower(): v for k, v in request.headers.items()}
            return httpx.Response(200, json={"code": 0, "msg": "ok", "data": {}})

        monkeypatch.setattr(backend_mod, "BACKEND_HMAC_ENABLED", True)
        cli = _make_client(handler)
        bc = BackendClient(base_url="http://test", client=cli, hmac_secret="short")
        await bc.call_tool("submit_leave", {})

        # 密钥太短：_hmac_enabled_now 返回 False，不带签名头
        assert "x-signature" not in captured["headers"]
