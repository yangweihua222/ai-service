# -*- coding: utf-8 -*-
"""MCP Server：将工具集封装为标准 MCP 工具（stdio / SSE / streamable-http）。
基于官方 mcp SDK（pip install mcp）。与 Agent 工具注册共享同一份定义（registry）。

启动：python -m app.mcp_server
（MCP 客户端如 Claude Desktop / Cursor / 自研 Agent 可通过 stdio 调用本服务工具）

📌 面试考点（对应《技术点深挖备考库》）：
  · 技术点5 MCP 协议（tools/resources/prompts 原语；stdio/SSE 传输） → 见 build_server() / list_tools() / call_tool()
  · 实战⑫ MCP 真实可跑（装 SDK 即起真 Server，非仅模式对齐）        → 见 _HAS_MCP 优雅降级 + 共享 registry
  · 工程点：SDK v1/v2 API 不同（v1 Server 装饰器 / v2 MCPServer.add_tool），此处做双路径兼容
"""
import inspect

# ---------- SDK 探测：优先 v2(MCPServer)，回退 v1(Server + 装饰器) ----------
_MCP_V2 = False
try:
    from mcp.server.mcpserver import MCPServer
    _MCP_V2 = True
    _HAS_MCP = True
except ImportError:
    try:
        from mcp.server import Server
        from mcp.server.stdio import stdio_server
        from mcp.types import TextContent, Tool
        _HAS_MCP = True
    except ImportError:
        _HAS_MCP = False

from app.tools_registry import registry
from app.tools import register_tools
from app.config import DEV_ROLE, DEV_TENANT, DEV_USER, validate_security_config, validate_redis_config

# 📌 P0 必修 (2026-09-08 code review): MCP Server 启动也走安全配置校验
#   stdio 模式允许 AUTH_MODE=dev（Claude Desktop/Cursor 进程内单租户场景）；
#   但即便 stdio 也强制校验 AUTH_MODE 合法 + jwt 模式密钥长度。
validate_security_config()
# 📌 启动期校验 Redis 后端配置（MCP Server 也接同一套治理/记忆）
validate_redis_config()

register_tools()

# 单例：避免每次调用都重建 Server
_server = None
# 启动期打印过 WARNING 后置 True（避免每次调用刷屏）
_dev_ctx_warned = False


def _stdio_dev_context_fallback():
    """⚠️ MCP stdio 模式专用兜底：从 app.config 读 DEV_TENANT/DEV_USER/DEV_ROLE 拼身份。

    📌 P0 必修 (2026-09-08 code review):
      - 协议层事实：stdio 是进程内单租户调用，Claude Desktop/Cursor 默认就是这种模式。
      - **绝不能用于 SSE/streamable-http 多客户端模式**——一旦 server 暴露成网络服务，
        所有客户端会共享同一 tenant_id（多租户崩塌）。
      - 未来切到 SSE/streamable-http 时，本函数应改为从 MCP metadata/headers 透传身份：
        ```python
        def _resolve_mcp_context(meta: dict) -> RequestContext:
            return RequestContext(
                tenant_id=meta.get("X-Tenant-Id") or "",
                user_id=meta.get("X-User-Id") or "",
                role=Role(meta.get("X-Role") or "student"),
            )
        ```
        然后在 _fn / call_tool 里把 meta 作为参数传进来。
    """
    global _dev_ctx_warned
    if not _dev_ctx_warned:
        import logging
        logging.getLogger(__name__).warning(
            "[security] mcp_server 使用 stdio dev 兜底身份（%s/%s/%s）。"
            " ⚠️ 仅 stdio 模式安全；切 SSE/streamable-http 前必须改为从 metadata 透传身份。",
            DEV_TENANT, DEV_USER, DEV_ROLE,
        )
        _dev_ctx_warned = True
    from app.models import RequestContext, Role
    return RequestContext(DEV_TENANT, DEV_USER, Role(DEV_ROLE))


def _make_tool_fn(spec):
    """把 registry 里的工具 spec 包装成带精确类型注解的函数。
    ⚙️ 为什么动态生成：MCP v2 的 add_tool 由函数签名与类型注解推导 input schema；
       而我们的工具是统一的 (args: dict, ctx) 形态，需按 spec.parameters 造出显式参数，
       才能让 MCP 客户端看到可读的参数列表（而不是一个笼统的 object）。"""
    props = (spec.parameters or {}).get("properties", {})
    type_map = {"integer": int, "number": float, "boolean": bool}
    params = []
    ann = {"return": str}
    for pname, pschema in props.items():
        ptype = type_map.get(pschema.get("type"), str)
        params.append(
            inspect.Parameter(
                pname, inspect.Parameter.POSITIONAL_OR_KEYWORD, default="", annotation=ptype
            )
        )
        ann[pname] = ptype

    async def _fn(**kwargs):
        # 📌 上下文解析：未来从 MCP 协议层 metadata 透传身份时，本行替换为 _resolve_mcp_context(meta)
        return await spec.execute({k: v for k, v in kwargs.items() if v != ""}, _stdio_dev_context_fallback())

    _fn.__signature__ = inspect.Signature(params)
    _fn.__annotations__ = ann
    _fn.__name__ = spec.name
    _fn.__doc__ = spec.description
    return _fn


def build_server():
    if not _HAS_MCP:
        raise RuntimeError("未安装 mcp SDK，请先执行：pip install mcp")
    # 📌 技术点5 MCP：把 registry 全部工具封装为标准 MCP Server(tools 原语)，供 Claude/Cursor 即插即用
    global _server
    if _server is not None:
        return _server

    if _MCP_V2:
        app = MCPServer("ai-teaching-assistant")
        for spec in registry.all():
            app.add_tool(_make_tool_fn(spec), name=spec.name, description=spec.description)
        _server = app
        return app

    app = Server("ai-teaching-assistant")

    @app.list_tools()
    async def list_tools():
        return [
            Tool(name=t.name, description=t.description, inputSchema=t.parameters)
            for t in registry.all()
        ]

    @app.call_tool()
    async def call_tool(name: str, arguments: dict):
        spec = registry.get(name)
        if not spec or not spec.execute:
            return [TextContent(type="text", text=f"未知工具: {name}")]
        # 📌 上下文解析：未来 SSE/streamable-http 时从 arguments 或 metadata 解析 tenant/user/role 注入 ctx
        result = await spec.execute(arguments or {}, _stdio_dev_context_fallback())
        return [TextContent(type="text", text=result)]

    _server = app
    return app


async def list_tools():
    """列出 MCP 暴露的工具（自测/健康检查用）。"""
    app = build_server()
    tools = await app.list_tools()
    out = []
    for t in tools:
        schema = getattr(t, "input_schema", None) or getattr(t, "inputSchema", {})
        out.append({"name": t.name, "description": t.description, "params": list((schema or {}).get("properties", {}))})
    return out


async def call_tool(name: str, arguments: dict) -> str:
    """调用 MCP 工具（自测/健康检查用），返回文本结果。"""
    app = build_server()
    res = await app.call_tool(name, arguments or {})
    # v2 返回 (CallToolResult, structured)；v1 直接返回 content 列表
    if isinstance(res, tuple):
        res = res[0]
    content = getattr(res, "content", res)  # CallToolResult → .content
    chunks = [getattr(c, "text", str(c)) for c in content]
    return "\n".join(chunks)


def run_stdio():
    app = build_server()
    if _MCP_V2:
        app.run(transport="stdio")
    else:
        import asyncio

        async def _run():
            async with stdio_server() as (read, write):
                await app.run(read, write, app.create_initialization_options())

        asyncio.run(_run())


if __name__ == "__main__":
    run_stdio()
