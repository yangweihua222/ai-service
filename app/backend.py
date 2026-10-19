# -*- coding: utf-8 -*-
"""后端客户端：调用医院 Laravel 平台的「Dify 工具回调」接口。

📌 为什么这样做（对应《技术点深挖备考库》· 实战③ / 技术点7/9）：
  小萱（Dify 智能体）的本质就是「LLM 编排 + Laravel 工具回调」。
  本项目用自研 ReAct 当脑子，复用同一套后端工具接口，做到：
    - 业务重活（请假建单 / 审批人分配 / 站内信）全在 Laravel，不重写；
    - AI 服务只负责「理解意图 → 选工具 → 调接口 → 汇总」。
  接口均为裸路由（routes/api.php 第 22–30 行，无 auth 中间件），
  由 Dify server-to-server 调用，故 AI 服务用 HTTP 直接调用即可，无需令牌。

  调用契约（与 DifyController 一致）：
    POST {BACKEND_BASE_URL}/dify/tools/submit_leave
         body: {user_id(学号), reason, start_time, end_time}
    POST {BACKEND_BASE_URL}/dify/tools/get_duty_schedule
         body: {user_id, start_time, end_time}
    POST {BACKEND_BASE_URL}/dify/tools/get_absent_times
         body: {user_id, start_date, end_date}
    POST {BACKEND_BASE_URL}/dify/tools/get_teaching_event
         body: {user_id, start_date, end_date}
  返回：Helper::SuccessJsonResponse($data) → 形如 {"code":0,"msg":"ok","data":<payload>}
"""
from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any, Dict, Optional

import httpx

from app.config import (
    BACKEND_BASE_URL,
    BACKEND_HMAC_ENABLED,
    BACKEND_HMAC_SKEW_SEC,
    BACKEND_HMAC_SECRET,
    BACKEND_TIMEOUT,
)
from app.observability import get_trace_id

# 小萱后端暴露的工具名（对应 DifyController 的方法）
TOOL_SUBMIT_LEAVE = "submit_leave"
TOOL_DUTY_SCHEDULE = "get_duty_schedule"
TOOL_ABSENT_TIMES = "get_absent_times"
TOOL_TEACHING_EVENT = "get_teaching_event"


# ============ P0 必修 (2026-09-08 code review): server-to-server HMAC 签名 ============
# 📌 防伪造：任何能访问 BACKEND_BASE_URL 的人都可伪造 submit_leave 等写操作。
#    协议：X-Timestamp: <unix-seconds> + X-Signature: hmac_sha256(secret, canonical)
#    canonical = "{method}\n{path}\n{ts}\n{sha256(body_bytes)}"
#    验签方（Laravel）：ts 与 now 偏差 > SKEW → 拒；sig 不匹配 → 拒。
#    开关：BACKEND_HMAC_ENABLED=false 可临时关闭（Laravel 验签未上线时）。
def _hmac_sign(method: str, path: str, body_bytes: bytes, ts: str, secret: str) -> str:
    body_hash = hashlib.sha256(body_bytes).hexdigest()
    canonical = f"{method.upper()}\n{path}\n{ts}\n{body_hash}"
    return hmac.new(secret.encode("utf-8"), canonical.encode("utf-8"), hashlib.sha256).hexdigest()


def _hmac_enabled_now(secret: str) -> bool:
    """运行时判断是否启用 HMAC：开关 + 密钥长度同时满足。"""
    return bool(
        BACKEND_HMAC_ENABLED
        and secret
        and len(secret.encode("utf-8")) >= 32
    )


class BackendClient:
    """调用 Laravel /api/dify/tools/* 的异步客户端。

    📌 实战③ 工具调用异常：网络/超时/HTTP 错误统一抛出，由 Agent 层兜底（agent.py），
       不会让整个对话崩溃；真实调用建议加超时（BACKEND_TIMEOUT）与重试（此处留扩展位）。"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        client: Optional[httpx.AsyncClient] = None,
        hmac_secret: Optional[str] = None,
    ):
        self.base_url = (base_url or BACKEND_BASE_URL or "").rstrip("/")
        self.timeout = timeout or BACKEND_TIMEOUT
        self._client = client  # 测试可注入 MockTransport
        # 优先用注入值（测试 / Laravel 多租户场景可单独配），否则用全局 BACKEND_HMAC_SECRET
        self._hmac_secret = hmac_secret if hmac_secret is not None else BACKEND_HMAC_SECRET

    @property
    def configured(self) -> bool:
        """未配置基址时，真实工具应降级（见 tools.py 的友好提示）。"""
        return bool(self.base_url)

    async def call_tool(self, tool: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if not self.configured:
            raise RuntimeError("BACKEND_BASE_URL 未配置：无法调用真实后端工具")
        url = f"{self.base_url}/dify/tools/{tool}"
        # 📌 HMAC 签名要稳定：先手动 json.dumps 一份（ensure_ascii=False 与 httpx 默认 json= 行为一致）
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        # path 仅为 "/dify/tools/{tool}"（不含 base_url），与服务端验签口径保持一致
        path = f"/dify/tools/{tool}"
        tid = get_trace_id()
        headers = {"X-Trace-Id": tid} if tid else {}
        headers["Content-Type"] = "application/json"

        # 📌 HMAC 签名：开关 + 密钥就绪才打头；否则保持原行为（向后兼容，Laravel 端未上线时不影响）
        if _hmac_enabled_now(self._hmac_secret):
            ts = str(int(time.time()))
            sig = _hmac_sign("POST", path, body_bytes, ts, self._hmac_secret)
            headers["X-Timestamp"] = ts
            headers["X-Signature"] = sig
            # 时间窗常量：仅供 Laravel 端参考（实际验签在服务端做）
            headers["X-Signature-Skew"] = str(BACKEND_HMAC_SKEW_SEC)

        if self._client is not None:
            resp = await self._client.post(url, content=body_bytes, headers=headers)
            resp.raise_for_status()
            return resp.json()
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(url, content=body_bytes, headers=headers)
            resp.raise_for_status()
            return resp.json()

    @staticmethod
    def extract_data(resp: Dict[str, Any]) -> Any:
        """从 Laravel 信封中提取业务数据；取不到 data 时回退整包。"""
        if isinstance(resp, dict) and "data" in resp:
            return resp["data"]
        return resp


