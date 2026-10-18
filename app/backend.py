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
from typing import Any, Dict, List, Optional, Tuple

import httpx

from app.config import (
    BACKEND_BASE_URL,
    BACKEND_HMAC_ENABLED,
    BACKEND_HMAC_SKEW_SEC,
    BACKEND_HMAC_SECRET,
    BACKEND_TIMEOUT,
    XUANWU_API_BASE,
    XUANWU_NUM,
    XUANWU_PASSWORD,
    XUANWU_TIMEOUT,
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


class XuanwuBackend:
    """真实平台数据源（宣武实习管理平台，AES 加密协议）。

    📌 定位（技术点9「集成与兼容」/ 实战③ 工具调用异常）：
      真实平台接口是「Sanctum 登录 + 双层 base64 + AES-256-CBC 加密信封」（协议见
      app/xuanwu_client.py，由逆向平台前端 JS 确认）。本类把它适配成与
      BackendClient 同级的异步数据源：
        - 同步 requests 客户端经 asyncio.to_thread 包装成 async（不阻塞事件循环）；
        - 登录惰性执行、token 进程内复用，401 时自动重登一次再重试；
        - 值班接口无「按人查询」参数 → 客户端按姓名/日期过滤（分页拉全量后筛选）。

    与 BackendClient 的关系：XuanwuBackend 是「真实平台直连」，配置了 XUANWU_*
    三项即启用（见 tools.py 的优先级：真实平台 > Laravel 明文契约 > 未配置提示）。
    """

    #: 分页保护上限：最多翻 10 页（防止异常 total 导致长循环）
    MAX_PAGES = 10

    def __init__(
        self,
        base_url: Optional[str] = None,
        num: Optional[str] = None,
        password: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        self.base_url = (base_url or XUANWU_API_BASE or "").rstrip("/")
        self.num = num or XUANWU_NUM
        self.password = password or XUANWU_PASSWORD
        self.timeout = timeout or XUANWU_TIMEOUT
        self._client = None  # 惰性创建的 XuanwuClient（登录后缓存）

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.num and self.password)

    # ---------- 基础调用 ----------

    def _ensure_login(self):
        """惰性登录：首次调用（或重登）创建客户端并取 token。"""
        if self._client is None or not getattr(self._client, "token", None):
            from app.xuanwu_client import XuanwuClient  # 延迟导入：未装依赖时不影响其他模块

            client = XuanwuClient(self.base_url, timeout=self.timeout)
            client.login(self.num, self.password)
            self._client = client
        return self._client

    async def _call(self, method: str, **kwargs: Any) -> Dict[str, Any]:
        """调 XuanwuClient 的同步方法（线程池执行），401 自动重登重试一次。

        📌 全链路透传：trace_id 必须在进入 asyncio.to_thread 之前取出——
        contextvar 不会跨线程传播（之前踩过的坑）。取到后作为显式参数传入
        _sync_call，再注入 XuanwuClient.trace_id，最终随 HTTP 头 X-Trace-Id
        发往平台，使「AI 助手 → 宣武平台」在实战里是同一条 trace。
        """
        import asyncio

        tid = get_trace_id()  # 线程边界前取值
        try:
            return await asyncio.to_thread(self._sync_call, method, trace_id=tid, **kwargs)
        except RuntimeError:
            # asyncio.to_thread 不可用时退化
            return self._sync_call(method, trace_id=tid, **kwargs)

    def _sync_call(self, method: str, trace_id: str = "", **kwargs: Any) -> Dict[str, Any]:
        client = self._ensure_login()
        client.trace_id = trace_id  # 注入，供 _headers 随请求发出
        try:
            return getattr(client, method)(**kwargs)
        except Exception as e:  # 401/403：token 过期 → 重登一次再试
            if "401" in str(e) or "403" in str(e):
                self._client = None
                client = self._ensure_login()
                client.trace_id = trace_id  # 重登后新 client 也要带上
                return getattr(client, method)(**kwargs)
            raise

    # ---------- 业务方法（供 tools.py 调用） ----------

    async def query_duty(
        self,
        name: str = "",
        start_time: str = "",
        end_time: str = "",
        page_size: int = 200,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """值班排班查询：分页拉取 /duty/scheduleScheduling，客户端按姓名/日期过滤。

        :param name:      姓名精确匹配（值班医生或值班学生）
        :param start_time: 起始日期 YYYY-MM-DD（含），空则不过滤
        :param end_time:   截止日期 YYYY-MM-DD（含），空则不过滤
        :return: (过滤后的排班列表, 平台总条数)
        """
        items: List[Dict[str, Any]] = []
        total = 0
        for page in range(1, self.MAX_PAGES + 1):
            resp = await self._call(
                "duty_schedule", page=page, limit=page_size
            )
            data = (resp or {}).get("data") or {}
            batch = data.get("list") or []
            items.extend(batch)
            total = data.get("total") or len(items)
            if not batch or len(items) >= total:
                break

        def _match(it: Dict[str, Any]) -> bool:
            if name:
                student = (it.get("student") or {}).get("name") or ""
                doctor = it.get("doctor_name") or ""
                if name not in (student, doctor):
                    return False
            date = it.get("date") or ""
            if start_time and date < start_time:
                return False
            if end_time and date > end_time:
                return False
            return True

        return [it for it in items if _match(it)], total

    async def query_teaching_events(
        self,
        start_date: str = "",
        end_date: str = "",
        page_size: int = 200,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """教学活动查询：分页拉取 /teaching-events，客户端按日期过滤。

        平台列表接口无「日期范围」参数，故按 start_time/estimated_start_time
        的前 10 位（YYYY-MM-DD）做客户端过滤。
        """
        items: List[Dict[str, Any]] = []
        total = 0
        for page in range(1, self.MAX_PAGES + 1):
            resp = await self._call("teaching_events", page=page, limit=page_size)
            data = (resp or {}).get("data") or {}
            batch = data.get("list") or []
            items.extend(batch)
            total = data.get("total") or len(items)
            if not batch or len(items) >= total:
                break

        def _match(it: Dict[str, Any]) -> bool:
            t = (it.get("start_time") or it.get("estimated_start_time") or "")[:10]
            if start_date and t < start_date:
                return False
            if end_date and t > end_date:
                return False
            return True

        return [it for it in items if _match(it)], total

    async def query_absent(
        self,
        name: str = "",
        start_date: str = "",
        end_date: str = "",
        page_size: int = 200,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """缺勤/打卡统计查询：调 /attendances/statistics/statisticsCheckInListAll。

        平台该接口原生支持 name/start/end 过滤（见逆向 attendance api 文件），
        直接透传后分页拉全量，返回「学员×日期×班次」粒度的打卡统计。
        """
        items: List[Dict[str, Any]] = []
        total = 0
        for page in range(1, self.MAX_PAGES + 1):
            resp = await self._call(
                "attendance_statistics",
                page=page,
                limit=page_size,
                name=name,
                start=start_date,
                end=end_date,
            )
            data = (resp or {}).get("data") or {}
            batch = data.get("list") or []
            items.extend(batch)
            total = data.get("total") or len(items)
            if not batch or len(items) >= total:
                break
        return items, total
