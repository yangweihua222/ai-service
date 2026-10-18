# -*- coding: utf-8 -*-
"""LLM 抽象层：可插拔 chat，含 Mock 与 OpenAI 兼容适配器。

- MockLLM：本地确定性生成（不联网、零依赖），用于 mock 模式跑通完整链路。
- OpenAICompatLLM：OpenAI 兼容接口适配器（DeepSeek / 通义 / 智谱 等复用），
  需要 OPENAI_API_KEY；用标准库 urllib 实现，无需额外依赖。

📌 面试考点（对应《技术点深挖备考库》）：
  · 技术点30 技术选型（LLM 抽象、模型可热插拔、不绑死厂商）     → 见 get_llm() / LLMClient
  · 实战⑤ Token 成本优化（简单任务用小模型分担）               → 见 OpenAICompatLLM（可接小模型做 query改写/rerank/反思）
  · 实战⑨ 反思用小模型（高并发时主模型只干规划+生成）          → 见 OpenAICompatLLM 注释
"""
import asyncio
import json
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from app import observability
from app.governance import llm_governance
from app.config import (
    LLM_ENABLE_THINKING,
    LLM_MODE,
    OPENAI_API_BASE,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    OPENAI_MODEL_FAST,
)


@dataclass
class Message:
    role: str                     # system | user | assistant | tool
    content: str
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    tool_calls: Optional[List["ToolCall"]] = None


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class LLMResponse:
    content: str
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)  # {prompt, completion, total}：token 成本统计


@dataclass
class StreamChunk:
    """流式响应的一个分片：正文增量 delta 与（结束时）完整 tool_calls 二选一或同时出现。"""
    delta: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    finish_reason: str = ""
    usage: dict = field(default_factory=dict)  # 流结束时（最后一个 chunk）可能携带 token 用量


class LLMClient(ABC):
    @abstractmethod
    async def chat(self, messages: List[Message], tools=None, temperature: Optional[float] = None) -> LLMResponse:
        ...

    async def stream_chat(self, messages: List[Message], tools=None):
        """流式生成：逐 token yield StreamChunk。

        默认实现为非流式降级（先完整 chat 再一次性 yield），
        OpenAI 兼容适配器会覆盖为真正的 SSE 逐 token 流式。
        """
        resp = await self.chat(messages, tools=tools)
        if resp.content:
            yield StreamChunk(delta=resp.content)
        if resp.tool_calls:
            yield StreamChunk(tool_calls=resp.tool_calls, finish_reason=resp.finish_reason)


class MockLLM(LLMClient):
    """本地确定性生成：用于 mock 模式下的『综合回答』与『最终总结』。"""

    async def chat(self, messages: List[Message], tools=None, temperature: Optional[float] = None) -> LLMResponse:
        obs = [m.content for m in messages if m.role == "tool"]
        if obs:
            return LLMResponse(content="【综合回答】\n" + "\n---\n".join(obs))
        last_user = ""
        for m in reversed(messages):
            if m.role == "user":
                last_user = m.content
                break
        return LLMResponse(
            content=f"（MockLLM）已基于工具结果生成回答。原始提问：{last_user}"
        )


class OpenAICompatLLM(LLMClient):
    """OpenAI 兼容接口适配器。真实调用 POST {base}/chat/completions。"""

    def __init__(self, model: Optional[str] = None):
        if not OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY 未配置，无法使用 openai 模式")
        # 📌 模型路由（A' ①）：决策/路由步可传快模型（如 qwen-turbo），综合生成用主模型
        self.model = model or OPENAI_MODEL

    def _build_payload(self, messages: List[Message], tools=None, temperature: Optional[float] = None) -> dict:
        """组装请求体（流式/非流式共用），含多轮闭环所需的 tool_call_id 与 tool_calls 回传。
        temperature：可选覆盖（默认 0.2）；评测裁判传 0 求确定性输出。"""
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": m.role,
                    "content": m.content,
                    **({"tool_call_id": m.tool_call_id} if m.tool_call_id else {}),
                    **(
                        {
                            "tool_calls": [
                                {
                                    "id": tc.id if isinstance(tc, ToolCall) else tc.get("id", ""),
                                    "type": "function",
                                    "function": {
                                        # 📌 双保险：历史消息从 Redis 反序列化回来时 tool_calls 可能是 dict，
                                        #    这里统一按两种形态取值，避免上游少一次类型转换就整条链路崩。
                                        "name": tc.name if isinstance(tc, ToolCall) else tc.get("name", ""),
                                        "arguments": json.dumps(
                                            (tc.arguments if isinstance(tc, ToolCall) else tc.get("arguments"))
                                            or {}, ensure_ascii=False
                                        ),
                                    },
                                }
                                for tc in m.tool_calls
                            ]
                        }
                        if m.tool_calls
                        else {}
                    ),
                }
                for m in messages
            ],
            "temperature": 0.2 if temperature is None else temperature,
            # 关闭思维链（qwen3 系列）：工具选型 + 结构化综合无需深度推理，关掉显著降低延迟
            "enable_thinking": LLM_ENABLE_THINKING,
        }
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    },
                }
                for t in tools
            ]
            payload["tool_choice"] = "auto"
        return payload

    def _parse_usage(self, data: dict) -> dict:
        """解析 token 用量，兼容 OpenAI 标准（prompt_tokens）与部分厂商（input_tokens）字段名。"""
        u = data.get("usage") or {}
        prompt = u.get("prompt_tokens", u.get("input_tokens", 0))
        completion = u.get("completion_tokens", u.get("output_tokens", 0))
        total = u.get("total_tokens", 0) or (prompt + completion)
        return {"prompt": int(prompt or 0), "completion": int(completion or 0), "total": int(total or 0)}

    def _parse_tool_calls(self, msg: dict) -> List[ToolCall]:
        tool_calls: List[ToolCall] = []
        for tc in msg.get("tool_calls") or []:
            if tc.get("type") == "function":
                fn = tc["function"]
                tool_calls.append(
                    ToolCall(
                        id=tc["id"],
                        name=fn["name"],
                        arguments=json.loads(fn.get("arguments") or "{}"),
                    )
                )
        return tool_calls

    def _request(self, messages: List[Message], tools=None, temperature: Optional[float] = None) -> LLMResponse:
        payload = self._build_payload(messages, tools, temperature)
        req = urllib.request.Request(
            f"{OPENAI_API_BASE}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {OPENAI_API_KEY}",
                "Content-Type": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = data["choices"][0]["message"]
        content = msg.get("content") or ""
        usage = self._parse_usage(data)
        return LLMResponse(
            content=content,
            tool_calls=self._parse_tool_calls(msg),
            finish_reason=data["choices"][0].get("finish_reason", "stop"),
            usage=usage,
        )

    async def chat(self, messages: List[Message], tools=None, temperature: Optional[float] = None) -> LLMResponse:
        # 📌 服务治理三道闸门：熔断 → QPS 令牌桶 → Token 预算（任一触发抛异常，由上层降级）
        # temperature 透传给 _build_payload（评测裁判传 0）
        llm_governance.before_call(messages)
        try:
            with observability.span("llm.chat", model=self.model):
                resp = await asyncio.to_thread(self._request, messages, tools, temperature)
        except Exception:
            llm_governance.record_failure()  # ⚠️ 熔断计数：网络错/超时/限流 429 都算失败
            raise
        llm_governance.record_success(messages, resp.usage)
        # ⚠️ add_usage 必须在这里（异步上下文）调用：_request 在线程池跑，contextvar 不会跨线程传播
        observability.add_usage(resp.usage.get("prompt", 0), resp.usage.get("completion", 0), resp.usage.get("total", 0))
        return resp

    async def stream_chat(self, messages: List[Message], tools=None):
        """真正的 token 级流式：POST stream=True，逐 SSE 增量 yield 正文，
        同时按 index 聚合 tool_calls（流结束后作为最后一个 chunk 吐出）。
        stream_options.include_usage 让最后一个 chunk 带回 token 用量（成本统计）。"""
        import httpx

        # 📌 服务治理三道闸门（同 chat）：熔断 → QPS 令牌桶 → Token 预算
        llm_governance.before_call(messages)
        payload = self._build_payload(messages, tools)
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        tool_parts: dict = {}
        finish_reason = "stop"
        usage: dict = {}
        try:
            with observability.span("llm.stream_chat", model=self.model):
                async with httpx.AsyncClient(timeout=120.0) as client:
                    async with client.stream(
                        "POST", f"{OPENAI_API_BASE}/chat/completions", json=payload, headers=headers
                    ) as resp:
                        resp.raise_for_status()
                        async for line in resp.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                break
                            obj = json.loads(data)
                            # include_usage 的最后一个 chunk：choices 为空、仅带 usage
                            if obj.get("usage"):
                                usage = self._parse_usage(obj)
                                continue
                            choices = obj.get("choices") or []
                            if not choices:
                                continue
                            choice = choices[0]
                            delta = choice.get("delta") or {}
                            if delta.get("content"):
                                yield StreamChunk(delta=delta["content"])
                            for tc in delta.get("tool_calls") or []:
                                idx = tc.get("index", 0)
                                part = tool_parts.setdefault(idx, {"id": "", "name": "", "args": ""})
                                if tc.get("id"):
                                    part["id"] = tc["id"]
                                fn = tc.get("function") or {}
                                if fn.get("name"):
                                    part["name"] += fn["name"]
                                if fn.get("arguments"):
                                    part["args"] += fn["arguments"]
                            if choice.get("finish_reason"):
                                finish_reason = choice["finish_reason"]
        except Exception:
            llm_governance.record_failure()  # ⚠️ 熔断计数：流中断/超时/HTTP 错误
            raise
        llm_governance.record_success(messages, usage)  # usage 在流末 chunk 才带回，这里回填 token 预算
        observability.add_usage(usage.get("prompt", 0), usage.get("completion", 0), usage.get("total", 0))
        calls: List[ToolCall] = []
        for idx in sorted(tool_parts):
            p = tool_parts[idx]
            try:
                args = json.loads(p["args"]) if p["args"] else {}
            except json.JSONDecodeError:
                args = {}
            calls.append(ToolCall(id=p["id"] or f"call_{idx}", name=p["name"], arguments=args))
        if calls:
            yield StreamChunk(tool_calls=calls, finish_reason=finish_reason, usage=usage)
        elif usage:
            yield StreamChunk(usage=usage)


def get_llm(model: Optional[str] = None) -> LLMClient:
    # 📌 技术点30 选型：LLM 抽象为接口，mock/openai 可热插拔，不绑死厂商；真实接入只换 base_url+key
    # 📌 模型路由（A' ①）：传 model 可构造指定模型的客户端（决策步用快模型、综合步用主模型）
    if LLM_MODE == "openai":
        return OpenAICompatLLM(model=model)
    return MockLLM()
