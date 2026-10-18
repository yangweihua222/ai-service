# -*- coding: utf-8 -*-
"""Redis 后端实现：会话历史 + 反思笔记 + TTL + 多租户 key。

📌 Redis 数据结构：
  · 会话历史 key:  ai:mem:<make_key>      → String (JSON 序列化) + EX TTL
  · 反思笔记 key:   ai:refl:<make_key>     → String + EX TTL（更长 TTL，反思跨 session 复用）
  · 多租户隔离：    make_key 已含 tenant_id（物理隔离，不是 WHERE 过滤）

📌 与 SQLite 后端对照：
  · 多 worker 一致：Redis 是单点 source of truth（连接池共享）
  · TTL 由 Redis 自动过期（SQLite 是 lazy cleanup_expired）
  · fakeredis 2.x 兼容 redis-py 8.x 接口（pip install fakeredis>=2.0 即可测试）
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from typing import List, Optional, Tuple

from app.llm import Message, ToolCall
from app.store_base import MemoryBackendABC

_log = logging.getLogger(__name__)


class RedisBackend(MemoryBackendABC):
    """Redis 后端（生产级别）：单点 source of truth + 自动 TTL + 多 worker 共享。

    连接：
      · url 形如 redis://[:password]@host:port/db
      · 内部 redis-py 客户端（decode_responses=True 处理 str/bytes）
    """

    PREFIX_HISTORY = "ai:mem:"
    PREFIX_REFLECTION = "ai:refl:"

    # 反思笔记 TTL 拉长（24h），便于跨 session 复用
    REFLECTION_TTL_SECONDS = 86400

    def __init__(self, url: str, socket_timeout: float = 2.0, client=None):
        """client: 测试可传 FakeRedis 实例跳过 url+ping（生产代码不传）。"""
        import redis  # redis-py 5/8 兼容
        self._url = url
        if client is not None:
            self._client = client
        else:
            if not url:
                raise RuntimeError("RedisBackend 需要 url（生产 redis://host:port/0）")
            self._client = redis.Redis.from_url(
                url, decode_responses=True, socket_timeout=socket_timeout
            )
            # 启动期校验连通性（fail-fast）
            try:
                self._client.ping()
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Redis 不可达 url={url}: {e}") from e

    # ---------------- helper ----------------
    @staticmethod
    def _msg_to_dict(m) -> dict:
        if isinstance(m, dict):
            return m
        try:
            return asdict(m)
        except Exception:  # noqa: BLE001
            return {"role": getattr(m, "role", "user"),
                    "content": getattr(m, "content", "")}

    def _hist_key(self, key: str) -> str:
        return f"{self.PREFIX_HISTORY}{key}"

    def _refl_key(self, key: str) -> str:
        return f"{self.PREFIX_REFLECTION}{key}"

    # ---------------- 接口实现 ----------------
    def upsert_history(self, key, tenant_id, user_id, session_id, messages):
        blob = json.dumps(
            [self._msg_to_dict(m) for m in messages],
            ensure_ascii=False, default=str,
        )
        k = self._hist_key(key)
        # SET + EX 用 pipeline 保证原子（避免两个命令间过期）
        pipe = self._client.pipeline()
        pipe.set(k, blob)
        # TTL 由业务层决定：默认 1h（Redis 自己管过期）
        pipe.expire(k, int(self._ttl_for_history()))
        pipe.execute()

    def load_history(self, key):
        blob = self._client.get(self._hist_key(key))
        if blob is None:
            return None
        try:
            raw = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            _log.warning("redis.load_history: json 反序列化失败 key=%s", key)
            return []
        return [self._dict_to_msg(m) if isinstance(m, dict) else m for m in raw]

    @staticmethod
    def _dict_to_msg(d: dict) -> Message:
        """反序列化时**还原嵌套对象类型**。

        ⚠️ 2026-09-11 修的生产级 bug：历史消息经 JSON 序列化再读回来时，
            `tool_calls` 只会是一堆 dict；如果直接 `Message(**m)` 塞进去，
            第二轮对话 `llm._build_payload()` 访问 `tc.id` 就会抛
            `AttributeError: 'dict' object has no attribute 'id'` —— 多轮对话直接崩。
            进程内（不持久化）时不复现，所以这个坑只在真正接了 Redis 之后才暴露。
        """
        tcs = d.get("tool_calls") or None
        if tcs:
            restored = []
            for t in tcs:
                if isinstance(t, ToolCall):
                    restored.append(t)
                elif isinstance(t, dict):
                    restored.append(ToolCall(
                        id=t.get("id") or "",
                        name=t.get("name") or "",
                        arguments=t.get("arguments") or {},
                    ))
            tcs = restored
        return Message(
            role=d.get("role", "user"),
            content=d.get("content") or "",
            tool_call_id=d.get("tool_call_id"),
            name=d.get("name"),
            tool_calls=tcs,
        )

    def upsert_reflection(self, key, tenant_id, user_id, session_id, note):
        k = self._refl_key(key)
        pipe = self._client.pipeline()
        pipe.set(k, note)
        pipe.expire(k, self.REFLECTION_TTL_SECONDS)
        pipe.execute()

    def load_reflection(self, key):
        return self._client.get(self._refl_key(key))

    def clear_reflection(self, key):
        self._client.delete(self._refl_key(key))

    def cleanup_expired(self, ttl_seconds: float) -> Tuple[int, List[str]]:
        """Redis 的过期由服务端自己管理，这里做一次全量扫描兜底。

        用 SCAN（不阻塞 Redis）+ 解析 key 拿 tenant/user/session 信息。
        返回的 deleted_keys 让上层同步 in-mem 缓存。
        """
        deleted: List[str] = []
        # 仅扫描历史 key（反思笔记有更长 TTL 单独管）
        for k in self._client.scan_iter(match=f"{self.PREFIX_HISTORY}*"):
            ttl = self._client.ttl(k)
            # ttl=-1 无过期；ttl=-2 key 已过期但 SCAN 还能扫到；ttl>0 还有效
            if ttl == -2 or ttl == -1:
                # key 不存在（-2）或无过期（-1）—— 视为待清理
                raw_key = k[len(self.PREFIX_HISTORY):]
                deleted.append(raw_key)
                self._client.delete(k)
        # 业务层 ttl_seconds 仅作日志提示（Redis 自己按 EXPIRE 控时长）
        return len(deleted), deleted

    def list_sessions(self, tenant_id, user_id) -> List[str]:
        """SCAN ai:mem:tenant={t}:user={u}:* 拿 session_id。"""
        prefix = f"{self.PREFIX_HISTORY}tenant={tenant_id}:user={user_id}:session="
        sessions = []
        for k in self._client.scan_iter(match=f"{prefix}*"):
            raw = k[len(self.PREFIX_HISTORY):]
            sid = raw.split("session=", 1)[-1]
            if sid:
                sessions.append(sid)
        return sessions

    def close(self):
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass

    # ---------------- TTL helper ----------------
    @staticmethod
    def _ttl_for_history() -> int:
        """历史 key 的 TTL：默认 1h，可通过 MEMORY_REDIS_TTL_SECONDS 覆盖。"""
        import os
        try:
            return int(os.environ.get("MEMORY_REDIS_TTL_SECONDS", "3600"))
        except ValueError:
            return 3600