# -*- coding: utf-8 -*-
"""对话记忆：会话内消息存储 + 长期摘要压缩 + 多租户隔离 + TTL + 可插拔持久化后端。

📌 面试考点（对应《技术点深挖备考库》）：
  · 实战⑩ 记忆系统（短期 + 长期摘要 + 工作/会话/长期三层思想） → MemoryStore / compact()
  · 上下文膨胀优化（滑动窗口 + 摘要压缩 + TTL 清理）            → compact()
  · 多租户隔离（P0 修复：key 含 tenant_id + user_id）             → _make_key()
  · 多副本持久化（backend=redis：多 worker 共享 + TTL + 服务端过期）      → MemoryBackendABC
  · 反思机制配套（反思笔记也走同一持久化）                        → save_reflection() / load_reflection()
  · 接口/实现分离（业务代码不感知 backend 是 Redis 还是进程内）      → store_base.MemoryBackendABC

⚠️ 诚实边界：
  · 2026-09-11 起**已移除 SQLite 后端**：backend 只有 redis（生产）与 memory（单测/降级，进程内，重启即丢）。
  · 摘要压缩会引入一次额外 LLM 调用（qwen-turbo，省钱）。
  · 没做用户画像/跨会话长期记忆（教学场景够用；生产请升级到 Redis + 向量画像）。
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import threading
import time
from dataclasses import asdict
from typing import Dict, List, Optional

from app.llm import Message
from app.store_base import MemoryBackendABC

_log = logging.getLogger(__name__)


class MemoryStore:
    """会话内消息存储 + LLM 摘要压缩 + 多租户隔离。

    关键点（面试口径）：
      · 写入双写：内存 Dict + 后端（Redis），后端是 source of truth，内存只是缓存
      · Key 含 tenant_id + user_id + session_id —— 跨租户绝对隔离
      · compact() 超阈值（默认 12 条）→ 调 qwen-turbo 生成摘要作为 system 消息保留
      · TTL 过期清理（默认 1 小时）
      · backend 可选 redis（生产）/ memory（进程内，单测与降级）
    """

    DEFAULT_KEEP_LAST = 6
    DEFAULT_SUMMARIZE_THRESHOLD = 12
    DEFAULT_TTL_SECONDS = 3600

    def __init__(self, db_path=None, keep_last=DEFAULT_KEEP_LAST,
                 summarize_threshold=DEFAULT_SUMMARIZE_THRESHOLD,
                 ttl_seconds=DEFAULT_TTL_SECONDS,
                 summarizer=None, backend: str = "redis",
                 redis_url: Optional[str] = None):
        """summarizer: 可选 callable(messages) -> str 或 awaitable。
        未传则跳过摘要（仅做滑动窗口裁剪）——便于单测隔离。

        backend: 'redis'（默认，生产级别多 worker 共享）/ 'memory'（进程内，单测与降级用）
        redis_url: 仅当 backend='redis' 时需要，如 redis://localhost:6379/0
        """
        self._backend = None
        self._db_path = db_path
        self._backend_kind = backend
        self._redis_url = redis_url
        self._redis_client_override = None  # 测试用 fakeredis 注入
        self._inmem: Dict[str, list] = {}
        self._failed_persist = False
        self.keep_last = keep_last
        self.summarize_threshold = summarize_threshold
        self.ttl_seconds = ttl_seconds
        self.summarizer = summarizer

    @staticmethod
    def _make_key(tenant_id: str, user_id: str, session_id: str) -> str:
        """P0 修复：key 必须含 tenant_id，杜绝跨租户上下文泄漏。"""
        return f"tenant={tenant_id}:user={user_id}:session={session_id}"

    def _ensure_backend(self):
        """懒加载后端：第一次写/读时实例化（避免启动期连不上直接崩）。"""
        if self._backend is not None:
            return self._backend
        if self._failed_persist:
            return None
        if self._backend_kind == "redis":
            try:
                from app.memory_redis import RedisBackend
                # 优先用测试注入的 client（fakeredis）
                client = self._redis_client_override
                if client is not None:
                    self._backend = RedisBackend(
                        url=self._redis_url or "redis://override/0",
                        client=client,
                    )
                else:
                    self._backend = RedisBackend(self._redis_url)
                return self._backend
            except Exception as e:  # noqa: BLE001
                _log.error("memory.redis 不可用 (%s)，降级为内存模式", e)
                self._failed_persist = True
                return None
        # 📌 2026-09-11：彻底移除 SQLite 分支。可选后端只剩两个：
        #    redis = 生产（多 worker 共享 + Lua 原子化）；memory = 进程内（单测/CI/降级）。
        #    进程内模式不做任何持久化，重启即丢，这是刻意的取舍而不是缺陷。
        return None

    def load(self, tenant_id, user_id, session_id):
        key = self._make_key(tenant_id, user_id, session_id)
        backend = self._ensure_backend()
        if backend:
            persisted = backend.load_history(key)
            if persisted is not None:
                self._inmem[key] = persisted
                return list(persisted)
        return list(self._inmem.get(key, []))

    def append(self, tenant_id, user_id, session_id, msg: Message) -> None:
        key = self._make_key(tenant_id, user_id, session_id)
        self._inmem.setdefault(key, []).append(msg)
        self._save(tenant_id, user_id, session_id, key)

    def save(self, tenant_id, user_id, session_id, history) -> None:
        key = self._make_key(tenant_id, user_id, session_id)
        self._inmem[key] = history
        self._save(tenant_id, user_id, session_id, key)

    def _save(self, tenant_id, user_id, session_id, key) -> None:
        backend = self._ensure_backend()
        if backend:
            backend.upsert_history(key, tenant_id, user_id, session_id, self._inmem[key])

    def compact(self, tenant_id, user_id, session_id) -> None:
        """滑动窗口 + LLM 摘要压缩。

        1. 如果历史超阈值且有 summarizer → 把前 N-keep_last 条压缩成 system 摘要插头部
        2. 否则仅滑动窗口
        3. 持久化
        """
        key = self._make_key(tenant_id, user_id, session_id)
        history = self._inmem.get(key, [])
        if not history:
            return

        if self.summarizer and len(history) > self.summarize_threshold:
            early = history[:-self.keep_last]
            late = history[-self.keep_last:]
            try:
                summary_text = self._summarize_sync(early)
            except Exception as e:  # noqa: BLE001
                _log.warning("memory.compact 摘要失败 (%s)，跳过压缩", e)
                summary_text = None
            if summary_text:
                summary_msg = Message(role="system", content=f"[历史摘要] {summary_text}")
                new_history = [summary_msg] + late
                self._inmem[key] = new_history
                self._save(tenant_id, user_id, session_id, key)
                return

        if len(history) > self.keep_last:
            self._inmem[key] = history[-self.keep_last:]
            self._save(tenant_id, user_id, session_id, key)

    def _summarize_sync(self, messages):
        """同步调用 summarizer（可传 async 函数）。"""
        result = self.summarizer(messages)
        if asyncio.iscoroutine(result):
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    _log.warning("memory.compact: 事件循环已运行，跳过异步摘要")
                    return None
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            return loop.run_until_complete(result)
        return result

    def save_reflection(self, tenant_id, user_id, session_id, note: str) -> None:
        key = self._make_key(tenant_id, user_id, session_id)
        backend = self._ensure_backend()
        if backend:
            backend.upsert_reflection(key, tenant_id, user_id, session_id, note)
        else:
            setattr(self, f"_refl_{key}", note)

    def load_reflection(self, tenant_id, user_id, session_id):
        key = self._make_key(tenant_id, user_id, session_id)
        backend = self._ensure_backend()
        if backend:
            return backend.load_reflection(key)
        return getattr(self, f"_refl_{key}", None)

    def clear_reflection(self, tenant_id, user_id, session_id) -> None:
        key = self._make_key(tenant_id, user_id, session_id)
        backend = self._ensure_backend()
        if backend:
            backend.clear_reflection(key)
        if hasattr(self, f"_refl_{key}"):
            delattr(self, f"_refl_{key}")

    def cleanup_expired(self):
        """清理过期 session，返回清理条数；同步 in-mem 防止脏读。"""
        backend = self._ensure_backend()
        if not backend:
            return 0
        count, deleted_keys = backend.cleanup_expired(self.ttl_seconds)
        for k in deleted_keys:
            self._inmem.pop(k, None)
        return count

    def list_sessions(self, tenant_id, user_id):
        backend = self._ensure_backend()
        if not backend:
            prefix = f"tenant={tenant_id}:user={user_id}:session="
            return [
                k.split("session=", 1)[-1]
                for k in self._inmem.keys()
                if k.startswith(prefix)
            ]
        return backend.list_sessions(tenant_id, user_id)