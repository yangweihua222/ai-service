# -*- coding: utf-8 -*-
"""记忆与治理后端抽象基类（生产级别可替换：SQLite / Redis）。

📌 面试考点（对应《技术点深挖备考库》2.9 / 4.4）：
  · 接口/实现分离 —— 业务代码不感知 backend 是 SQLite 还是 Redis
  · 多租户隔离 —— key 强制含 tenant_id（key 拼接是物理隔离，不是 WHERE 过滤）
  · 进程内 Dict 是 cache，持久化层是 source of truth
  · 生产换 Redis 三件套：限流 Lua 原子化 / 熔断状态集中 / 记忆 HSET + TTL

⚠️ 当前抽象：
  · MemoryBackendABC：会话历史 + 反思笔记（SQLite / Redis 都实现同一接口）
  · GovernanceBackendABC：令牌桶 + Token 预算 + 熔断（Lua 原子化保证多 worker 一致）
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple


class MemoryBackendABC(ABC):
    """记忆持久化后端抽象接口。

    设计要点：
      · key 包含 tenant_id + user_id + session_id（由 MemoryStore._make_key 拼好传入）
      · 不依赖外部时钟——只跟单调时间戳走（Redis 用 server time，SQLite 用 time.time()）
      · 所有方法同步（业务代码调 MemoryStore 的异步包装层）
    """

    @abstractmethod
    def upsert_history(self, key: str, tenant_id: str, user_id: str,
                       session_id: str, messages) -> None:
        """覆盖写入整段会话历史（业务层负责构造 messages 列表）。"""

    @abstractmethod
    def load_history(self, key: str) -> Optional[list]:
        """读取会话历史；不存在返回 None，序列化失败返回空 list。"""

    @abstractmethod
    def upsert_reflection(self, key: str, tenant_id: str, user_id: str,
                          session_id: str, note: str) -> None:
        """覆盖写入反思笔记。"""

    @abstractmethod
    def load_reflection(self, key: str) -> Optional[str]:
        """读取反思笔记；不存在返回 None。"""

    @abstractmethod
    def clear_reflection(self, key: str) -> None:
        """清除反思笔记。"""

    @abstractmethod
    def cleanup_expired(self, ttl_seconds: float) -> Tuple[int, List[str]]:
        """清理过期 session；返回 (清理条数, 被删的 key 列表)。"""

    @abstractmethod
    def list_sessions(self, tenant_id: str, user_id: str) -> List[str]:
        """列出某租户某用户的所有 session_id。"""

    @abstractmethod
    def close(self) -> None:
        """释放连接（Redis 关连接池，SQLite 关文件）。"""


class GovernanceBackendABC(ABC):
    """LLM 服务治理后端抽象接口（令牌桶 + Token 预算 + 熔断）。

    生产级别要求：
      · 多 worker 一致：原子扣减（Redis 用 Lua / SQLite 用事务）
      · 不依赖外部时钟：Lua 内取 server time / SQLite 用 time.monotonic 序列号
      · 错误不静默：Redis 挂了要让上层感知（连接异常向上抛）
    """

    # ========== 令牌桶 ==========
    @abstractmethod
    def token_bucket_acquire(self, qps: float, burst: int) -> bool:
        """尝试取一个令牌；返回 True=拿到，False=被限。"""

    @abstractmethod
    def token_bucket_snapshot(self, qps: float, burst: int) -> dict:
        """查看当前桶状态：{tokens_left, qps, burst}。"""

    # ========== Token 预算（60s 滚动窗口）==========
    @abstractmethod
    def token_budget_reserve(self, budget: int, window_s: float, estimate: int) -> bool:
        """预占 estimate 个 token；返回 True=成功，False=超额。"""

    @abstractmethod
    def token_budget_settle(self, budget: int, window_s: float,
                            actual: int, estimate: int) -> None:
        """响应后用真实 usage 回填（多退少补）。"""

    @abstractmethod
    def token_budget_used(self, budget: int, window_s: float) -> int:
        """当前窗口已消耗 token 数（用于观测）。"""

    # ========== 熔断器 ==========
    @abstractmethod
    def breaker_before_call(self, name: str, cooldown_s: float,
                            halfopen_probes: int) -> Tuple[bool, str]:
        """调用前检查；返回 (allowed, state)。state ∈ {closed, open, half_open}。"""

    @abstractmethod
    def breaker_record_success(self, name: str) -> None:
        """记录一次成功；HALF_OPEN 探测成功会恢复到 CLOSED。"""

    @abstractmethod
    def breaker_record_failure(self, name: str) -> None:
        """记录一次失败；超阈值会触发 OPEN。"""

    @abstractmethod
    def breaker_snapshot(self, name: str, window_s: float,
                         cooldown_s: float) -> dict:
        """当前熔断状态快照。"""

    # ========== 生命周期 ==========
    @abstractmethod
    def close(self) -> None:
        """释放连接。"""