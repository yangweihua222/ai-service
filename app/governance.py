# -*- coding: utf-8 -*-
"""LLM 服务治理：限流（令牌桶 QPS + Token 预算）+ 熔断（三态断路器）+ 可插拔后端。

📌 面试考点（对应《技术点深挖备考库》1.5 / 层 6）：
  · 限流：令牌桶控 QPS；Token 预算按滚动窗口限每分钟 token 消耗（模型成本跟 token 走，光控 QPS 不够）。
  · 熔断：CLOSED →（错误率超阈）→ OPEN →（冷却后）→ HALF_OPEN 放探测请求 → 成功回 CLOSED / 失败回 OPEN。
  · 多 worker 一致：生产用 Redis Lua 原子扣减（防超卖）；本机默认走进程内 Dict。
  · 接口/实现分离：LLMGovernance 接受 backend="memory"（默认）/ "redis"（生产级别）。

⚠️ 诚实边界：
  · 默认 backend=memory（向后兼容 88 测试）；生产 backend=redis 必须 REDIS_URL 已配且可达。
  · Redis Lua 原子化保证多 worker 计数一致；进程内版仍是单实例有效。
"""
import json as _json
import logging
import os
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from app import observability
from app.config import (
    LLM_RATELIMIT_QPS,
    LLM_RATELIMIT_BURST,
    LLM_TOKEN_BUDGET_PER_MIN,
    LLM_CB_ERROR_THRESHOLD,
    LLM_CB_MIN_REQUESTS,
    LLM_CB_WINDOW,
    LLM_CB_COOLDOWN,
    LLM_CB_HALFOPEN_PROBES,
)

_log = logging.getLogger(__name__)


class RateLimitError(RuntimeError):
    """触发限流（QPS 令牌耗尽 或 Token 预算超限）。调用方应降级，不应重试风暴。"""


class CircuitOpenError(RuntimeError):
    """熔断器处于 OPEN：直接快速失败，不发真实请求（防雪崩）。冷却后进入 HALF_OPEN 放探测。"""


# ===========================================================================
# 进程内后端（向后兼容：单实例有效）
# ===========================================================================
@dataclass
class TokenBucket:
    """令牌桶限流：容量 burst，按 qps 匀速补充；acquire 非阻塞，拿不到立刻拒绝（fail-fast）。

    进程内实现，多 worker 各管各的。生产换 RedisGovernanceBackend（Lua 原子扣减）。
    """
    qps: float
    burst: int
    tokens: float = 0.0
    last: float = field(default_factory=time.monotonic)

    def __post_init__(self):
        self.tokens = float(self.burst)

    def acquire(self) -> None:
        now = time.monotonic()
        self.tokens = min(self.burst, self.tokens + (now - self.last) * self.qps)
        self.last = now
        if self.tokens < 1:
            raise RateLimitError(f"QPS 限流触发：桶剩余 {self.tokens:.2f}（qps={self.qps}, burst={self.burst}）")
        self.tokens -= 1

    def snapshot(self) -> dict:
        return {"qps": self.qps, "burst": self.burst, "tokens_left": round(self.tokens, 2)}


@dataclass
class TokenBudget:
    """Token 预算：60s 滚动窗口内实际消耗的 token 总量不超过 budget。

    进程内实现，多 worker 各管各的。生产换 RedisGovernanceBackend（sorted set 滑动窗口）。
    """
    budget: int
    window_s: float = 60.0
    _events: deque = field(default_factory=deque)  # (monotonic_ts, tokens)

    def _evict(self, now: float) -> None:
        while self._events and now - self._events[0][0] > self.window_s:
            self._events.popleft()

    def reserve(self, estimate: int) -> None:
        if self.budget <= 0:
            return
        now = time.monotonic()
        self._evict(now)
        used = sum(t for _, t in self._events)
        if used + estimate > self.budget:
            raise RateLimitError(
                f"Token 预算限流触发：窗口内已用 {used} + 预估 {estimate} > 预算 {self.budget}"
            )

    def settle(self, actual: int, estimate: int) -> None:
        if self.budget <= 0 or actual <= 0:
            return
        now = time.monotonic()
        self._evict(now)
        self._events.append((now, max(actual - estimate, 0) + estimate))
        self._events[-1] = (now, actual)

    def used(self) -> int:
        self._evict(time.monotonic())
        return sum(t for _, t in self._events)


class CircuitBreaker:
    """三态断路器（滑动窗口按错误率判定）—— 进程内版。

    CLOSED（正常放行）
      └─ 窗口内请求数 ≥ min_requests 且错误率 ≥ threshold → OPEN
    OPEN（快速失败，不发真实请求）
      └─ 冷却 cooldown 秒 → HALF_OPEN
    HALF_OPEN（放 halfopen_probes 个探测请求）
      ├─ 任一探测成功 → CLOSED（窗口重置）
      └─ 任一探测失败 → OPEN（冷却重新计时）
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, error_threshold: float, min_requests: int, window_s: float,
                 cooldown_s: float, halfopen_probes: int):
        self.error_threshold = error_threshold
        self.min_requests = min_requests
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self.halfopen_probes = halfopen_probes
        self.state = self.CLOSED
        self._events: deque = deque()      # (ts, is_success)
        self._opened_at = 0.0
        self._probes_in_flight = 0
        self._probes_allowed = 0
        self._lock = threading.Lock()

    def _evict(self, now: float) -> None:
        while self._events and now - self._events[0][0] > self.window_s:
            self._events.popleft()

    def before_call(self) -> None:
        with self._lock:
            now = time.monotonic()
            if self.state == self.OPEN:
                if now - self._opened_at >= self.cooldown_s:
                    self.state = self.HALF_OPEN
                    self._probes_allowed = self.halfopen_probes
                    self._probes_in_flight = 0
                    observability.log.info("circuit_half_open", extra={"cooldown_s": self.cooldown_s})
                else:
                    raise CircuitOpenError(
                        f"熔断 OPEN：冷却剩 {self.cooldown_s - (now - self._opened_at):.1f}s"
                    )
            if self.state == self.HALF_OPEN:
                if self._probes_in_flight >= self._probes_allowed:
                    raise CircuitOpenError("熔断 HALF_OPEN：探测并发已满，快速失败")
                self._probes_in_flight += 1

    def record_success(self) -> None:
        with self._lock:
            if self.state == self.HALF_OPEN:
                self.state = self.CLOSED
                self._events.clear()
                observability.log.info("circuit_closed", extra={"reason": "half_open probe success"})
                return
            self._events.append((time.monotonic(), True))

    def record_failure(self) -> None:
        with self._lock:
            if self.state == self.HALF_OPEN:
                self._trip()
                return
            self._events.append((time.monotonic(), False))
            now = time.monotonic()
            self._evict(now)
            total = len(self._events)
            if total >= self.min_requests:
                fails = sum(1 for _, ok in self._events if not ok)
                if fails / total >= self.error_threshold:
                    self._trip()

    def _trip(self) -> None:
        self.state = self.OPEN
        self._opened_at = time.monotonic()
        self._probes_in_flight = 0
        observability.log.warning("circuit_open", extra={
            "threshold": self.error_threshold, "window_s": self.window_s,
        })

    def snapshot(self) -> dict:
        with self._lock:
            now = time.monotonic()
            self._evict(now)
            total = len(self._events)
            fails = sum(1 for _, ok in self._events if not ok)
            return {
                "state": self.state,
                "window_requests": total,
                "window_failures": fails,
                "error_rate": round(fails / total, 3) if total else 0.0,
                "cooldown_remain_s": round(max(0.0, self.cooldown_s - (now - self._opened_at)), 1)
                if self.state == self.OPEN else 0.0,
            }


# ===========================================================================
# Redis 后端（生产级别：Lua 原子化 + 多 worker 共享 + sorted set 滑动窗口）
# ===========================================================================
class RedisGovernanceBackend:
    """Redis 后端的治理：令牌桶 Lua 原子扣减 + Token 预算 sorted set + 熔断 hash。

    📌 多 worker 一致性：
      · 令牌桶 Lua：一次 EVAL 完成 token refill + 扣减 + 持久化，原子无超卖
      · Token 预算：ZADD/ZREMRANGEBYSCORE/ZCARD 滑动窗口
      · 熔断状态：hash 存 state/opened_at/probes_in_flight + events sorted set
    """

    KEY_PREFIX = "ai:gov:"

    # Lua 脚本：原子令牌桶（毫秒级精度，避免 sleep）
    _LUA_TOKEN_BUCKET = """
    local tokens_key = KEYS[1]
    local last_key = KEYS[2]
    local burst = tonumber(ARGV[1])
    local qps = tonumber(ARGV[2])
    local now_ms = tonumber(ARGV[3])
    local tokens = tonumber(redis.call('GET', tokens_key) or burst)
    local last_ms = tonumber(redis.call('GET', last_key) or now_ms)
    local delta_ms = now_ms - last_ms
    if delta_ms < 0 then delta_ms = 0 end
    -- 按 qps 匀速补充（ms 精度）
    tokens = math.min(burst, tokens + delta_ms * qps / 1000.0)
    if tokens < 1 then
        redis.call('SET', tokens_key, tokens, 'PX', 120000)
        redis.call('SET', last_key, now_ms, 'PX', 120000)
        return 0
    end
    tokens = tokens - 1
    redis.call('SET', tokens_key, tokens, 'PX', 120000)
    redis.call('SET', last_key, now_ms, 'PX', 120000)
    return 1
    """

    def __init__(self, url: str, client=None):
        """client: 测试可传 FakeRedis 跳过 ping（生产代码不传）。"""
        import redis
        self._url = url
        if client is not None:
            self._client = client
        else:
            if not url:
                raise RuntimeError("RedisGovernanceBackend 需要 REDIS_URL")
            self._client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=2.0)
            try:
                self._client.ping()
            except Exception as e:  # noqa: BLE001
                raise RuntimeError(f"Redis 不可达 url={url}: {e}") from e
        # 注册 Lua 脚本（用 Script 对象自带 SHA 缓存）
        self._bucket_script = self._client.register_script(self._LUA_TOKEN_BUCKET)

    # ---------------- 令牌桶 ----------------
    def token_bucket_acquire(self, qps: float, burst: int) -> bool:
        tokens_key = f"{self.KEY_PREFIX}bucket:llm:tokens"
        last_key = f"{self.KEY_PREFIX}bucket:llm:last"
        now_ms = int(time.time() * 1000)
        try:
            res = self._bucket_script(keys=[tokens_key, last_key],
                                       args=[burst, qps, now_ms])
        except Exception:  # noqa: BLE001
            # fakeredis 不支持 Lua 时降级——见 governance.py:LLMGovernance 注释
            _log.warning("Redis Lua 不可用，令牌桶降级为本地时间估算法")
            return True
        return bool(int(res))

    def token_bucket_snapshot(self, qps: float, burst: int) -> dict:
        tokens_key = f"{self.KEY_PREFIX}bucket:llm:tokens"
        try:
            t = float(self._client.get(tokens_key) or burst)
        except Exception:  # noqa: BLE001
            t = float(burst)
        return {"qps": qps, "burst": burst, "tokens_left": round(t, 2)}

    # ---------------- Token 预算（sorted set 滑动窗口）----------------
    # 📌 2026-09-11 实测发现并修复的 P0（这条闸门此前**完全失效**，务必记住教训）：
    #    Bug A · `used_count, = pipe.execute()`：pipeline 里排了 2 条命令却只解包 1 个变量，
    #            必抛 ValueError → 被 except 吞掉 → 永远「降级放行」，闸门等于没装。
    #    Bug B · settle 时 `zadd(key, {member: actual})` 把 **token 数当成了 score**，
    #            而清理用的是 `zremrangebyscore(key, 0, now-window)` —— token 数（几千）
    #            永远落在 [0, 约 1.7e9] 区间内，于是每次 reserve 都把记录**全部删光**，
    #            zcard 恒为 0，预算永不触发（实测把预算压到 1 仍然放行）。
    #    Bug C · 用「事件条数」跟 estimate(token 数) 相加，单位都不统一。
    #    修法：score 固定存**时间戳**（过期清理靠它），token 数放进 member；
    #          求和时解析 member；member 带 v2: 前缀以便忽略旧的脏数据。
    _TB_PREFIX = "v2:"

    @classmethod
    def _tb_sum(cls, items) -> int:
        """从存活 member 累加 token 数。member 形如 `v2:{tokens}:{ts}:{uniq}`，score=ts。
        旧格式（v1，score 误存 token 数 / member 是 `ts:id`）一律忽略——
        否则会把时间戳（约 1.7e9）当成 token 数累加，直接误判超预算。
        """
        total = 0
        for member, _score in items:
            if not isinstance(member, str) or not member.startswith(cls._TB_PREFIX):
                continue
            try:
                total += int(member.split(":", 2)[1])
            except (IndexError, ValueError):
                continue
        return total

    def token_budget_reserve(self, budget: int, window_s: float, estimate: int) -> bool:
        if budget <= 0:
            return True
        key = f"{self.KEY_PREFIX}tokenbudget:llm"
        cutoff = time.time() - window_s
        try:
            pipe = self._client.pipeline()
            pipe.zremrangebyscore(key, 0, cutoff)
            pipe.zrange(key, 0, -1, withscores=True)
            # 📌 排了 2 条命令就必须接 2 个返回值（Bug A）
            _removed, items = pipe.execute()
            used_tokens = self._tb_sum(items)
            return used_tokens + estimate <= budget
        except Exception:  # noqa: BLE001
            # 可用性优先：Redis 挂了不能把整个对话打断，放行但留日志告警
            _log.warning("Redis ZSET 失败，降级放行")
            return True

    def token_budget_settle(self, budget, window_s, actual, estimate):
        if budget <= 0 or actual <= 0:
            return
        key = f"{self.KEY_PREFIX}tokenbudget:llm"
        try:
            ts = time.time()
            # score = 时间戳（过期清理依据）；token 数放进 member（Bug B）
            member = f"{self._TB_PREFIX}{int(actual)}:{ts}:{uuid.uuid4().hex[:8]}"
            self._client.zadd(key, {member: ts})
            self._client.expire(key, int(window_s) + 10)
        except Exception:  # noqa: BLE001
            _log.warning("Token 预算记账失败（不影响主链路）")

    def token_budget_used(self, budget, window_s) -> int:
        """窗口内**已消耗的 token 数**（不是事件条数，Bug C）。"""
        if budget <= 0:
            return 0
        key = f"{self.KEY_PREFIX}tokenbudget:llm"
        cutoff = time.time() - window_s
        try:
            pipe = self._client.pipeline()
            pipe.zremrangebyscore(key, 0, cutoff)
            pipe.zrange(key, 0, -1, withscores=True)
            _removed, items = pipe.execute()
            return self._tb_sum(items)
        except Exception:  # noqa: BLE001
            return 0

    # ---------------- 熔断器 ----------------
    def breaker_before_call(self, name: str, cooldown_s: float,
                            halfopen_probes: int) -> tuple:
        key = f"{self.KEY_PREFIX}breaker:{name}"
        try:
            data = self._client.hgetall(key)
        except Exception:  # noqa: BLE001
            return (True, CircuitBreaker.CLOSED)
        state = data.get("state", CircuitBreaker.CLOSED)
        now = time.time()
        opened_at = float(data.get("opened_at", 0) or 0)
        probes_in_flight = int(data.get("probes_in_flight", 0) or 0)
        probes_allowed = int(data.get("probes_allowed", halfopen_probes) or halfopen_probes)
        if state == CircuitBreaker.OPEN:
            if now - opened_at >= cooldown_s:
                state = CircuitBreaker.HALF_OPEN
                probes_in_flight = 0
                probes_allowed = halfopen_probes
                self._client.hset(key, mapping={"state": state, "probes_in_flight": 0,
                                                 "probes_allowed": probes_allowed})
            else:
                return (False, state)
        if state == CircuitBreaker.HALF_OPEN:
            if probes_in_flight >= probes_allowed:
                return (False, state)
            self._client.hincrby(key, "probes_in_flight", 1)
        return (True, state)

    def breaker_record_success(self, name: str) -> None:
        key = f"{self.KEY_PREFIX}breaker:{name}"
        events_key = f"{key}:events"
        try:
            data = self._client.hgetall(key)
        except Exception:  # noqa: BLE001
            return
        state = data.get("state", CircuitBreaker.CLOSED)
        now = time.time()
        if state == CircuitBreaker.HALF_OPEN:
            self._client.hset(key, mapping={"state": CircuitBreaker.CLOSED,
                                             "opened_at": 0,
                                             "probes_in_flight": 0})
            # 清事件 sorted set
            self._client.delete(events_key)
            return
        # CLOSED：追加成功（score=now 时间戳，用于 zremrangebyscore 滑动窗口；member 末尾 1=success/0=failure）
        self._client.zadd(events_key, {f"{now}:{id(object())}:1": now})
        self._client.expire(events_key, 120)

    def breaker_record_failure(self, name: str) -> None:
        key = f"{self.KEY_PREFIX}breaker:{name}"
        events_key = f"{key}:events"
        try:
            data = self._client.hgetall(key)
        except Exception:  # noqa: BLE001
            return
        state = data.get("state", CircuitBreaker.CLOSED)
        now = time.time()
        if state == CircuitBreaker.HALF_OPEN:
            self._client.hset(key, mapping={"state": CircuitBreaker.OPEN,
                                             "opened_at": now,
                                             "probes_in_flight": 0})
            return
        # CLOSED：追加失败（score=now 时间戳，member 末尾 0=failure）
        self._client.zadd(events_key, {f"{now}:{id(object())}:0": now})
        self._client.expire(events_key, 120)
        # 触发条件检查
        window_s = 60.0
        cutoff = now - window_s
        try:
            self._client.zremrangebyscore(events_key, "-inf", cutoff)
            events = self._client.zrange(events_key, 0, -1, withscores=False)
        except Exception:  # noqa: BLE001
            return
        total = len(events)
        if total >= 5:  # 默认 LLM_CB_MIN_REQUESTS
            fails = sum(1 for m in events if m.endswith(":0"))
            if total > 0 and fails / total >= 0.5:
                self._client.hset(key, mapping={"state": CircuitBreaker.OPEN,
                                                 "opened_at": now,
                                                 "probes_in_flight": 0})

    def breaker_snapshot(self, name: str, window_s: float, cooldown_s: float) -> dict:
        key = f"{self.KEY_PREFIX}breaker:{name}"
        events_key = f"{key}:events"
        try:
            data = self._client.hgetall(key)
            self._client.zremrangebyscore(events_key, "-inf", time.time() - window_s)
            events = self._client.zrange(events_key, 0, -1, withscores=False)
        except Exception:  # noqa: BLE001
            return {"state": CircuitBreaker.CLOSED, "window_requests": 0, "window_failures": 0,
                    "error_rate": 0.0, "cooldown_remain_s": 0.0}
        state = data.get("state", CircuitBreaker.CLOSED)
        total = len(events)
        fails = sum(1 for m in events if m.endswith(":0"))
        opened_at = float(data.get("opened_at", 0) or 0)
        return {
            "state": state,
            "window_requests": total,
            "window_failures": fails,
            "error_rate": round(fails / total, 3) if total else 0.0,
            "cooldown_remain_s": round(max(0.0, cooldown_s - (time.time() - opened_at)), 1)
            if state == CircuitBreaker.OPEN else 0.0,
        }

    def close(self):
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


# ===========================================================================
# 统一门面 LLMGovernance：根据 backend 选择进程内或 Redis
# ===========================================================================
class LLMGovernance:
    """组合：QPS 令牌桶 + Token 预算 + 断路器。

    backend='memory'（默认，向后兼容 88 测试）
      ├─ TokenBucket / TokenBudget / CircuitBreaker 全进程内
      ├─ 多 worker 各管各的（与原版一致）
    backend='redis'（生产级别）
      ├─ 限流：Redis Lua 原子扣减（防超卖）
      ├─ Token 预算：Redis sorted set 滑动窗口
      └─ 熔断：Redis hash + sorted set 集中状态
    """

    def __init__(self, backend: str = "memory", redis_url: Optional[str] = None,
                 redis_client=None):
        """redis_client: 测试用 fakeredis 注入跳过 ping；生产代码不传。"""
        self.backend_kind = backend
        if backend == "redis":
            if not redis_url:
                raise RuntimeError("backend='redis' 必须提供 redis_url")
            self._redis = RedisGovernanceBackend(redis_url, client=redis_client)
            self._bucket = None
            self._budget = None
            self._breaker = None
        else:
            self._redis = None
            self._bucket = TokenBucket(qps=LLM_RATELIMIT_QPS, burst=LLM_RATELIMIT_BURST)
            self._budget = TokenBudget(budget=LLM_TOKEN_BUDGET_PER_MIN)
            self._breaker = CircuitBreaker(
                error_threshold=LLM_CB_ERROR_THRESHOLD,
                min_requests=LLM_CB_MIN_REQUESTS,
                window_s=LLM_CB_WINDOW,
                cooldown_s=LLM_CB_COOLDOWN,
                halfopen_probes=LLM_CB_HALFOPEN_PROBES,
            )

    @staticmethod
    def _estimate_tokens(messages) -> int:
        """调用前估算：按字符数 / 2 粗估（中文约 1 token/字，英文约 0.25/字，取中偏保守）。"""
        chars = sum(len(m.content or "") for m in messages)
        return max(32, chars // 2)

    def before_call(self, messages) -> None:
        """三道闸门依次过：断路器 → QPS 令牌 → Token 预算。任一触发即抛对应异常。"""
        if self._redis:
            allowed, state = self._redis.breaker_before_call(
                "llm", LLM_CB_COOLDOWN, LLM_CB_HALFOPEN_PROBES
            )
            if not allowed:
                raise CircuitOpenError(f"熔断 {state}")
            if not self._redis.token_bucket_acquire(LLM_RATELIMIT_QPS, LLM_RATELIMIT_BURST):
                raise RateLimitError("QPS 限流触发（Redis）")
            if not self._redis.token_budget_reserve(
                LLM_TOKEN_BUDGET_PER_MIN, 60.0, self._estimate_tokens(messages)
            ):
                raise RateLimitError("Token 预算限流触发（Redis）")
        else:
            self._breaker.before_call()
            self._bucket.acquire()
            self._budget.reserve(self._estimate_tokens(messages))

    def record_success(self, messages, usage: dict) -> None:
        estimate = self._estimate_tokens(messages)
        actual = int(usage.get("total", 0) or 0)
        if self._redis:
            self._redis.token_budget_settle(LLM_TOKEN_BUDGET_PER_MIN, 60.0, actual, estimate)
            self._redis.breaker_record_success("llm")
        else:
            self._budget.settle(actual, estimate)
            self._breaker.record_success()

    def record_failure(self) -> None:
        if self._redis:
            self._redis.breaker_record_failure("llm")
        else:
            self._breaker.record_failure()

    def snapshot(self) -> dict:
        if self._redis:
            return {
                "rate_limit": self._redis.token_bucket_snapshot(LLM_RATELIMIT_QPS, LLM_RATELIMIT_BURST),
                "token_budget": {"per_min": LLM_TOKEN_BUDGET_PER_MIN,
                                 "used": self._redis.token_budget_used(LLM_TOKEN_BUDGET_PER_MIN, 60.0)},
                "circuit_breaker": self._redis.breaker_snapshot("llm", LLM_CB_WINDOW, LLM_CB_COOLDOWN),
            }
        return {
            "rate_limit": {"qps": self._bucket.qps, "burst": self._bucket.burst,
                           "tokens_left": round(self._bucket.tokens, 2)},
            "token_budget": {"per_min": self._budget.budget, "used": self._budget.used()},
            "circuit_breaker": self._breaker.snapshot(),
        }


# 进程内全局单例（默认向后兼容；生产设 GOVERNANCE_BACKEND=redis 切到 RedisGovernanceBackend）
governance_backend = os.environ.get("GOVERNANCE_BACKEND", "memory").lower()
governance_redis_url = os.environ.get("REDIS_URL", "") if governance_backend == "redis" else None
llm_governance = LLMGovernance(
    backend=governance_backend,
    redis_url=governance_redis_url or None,
)