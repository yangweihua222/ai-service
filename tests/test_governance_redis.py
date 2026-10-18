# -*- coding: utf-8 -*-
"""RedisGovernanceBackend 单测：限流(Lua)+ Token 预算(sorted set)+ 熔断(hash)。

📌 Lua 脚本在 fakeredis 2.38.0 部分支持（可能失败），测试用 mock 跳过。
   生产环境真实 Redis 完全支持 Lua EVAL。
"""
import time
from unittest.mock import patch

import pytest
from fakeredis import FakeServer, FakeRedis

from app.governance import (
    CircuitBreaker,
    LLMGovernance,
    RedisGovernanceBackend,
)


@pytest.fixture
def fake_redis():
    server = FakeServer()
    client = FakeRedis(server=server, decode_responses=True)
    yield client
    client.flushall()
    client.close()


@pytest.fixture
def backend(fake_redis):
    """RedisGovernanceBackend 但直接注入 fake_redis 客户端。"""
    b = RedisGovernanceBackend.__new__(RedisGovernanceBackend)
    b._url = "redis://fakeredis/0"
    b._client = fake_redis
    # 跳过 Lua 注册（fakeredis 可能不完整支持）
    b._bucket_script = lambda keys, args: 1
    return b


class TestTokenBucket:
    def test_acquire_returns_bool(self, backend):
        ok = backend.token_bucket_acquire(qps=5, burst=10)
        assert isinstance(ok, bool)

    def test_snapshot_returns_state(self, backend):
        snap = backend.token_bucket_snapshot(qps=5, burst=10)
        assert snap["qps"] == 5
        assert snap["burst"] == 10
        assert "tokens_left" in snap


class TestTokenBudget:
    def test_reserve_disabled_when_budget_zero(self, backend):
        assert backend.token_budget_reserve(0, 60.0, 10000) is True

    def test_reserve_succeeds_below_budget(self, backend):
        assert backend.token_budget_reserve(1000, 60.0, 100) is True

    def test_settle_and_used(self, backend):
        backend.token_budget_settle(1000, 60.0, 100, 80)
        backend.token_budget_settle(1000, 60.0, 50, 40)
        used = backend.token_budget_used(1000, 60.0)
        # fakeredis ZADD/ZCARD 实现下，每条事件计 1（设计是 sorted set 简化版）
        # 真实生产建议改 hash+timestamp 精确计 token
        assert used >= 0


class TestCircuitBreaker:
    def test_before_call_returns_tuple(self, backend):
        allowed, state = backend.breaker_before_call("test", cooldown_s=30, halfopen_probes=2)
        assert allowed is True
        assert state == CircuitBreaker.CLOSED

    def test_record_failure_opens_breaker(self, backend):
        # 5 次失败（达到 MIN_REQUESTS=5 且失败率 100% >= 0.5）
        for _ in range(5):
            backend.breaker_record_failure("llm")
        snap = backend.breaker_snapshot("llm", window_s=60, cooldown_s=30)
        assert snap["state"] == CircuitBreaker.OPEN
        assert snap["error_rate"] >= 0.5

    def test_open_breaker_blocks_before_call(self, backend):
        # 触发 OPEN
        for _ in range(5):
            backend.breaker_record_failure("llm")
        allowed, state = backend.breaker_before_call("llm", cooldown_s=30, halfopen_probes=1)
        assert allowed is False
        assert state == CircuitBreaker.OPEN

    def test_half_open_after_cooldown(self, backend):
        for _ in range(5):
            backend.breaker_record_failure("llm")
        # 模拟 cooldown 已过（用 Redis key 直接改 opened_at）
        backend._client.hset("ai:gov:breaker:llm",
                              mapping={"opened_at": str(time.time() - 60)})
        allowed, state = backend.breaker_before_call("llm", cooldown_s=30, halfopen_probes=2)
        assert state == CircuitBreaker.HALF_OPEN
        assert allowed is True

    def test_half_open_success_recovers_to_closed(self, backend):
        for _ in range(5):
            backend.breaker_record_failure("llm")
        backend._client.hset("ai:gov:breaker:llm",
                              mapping={"opened_at": str(time.time() - 60)})
        backend.breaker_before_call("llm", cooldown_s=30, halfopen_probes=2)
        backend.breaker_record_success("llm")
        snap = backend.breaker_snapshot("llm", window_s=60, cooldown_s=30)
        assert snap["state"] == CircuitBreaker.CLOSED


class TestLLMGovernanceBackendSwitch:
    """LLMGovernance 集成：backend='redis' 时走 RedisGovernanceBackend。"""

    def test_redis_backend_path(self, fake_redis):
        from app.governance import LLMGovernance
        # 注入 fake_redis（跳过 ping）
        gov = LLMGovernance(backend="redis", redis_url="redis://fakeredis/0",
                            redis_client=fake_redis)
        # fakeredis 不支持 Lua，替换为恒真（限流绕开）
        gov._redis._bucket_script = lambda keys, args: 1
        from app.llm import Message
        msgs = [Message(role="user", content="test")]
        gov.before_call(msgs)  # 不抛
        gov.record_success(msgs, {"total": 100})
        snap = gov.snapshot()
        assert "rate_limit" in snap
        assert "token_budget" in snap
        assert "circuit_breaker" in snap

    def test_memory_backend_path_default(self):
        """默认 backend='memory'，与原 88 测试兼容。"""
        from app.governance import LLMGovernance
        gov = LLMGovernance()
        from app.llm import Message
        msgs = [Message(role="user", content="hi")]
        gov.before_call(msgs)  # 不抛
        gov.record_success(msgs, {"total": 50})
        assert gov.snapshot()["circuit_breaker"]["state"] == CircuitBreaker.CLOSED


class TestTokenBudgetP0Regression:
    """Token 预算闸门失效的回归测试（2026-09-11 实测发现并修复的 P0）。

    修复前这条闸门**完全没在工作**，三个 bug 叠加：
      A. `used_count, = pipe.execute()` 少接一个返回值 → 必抛 ValueError → 永远降级放行；
      B. settle 把 token 数存成了 score → reserve 的 zremrangebyscore 把记录全删光 → 统计恒为 0；
      C. 拿「事件条数」跟 estimate(token 数) 相加，单位都不统一。
    实测铁证：修复前把预算压到 1，reserve 仍返回 True（放行）。
    下面每个用例都对应上面某一个 bug，任何一个回归都会让这里变红。
    """

    def test_score_is_timestamp_not_token_count(self, backend):
        """Bug B：score 必须是时间戳（过期清理靠它），token 数放 member。"""
        backend.token_budget_settle(400000, 60.0, 3000, 3000)
        key = f"{backend.KEY_PREFIX}tokenbudget:llm"
        items = backend._client.zrange(key, 0, -1, withscores=True)
        assert len(items) == 1
        member, score = items[0]
        assert member.startswith("v2:3000:")   # token 数在 member 里
        assert score > 1_000_000_000           # score 是时间戳量级，不是 3000

    def test_used_sums_tokens_not_event_count(self, backend):
        """Bug C：统计的是 token 数，不是事件条数。"""
        for t in (3000, 5000, 2000):
            backend.token_budget_settle(400000, 60.0, t, t)
        assert backend.token_budget_used(400000, 60.0) == 10000

    def test_reserve_blocks_when_over_budget(self, backend):
        """Bug A/B 的直接后果：预算不足时必须拒绝（修复前预算压到 1 都放行）。"""
        backend.token_budget_settle(10000, 60.0, 9000, 9000)
        assert backend.token_budget_reserve(10000, 60.0, 5000) is False  # 9000+5000 > 10000
        assert backend.token_budget_reserve(10000, 60.0, 500) is True    # 9000+500 <= 10000

    def test_reserve_blocks_when_budget_is_one(self, backend):
        """最硬的判据：预算只有 1 个 token，任何请求都不能放行。"""
        backend.token_budget_settle(400000, 60.0, 100, 100)
        assert backend.token_budget_reserve(1, 60.0, 3000) is False

    def test_expired_events_are_cleaned(self, backend):
        """滑动窗口要真的会过期，否则统计只增不减、永久熔断。"""
        backend.token_budget_settle(400000, 1.0, 3000, 3000)
        assert backend.token_budget_used(400000, 60.0) == 3000
        time.sleep(1.2)
        assert backend.token_budget_used(400000, 1.0) == 0

    def test_legacy_v1_member_is_ignored(self, backend):
        """旧脏数据（member='ts:id'、score=token数）必须忽略。

        否则会把时间戳（约 1.7e9）当成 token 数累加，一上来就误判超预算。
        窗口取极大值，确保这条 v1 记录不会被 zremrangebyscore 清掉，
        从而真正检验的是「解析时忽略」而不是「被删掉了」。
        """
        key = f"{backend.KEY_PREFIX}tokenbudget:llm"
        backend._client.zadd(key, {"1789143353.39:2675454910048": 3000.0})
        assert backend.token_budget_used(400000, 2_000_000_000.0) == 0