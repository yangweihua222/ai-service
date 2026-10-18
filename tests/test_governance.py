# -*- coding: utf-8 -*-
"""LLM 服务治理单测：令牌桶限流 / Token 预算 / 熔断三态流转 / Agent 降级兜底。"""
import asyncio
import pytest

from app.governance import (
    RateLimitError, CircuitOpenError, TokenBucket, TokenBudget, CircuitBreaker,
)
from app.llm import Message, MockLLM
from app.agent import LLMPlanner, GOV_DEGRADE_MSG


# ---------- 令牌桶 ----------
def test_token_bucket_rejects_when_empty():
    tb = TokenBucket(qps=0.01, burst=2)  # 补充极慢，等效"只许 2 个"
    tb.acquire()
    tb.acquire()
    with pytest.raises(RateLimitError):
        tb.acquire()


def test_token_bucket_refills_over_time():
    import time as _t
    tb = TokenBucket(qps=100, burst=1)
    tb.acquire()
    with pytest.raises(RateLimitError):
        tb.acquire()
    _t.sleep(0.03)  # 100/s → 30ms 补 3 个
    tb.acquire()  # 不再抛错


# ---------- Token 预算 ----------
def test_token_budget_blocks_when_exceeded():
    bud = TokenBudget(budget=100)
    bud.reserve(60)
    bud.settle(60, 60)
    with pytest.raises(RateLimitError):
        bud.reserve(50)  # 60 + 50 > 100


def test_token_budget_disabled():
    bud = TokenBudget(budget=0)
    bud.reserve(10**9)  # 0 = 不启用，永不拦


# ---------- 熔断三态 ----------
def _breaker():
    return CircuitBreaker(error_threshold=0.5, min_requests=4, window_s=60,
                          cooldown_s=0.05, halfopen_probes=1)


def test_breaker_trips_on_error_rate():
    cb = _breaker()
    for _ in range(2):
        cb.before_call(); cb.record_success()
    for _ in range(2):
        cb.before_call(); cb.record_failure()
    # 4 个请求、50% 失败 ≥ 阈值 → OPEN；立刻再调用应快速失败（不发真实请求）
    assert cb.state == cb.OPEN
    with pytest.raises(CircuitOpenError):
        cb.before_call()


def test_breaker_half_open_recovers():
    import time as _t
    cb = _breaker()
    for _ in range(4):
        cb.before_call(); cb.record_failure()
    with pytest.raises(CircuitOpenError):
        cb.before_call()
    _t.sleep(0.06)  # 超过 cooldown → 下次 before_call 转 HALF_OPEN
    cb.before_call()      # 第 1 个探测请求放行
    with pytest.raises(CircuitOpenError):
        cb.before_call()  # 探测并发 1 已满
    cb.record_success()   # 探测成功 → CLOSED
    assert cb.state == cb.CLOSED


def test_breaker_half_open_retrip():
    import time as _t
    cb = _breaker()
    for _ in range(4):
        cb.before_call(); cb.record_failure()
    _t.sleep(0.06)
    cb.before_call()
    cb.record_failure()   # 探测失败 → 重新 OPEN
    assert cb.state == cb.OPEN


# ---------- Agent 降级兜底 ----------
class _FailingLLM(MockLLM):
    async def chat(self, messages, tools=None):
        raise CircuitOpenError("熔断 OPEN：测试")


def test_planner_degrades_on_circuit_open():
    planner = LLMPlanner(llm=_FailingLLM())
    action = asyncio.run(planner.decide("帮我查缺勤", [], []))
    assert action.finish is True
    assert action.thought == GOV_DEGRADE_MSG
