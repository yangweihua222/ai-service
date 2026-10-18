# -*- coding: utf-8 -*-
"""RedisBackend 单测：会话历史 + 反思笔记 + 多租户隔离 + TTL。

📌 用 fakeredis 2.x 模拟真实 Redis（pip install fakeredis>=2.0）。
   跑测试无需起 Redis 服务。
"""
import json
import time

import pytest
from fakeredis import FakeServer
from fakeredis import FakeRedis

from app.llm import Message, ToolCall, OpenAICompatLLM
from app.memory_redis import RedisBackend


@pytest.fixture
def fake_server():
    """每个测试一个全新 FakeServer，确保 key namespace 隔离。"""
    server = FakeServer()
    client = FakeRedis(server=server, decode_responses=True)
    yield client
    client.flushall()


@pytest.fixture
def backend(fake_server):
    """RedisBackend 实例，但直接用 fake_server 替代 redis-py 客户端。"""
    backend = RedisBackend.__new__(RedisBackend)  # 跳过 __init__ 的 ping 校验
    backend._client = fake_server
    backend._url = "redis://fakeredis-test/0"
    return backend


def _msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


class TestHistoryCRUD:
    def test_upsert_and_load_history(self, backend):
        msgs = [_msg("user", "你好"), _msg("assistant", "有什么可以帮你的？")]
        backend.upsert_history("k1", "t1", "u1", "s1", msgs)
        loaded = backend.load_history("k1")
        assert len(loaded) == 2
        assert loaded[0].role == "user"
        assert loaded[0].content == "你好"

    def test_load_missing_returns_none(self, backend):
        assert backend.load_history("nonexistent") is None

    def test_overwrite_history(self, backend):
        backend.upsert_history("k1", "t", "u", "s", [_msg("user", "v1")])
        backend.upsert_history("k1", "t", "u", "s",
                                [_msg("user", "v2"), _msg("assistant", "v2-resp")])
        loaded = backend.load_history("k1")
        assert len(loaded) == 2
        assert loaded[0].content == "v2"


class TestTenantIsolation:
    """多租户隔离：key 强制含 tenant_id（P0 修复——物理隔离不是 WHERE 过滤）。"""

    def test_same_session_id_different_tenants_isolated(self, backend):
        backend.upsert_history("tenant=A:user=u:session=shared",
                               "A", "u", "shared", [_msg("user", "A 的对话")])
        backend.upsert_history("tenant=B:user=u:session=shared",
                               "B", "u", "shared", [_msg("user", "B 的对话")])
        assert backend.load_history("tenant=A:user=u:session=shared")[0].content == "A 的对话"
        assert backend.load_history("tenant=B:user=u:session=shared")[0].content == "B 的对话"


class TestReflection:
    def test_save_and_load_reflection(self, backend):
        backend.upsert_reflection("k1", "t", "u", "s", "工具选错了")
        assert backend.load_reflection("k1") == "工具选错了"

    def test_clear_reflection(self, backend):
        backend.upsert_reflection("k1", "t", "u", "s", "test")
        backend.clear_reflection("k1")
        assert backend.load_reflection("k1") is None

    def test_reflection_isolation_by_tenant(self, backend):
        backend.upsert_reflection("k1", "tenant-A", "u", "s", "A 的反思")
        backend.upsert_reflection("k2", "tenant-B", "u", "s", "B 的反思")
        assert backend.load_reflection("k1") == "A 的反思"
        assert backend.load_reflection("k2") == "B 的反思"


class TestTTL:
    def test_ttl_set_on_history(self, backend):
        backend.upsert_history("k1", "t", "u", "s", [_msg("user", "hi")])
        ttl = backend._client.ttl("ai:mem:k1")
        # TTL > 0（已设过期）且 ≤ 3600
        assert 0 < ttl <= 3600

    def test_ttl_set_on_reflection(self, backend):
        backend.upsert_reflection("k1", "t", "u", "s", "n")
        ttl = backend._client.ttl("ai:refl:k1")
        # 反思 TTL 24h
        assert 0 < ttl <= 86400

    def test_cleanup_expired_returns_keys(self, backend):
        # 写一个无 TTL 的 key（手动）
        backend._client.set("ai:mem:orphan:1", "stale")
        count, keys = backend.cleanup_expired(3600)
        # 无过期 key 被清
        assert "orphan:1" in keys


class TestListSessions:
    def test_list_sessions_per_tenant_user(self, backend):
        backend.upsert_history("tenant=t:user=u:session=s1", "t", "u", "s1",
                               [_msg("user", "1")])
        backend.upsert_history("tenant=t:user=u:session=s2", "t", "u", "s2",
                               [_msg("user", "2")])
        sessions = backend.list_sessions("t", "u")
        assert sorted(sessions) == ["s1", "s2"]

    def test_list_sessions_does_not_leak_across_users(self, backend):
        backend.upsert_history("tenant=t:user=u1:session=s1", "t", "u1", "s1",
                               [_msg("user", "1")])
        backend.upsert_history("tenant=t:user=u2:session=s1", "t", "u2", "s1",
                               [_msg("user", "2")])
        assert backend.list_sessions("t", "u1") == ["s1"]
        assert backend.list_sessions("t", "u2") == ["s1"]
        # 同 session_id 但不同 user 不混


class TestMemoryStoreBackendSwitch:
    """MemoryStore 集成：backend='redis' 时走 RedisBackend（端到端）。"""

    def test_memorystore_with_redis_backend(self, fake_server):
        from app.memory import MemoryStore
        store = MemoryStore(backend="redis", redis_url="redis://fakeredis/0")
        # 注入 fake_server 客户端（跳过 ping）
        store._redis_client_override = fake_server
        store.append("tenant-A", "u", "s", _msg("user", "hello"))
        msgs = store.load("tenant-A", "u", "s")
        assert len(msgs) == 1
        assert msgs[0].content == "hello"
        # 新建 MemoryStore 实例（模拟重启），仍能读到
        store2 = MemoryStore(backend="redis", redis_url="redis://fakeredis/0")
        store2._redis_client_override = fake_server
        loaded = store2.load("tenant-A", "u", "s")
        assert loaded[0].content == "hello"

    def test_memorystore_default_backend_is_process_memory(self, tmp_path):
        """默认 backend=memory（进程内，零外部依赖；生产一律 redis）。"""
        from app.memory import MemoryStore
        store = MemoryStore()
        store.append("t", "u", "s", _msg("user", "v"))
        assert store.load("t", "u", "s")[0].content == "v"

class TestToolCallsRoundTrip:
    """2026-09-11 修的生产级 bug：**多轮对话在真正落盘后崩**。

    现象：历史消息经 JSON 落 Redis 再读回，`tool_calls` 退化成一堆 dict；
    第二轮 `llm._build_payload()` 访问 `tc.id` 直接抛
    `AttributeError: 'dict' object has no attribute 'id'`。
    进程内（不持久化）时永远不复现，所以这个坑只在接了真 Redis 之后才暴露。
    """

    def test_tool_calls_restored_as_objects_after_reload(self, backend):
        history = [
            _msg("user", "糖尿病酮症酸中毒怎么处理？"),
            Message(
                role="assistant",
                content="",
                tool_calls=[ToolCall(id="call_1", name="query_knowledge_base",
                                      arguments={"query": "DKA 处理"})],
            ),
            _msg("tool", "补液首选生理盐水"),
        ]
        key = "tenant=t:user=u:session=s"
        backend.upsert_history(key, "t", "u", "s", history)
        loaded = backend.load_history(key)

        tc = loaded[1].tool_calls[0]
        assert isinstance(tc, ToolCall), (
            "反序列化必须还原成 ToolCall 对象；否则第二轮对话会对 dict 取 .id 直接崩"
        )
        assert tc.name == "query_knowledge_base"

    def test_rebuild_payload_after_reload_does_not_crash(self, backend):
        """回归验证：load 回来的历史直接送进 _build_payload 不能崩。"""
        backend.upsert_history(
            "tenant=t:user=u:session=s", "t", "u", "s",
            [_msg("user", "查一下"),
             Message(role="assistant", content="",
                     tool_calls=[ToolCall(id="c1", name="query_usage_guide",
                                          arguments={"query": "怎么请假"})])],
        )
        loaded = backend.load_history("tenant=t:user=u:session=s")

        llm = OpenAICompatLLM.__new__(OpenAICompatLLM)  # 跳过 __init__（不联网）
        llm.model = "unit-test"
        payload = llm._build_payload(loaded)
        fn = payload["messages"][1]["tool_calls"][0]["function"]
        assert fn["name"] == "query_usage_guide"
        assert json.loads(fn["arguments"]) == {"query": "怎么请假"}
