# -*- coding: utf-8 -*-
"""MemoryStore 多租户隔离 + 持久化 + 摘要压缩 + TTL 单元测试。"""
import gc
import os
import tempfile

import pytest

from app.llm import Message
from app.memory import MemoryStore


@pytest.fixture
def tmp_db():
    fd, path = tempfile.mkstemp(suffix=".tmpdb")
    os.close(fd)
    yield path
    gc.collect()  # 触发 SQLite 连接释放
    if os.path.exists(path):
        try:
            os.remove(path)
        except OSError:
            pass


def _make_msg(role: str, content: str) -> Message:
    return Message(role=role, content=content)


class TestKeyIsolation:
    """多租户隔离（P0 修复）：key 必须含 tenant_id。"""

    def test_same_session_id_different_tenants_isolated(self, tmp_db):
        store = MemoryStore(backend="memory")
        store.append("tenant-A", "user1", "shared", _make_msg("user", "A 的对话"))
        msgs_b = store.load("tenant-B", "user1", "shared")
        assert msgs_b == [], "跨租户应返回空，禁止泄漏"
        msgs_a = store.load("tenant-A", "user1", "shared")
        assert len(msgs_a) == 1
        assert msgs_a[0].content == "A 的对话"

    def test_same_session_id_different_users_isolated(self, tmp_db):
        store = MemoryStore(backend="memory")
        store.append("tenant-A", "user1", "s1", _make_msg("user", "user1 对话"))
        msgs = store.load("tenant-A", "user2", "s1")
        assert msgs == []


class TestPersistence:
    """进程内（memory）后端的语义：同一实例内读写一致，**不跨实例共享**。

    ⚠️ 2026-09-11：SQLite 后端下线后，这两个用例从"验证跨实例持久化"改写为
       "固化进程内语义"——真正验证持久化的用例在 tests/test_memory_redis.py（用 fakeredis）。
    """

    def test_append_reloads_within_same_instance(self, tmp_db):
        store = MemoryStore(backend="memory")
        store.append("t", "u", "s", _make_msg("user", "hello"))
        msgs = store.load("t", "u", "s")
        assert len(msgs) == 1
        assert msgs[0].content == "hello"

    def test_process_memory_not_shared_across_instances(self, tmp_db):
        """进程内后端刻意不跨实例：新实例读到的是空的（重启丢记忆，这是已知取舍）。"""
        store = MemoryStore(backend="memory")
        history = [
            _make_msg("user", "msg1"),
            _make_msg("assistant", "reply1"),
            _make_msg("user", "msg2"),
            _make_msg("assistant", "reply2"),
        ]
        store.save("t", "u", "s", history)
        assert [m.content for m in store.load("t", "u", "s")] == ["msg1", "reply1", "msg2", "reply2"]

        # 新实例 = 模拟重启：进程内模式下读不到（要持久就必须 redis，见 test_memory_redis.py）
        store2 = MemoryStore(backend="memory")
        assert store2.load("t", "u", "s") == []


class TestCompaction:
    def test_no_compact_below_threshold(self, tmp_db):
        store = MemoryStore(backend="memory", summarize_threshold=10, keep_last=3)
        for i in range(5):
            store.append("t", "u", "s", _make_msg("user", f"msg-{i}"))
        store.compact("t", "u", "s")
        msgs = store.load("t", "u", "s")
        assert len(msgs) == 3
        assert msgs[-1].content == "msg-4"

    def test_compact_with_summarizer_inserts_summary(self, tmp_db):
        def fake_summarizer(msgs):
            return f"摘要：{len(msgs)} 条历史对话"

        store = MemoryStore(
            backend="memory", summarize_threshold=5, keep_last=2,
            summarizer=fake_summarizer,
        )
        for i in range(8):
            role = "user" if i % 2 == 0 else "assistant"
            store.append("t", "u", "s", _make_msg(role, f"msg-{i}"))
        store.compact("t", "u", "s")
        msgs = store.load("t", "u", "s")
        assert len(msgs) == 3
        assert msgs[0].role == "system"
        assert "摘要" in msgs[0].content
        assert msgs[1].content == "msg-6"
        assert msgs[2].content == "msg-7"

    def test_compact_failed_summarizer_keeps_history(self, tmp_db):
        def bad_summarizer(msgs):
            raise RuntimeError("LLM 失败")

        store = MemoryStore(
            backend="memory", summarize_threshold=3, keep_last=2,
            summarizer=bad_summarizer,
        )
        for i in range(8):
            store.append("t", "u", "s", _make_msg("user", f"msg-{i}"))
        store.compact("t", "u", "s")
        msgs = store.load("t", "u", "s")
        assert len(msgs) == 2
        assert msgs[-1].content == "msg-7"


class TestTTL:
    def test_cleanup_expired_is_backend_duty(self, tmp_db):
        """进程内后端没有服务端过期能力：cleanup_expired 返回 0。

        ⚠️ 这不是缺陷而是分工——TTL 与过期清理本来就是**后端（Redis）的职责**，
           生产环境靠 Redis 的 SET+EX 自动过期，不需要应用侧扫。（过期清理的真实验证见
           tests/test_memory_redis.py::TestTTL）
        """
        store = MemoryStore(backend="memory", ttl_seconds=0.1)
        store.append("t", "u", "s1", _make_msg("user", "old"))
        import time
        time.sleep(0.2)
        assert store.cleanup_expired() == 0
        # 进程内模式下数据仍在（未被误删），过期交给 Redis 服务端
        assert len(store.load("t", "u", "s1")) == 1


class TestReflection:
    def test_save_and_load_reflection(self, tmp_db):
        store = MemoryStore(backend="memory")
        store.save_reflection("t", "u", "s", "工具选错了，应该用 query_teaching_data")
        note = store.load_reflection("t", "u", "s")
        assert note == "工具选错了，应该用 query_teaching_data"

    def test_clear_reflection(self, tmp_db):
        store = MemoryStore(backend="memory")
        store.save_reflection("t", "u", "s", "test")
        store.clear_reflection("t", "u", "s")
        assert store.load_reflection("t", "u", "s") is None

    def test_reflection_isolation_by_tenant(self, tmp_db):
        store = MemoryStore(backend="memory")
        store.save_reflection("tenant-A", "u", "s", "A 的反思")
        store.save_reflection("tenant-B", "u", "s", "B 的反思")
        assert store.load_reflection("tenant-A", "u", "s") == "A 的反思"
        assert store.load_reflection("tenant-B", "u", "s") == "B 的反思"


class TestMakeKey:
    def test_make_key_format(self):
        key = MemoryStore._make_key("tenant-A", "user1", "session1")
        assert key == "tenant=tenant-A:user=user1:session=session1"

    def test_different_inputs_yield_different_keys(self):
        keys = {
            MemoryStore._make_key("t1", "u", "s"),
            MemoryStore._make_key("t2", "u", "s"),
            MemoryStore._make_key("t", "u1", "s"),
            MemoryStore._make_key("t", "u", "s1"),
            MemoryStore._make_key("t", "u", "s2"),
        }
        assert len(keys) == 5
