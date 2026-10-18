# -*- coding: utf-8 -*-
"""fakeredis fixture：测试用纯内存 Redis（无需本地 Redis 服务）。"""
import os

import pytest

# 默认测试用 fakeredis（无需真实服务）
os.environ.setdefault("REDIS_URL", "redis://fakeredis-test/0")
# 2026-09-11：SQLite 后端已下线，测试默认用进程内（memory），Redis 相关测试再显式切 redis
os.environ.setdefault("MEMORY_BACKEND", "memory")
os.environ.setdefault("GOVERNANCE_BACKEND", "memory")


@pytest.fixture
def fake_redis():
    """纯内存 Redis 客户端（fakeredis）。

    📌 与真实 Redis 行为一致：
      · SET/GET/HSET/HGET/HGETALL/ZADD/ZRANGEBYSCORE/PIPELINE 全支持
      · Lua 脚本 fakeredis 2.x 部分支持；我们测试会 mock 跳过
      · 自动过期（fakeredis 不真过期，靠 scan_iter 兜底清理）
    """
    import fakeredis
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, decode_responses=True)
    yield client
    client.flushall()
    client.close()