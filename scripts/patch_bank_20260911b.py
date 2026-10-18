# -*- coding: utf-8 -*-
"""把 2026-09-11 的「SQLite 下线 + 多轮对话 bug」结论写进备考库 2.9 / 4.4。"""
import io
import sys

BANK = r"D:\简历\技术点深挖备考库.md"

REPLACES = [
    # --- 2.9 话术里的 backend 选项 ---
    ('> ① **接口层**`MemoryStore(backend="sqlite"|"redis")` + **抽象基类** `MemoryBackendABC`（`app/store_base.py`）—— 业务代码不感知 backend 是 SQLite 还是 Redis',
     '> ① **接口层**`MemoryStore(backend="redis"|"memory")` + **抽象基类** `MemoryBackendABC`（`app/store_base.py`）—— 业务代码不感知 backend 是 Redis 还是进程内'),

    ('> ② **SQLite 实现**（本地 demo / 单 worker 兜底）：WAL + NORMAL + 同库两张表 `session_history / session_reflection`',
     '> ② **进程内实现**（`backend="memory"`，只用于单测/CI 与降级兜底）：进程内 Dict，**刻意不做持久化**——重启丢记忆、多 worker 各存各的（这是明确取舍，见下方诚实边界）'),

    ('> **接口/实现分离是核心**——切 backend 一行 env 变量，**零代码改动**；单 worker + 100 学员试点用 SQLite 零依赖够用，多 worker 真生产直接 `MEMORY_BACKEND=redis + REDIS_URL=redis://...` 切换。"',
     '> **接口/实现分离是核心**——切 backend 一行 env 变量，**零代码改动**。2026-09-11 起**SQLite 后端已彻底下线**：只剩 `redis`（生产，多 worker 共享 + 服务端 TTL）和 `memory`（进程内，单测与降级）。理由很简单：SQLite 属于"既不像 demo 又不像生产"的中间态——多副本部署时它照样不共享，留着只会让人误以为它是可选项。"'),

    # --- 知识点表格 ---
    ('| **SQLite 后端** | WAL + NORMAL；零外部依赖；本地 demo 兜底                                       |',
     '| **进程内后端** | `backend="memory"`；零外部依赖、**不做持久化**（重启丢、多副本不共享），单测与降级用  |'),

    ('| **TTL**     | Redis EX 默认 3600s / SQLite `MEMORY_TTL_S` 默认 24h；过期服务端自动清理                |',
     '| **TTL**     | Redis `SET + EX 3600`；**过期由 Redis 服务端负责**，应用侧不扫（进程内模式无此能力）     |'),

    ('| **反思笔记表**   | `session_reflection`（SQLite）/ `ai:refl:<key>`（Redis）；跨请求复用反思笔记         |',
     '| **反思笔记**     | `ai:refl:<key>`（Redis，带 TTL）；与会话历史同源，跨请求复用反思笔记                  |'),
]

# 2.9 章节末尾追加的补记（多轮对话 bug）
ADD_29 = """
### ★ 2026-09-11 实测补记：接了真 Redis 之后才暴露的多轮对话崩溃

**生产做法**：凡是"对象 → JSON → 对象"的地方，**反序列化那一侧必须把嵌套类型还原回来**。这就要求 cyclic 有回归测试盯着——因为进程内（不落盘）时永远不会复现。

**我们现状**：2026-09-11 修复，并在 `app/llm.py::_build_payload` 加了第二层防御（按对象/字典两种情况取值）；回归测试在 `tests/test_memory_redis.py::TestToolCallsRoundTrip`（2 个用例）。

**事故经过（真实）**：对话历史落 Redis 时 `asdict(Message)` 把 `tool_calls` 序列化成 dict；读回来时老代码直接 `Message(**m)`，于是 `tool_calls` 变成一堆 dict。第二轮请求构造 Function Calling 的 payload 时执行 `tc.id` →

```
AttributeError: 'dict' object has no attribute 'id'
```

**多轮直接崩**。而单 invoke 用进程内 Dict 保存的是对象引用，永远不序列化，**所以这个 bug 在没接 Redis 之前一点都看不出来**——这正是"本地跑得好好的，上生产第二轮就炸"的典型场景。

**验证方式（不是写完就说修好了）**：把修复**故意回退**成 `Message(**m)` 重跑回归测试 → 测试确实报错；改回来 → 2 个用例通过。测试能抓住 bug 才算数。

**面试口径**：

> "接完 Redis 我抓到一个必须在生产才会炸的 bug：历史消息存进 Redis 再读回来，工具调用记录退化成了普通字典，第二轮再去取它的 id 直接崩 AttributeError，**多轮对话全挂**。这个坑在进程内模式下永远不会复现，因为没经过序列化。修法是反序列化的时候把嵌套对象还原回来，另外在构造请求那层按两种形态都取一次，做双保险。更重要的是我补了回归测试，而且验证过——把修复故意回退，测试确实报错，说明这个测试不是摆设。"

**追问预案**：

- "为什么不用 pydantic 统一序列化？" → 我们的 Message/ToolCall 是 dataclass（为了少一层依赖、闪电 tabulation 好改）；代价正是反序列化要自己兜类型。真要长期维护，换成 pydantic v2 的 model_validate 会更省心，这是重构方向之一，但目前靠这一层 + 回归测试兜住。
- "除了 tool_calls 还有别的地方会丢类型吗？" → 同一类风险点还有：反思笔记（纯字符串不受影响）、tool_call_id / name（基础类型不受影响）。真正值得盯的就是嵌套对象字段，目前只有 tool_calls 一处。
"""


def main():
    t = io.open(BANK, encoding="utf-8").read()
    miss = 0
    for a, b in REPLACES:
        if a not in t:
            print("!! MISS:", a[:60], file=sys.stderr)
            miss += 1
            continue
        t = t.replace(a, b)

    lines = t.split("\n")
    anchor = "## 2.9 记忆系统怎么设计？存哪了？（Redis 三件套 · 可插拔 backend）"
    idxs = [i for i, l in enumerate(lines) if l.strip() == anchor]
    if len(idxs) != 1:
        print(f"!! anchor 2.9 不唯一: {len(idxs)}", file=sys.stderr)
        sys.exit(1)
    i = idxs[0]
    j = i + 1
    while j < len(lines) and not lines[j].startswith("## "):
        j += 1
    while j - 1 > i and lines[j - 1].strip() == "":
        j -= 1
    lines[j:j] = [""] + ADD_29.strip("\n").split("\n")
    t = "\n".join(lines)

    io.open(BANK, "w", encoding="utf-8", newline="\n").write(t)
    print(f"[done] MISS={miss}；总行数 -> {len(t.split(chr(10)))}")


if __name__ == "__main__":
    main()
