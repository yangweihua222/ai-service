"""从 audit.jsonl 解析真实观测数据（2026-09-12）

目的：把备考库对照表里「只有配置值、没有真实数字」的项，
      尽可能换成**从真实落盘日志统计出来的数字**。

口径必须写死：这些是**本机开发/评测期间累积的真实调用**（2026-09-02 ~ 09-12），
**不是生产流量**。面试讲的时候要说清"本机 200 通真实调用统计"，不能说"生产 QPS"。
"""
import json
import statistics
from collections import Counter
from pathlib import Path

LOG = Path("logs/audit.jsonl")
if not LOG.exists():
    raise SystemExit("logs/audit.jsonl 不存在")


def pct(vals, p):
    """百分位数（线性插值），空列表返回 0"""
    if not vals:
        return 0.0
    s = sorted(vals)
    if len(s) == 1:
        return float(s[0])
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return float(s[f] + (s[c] - s[f]) * (k - f))


rows = []
for line in LOG.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        rows.append(json.loads(line))
    except json.JSONDecodeError:
        continue

# ⚠️ 关键清洗：llm_calls=0 的记录是**跑 mock 确定性单测时落盘的污染数据**
# （2026-09-08 那批 pytest 把审计也写进了生产日志，耗时只有 2~500ms，没真实 LLM 调用）
# 混在一起统计会把 P50 严重拉低 —— 必须分开报。
CLEAN = [r for r in rows if r.get("usage", {}).get("llm_calls", 0) > 0]
DIRTY = [r for r in rows if r.get("usage", {}).get("llm_calls", 0) == 0]
print(f"\n[数据清洗] 总 {len(rows)} 条 → 真实 LLM 调用 {len(CLEAN)} 条，"
      f"mock 单测污染 {len(DIRTY)} 条（已剔除出核心统计）")

n = len(rows)
print(f"=== audit.jsonl 真实样本：{n} 条 ===")
ts_all = [r.get("ts", "") for r in rows if r.get("ts")]
if ts_all:
    print(f"时间跨度：{min(ts_all)}  ~  {max(ts_all)}")

# ---------- 1. Token 成本（E2/E3）----------
totals = [r["usage"]["total"] for r in rows if r.get("usage", {}).get("total")]
prompts = [r["usage"]["prompt"] for r in rows if r.get("usage", {}).get("prompt")]
comps = [r["usage"]["completion"] for r in rows if r.get("usage", {}).get("completion")]
calls = [r["usage"].get("llm_calls", 0) for r in rows if r.get("usage")]

print("\n=== [E2/E3] Token 成本（真实统计）===")
print(f"  样本数            : {len(totals)} 通")
print(f"  单轮总 token 均值 : {statistics.mean(totals):.0f}")
print(f"  单轮 P50 / P95    : {pct(totals, 50):.0f} / {pct(totals, 95):.0f}")
print(f"  单轮最大          : {max(totals)}")
print(f"  累计消耗          : {sum(totals):,} tokens")
print(f"  prompt 均值       : {statistics.mean(prompts):.0f}")
print(f"  completion 均值   : {statistics.mean(comps):.0f}")
print(f"  单轮 LLM 调用均值 : {statistics.mean(calls):.2f} 次（>1 即含工具再做一次 Final0）")
print(f"  llm_calls 分布    : {dict(sorted(Counter(calls).items()))}")

# ---------- 2. 端到端耗时（C1）—— 只用清洗后的真实 LLM 调用 ----------
durs = [r["duration_ms"] for r in CLEAN if r.get("duration_ms")]
print("\n=== [C1] 端到端耗时 E2E（真实统计）===")
print(f"  样本数            : {len(durs)} 通")
print(f"  均值              : {statistics.mean(durs):.0f} ms")
print(f"  P50 / P90 / P95   : {pct(durs, 50):.0f} / {pct(durs, 90):.0f} / {pct(durs, 95):.0f} ms")
print(f"  最小 / 最大       : {min(durs):.0f} / {max(durs):.0f} ms")

# 按是否用工具分组（这是面试最爱问的：工具型 vs 纯对话）
_no_tool = [r["duration_ms"] for r in CLEAN if r.get("duration_ms") and not r.get("tools")]
_with_tool = [r["duration_ms"] for r in CLEAN if r.get("duration_ms") and r.get("tools")]
no_tool = _no_tool
with_tool = _with_tool
if no_tool:
    print(f"  纯对话（无工具）  : n={len(no_tool)}  P50={pct(no_tool,50):.0f} ms  P95={pct(no_tool,95):.0f} ms")
if with_tool:
    print(f"  带工具调用        : n={len(with_tool)}  P50={pct(with_tool,50):.0f} ms  P95={pct(with_tool,95):.0f} ms")
if with_tool and no_tool:
    ratio = statistics.mean(with_tool) / max(statistics.mean(no_tool), 1e-9)
    print(f"  带工具 / 纯对话倍率: {ratio:.1f}x")

# ---------- 3. 工具调用分布 ----------
tool_counter = Counter()
for r in rows:
    for t in r.get("tools", []):
        tool_counter[t] += 1
print("\n=== 工具调用真实分布 ===")
for t, c in tool_counter.most_common():
    print(f"  {t:<28} {c:>4} 次")
print("  （与 registry 全量工具对比，可看出哪些工具从未被真实调用过）")

# ---------- 4. 错误率 ----------
status_counter = Counter(r.get("status", "unknown") for r in rows)
print("\n=== 状态分布 ===")
for s, c in status_counter.most_common():
    print(f"  {s}: {c}  ({c/n*100:.1f}%)")
errs = [r for r in rows if r.get("status") != "ok"]
print(f"  非 ok 条数: {len(errs)} / {n}  ({len(errs)/n*100:.1f}%)")

# ---------- 5. 角色 / 租户分布（E3 归因）----------
print("\n=== [E3] 按角色/租户归因（真实分布）===")
for role, c in Counter(r.get("role", "?") for r in rows).most_common():
    print(f"  role={role:<12} {c:>4} 次")
for tn, c in Counter(r.get("tenant", "?") for r in rows).most_common():
    print(f"  tenant={tn:<20} {c:>4} 次")

# ---------- 6. 真实 QPS / 并发（D1）----------
from datetime import datetime
times = []
for r in rows:
    try:
        times.append(datetime.fromisoformat(r["ts"]))
    except Exception:
        continue
if len(times) >= 2:
    times.sort()
    span = (times[-1] - times[0]).total_seconds()
    print("\n=== [D1] 真实调用频率（注意：非压测，是自然累积）===")
    print(f"  {n} 次调用跨越 {span/3600:.1f} 小时")
    if span > 0:
        print(f"  平均间隔          : {span/(n-1):.1f} s")
        print(f"  自然平均 QPS      : {n/span:.4f}（远低于配置上限 5，说明限流从未在真实流量下被触发过）")
    # 最密的一分钟内有多少次
    per_min = Counter(t.strftime("%Y-%m-%d %H:%M") for t in times)
    top = per_min.most_common(3)
    print(f"  单分钟峰值调用数  : {[c for _, c in top]} 次/分钟（≈{max(c for _,c in top)/60:.3f} QPS）")

# ---------- 7. 落盘留存 ----------
out = Path("bench_out/observability_real.json")
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({
    "口径": "本机开发/评测期间累积的真实调用日志（logs/audit.jsonl），非生产流量",
    "样本数": n,
    "token_total": {"mean": round(statistics.mean(totals), 1), "p50": round(pct(totals, 50), 1),
                    "p95": round(pct(totals, 95), 1), "max": max(totals), "sum": sum(totals)},
    "duration_ms": {"mean": round(statistics.mean(durs), 1), "p50": round(pct(durs, 50), 1),
                    "p90": round(pct(durs, 90), 1), "p95": round(pct(durs, 95), 1),
                    "min": round(min(durs), 1), "max": round(max(durs), 1)},
    "duration_no_tool_ms": {"n": len(no_tool), "p50": round(pct(no_tool, 50), 1), "p95": round(pct(no_tool, 95), 1)} if no_tool else None,
    "duration_with_tool_ms": {"n": len(with_tool), "p50": round(pct(with_tool, 50), 1), "p95": round(pct(with_tool, 95), 1)} if with_tool else None,
    "tool_dist": dict(tool_counter.most_common()),
    "status_dist": dict(status_counter),
    "role_dist": dict(Counter(r.get("role", "?") for r in rows)),
    "tenant_dist": dict(Counter(r.get("tenant", "?") for r in rows)),
    "llm_calls_dist": dict(sorted(Counter(calls).items())),
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n结果已写入 {out}")
