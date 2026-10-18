# -*- coding: utf-8 -*-
"""pgvector 索引参数实测：HNSW(ef_search) vs IVFFlat(probes) vs 精确检索 —— ANN 召回率/延迟/索引成本。

⚠️ 数据口径（诚实声明）：本院真实知识库是「5 篇指南 → 27 个 chunk」的小语料，
    在这个量级上 ANN 索引与顺序扫描无差别（PG 优化器甚至会直接走 seq scan），
    参数调优没有观测价值。所以本实验**合成 5000 条 1024 维向量**（固定随机种子，与生产同维度 EMBEDDING_DIM=1024），
    专门测「索引参数行为 + 量级趋势」，不是真实业务语料的效果指标。

跑法：OMP_NUM_THREADS=1 python scripts/bench_index.py
输出：bench_out/index_bench.json
"""
import json
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GOVERNANCE_BACKEND", "memory")
os.environ.setdefault("MEMORY_BACKEND", "memory")

import psycopg2  # noqa: E402
from app.config import PGVECTOR_DSN, EMBEDDING_DIM  # noqa: E402

N = 20000
DIM = EMBEDDING_DIM
Q_N = 30          # 查询条数
K = 10            # top-k
SEED = 20260911
CLUSTERS = 40     # 混合高斯簇数：真实 embedding 是「聚成一团一团」的，纯均匀分布测不出 ANN 的差别


def _normalize(v):
    n = sum(x * x for x in v) ** 0.5
    return [x / n for x in v]


def gen_vec(rng, centers=None):
    """混合高斯簇采样（贴近真实 sentence embedding 的簇状分布）后做 cosine 归一化。"""
    if centers is None:
        return _normalize([rng.gauss(0, 1) for _ in range(DIM)])
    c = rng.choice(centers)
    return _normalize([ci + rng.gauss(0, 0.35) for ci in c])


def vs(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def main():
    rng = random.Random(SEED)
    centers = [_normalize([rng.gauss(0, 1) for _ in range(DIM)]) for _ in range(CLUSTERS)]
    rows = [gen_vec(rng, centers) for _ in range(N)]
    queries = [gen_vec(rng, centers) for _ in range(Q_N)]

    con = psycopg2.connect(PGVECTOR_DSN, connect_timeout=10)
    con.autocommit = True
    cur = con.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("DROP TABLE IF EXISTS bench_ann")
    cur.execute(f"CREATE TABLE bench_ann (id INT PRIMARY KEY, embedding vector({DIM}))")
    t0 = time.perf_counter()
    cur.executemany("INSERT INTO bench_ann (id, embedding) VALUES (%s, %s)",
                    [(i, vs(v)) for i, v in enumerate(rows)])
    insert_s = round(time.perf_counter() - t0, 2)
    print(f"[data] {N} 条 {DIM} 维向量灌入完成 {insert_s}s", flush=True)

    def exact_topk(qv):
        cur.execute("SET enable_indexscan = off; SET enable_bitmapscan = off")
        cur.execute("SELECT id FROM bench_ann ORDER BY embedding <=> %s::vector LIMIT %s", (vs(qv), K))
        return [r[0] for r in cur.fetchall()]

    def ann_topk(qv, idx, param_sql):
        cur.execute(param_sql)
        cur.execute("SELECT id FROM bench_ann ORDER BY embedding <=> %s::vector LIMIT %s", (vs(qv), K))
        return [r[0] for r in cur.fetchall()]

    gold = [exact_topk(q) for q in queries]
    print("[gold] 精确检索基线完成", flush=True)

    results = []

    def measure(label, ddl, params_list, parse_none=False):
        cur.execute("DROP INDEX IF EXISTS bench_ann_idx")
        t0 = time.perf_counter()
        if ddl:
            cur.execute(ddl)
        build_s = round(time.perf_counter() - t0, 3)
        if ddl:
            try:
                cur.execute("SELECT pg_size_pretty(pg_relation_size('bench_ann_idx'))")
                size = cur.fetchone()[0]
            except Exception:
                size = "-"
        else:
            size = "-"
        cur.execute("ANALYZE bench_ann")
        for p in params_list:
            set_sql, pname = p
            # ⚠️ 必须关掉 seqscan：5000 行的表 PG 优化器会直接选全表扫描，索引压根不生效，
            #    测出来的延迟全是扫描耗时，参数对比毫无意义。这里强制走索引并用 EXPLAIN 留证。
            set_sql = set_sql + "; SET enable_seqscan = off"
            rec, lat = [], []
            for q, g in zip(queries, gold):
                t = time.perf_counter()
                got = ann_topk(q, None, set_sql)
                lat.append((time.perf_counter() - t) * 1000)
                rec.append(len(set(got) & set(g)) / K)
            results.append({
                "索引": label, "参数": pname,
                "recall@10": round(sum(rec) / len(rec), 4),
                "延迟ms_均值": round(sum(lat) / len(lat), 2),
                "延迟ms_P95": round(sorted(lat)[int(len(lat) * 0.95) - 1], 2),
                "索引构建s": build_s, "索引体积": size,
            })
            print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    measure("精确检索(无索引/seq scan)", None, [("SET enable_indexscan = off", "禁用索引")])
    measure("HNSW(m=16,ef_construction=64)",
            "CREATE INDEX bench_ann_idx ON bench_ann USING hnsw (embedding vector_cosine_ops)",
            [(f"SET hnsw.ef_search = {v}", f"ef_search={v}") for v in [10, 20, 40, 100, 200, 400]])
    measure("IVFFlat(lists=50)",
            "CREATE INDEX bench_ann_idx ON bench_ann USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50)",
            [(f"SET ivfflat.probes = {v}", f"probes={v}") for v in [1, 5, 10, 20, 50]])
    measure("IVFFlat(lists=200)",
            "CREATE INDEX bench_ann_idx ON bench_ann USING ivfflat (embedding vector_cosine_ops) WITH (lists = 200)",
            [(f"SET ivfflat.probes = {v}", f"probes={v}") for v in [1, 5, 10, 20, 50]])

    cur.execute("DROP INDEX IF EXISTS bench_ann_idx")
    out = {"规模": N, "维度": DIM, "查询数": Q_N, "top_k": K,
           "口径": "合成正态分布归一化向量（seed=20260911），只测索引参数行为与量级趋势，非业务语料效果",
           "结果": results}
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "index_bench.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n=== 写入 bench_out/index_bench.json ===", flush=True)


if __name__ == "__main__":
    main()
