# -*- coding: utf-8 -*-
"""混合检索到底有没有用？—— 向量单路 / BM25 单路 / RRF 混合 三方对比。

跑法：OMP_NUM_THREADS=1 python scripts/bench_fusion.py
输出：bench_out/fusion_bench.json

⚠️ 为什么要在长文档语料上做：本院现有 5 篇短指南只有 5 个 chunk，
   候选池太小导致"召回谁都一样"，任何融合参数都拉不开差距（已在短题库上验证：三种模式指标几乎重合）。
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GOVERNANCE_BACKEND", "memory")
os.environ.setdefault("MEMORY_BACKEND", "memory")

import psycopg2  # noqa: E402
from app.config import PGVECTOR_DSN  # noqa: E402
from app.rag import BM25, _rrf, _tokenize, get_embedder  # noqa: E402
from scripts.bench_chunk_size import LONG_DOCS, PARA_QUESTIONS, build_flat, split_recursive, para_sentences, chunks_para_sets, metrics  # noqa: E402

TOP_K = 3
POOL = 9


def vs(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def main():
    docs, labels = build_flat()
    psent = para_sentences(docs)
    flat = [(q, [i]) for i, qs in enumerate(PARA_QUESTIONS) for q in qs]
    queries = [q for q, _ in flat]
    golds = [g for _, g in flat]

    tenant = "bench-fusion-recursive"
    chunks = split_recursive(docs, labels)
    texts = [c for c, _ in chunks]
    para_sets = chunks_para_sets(texts, psent)

    ef = get_embedder()
    embs = [ef.encode_one(c) for c in texts]
    with psycopg2.connect(PGVECTOR_DSN, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM kb_chunks WHERE kb='long' AND tenant_id=%s", (tenant,))
            cur.executemany(
                "INSERT INTO kb_chunks (kb,tenant_id,doc_idx,content,embedding) "
                "VALUES ('long',%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                [(tenant, i, c, vs(embs[i])) for i, c in enumerate(texts)])

    bm = BM25(texts)
    qvecs = {q: vs(ef.encode_one(q)) for q in dict.fromkeys(queries)}
    vec_all, bm_all = [], []
    for q in queries:
        with psycopg2.connect(PGVECTOR_DSN, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT doc_idx FROM kb_chunks WHERE kb='long' AND tenant_id=%s "
                            "ORDER BY embedding <=> %s::vector LIMIT 20", (tenant, qvecs[q]))
                vec_all.append([r[0] for r in cur.fetchall()])
        bm_all.append([i for s, i in sorted(bm.scores(q), key=lambda x: -x[0])[:20] if s > 0])

    results = []
    modes = [("vector", "仅向量召回"), ("bm25", "仅 BM25 召回"),
             ("hybrid", "RRF 混合（现网）")]
    removal = [("hybrid_norerank", "RRF 混合 · 关闭 Rerank")]
    for mode, name in modes:
        t0 = time.perf_counter()
        ranked_sets = []
        for vec, bids, q in zip(vec_all, bm_all, queries):
            v_idx, b_idx = vec[:POOL], bids[:POOL]
            if mode == "vector":
                fused = [(i, 1.0 / (60 + r + 1)) for r, i in enumerate(v_idx)]
            elif mode == "bm25":
                fused = [(i, 1.0 / (60 + r + 1)) for r, i in enumerate(b_idx)]
            else:
                fused = _rrf([v_idx, b_idx], k=60)
            cands = [(s, i, texts[i]) for i, s in fused][:TOP_K * 3]
            qtok = set(_tokenize(q))
            cands.sort(key=lambda x: (len(qtok & set(_tokenize(x[2]))), x[0]), reverse=True)  # 现网 rerank
            ranked_sets.append([para_sets[i] for _, i, _ in cands])
        dt = time.perf_counter() - t0
        row = {"配置": name, "单题ms": round(dt / len(queries) * 1000, 1)}
        row.update(metrics(ranked_sets, golds))
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    # 关闭 Rerank 的对照（混合模式）
    ranked_sets = []
    for vec, bids, q in zip(vec_all, bm_all, queries):
        fused = _rrf([vec[:POOL], bids[:POOL]], k=60)
        ranked_sets.append([para_sets[i] for i, s in fused])
    row = {"配置": removal[0][1], "单题ms": 0}
    row.update(metrics(ranked_sets, golds))
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

    # RRF k 敏感性
    for k in [10, 30, 60, 100]:
        ranked_sets = []
        for vec, bids, q in zip(vec_all, bm_all, queries):
            fused = _rrf([vec[:POOL], bids[:POOL]], k=k)
            cands = [(s, i, texts[i]) for i, s in fused][:TOP_K * 3]
            qtok = set(_tokenize(q))
            cands.sort(key=lambda x: (len(qtok & set(_tokenize(x[2]))), x[0]), reverse=True)
            ranked_sets.append([para_sets[i] for _, i, _ in cands])
        row = {"配置": f"RRF k={k}", "单题ms": 0}
        row.update(metrics(ranked_sets, golds))
        results.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "fusion_bench.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n=== 写入 bench_out/fusion_bench.json ===", flush=True)


if __name__ == "__main__":
    main()
