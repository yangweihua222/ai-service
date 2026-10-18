# -*- coding: utf-8 -*-
"""RAG 检索参数横向实测（真实跑出来的对比数据，非拍脑袋选参）。

跑法（Windows 必须 OMP_NUM_THREADS=1，否则 torch 偶发段错误）：
    OMP_NUM_THREADS=1 python scripts/bench_retrieval.py

实验分组：
  A 分块策略对比：整篇不切 / 按句 / 按逗号短句 / 递归字符 / 定长(4 档)
  B 召回链路参数：单路(向量 / BM25) vs RRF 混合；召回池 3/5/9/15/30；RRF k=10/30/60/100
判据（重要）：标注集 gold 是「文档 id」，而不同切分产生的 chunk 数不同，
    所以统一用**文档级**判据（召回的 chunk 属于 gold 文档即算命中），否则切分越碎天然越吃亏、不公平。

输出：bench_out/retrieval_bench.json
"""
import json
import math
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GOVERNANCE_BACKEND", "memory")
os.environ.setdefault("MEMORY_BACKEND", "memory")

import psycopg2  # noqa: E402

from app import rag  # noqa: E402
from app.rag import DOCUMENTS, USAGE_GUIDE, RAG_EVAL, USAGE_EVAL, BM25, _rrf, _tokenize, get_embedder  # noqa: E402
from app.config import PGVECTOR_DSN, EMBEDDING_DIM  # noqa: E402

VECTOR_TOP = 50      # 每路召回的候选池上限（足够覆盖所有 K 组合）
BM25_TOP = 50
RAG_TOP_K_DEFAULT = rag.RAG_TOP_K


# ---------------- 分块策略 ----------------
def _fixed(text, size, overlap):
    out, i, step = [], 0, max(size - overlap, 1)
    while i < len(text):
        out.append(text[i:i + size])
        i += step
    return out


def _split_by(text, pattern, min_len=8):
    parts = re.split(pattern, text)
    return [p.strip() for p in parts if p and p.strip() and len(p.strip()) >= min_len]


def strat_none(t):
    """整篇不切：每篇文档 1 个 chunk（对照基线，模拟'不分块直接 embedding'）。"""
    return [t]


def strat_sentence(t):
    """按句切分：句号/分号/叹号/问号是一句话的自然边界。"""
    return _split_by(t, r"(?<=[。；！？])")


def strat_comma(t):
    """按逗号/顿号切成短片段（粒度最细，考察'切太碎'的影响）。"""
    return _split_by(t, r"(?<=[，,、])")


def strat_recursive(t, size=120, overlap=30):
    """递归字符切分（LangChain RecursiveCharacterTextSplitter 同思路）：
    优先按段落 → 句号 → 逗号降级切；仍超过 size 才退化为定长滑动窗口。"""
    if len(t) <= size:
        return [t]
    chunks = []
    for piece in [_split_by(t, r"(?<=[。；！？])"), [t]][0]:
        if len(piece) <= size:
            chunks.append(piece)
        else:
            sub = _split_by(piece, r"(?<=[，,、])")
            if sub and all(len(s) <= size for s in sub):
                chunks.extend(sub)
            else:
                chunks.extend(_fixed(piece, size, overlap))
    return [c for c in chunks if c]


STRATEGIES = {
    "A0 整篇不切(对照)": lambda docs: [(c, i) for i, d in enumerate(docs) for c in strat_none(d)],
    "A1 按句切分": lambda docs: [(c, i) for i, d in enumerate(docs) for c in strat_sentence(d)],
    "A2 按逗号短句": lambda docs: [(c, i) for i, d in enumerate(docs) for c in strat_comma(d)],
    "A3 递归字符(≤120)": lambda docs: [(c, i) for i, d in enumerate(docs) for c in strat_recursive(d)],
    "A4 定长64/重叠16": lambda docs: [(c, i) for i, d in enumerate(docs) for c in _fixed(d, 64, 16)],
    "A5 定长120/重叠30(现网)": lambda docs: [(c, i) for i, d in enumerate(docs) for c in _fixed(d, 120, 30)],
    "A6 定长256/重叠50": lambda docs: [(c, i) for i, d in enumerate(docs) for c in _fixed(d, 256, 50)],
    "A7 定长512/重叠100": lambda docs: [(c, i) for i, d in enumerate(docs) for c in _fixed(d, 512, 100)],
}


# ---------------- 指标（文档级） ----------------
def doc_seq(ranked_idx, chunk2doc):
    """chunk 排名 → 文档排名（去重保序）。"""
    seen, out = set(), []
    for i in ranked_idx:
        d = chunk2doc[i]
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


def metrics(ranked_lists, gold_lists, ks=(1, 3, 5)):
    n = len(ranked_lists)
    res = {"n": n}
    for k in ks:
        hit = sum(1.0 for r, g in zip(ranked_lists, gold_lists) if set(g) & set(r[:k])) / n
        res[f"hit@{k}"] = round(hit, 4)
    mrr = 0.0
    for r, g in zip(ranked_lists, gold_lists):
        gs = set(g)
        for pos, d in enumerate(r, 1):
            if d in gs:
                mrr += 1.0 / pos
                break
    res["mrr@10"] = round(mrr / n, 4)
    # NDCG@3（二值相关）
    ndcg = 0.0
    for r, g in zip(ranked_lists, gold_lists):
        gs = set(g)
        dcg = sum(1.0 / math.log2(p + 2) for p, d in enumerate(r[:3]) if d in gs)
        idcg = sum(1.0 / math.log2(p + 2) for p in range(min(len(gs), 3)))
        ndcg += dcg / idcg if idcg else 0.0
    res["ndcg@3"] = round(ndcg / n, 4)
    return res


# ---------------- 数据库操作 ----------------
def rebuild(tenant, kb, chunks_with_doc):
    """清空该租户的向量，按新切分灌库返回 {i: (text, doc_id)}。"""
    ef = get_embedder()
    # ⚠️ 逐条 encode + 显式回收：本机内存紧张（8G），bge-m3 批量推理峰值会 OOM
    embs = []
    for c, _ in chunks_with_doc:
        embs.append(ef.encode_one(c))
    import gc
    gc.collect()
    with psycopg2.connect(PGVECTOR_DSN, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM kb_chunks WHERE kb=%s AND tenant_id=%s", (kb, tenant))
            sql = ("INSERT INTO kb_chunks (kb, tenant_id, doc_idx, content, embedding) "
                   "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (kb, tenant_id, doc_idx) DO NOTHING")
            rows = [(kb, tenant, i, c, "[" + ",".join(f"{x:.6f}" for x in embs[i]) + "]")
                    for i, (c, _) in enumerate(chunks_with_doc)]
            cur.executemany(sql, rows)
    return [c for c, _ in chunks_with_doc], [d for _, d in chunks_with_doc]


def vector_search(tenant, kb, qvec_str, n):
    sql = ("SELECT doc_idx, 1-(embedding <=> %s::vector) FROM kb_chunks "
           "WHERE kb=%s AND tenant_id=%s ORDER BY embedding <=> %s::vector LIMIT %s")
    with psycopg2.connect(PGVECTOR_DSN, connect_timeout=5) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, (qvec_str, kb, tenant, qvec_str, n))
            return [(int(i), float(s)) for i, s in cur.fetchall()]


def vec_str(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


# ---------------- 主流程 ----------------
def run_dataset(name, docs, eval_set, kb):
    ef = get_embedder()
    queries = [q for q, _ in eval_set]
    golds = [g for _, g in eval_set]
    print(f"\n===== {name}（{len(queries)} 题 / {len(docs)} 篇）=====", flush=True)
    qvecs = {q: vec_str(ef.encode_one(q)) for q in dict.fromkeys(queries)}
    print(f"[embed] query 向量化完成 {len(qvecs)} 条", flush=True)

    table = []
    cache = {}
    for strat, fn in STRATEGIES.items():
        tenant = f"bench-{kb}-{strat.split()[0]}"
        t0 = time.perf_counter()
        chunks, chunk2doc = rebuild(tenant, kb, fn(docs))
        build_s = time.perf_counter() - t0

        bm = BM25(chunks)
        t0 = time.perf_counter()
        vec_per_q, bm_per_q = [], []
        for q in queries:
            vec_per_q.append(vector_search(tenant, kb, qvecs[q], min(VECTOR_TOP, len(chunks))))
            bm_per_q.append(sorted(bm.scores(q), key=lambda x: -x[0])[:BM25_TOP])
        search_s = time.perf_counter() - t0
        cache[strat] = (chunks, chunk2doc, vec_per_q, bm_per_q)

        avg_len = round(sum(len(c) for c in chunks) / len(chunks), 1)
        row = {"配置": strat, "chunk数": len(chunks), "平均长度": avg_len,
               "灌库+向量化s": round(build_s, 1), "150题检索s": round(search_s, 1),
               "单题ms": round(search_s / len(queries) * 1000, 1)}
        row.update(_eval_chain_full(chunks, chunk2doc, vec_per_q, bm_per_q,
                                    queries, golds, pool=9, rrf_k=60, mode="hybrid"))
        table.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    return table, cache


def _fuse(chunks, vec_per_q, bm_per_q, pool, rrf_k, mode="hybrid"):
    """按 (pool, rrf_k, mode) 生成每题的候选排序 [(idx, score), ...]。"""
    out = []
    for vec, bms in zip(vec_per_q, bm_per_q):
        v_idx = [i for i, _ in vec][:pool]
        b_idx = [i for s, i in bms if s > 0][:pool]
        if mode == "vector":
            fused = [(i, 1.0 / (rrf_k + r + 1)) for r, i in enumerate(v_idx)]
        elif mode == "bm25":
            fused = [(i, 1.0 / (rrf_k + r + 1)) for r, i in enumerate(b_idx)]
        else:
            fused = _rrf([v_idx, b_idx], k=rrf_k)
        out.append([(i, s) for i, s in fused])
    return out


def _rerank_like(items, chunks):
    """复刻生产 rerank：按「查询关键词命中数」优先、再用融合分兜底排序。"""
    def hit(c):
        return len(RAG_TOP_K_DEFAULT and set(_tokenize(c)) & set(_tokenize(_LAST_Q[0])))
    return sorted(items, key=lambda x: (hit(x[2]), x[0]), reverse=True)


_LAST_Q = [""]


def main():
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_out")
    os.makedirs(out_dir, exist_ok=True)
    tab, cache = run_dataset("临床知识库 RAG_EVAL", DOCUMENTS, RAG_EVAL, "clinical")

    # ---- B 组：链路参数（在现网配置 A5 上，复用已算好的召回缓存）----
    print("\n===== B 组：召回链路参数（基线=定长120/30）=====", flush=True)
    chunks, chunk2doc, vec_per_q, bm_per_q = cache["A5 定长120/重叠30(现网)"]
    golds = [g for _, g in RAG_EVAL]
    queries = [q for q, _ in RAG_EVAL]
    btable = []
    for mode, mname in [("vector", "仅向量召回"), ("bm25", "仅BM25召回"), ("hybrid", "RRF混合(现网)")]:
        for pool in [3, 5, 9, 15, 30]:
            for k in [60]:
                row = {"配置": f"{mname}｜池{pool}｜RRF k={k}"}
                row.update(_eval_chain_full(chunks, chunk2doc, vec_per_q, bm_per_q, queries, golds, pool, k, mode))
                btable.append(row)
                print(json.dumps(row, ensure_ascii=False), flush=True)
    for k in [10, 30, 60, 100]:
        row = {"配置": f"RRF混合｜池9｜RRF k={k}"}
        row.update(_eval_chain_full(chunks, chunk2doc, vec_per_q, bm_per_q, queries, golds, 9, k, "hybrid"))
        btable.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    # 关掉 rerank 的对照
    row = {"配置": "RRF混合｜池9｜RRF k=60｜关闭Rerank"}
    row.update(_eval_chain_full(chunks, chunk2doc, vec_per_q, bm_per_q, queries, golds, 9, 60, "hybrid", rerank=False))
    btable.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

    with open(os.path.join(out_dir, "retrieval_bench.json"), "w", encoding="utf-8") as f:
        json.dump({"A_分块策略": tab, "B_链路参数": btable}, f, ensure_ascii=False, indent=2)
    print("\n=== 写入 bench_out/retrieval_bench.json ===", flush=True)


def _eval_chain_full(chunks, chunk2doc, vec_per_q, bm_per_q, queries, golds, pool, rrf_k, mode, top_k=3, rerank=True):
    global _LAST_Q
    cands = _fuse(chunks, vec_per_q, bm_per_q, pool, rrf_k, mode)
    ranked = []
    for q, cands_q in zip(queries, cands):
        items = [(s, i, chunks[i]) for i, s in cands_q][:top_k * 3]
        if rerank:
            _LAST_Q[0] = q
            items = _rerank_like(items, chunks)
        ranked.append(doc_seq([i for _, i, _ in items], chunk2doc))
    return metrics(ranked, golds)


if __name__ == "__main__":
    main()
