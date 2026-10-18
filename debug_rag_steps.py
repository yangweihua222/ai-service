# -*- coding: utf-8 -*-
"""RAG 检索链路分步透视：把 chunk / 向量召回 / BM25 / RRF / Rerank / 端到端 每一步的中间结果打出来，
用于验证"每一步检索对不对"。用法：python debug_rag_steps.py"""
import asyncio
from app import rag
from app.rag import PgVectorRetriever

TENANT = "debug-demo"
QUERY = "糖尿病酮症酸中毒怎么处理"

r = PgVectorRetriever(TENANT, "clinical")

print("========== 0. 文档分块 chunk_text ==========")
corpus = r.corpus()
print(f"语料分块数: {len(corpus)}  (CHUNK_SIZE=120 / OVERLAP=30)")
for i in range(min(2, len(corpus))):
    print(f"  [{i}] {corpus[i][:80]}")

print("\n========== 1. 向量召回 retrieve() (pgvector HNSW, cosine) Top-3 ==========")
vec = r.retrieve(QUERY, top_k=3)
for sim, idx, doc in vec:
    print(f"  sim={sim:.4f}  idx={idx}  :: {doc[:48]}")

print("\n========== 2. BM25 关键词召回 BM25.scores() Top-3 ==========")
bm = sorted(r.bm25().scores(QUERY), key=lambda x: -x[0])[:3]
for s, i in bm:
    print(f"  bm25={s:.4f}  idx={i}  :: {corpus[i][:48]}")

print("\n========== 3. RRF 融合 _rrf(k=60) Top-5 ==========")
fused = rag._rrf([[i for _, i, _ in vec], [i for s, i in bm if s > 0]])
for idx, score in fused[:5]:
    print(f"  rrf={score:.4f}  idx={idx}  :: {corpus[idx][:48]}")

print("\n========== 4. Rerank 精排 rerank() Top-3 (带引用) ==========")
cands = r.hybrid_retrieve(QUERY)
reranked = r.rerank(QUERY, cands)
for i, (_, _, c) in enumerate(reranked[:3], 1):
    print(f"  [{i}] {c[:55]}")

print("\n========== 5. 端到端 rag_search() ==========")
print(asyncio.run(rag.rag_search(QUERY, TENANT)))

print("\n========== 6. 离线召回评测 eval_rag() ==========")
res = rag.eval_rag()
print(f"Top-1 命中: {res['hit']}/{res['total']} = {res['rate']}")
for d in res["details"]:
    print(f"  {'OK ' if d['top1_hit'] else 'XX '} {d['query']} -> 期望含「{d['expect']}」")
