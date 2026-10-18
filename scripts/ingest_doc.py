"""文档入库脚本：把新文档分块 → Embedding(bge-m3) → 写入 pgvector(kb_chunks)。

这是"知识库新增文档"的标准路径，与 RAG 检索链路完全一致：
    文档文本 → chunk_text(120字/30字重叠) → BgeEmbedder 编码 1024 维 → INSERT kb_chunks

用法示例：
  python scripts/ingest_doc.py --file docs/学生实习大纲.md                  # 并入临床知识库（默认）
  python scripts/ingest_doc.py --text "直接传文本，不走文件"                # 直接传文本
  python scripts/ingest_doc.py --file xxx.md --kb usage                    # 导入"使用说明"知识库
  python scripts/ingest_doc.py --file xxx.md --verify "实习出科考核怎么算分" # 导入后立即检索验证

设计要点：
  · 幂等：doc_idx 从该知识库当前最大编号 +1 开始，ON CONFLICT DO NOTHING，重复执行不产生重复数据
  · 多租户：按 tenant_id 写入，默认 capital-medical；换租户数据互不可见（行级隔离）
  · 知识域：kb 字段区分知识库（clinical=临床知识库 / usage=使用说明），新文档默认并入 clinical，
    这样助手检索临床/教学问题时能命中新内容
"""
import argparse
import os
import sys

# 保证从项目根目录运行时可 import app.*
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import psycopg2  # noqa: E402

from app.config import PGVECTOR_DSN  # noqa: E402
from app.rag import PgVectorRetriever, _vec_str, chunk_text, get_embedder  # noqa: E402


def _next_doc_idx(conn, kb: str, tenant_id: str) -> int:
    """该知识库当前最大 doc_idx + 1（新文档接着编号，避免覆盖已有 chunk）。"""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT COALESCE(MAX(doc_idx), -1) FROM kb_chunks WHERE kb = %s AND tenant_id = %s",
            (kb, tenant_id),
        )
        return int(cur.fetchone()[0]) + 1


def ingest(text: str, kb: str = "clinical", tenant_id: str = "capital-medical") -> dict:
    """入库主流程：确保表/索引 → 分块 → 编码 → 插入。返回统计信息。"""
    # 1) 确保表结构 + HNSW 索引存在（幂等）；同时触发该租户种子库（若为空）
    retriever = PgVectorRetriever(tenant_id, kb)

    # 2) 分块（与 RAG 检索链路同一套 chunk_text：120 字 / 30 字重叠）
    chunks = [c for c in chunk_text(text) if c.strip()]
    if not chunks:
        raise SystemExit("文档内容为空或全为空白，未导入任何数据。")

    # 3) 编码（bge-m3 正式模型，首次加载约 20s-2min；进程内只加载一次）
    ef = get_embedder()
    print(f"正在编码 {len(chunks)} 个 chunk（bge-m3 1024 维，首次加载模型需耐心）...")
    embs = ef(chunks)

    # 4) 写入 pgvector（doc_idx 接续编号，ON CONFLICT 幂等）
    conn = psycopg2.connect(PGVECTOR_DSN, connect_timeout=5)
    try:
        start = _next_doc_idx(conn, kb, tenant_id)
        sql = (
            "INSERT INTO kb_chunks (kb, tenant_id, doc_idx, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (kb, tenant_id, doc_idx) DO NOTHING"
        )
        inserted = 0
        with conn.cursor() as cur:
            for i, (c, e) in enumerate(zip(chunks, embs)):
                cur.execute(sql, (kb, tenant_id, start + i, c, _vec_str(e)))
                inserted += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return {"kb": kb, "tenant_id": tenant_id, "start_idx": start, "chunks": len(chunks), "inserted": inserted}


def verify(query: str, kb: str = "clinical", tenant_id: str = "capital-medical", top_k: int = 3):
    """导入后自检：用一条问题做混合检索（BM25 + 向量 + RRF），看能否命中新文档。"""
    retriever = PgVectorRetriever(tenant_id, kb)
    print(f"\n检索验证：query = 「{query}」（hybrid_retrieve, top_k={top_k}）")
    for sim, idx, content in retriever.hybrid_retrieve(query, top_k=top_k):
        print(f"  [doc_idx={idx}] sim={sim:.3f}  {content[:48]}...")


def main():
    ap = argparse.ArgumentParser(description="文档入库脚本（分块 → bge-m3 → pgvector）")
    ap.add_argument("--file", help="文档文件路径（txt/md）")
    ap.add_argument("--text", help="直接传入文档文本")
    ap.add_argument("--kb", default="clinical", help="知识库名：clinical(默认)/usage")
    ap.add_argument("--tenant", default="capital-medical", help="租户 ID（默认 capital-medical）")
    ap.add_argument("--verify", help="导入后执行的验证问题（可选）")
    args = ap.parse_args()

    if args.file and args.text:
        raise SystemExit("--file 与 --text 二选一")
    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        raise SystemExit("请提供 --file 或 --text")

    print(f"== 文档入库开始 == kb={args.kb} tenant={args.tenant} 字数={len(text)}")
    stat = ingest(text, kb=args.kb, tenant_id=args.tenant)
    print(f"入库完成：本次生成 {stat['chunks']} 个 chunk，实际插入 {stat['inserted']} 行（重复导入为 0）。")
    print(f"验证方式：pgAdmin 连 ai_service 库 → kb_chunks 表 → 过滤 kb='{stat['kb']}' AND tenant_id='{stat['tenant_id']}'")

    if args.verify:
        verify(args.verify, kb=args.kb, tenant_id=args.tenant)


if __name__ == "__main__":
    main()
