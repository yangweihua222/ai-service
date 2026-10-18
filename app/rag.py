# -*- coding: utf-8 -*-
"""RAG 检索模块：pgvector 向量库（PostgreSQL 扩展）+ bge-m3 正式 Embedding + 重排 + 引用溯源 + 评测集。

真实落地：向量存储用 pgvector（PostgreSQL 的向量扩展，HNSW ANN 检索），Embedding 用 **bge-m3 本地正式模型**
（sentence-transformers 加载，1024 维，L2 归一化；开发期可用 mock 本地哈希兜底，见 get_embedder）；
检索链路为 向量召回(pgvector) → BM25 关键词召回 → RRF 融合 → Rerank 精排 → 带引用回答。
对应简历「AI 教学助手 · 临床知识库(RAG)」与备考库第七章。

📌 面试考点（读代码前先看这里，对应《技术点深挖备考库》）：
  · 技术点10 文档分块 chunk + 重叠           → 见 chunk_text()
  · 技术点11 Embedding 向量化                → 见 BgeEmbedder（本地 bge-m3 正式模型 1024 维；mock 哈希为开发兜底）
  · 技术点12 向量检索                         → 见 PgVectorRetriever.retrieve()（pgvector 持久化 + HNSW ANN）
  · 技术点12-补 混合检索(BM25+RRF)            → 见 PgVectorRetriever.hybrid_retrieve()
  · 技术点12-⑥ 三种向量库选型（开发/生产统一 pgvector，Milvus 大规模才上）→ 见 PgVectorRetriever 类
  · 技术点13 Rerank 重排                     → 见 PgVectorRetriever.rerank()
  · 技术点14 引用溯源（带 [1][2]，防幻觉）    → 见 rag_search()/usage_search()
  · 技术点15 RAG 边界红线：RAG 只用于临床知识库(非结构化)；学员档案走 Text2SQL，绝不混用
  · 实战① 怎么验证 RAG 检索正确？(离线召回评测+引用校验+BadCase) → 见 eval_rag()/eval_usage()
    检索层四指标(Hit@K/Recall@K/Precision@K/MRR, chunk 精确匹配) + 生成层 LLM-as-judge(忠实度/切题度/上下文精度) → retrieval_metrics()/eval_generation()
  · 实战② RAG 检索不好怎么优化？(调 chunk / 混合检索 / Rerank / 查询改写) → 见各函数
  · 使用说明 RAG（与临床知识库并列的双知识库之一）：实习系统操作 FAQ 检索 → 见 usage_search()
  · 多租户隔离：不同 tenant 用行级过滤 `WHERE tenant_id=?`（pgvector 生产标准做法，与业务数据同库同表）
"""
import os
import re
import json
import math
import hashlib
from typing import List, Tuple

import numpy as np
import psycopg2

from app import observability as ob
from app.config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    RAG_TOP_K,
    EMBEDDING_MODE,
    EMBEDDING_MODEL,
    EMBEDDING_MODEL_PATH,
    EMBEDDING_DIM,
    PGVECTOR_DSN,
)
from app.llm import Message

# ============ 示例临床语料（教材/指南/病例模板等非结构化文档）============
DOCUMENTS = [
    "糖尿病酮症酸中毒（DKA）诊疗要点：立即建立静脉通路补液恢复血容量，首选生理盐水；"
    "胰岛素采用小剂量持续静脉输注，纠正高血糖同时密切监测血钾，防止低钾。",
    "脓毒症集束化治疗（Sepsis Bundle）：确诊 1 小时内留取血培养并使用广谱抗生素，"
    "维持平均动脉压不低于 65mmHg，乳酸升高者评估组织灌注。",
    "成人心肺复苏（CPR）要点：胸外按压深度 5-6 厘米，频率 100-120 次/分，"
    "按压与通气比例 30:2；尽早使用自动体外除颤器（AED）处理可除颤心律。",
    "高血压急症处理：收缩压超过 180mmHg 且伴靶器官损害时需静脉降压，"
    "首小时降幅不宜超过 25%，避免重要脏器灌注不足。",
    "抗生素合理使用原则：依据病原学证据与目标性选药，能窄谱不广谱；"
    "围手术期预防性用药应在切皮前 0.5-1 小时给药，疗程一般不超过 24 小时。",
]

# ============ 实习系统使用说明语料（操作 FAQ 等非结构化文档，与临床知识库并列构成双知识库）============
USAGE_GUIDE = [
    "如何提交请假：在「我的-请假申请」填写事由与起止日期后提交，系统自动生成审批流程并通知班主任/带教老师审核。",
    "如何查询值班安排：进入「轮转值班」页面，选择日期范围即可查看本周/本月科室值班表。",
    "如何查看轮转计划：在「轮转管理-我的轮转」中可查看当前科室、起止时间与轮转进度。",
    "考勤异常怎么办：若打卡异常或缺卡，可在「考勤-补卡申请」提交补卡，由带教老师审批。",
    "如何查看教学活动：带教老师可在「教学活动」模块发布与查看教学查房、病例讨论等安排。",
]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    """按字符滑窗分块，带重叠。真实场景常按句子/语义边界切，避免切断语义。
    📌 技术点10 文档分块 chunk + 重叠：重叠防一句话被切断丢语义；真实调 CHUNK_SIZE/CHUNK_OVERLAP 对比检索质量。"""
    if len(text) <= size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[\u4e00-\u9fa5]|[a-zA-Z0-9]+", text)


class LocalEmbeddingFunction:
    """本地稠密 Embedding（开发兜底）：字符/词级哈希映射到固定维度向量，归一化。
    📌 技术点11：默认已切 bge-m3 正式模型（见 BgeEmbedder）；此处仅在 EMBEDDING_MODE=mock 时使用，
       用于零依赖离线调试链路（检索质量弱于 Transformer 模型，不用于生产）。
    说明：这是'真'稠密向量（非词频稀疏），足以驱动向量召回；配合 Rerank（关键词命中）保证召回质量。
    ⚙️ 工程说明：Embedding 由我们显式算好再交给 pgvector 存与检索——pgvector 负责持久化 +
       HNSW ANN 索引，Embedder 职责仍在应用侧；换 bge/OpenAI 只改 get_embedder() 工厂。"""

    def __init__(self, dim: int = EMBEDDING_DIM):
        self.dim = dim

    def encode_one(self, text: str) -> List[float]:
        return self([text])[0]

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        out = []
        for t in texts:
            vec = np.zeros(self.dim, dtype=np.float32)
            toks = _tokenize(t)
            if not toks:
                toks = [t]
            for tok in toks:
                h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
            for i in range(len(toks) - 1):
                bg = toks[i] + toks[i + 1]
                h = int(hashlib.md5(bg.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 0.5
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            out.append(vec.tolist())
        return out


class BgeEmbedder:
    """本地 bge-m3 正式 Embedding（sentence-transformers 加载，1024 维，L2 归一化）。
    📌 技术点11 正式模型：BAAI/bge-m3 中文效果好、可本地部署（医疗数据不出域、合规好讲）；
       懒加载——首次调用才加载模型（约 2.2GB，加载 10~30s），进程内复用，避免反复加载。
    ⚙️ 切换点说明：Embedding 统一由应用侧算好再交给 pgvector 存与检索；换模型只改 get_embedder() 与
       EMBEDDING_DIM（维度变了需重建表 + HNSW 索引 + 重灌语料），上层检索接口一行不动。
    📌 面试口径：开发期用 mock（LocalEmbeddingFunction）零依赖跑通链路；生产接 bge-m3 正式模型。"""

    def __init__(
        self,
        model_name: str = EMBEDDING_MODEL,
        model_path: str = EMBEDDING_MODEL_PATH,
        dim: int = EMBEDDING_DIM,
    ):
        self.model_name = model_name
        self.model_path = model_path
        self.dim = dim
        self._model = None

    # 类级共享模型：进程内只加载一次（2.2GB 模型反复加载会卡死业务）
    _shared_model = None

    # ---------- 模型加载（懒加载 + 进程级单例 + 本地缓存）----------
    def _load(self):
        if self._model is not None:
            return
        if BgeEmbedder._shared_model is not None:
            self._model = BgeEmbedder._shared_model
            return
        if not os.path.isdir(self.model_path) or not os.listdir(self.model_path):
            self._download_model()
        # 📌 Windows + torch 环境块 bug（本机实测复现）：os.environ 条目 ≥20 时，
        #   连 `import sentence_transformers`（触发 torch 初始化）都会 Segfault
        #   （与变量内容无关，纯条目数触发；18 个 OK / 20 个崩）。.env 经 load_dotenv
        #   注入 23 个变量即超阈值。解法：import+加载模型前临时摘除 .env 注入的变量，
        #   完成后恢复——torch 只在初始化时读一次环境块，恢复不影响已加载模型（进程内单例）。
        env_backup: Dict[str, str] = {}
        try:
            from dotenv import dotenv_values
            dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
            for k in (dotenv_values(dotenv_path) or {}):
                if k in os.environ:
                    env_backup[k] = os.environ.pop(k)
        except Exception:
            env_backup = {}  # 兜底：摘除失败则按原逻辑加载（环境变量少的环境不受影响）
        try:
            try:
                from sentence_transformers import SentenceTransformer  # 触发 torch 初始化，须在摘除后
            except ImportError:
                raise RuntimeError(
                    "未安装 sentence-transformers：pip install sentence-transformers（bge 模式必需）"
                )
            self._model = SentenceTransformer(self.model_path)
        finally:
            os.environ.update(env_backup)
        BgeEmbedder._shared_model = self._model

    def _download_model(self):
        """首次使用：从 HuggingFace 下载模型到本地缓存目录（国内走 hf-mirror 镜像）。"""
        import subprocess
        import sys

        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        print(f"[BgeEmbedder] 首次使用，下载模型 {self.model_name} -> {self.model_path}（约 2GB，耐心等待）...")
        try:
            from huggingface_hub import snapshot_download
        except ImportError:
            raise RuntimeError("未安装 huggingface_hub：pip install huggingface_hub")
        os.makedirs(self.model_path, exist_ok=True)
        snapshot_download(repo_id=self.model_name, local_dir=self.model_path)

    # ---------- 统一接口（与 LocalEmbeddingFunction 同构）----------
    def encode_one(self, text: str) -> List[float]:
        return self([text])[0]

    def __call__(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return []
        self._load()
        embs = self._model.encode(
            list(texts), normalize_embeddings=True, show_progress_bar=False
        )
        return embs.astype(float).tolist()


def get_embedder() -> "object":
    """Embedder 工厂：按 EMBEDDING_MODE 返回对应实现（bge 正式 / mock 开发兜底 / openai 预留）。
    📌 技术点30 选型与取舍：模式由环境变量切换，业务代码不感知——这就是"可切换"的工程落地。"""
    mode = EMBEDDING_MODE.lower()
    if mode == "bge":
        return BgeEmbedder()
    if mode == "mock":
        return LocalEmbeddingFunction(dim=EMBEDDING_DIM)
    if mode == "openai":
        # 预留：接入 OpenAI 兼容接口（text-embedding-3-small 等）。写入/查询须同一模型。
        raise RuntimeError(
            "EMBEDDING_MODE=openai 尚未启用：如需接入请在 rag.py 实现 OpenAIEmbedder（或改回 bge/mock）"
        )
    raise RuntimeError(f"未知 EMBEDDING_MODE={mode}，可选：bge | mock | openai")


class BM25:
    """本地 BM25 关键词召回（零依赖实现）。
    📌 为什么还要 BM25：向量召回抓'语义相近'，但对专有名词/缩写（如 DKA、CPR、30:2）不敏感；
       BM25 抓'关键词精确命中'，两者互补。合起来就是混合检索（Hybrid Search）。"""

    def __init__(self, corpus: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.corpus = corpus
        self.toks = [_tokenize(c) for c in corpus]
        self.N = len(self.toks)
        self.avgdl = (sum(len(t) for t in self.toks) / self.N) if self.N else 0.0
        self.df = {}
        for t in self.toks:
            for w in set(t):
                self.df[w] = self.df.get(w, 0) + 1

    def idf(self, w: str) -> float:
        n = self.df.get(w, 0)
        return math.log(1 + (self.N - n + 0.5) / (n + 0.5))

    def scores(self, query: str) -> List[Tuple[float, int]]:
        q = _tokenize(query)
        out = []
        for i, t in enumerate(self.toks):
            tf = {}
            for w in t:
                tf[w] = tf.get(w, 0) + 1
            dl = len(t)
            s = 0.0
            for w in q:
                f = tf.get(w)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * dl / max(self.avgdl, 1e-9))
                s += self.idf(w) * f * (self.k1 + 1) / denom
            out.append((s, i))
        return out


def _rrf(rank_lists: List[List[int]], k: int = 60) -> List[Tuple[float, int]]:
    """RRF 倒数排名融合：把多路召回的'排名'合成统一分数，无需归一化各路分值。
    📌 为什么用 RRF：向量相似度(0~1) 与 BM25 分值(无上界)量纲不同，直接加权很难调；
       RRF 只看排名，稳健且免调参，是混合检索的工业常用做法。"""
    fused = {}
    for lst in rank_lists:
        for rank, idx in enumerate(lst, 1):
            fused[idx] = fused.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(fused.items(), key=lambda x: -x[1])


def _vec_str(vec: List[float]) -> str:
    """Python list → pgvector 文本格式 '[0.1,0.2,...]'。
    ⚙️ 为什么自己拼：psycopg2 默认把 list 适配成 PG 数组 {..}，而 pgvector 要求方括号文本，必须显式转换。"""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"


class PgVectorRetriever:
    """pgvector 持久化检索器；按知识库(kb) + 租户(tenant_id) 行级过滤（`WHERE tenant_id=?`）隔离。
    检索链路：向量召回(pgvector HNSW) + BM25 关键词召回 → RRF 融合 → Rerank 精排。
    📌 技术点12-⑥ 选型：开发/生产统一 pgvector（复用 PostgreSQL，事务一致、能连 DBeaver/Navicat），
       Milvus 只在数据量到亿级 / GPU 检索 / 跨机房高可用时才上。"""

    def __init__(self, tenant_id: str = "", kb: str = "clinical", dsn: str = ""):
        self.tenant_id = tenant_id or "default"
        self.kb = kb
        self._dsn = dsn or PGVECTOR_DSN
        if not self._dsn:
            raise RuntimeError(
                "未配置 PGVECTOR_DSN。请先启动 PostgreSQL 并安装 pgvector 扩展，"
                "然后在 .env 里填 PGVECTOR_DSN（例：postgresql://postgres:密码@localhost:5432/ai_service）"
            )
        self._ef = get_embedder()
        self._corpus = None
        self._bm25 = None
        self._ensure_schema()
        self._ensure_seed()

    # ---------- 连接与表结构 ----------
    def _connect(self):
        """短连接（本地/教学量级够用）；生产可换连接池（如 psycopg_pool）。
        📌 技术点30：数据库访问全部参数化（%s），禁止 f-string 拼 SQL，防注入。"""
        return psycopg2.connect(self._dsn, connect_timeout=5)

    def _ensure_schema(self):
        """建表 + HNSW 索引（幂等）。向量列类型 vector(EMBEDDING_DIM) 由 pgvector 扩展提供。
        📌 技术点12：HNSW 索引 = 近似最近邻（ANN）检索，是'真向量库'的核心能力；建在 cosine 距离算子之上。"""
        ddl_table = f"""
            CREATE TABLE IF NOT EXISTS kb_chunks (
                id BIGSERIAL PRIMARY KEY,
                kb TEXT NOT NULL,
                tenant_id TEXT NOT NULL,
                doc_idx INT NOT NULL,
                content TEXT NOT NULL,
                embedding vector({EMBEDDING_DIM}),
                UNIQUE (kb, tenant_id, doc_idx)
            )"""
        ddl_idx = (
            "CREATE INDEX IF NOT EXISTS idx_kb_chunks_hnsw "
            "ON kb_chunks USING hnsw (embedding vector_cosine_ops)"
        )
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(ddl_table)
                cur.execute(ddl_idx)

    def _ensure_seed(self):
        """租户知识库为空时灌入种子语料（幂等，靠 UNIQUE(kb, tenant_id, doc_idx) 兜底）。
        📌 技术点12：多租户下每个租户各自灌入自己的知识库，靠 tenant_id 行级隔离，杜绝串数据。"""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM kb_chunks WHERE kb = %s AND tenant_id = %s",
                    (self.kb, self.tenant_id),
                )
                if cur.fetchone()[0] > 0:
                    return
                docs = DOCUMENTS if self.kb == "clinical" else USAGE_GUIDE
                chunks = [c for d in docs for c in chunk_text(d)]
                embs = self._ef(chunks)
                sql = (
                    "INSERT INTO kb_chunks (kb, tenant_id, doc_idx, content, embedding) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (kb, tenant_id, doc_idx) DO NOTHING"
                )
                for i, (c, e) in enumerate(zip(chunks, embs)):
                    cur.execute(sql, (self.kb, self.tenant_id, i, c, _vec_str(e)))

    # ---------- 语料 / BM25 ----------
    def corpus(self) -> List[str]:
        """按 doc_idx 顺序还原全量 chunk（BM25 需要全量语料统计 idf/avgdl）。"""
        if self._corpus is None:
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT content FROM kb_chunks WHERE kb = %s AND tenant_id = %s ORDER BY doc_idx",
                        (self.kb, self.tenant_id),
                    )
                    self._corpus = [row[0] for row in cur.fetchall()]
        return self._corpus

    def bm25(self) -> BM25:
        if self._bm25 is None:
            self._bm25 = BM25(self.corpus())
        return self._bm25

    # ---------- 检索 ----------
    def _log_items(self, span: str, items: list, limit: int = 10, preview: int = 180) -> None:
        """把某一路召回的「具体命中内容」写入结构化日志，供 /observability 面板展开查看。

        📌 面试考点（可观测性）：Prometheus 指标只能回答"召回了几条、花了多少毫秒"，
           排查 bad case（答非所问/引用错）时必须能看到"到底召回的是哪几条、分数多少"，
           所以把召回明细随 trace_id 一起落 JSONL 日志，指标管趋势、日志管明细。
           只记前 limit 条 + 每条前 preview 字，避免日志膨胀拖慢检索链路。
        """
        payload = []
        for r, it in enumerate(items[:limit], 1):
            text = str(it.get("text", ""))
            payload.append({
                "rank": r,
                "doc_idx": it.get("doc_idx"),
                "score": it.get("score"),
                "text": text[:preview] + ("…" if len(text) > preview else ""),
            })
        ob.log.info(
            span + ".detail",
            extra={
                "span": span + ".detail",
                "kb": self.kb,
                "items": payload,
                "item_total": len(items),
            },
        )

    def retrieve(self, query: str, top_k: int = RAG_TOP_K):
        """向量检索：pgvector HNSW ANN 召回 Top-K（真向量库，非内存 mock）。
        📌 技术点12 向量检索：`embedding <=> %s::vector` 是 cosine 距离算子；
           租户隔离 = `WHERE kb = ? AND tenant_id = ?` 行级过滤（生产标准做法）。"""
        q_emb = _vec_str(self._ef.encode_one(query))
        sql = (
            "SELECT doc_idx, content, 1 - (embedding <=> %s::vector) AS sim "
            "FROM kb_chunks WHERE kb = %s AND tenant_id = %s "
            "ORDER BY embedding <=> %s::vector LIMIT %s"
        )
        with ob.span("rag.vector", kb=self.kb, tenant=self.tenant_id, top_k=top_k):
            with self._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(sql, (q_emb, self.kb, self.tenant_id, q_emb, top_k))
                    rows = cur.fetchall()
        out = [(float(sim), int(idx), doc) for idx, doc, sim in rows]
        # 明细埋点：这一路到底召回了哪几条（cosine 相似度越高越相关）
        self._log_items("rag.vector", [
            {"doc_idx": i, "score": round(s, 4), "text": d} for s, i, d in out
        ])
        return out

    def hybrid_retrieve(self, query: str, top_k: int = RAG_TOP_K, pool: int = 3):
        """混合检索：向量召回 + BM25 召回 → RRF 融合 → 取 Top-K。
        📌 技术点12-补 混合检索(BM25 + RRF)：先各自召回一个更大的候选池(pool 倍)，
           再 RRF 融合取前 K，兼顾语义泛化与关键词精确；随后交给 rerank 精排。"""
        import time as _t
        n = min(top_k * pool, len(self.corpus()))
        t0 = _t.perf_counter()
        vec = self.retrieve(query, top_k=n)  # 内部已 ob.span("rag.vector")
        with ob.span("rag.bm25", kb=self.kb):
            bm = sorted(self.bm25().scores(query), key=lambda x: -x[0])[:n]
        bm_idx = [i for s, i in bm if s > 0]
        # 明细埋点：关键词这一路命中的是哪几条、BM25 得分多少（只留 score>0 的有效命中）
        self._log_items("rag.bm25", [
            {"doc_idx": i, "score": round(float(s), 4), "text": self.corpus()[i]}
            for s, i in bm if s > 0
        ])
        with ob.span("rag.rrf", vec_count=len(vec), bm25_count=len(bm_idx)):
            fused = _rrf([[i for _, i, _ in vec], bm_idx])
        doc_of = {i: d for _, i, d in vec}
        for i in bm_idx:
            doc_of.setdefault(i, self.corpus()[i])
        out = [(score, i, doc_of[i]) for i, score in fused if i in doc_of]
        # 明细埋点：RRF 融合后的最终排名（两路都命中的会被顶上来）
        self._log_items("rag.rrf", [
            {"doc_idx": i, "score": round(float(s), 6), "text": doc_of[i]}
            for i, s in fused if i in doc_of
        ])
        ob.log.info("rag.search", extra={
            "span": "rag.search",
            "kb": self.kb,
            "query": query[:80],
            "vec_count": len(vec),
            "bm25_count": len(bm_idx),
            "fused_topk": len(out),
            "final_topk": top_k,
            "duration_ms": round((_t.perf_counter() - t0) * 1000, 2),
        })
        total_s = _t.perf_counter() - t0
        ob.record_rag("total", self.kb, total_s, len(vec))
        ob.record_rag("bm25_recall", self.kb, 0.0, len(bm_idx))
        ob.record_rag("fused", self.kb, 0.0, len(out))
        return out[:top_k]

    def rerank(self, query: str, candidates) -> List[Tuple[float, int, str]]:
        """二次重排：用查询关键词命中数微调顺序，模拟 Cross-Encoder 重排器。
        📌 技术点13 Rerank 重排：粗排召回量大但不精，rerank 用更细相关性判断提精度、降幻觉。"""
        qkw = set(_tokenize(query))

        def hit(c: str) -> int:
            ckw = set(_tokenize(c))
            return len(qkw & ckw)

        return sorted(candidates, key=lambda x: (hit(x[2]), x[0]), reverse=True)


async def rag_search(query: str, tenant_id: str = "") -> str:
    """端到端 RAG 检索 + 生成（带引用）。tenant_id 用于多租户语料隔离（行级过滤 WHERE tenant_id=?）。
    📌 技术点14 引用溯源：[1][2] 标注来源，前端可点击跳转，防模型编造；
       技术点15 边界红线：RAG 只用于临床知识库(非结构化)，学员档案走 Text2SQL。"""
    retriever = PgVectorRetriever(tenant_id, "clinical")
    cands = retriever.hybrid_retrieve(query)  # 内部已记 rag.search 汇总
    with ob.span("rag.rerank", kb="clinical", top_k=len(cands)):
        reranked = retriever.rerank(query, cands)
    # 明细埋点：精排后的最终顺序（这个顺序决定 [1][2] 引用编号）
    retriever._log_items("rag.rerank", [
        {"doc_idx": i, "score": round(float(s), 6), "text": c} for s, i, c in reranked
    ])
    parts = []
    for i, (_, _, c) in enumerate(reranked, 1):
        snippet = c[:60] + ("…" if len(c) > 60 else "")
        parts.append(f"[{i}] {snippet}")
    return "【临床知识库回答】\n" + "\n".join(parts)


# ============ 系统使用说明检索（双知识库之一，与临床知识库对称的 RAG 链路）============
async def usage_search(query: str, tenant_id: str = "") -> str:
    """端到端"系统使用说明"检索（带引用）。复用 PgVectorRetriever 的 chunk/embed/retrieve/rerank 链路，
    仅替换语料为 USAGE_GUIDE 与 kb='usage'。语义红线：使用说明 ≠ 临床知识，两者是并列双库，互不混用。
    📌 使用说明 RAG：实习系统操作 FAQ 检索（如何请假/查值班/看轮转/补卡/教学活动）。"""
    retriever = PgVectorRetriever(tenant_id, "usage")
    cands = retriever.hybrid_retrieve(query)  # 内部已记 rag.search 汇总
    with ob.span("rag.rerank", kb="usage", top_k=len(cands)):
        reranked = retriever.rerank(query, cands)
    retriever._log_items("rag.rerank", [
        {"doc_idx": i, "score": round(float(s), 6), "text": c} for s, i, c in reranked
    ])
    parts = []
    for i, (_, _, c) in enumerate(reranked, 1):
        snippet = c[:60] + ("…" if len(c) > 60 else "")
        parts.append(f"[{i}] {snippet}")
    return "【系统使用说明】\n" + "\n".join(parts)


# ============ 离线标注评测集（问题 → 期望召回的 doc_idx 集合）============
# 📌 判据升级：原「Top-1 关键词包含」→「chunk 精确匹配（doc_idx）」。
#    每库 150 题：每条语料覆盖多个知识点，每知识点配 术语全称/缩写/口语化/场景化 多种问法（业界 golden set 口径）。
#    语料 5 条/库（demo 规模）——题量是真实的回归门禁资产，数字不当效果结论讲。
RAG_EVAL = [
    # ---- chunk 0：DKA（补液/生理盐水/小剂量胰岛素/血钾监测/防低钾）----
    ("糖尿病酮症酸中毒怎么处理", [0]),
    ("DKA 补液首选什么液体", [0]),
    ("酮症酸中毒时胰岛素和血钾怎么处理", [0]),
    ("DKA 的诊疗要点有哪些", [0]),
    ("糖尿病酮症酸中毒为什么要立即补液", [0]),
    ("DKA 患者首先建立什么通路", [0]),
    ("酮症酸中毒补液首选盐水吗", [0]),
    ("DKA 的胰岛素怎么用", [0]),
    ("糖尿病酮症酸中毒胰岛素剂量原则", [0]),
    ("DKA 治疗要监测哪些电解质", [0]),
    ("酮症酸中毒为什么会关注低钾", [0]),
    ("DKA 补液恢复血容量", [0]),
    ("糖尿病酮症酸中毒补液原则", [0]),
    ("小剂量胰岛素持续输注是什么方案", [0]),
    ("DKA 纠正高血糖时防什么并发症", [0]),
    ("酮症酸中毒静脉输注胰岛素要点", [0]),
    ("DKA 液体治疗首选什么", [0]),
    ("糖尿病酮症酸中毒的血钾管理", [0]),
    ("酮症酸中毒用生理盐水吗", [0]),
    ("DKA 处理的第一步是什么", [0]),
    ("糖尿病酮症酸中毒诊疗常规", [0]),
    ("DKA 建立静脉通路", [0]),
    ("酮症酸中毒治疗中监测血钾的意义", [0]),
    ("DKA 首选什么补液", [0]),
    ("糖尿病酮症酸中毒抢救要点", [0]),
    ("酮症酸中毒持续静输胰岛素", [0]),
    ("DKA 高血糖怎么纠正", [0]),
    ("糖尿病酮症酸中毒防止低钾", [0]),
    ("酮症酸中毒血容量不足怎么办", [0]),
    ("DKA 临床处理要点", [0]),
    # ---- chunk 1：脓毒症（血培养/广谱抗生素/MAP≥65/乳酸/组织灌注）----
    ("脓毒症集束化治疗要点", [1]),
    ("脓毒症确诊 1 小时内要做什么", [1]),
    ("Sepsis Bundle 包括什么", [1]),
    ("脓毒症血培养什么时候留取", [1]),
    ("脓毒症广谱抗生素使用时机", [1]),
    ("脓毒症平均动脉压要维持多少", [1]),
    ("感染性休克血压目标是多少", [1]),
    ("脓毒症患者乳酸升高说明什么", [1]),
    ("集束化治疗 1 小时内给药指什么", [1]),
    ("脓毒症组织灌注怎么评估", [1]),
    ("血培养和广谱抗生素一起做吗", [1]),
    ("脓毒症 MAP 要维持 65mmHg 吗", [1]),
    ("乳酸升高要评估什么", [1]),
    ("脓毒症确诊后一小时的措施", [1]),
    ("Sepsis Bundle 具体内容", [1]),
    ("脓毒症血压维持目标", [1]),
    ("感染性休克的集束化治疗", [1]),
    ("脓毒症首个小时 bundle", [1]),
    ("留取血培养的时机是什么", [1]),
    ("广谱抗生素什么时候用", [1]),
    ("脓毒症灌注不足怎么评估", [1]),
    ("平均动脉压不低于 65 是什么要求", [1]),
    ("脓毒症治疗乳酸监测", [1]),
    ("确诊脓毒症 60 分钟内做什么", [1]),
    ("脓毒症一小时 bundle 要点", [1]),
    ("严重脓毒症的处理", [1]),
    ("脓毒症抗生素和血培养的关系", [1]),
    ("集束化治疗中血培养的要求", [1]),
    ("脓毒症血压低怎么办", [1]),
    ("Sepsis 治疗要点", [1]),
    # ---- chunk 2：CPR（深度5-6cm/频率100-120/比例30:2/AED）----
    ("成人心肺复苏按压深度", [2]),
    ("CPR 按压频率和通气比例是多少", [2]),
    ("什么情况要尽快用 AED", [2]),
    ("成人胸外按压频率是多少", [2]),
    ("心肺复苏按压通气比", [2]),
    ("胸外按压深度几厘米", [2]),
    ("CPR 按压深度和频率要求", [2]),
    ("成人复苏通气比例 30:2", [2]),
    ("自动体外除颤器什么时候用", [2]),
    ("心跳骤停怎么抢救", [2]),
    ("成人心肺复苏要点", [2]),
    ("按压频率 100-120 次/分", [2]),
    ("可除颤心律用什么设备", [2]),
    ("CPR 胸外按压规范", [2]),
    ("心肺复苏操作要点", [2]),
    ("成人按压每分钟多少次", [2]),
    ("AED 除颤时机", [2]),
    ("胸外按压 5-6 厘米", [2]),
    ("心肺复苏时按压和通气怎么配合", [2]),
    ("成人 CPR 操作参数", [2]),
    ("复苏时尽早除颤", [2]),
    ("心跳呼吸骤停按压深度", [2]),
    ("CPR 按压每分钟频率", [2]),
    ("除颤器使用时机", [2]),
    ("成人心脏骤停抢救要点", [2]),
    ("心肺复苏技术参数", [2]),
    ("胸外按压与人工呼吸比例", [2]),
    ("复苏按压的深度频率规范", [2]),
    ("AED 用于什么心律", [2]),
    ("成人 CPR 标准操作", [2]),
    # ---- chunk 3：高血压急症（>180+靶器官损害/静脉降压/首小时≤25%）----
    ("高血压急症首小时降幅", [3]),
    ("收缩压 180 以上伴靶器官损害怎么降", [3]),
    ("高血压急症静脉降压指征", [3]),
    ("血压降太快有什么风险", [3]),
    ("高血压急症首小时血压下降多少合适", [3]),
    ("收缩压超过 180 要静脉降压吗", [3]),
    ("高血压急症降压幅度", [3]),
    ("靶器官损害伴高血压怎么处理", [3]),
    ("高血压急症处理原则", [3]),
    ("首小时降压不超过 25%", [3]),
    ("高血压急症为什么要控制降压速度", [3]),
    ("血压骤降导致灌注不足", [3]),
    ("高血压急症的静脉用药", [3]),
    ("收缩压 180mmHg 要不要静脉降压", [3]),
    ("高血压急症重要脏器灌注", [3]),
    ("恶性高血压怎么降", [3]),
    ("高血压急症降压目标", [3]),
    ("静脉降压用于什么高血压", [3]),
    ("高血压急症降幅控制", [3]),
    ("伴靶器官损害的高血压", [3]),
    ("高血压急症一小时内的处理", [3]),
    ("高血压危象怎么处理", [3]),
    ("高血压急症首小时处理要点", [3]),
    ("血压 180 以上怎么降", [3]),
    ("高血压急症降压不要太快", [3]),
    ("靶器官损害时降压原则", [3]),
    ("高血压急症诊疗要点", [3]),
    ("收缩压很高伴器官损害", [3]),
    ("高血压急症静脉降压治疗", [3]),
    ("高血压急症血压管理", [3]),
    # ---- chunk 4：抗生素（病原学/窄谱优先/切皮前0.5-1h/疗程≤24h）----
    ("围手术期预防性抗生素什么时候给", [4]),
    ("抗生素窄谱广谱怎么选", [4]),
    ("预防性用药切皮前多久给", [4]),
    ("抗生素疗程一般不超过多久", [4]),
    ("抗生素合理使用原则", [4]),
    ("依据病原学证据选药", [4]),
    ("能窄谱不广谱是什么意思", [4]),
    ("围手术期抗生素给药时机", [4]),
    ("预防性抗生素给药时间窗", [4]),
    ("抗生素目标性选药", [4]),
    ("切皮前半小时到一小时抗生素", [4]),
    ("术后抗生素用几天", [4]),
    ("围手术期预防用药疗程", [4]),
    ("抗生素选择的依据", [4]),
    ("预防性抗生素 24 小时内停", [4]),
    ("广谱抗生素什么时候才用", [4]),
    ("手术前抗生素怎么用", [4]),
    ("抗生素使用原则和疗程", [4]),
    ("病原学指导的抗生素选择", [4]),
    ("围术期抗菌药物预防", [4]),
    ("抗生素窄谱优先", [4]),
    ("切皮前给药的意义", [4]),
    ("预防性抗感染用药", [4]),
    ("抗菌药怎么合理使用", [4]),
    ("围手术期感染预防用药时机", [4]),
    ("抗生素使用时长限制", [4]),
    ("目标性抗感染治疗", [4]),
    ("手术预防用药什么时候停", [4]),
    ("抗生素选药原则", [4]),
    ("围手术期抗生素管理", [4]),
]


def retrieval_metrics(ranked: List[List[int]], gold: List[List[int]], k: int = RAG_TOP_K) -> dict:
    """检索层四指标（Hit@K / Recall@K / Precision@K / MRR，chunk 精确匹配口径）。
    📌 ranked：每题重排后的 doc_idx 排名序列；gold：每题期望命中的 doc_idx 集合。
       Hit@K=前K有相关(0/1)；Recall@K=前K召回的相关数/应召回数；Precision@K=前K相关数/K；
       MRR=第一个相关结果排名的倒数（第1名=1.0，第3名=0.33）。"""
    n = len(ranked)
    hit = recall = precision = mrr = 0.0
    for r, g in zip(ranked, gold):
        top = list(r[:k])
        gs = set(g)
        h = len(gs & set(top))
        hit += 1.0 if h else 0.0
        recall += h / len(gs) if gs else 0.0
        precision += h / max(len(top), 1)
        for pos, idx in enumerate(r, 1):
            if idx in gs:
                mrr += 1.0 / pos
                break
    if not n:
        return {"k": k, "total": 0, "hit_rate": 0.0, "recall_at_k": 0.0,
                "precision_at_k": 0.0, "mrr": 0.0}
    return {
        "k": k,
        "total": n,
        "hit_rate": round(hit / n, 4),
        "recall_at_k": round(recall / n, 4),
        "precision_at_k": round(precision / n, 4),
        "mrr": round(mrr / n, 4),
    }


def _eval_retrieval(tenant_id: str, kb: str, eval_set: list) -> dict:
    """离线召回评测通用跑法：全链路（混合召回 + 重排）→ 取排名序列 → 四指标。"""
    retriever = PgVectorRetriever(tenant_id, kb)
    ranked_all, gold_all, details = [], [], []
    for q, gold in eval_set:
        cands = retriever.hybrid_retrieve(q, top_k=RAG_TOP_K)
        reranked = retriever.rerank(q, cands)
        ranked = [i for _, i, _ in reranked]
        ranked_all.append(ranked)
        gold_all.append(gold)
        details.append({"query": q, "gold": gold, "ranked": ranked,
                        "top1_hit": gold[0] in ranked[:1]})
    res = retrieval_metrics(ranked_all, gold_all)
    res["details"] = details
    return res


def eval_rag() -> dict:
    """临床库离线评测：检索层四指标（Hit@K/Recall@K/Precision@K/MRR，chunk 精确匹配判据）。
    📌 实战① 怎么验证 RAG 检索正确：离线标注集评测 + 引用校验 + BadCase 复盘；
       判据从「关键词包含」升级为「doc_idx 精确匹配」——回归门禁口径更严。"""
    return _eval_retrieval("eval-clinical", "clinical", RAG_EVAL)


# ============ 使用说明评测集（操作手册型题库：30 题，每条 FAQ 6 种真实问法）============
# 📌 题量刻意小于临床库——使用说明的问题空间本来就小（就 5 个功能点），
#    硬凑大题量反而失真；临床库知识点密度高，才是题量大头。
USAGE_EVAL = [
    # ---- chunk 0：请假（我的-请假申请/事由起止日期/班主任带教审批）----
    ("怎么提交请假", [0]),
    ("请假提交后谁来审核", [0]),
    ("事假怎么申请", [0]),
    ("请假需要带教老师同意吗", [0]),
    ("我想请假怎么操作", [0]),
    ("请假的填写入口在哪", [0]),
    # ---- chunk 1：值班（轮转值班页面/日期范围/科室值班表）----
    ("怎么查值班安排", [1]),
    ("本周科室值班表在哪看", [1]),
    ("怎么看本月排班", [1]),
    ("我的值班在哪里看", [1]),
    ("值班表按周查还是按月查", [1]),
    ("在哪里能看到值班名单", [1]),
    # ---- chunk 2：轮转计划（轮转管理-我的轮转/当前科室/起止时间/进度）----
    ("轮转计划在哪看", [2]),
    ("当前科室和轮转进度在哪查", [2]),
    ("我现在轮转到哪一站", [2]),
    ("轮转起止时间在哪看", [2]),
    ("怎么查轮转进度", [2]),
    ("轮转管理功能在哪", [2]),
    # ---- chunk 3：补卡（考勤-补卡申请/带教老师审批）----
    ("缺卡了怎么办", [3]),
    ("补卡申请谁审批", [3]),
    ("忘记打卡了怎么办", [3]),
    ("考勤异常如何处理", [3]),
    ("补卡在哪个页面", [3]),
    ("补卡申请怎么提交", [3]),
    # ---- chunk 4：教学活动（教学活动模块/教学查房/病例讨论）----
    ("教学查房安排在哪发布", [4]),
    ("病例讨论安排在哪看", [4]),
    ("教学活动模块在哪", [4]),
    ("带教老师怎么发布教学活动", [4]),
    ("教学查房时间在哪", [4]),
    ("实习教学活动安排", [4]),
]


def eval_usage() -> dict:
    """使用说明库离线评测（与 eval_rag 对称：检索层四指标 + chunk 精确匹配判据）。"""
    return _eval_retrieval("eval-usage", "usage", USAGE_EVAL)


# ============ 生成层评测（LLM-as-judge 自研裁判；无参考答案，RAGAS 同思路）============
def _parse_json_list(text: str) -> list:
    """从裁判输出提取 JSON 数组：容忍 ```json 围栏与前后缀噪声（模型输出不保证纯净 JSON）。"""
    t = re.sub(r"```(json)?", "", text, flags=re.IGNORECASE).strip()
    s, e = t.find("["), t.rfind("]")
    if s == -1 or e <= s:
        return []
    try:
        v = json.loads(t[s:e + 1])
        return v if isinstance(v, list) else []
    except json.JSONDecodeError:
        return []


async def _judge_chat(llm, system: str, user: str) -> str:
    """裁判统一调用（temperature=0 求确定性输出；非流式一次拿全便于严格 JSON 解析）。"""
    resp = await llm.chat(
        [Message(role="system", content=system), Message(role="user", content=user)],
        temperature=0.0,
    )
    return (resp.content or "").strip()


async def judge_faithfulness(llm, answer: str, contexts: List[str]) -> float:
    """忠实度 Faithfulness（两步裁定）：①从答案提取事实断言 ②逐条判断是否被检索结果支撑。
    返回 支撑断言数/总断言数（0~1）；答案无事实断言时返回 1.0（无从判假）。"""
    claims_text = await _judge_chat(
        llm,
        "你是评测裁判。从回答中提取全部事实性断言（观点/数字/结论），"
        "输出 JSON 字符串数组；没有断言输出 []。只输出 JSON。",
        answer,
    )
    claims = [str(c) for c in _parse_json_list(claims_text) if c]
    if not claims:
        return 1.0
    verdict_text = await _judge_chat(
        llm,
        "你是评测裁判。逐条判断每个断言是否能由检索结果支持。"
        "输出 JSON 布尔数组，与断言顺序一一对应（true=有依据 / false=无依据）。只输出 JSON。",
        "断言：" + json.dumps(claims, ensure_ascii=False)
        + "\n\n检索结果：\n" + "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1)),
    )
    verdicts = _parse_json_list(verdict_text)
    if not verdicts:
        return 0.0
    supported = sum(1 for v in verdicts[:len(claims)] if v is True or v == "true" or v == 1)
    return supported / len(claims)


async def judge_answer_relevancy(llm, question: str, answer: str) -> float:
    """切题度 Answer Relevancy（单次裁定）：答案是否回答了问题。返回 0/1。"""
    text = await _judge_chat(
        llm,
        '你是评测裁判。判断回答是否切题地回答了问题（答非所问/拒绝回答=否）。'
        '输出 JSON 对象 {"relevant": true/false}。只输出 JSON。',
        f"问题：{question}\n\n回答：{answer}",
    )
    m = re.search(r'"?relevant"?\s*[:：]\s*(true|false)', text, re.IGNORECASE)
    return 1.0 if (m and m.group(1).lower() == "true") else 0.0


async def judge_context_precision(llm, question: str, contexts: List[str]) -> float:
    """上下文精度 Context Precision（RAGAS 口径）：逐片段判相关 → 相关片段排名越靠前分越高。
    CP = mean(top-r 内相关数 / r)，仅对相关片段的排名 r 取均值；无相关片段返回 0。"""
    verdict_text = await _judge_chat(
        llm,
        "你是评测裁判。逐条判断每段检索结果与问题是否相关。"
        "输出 JSON 布尔数组，与片段顺序一一对应（true=相关 / false=不相关）。只输出 JSON。",
        "问题：" + question + "\n\n检索结果：\n"
        + "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1)),
    )
    verdicts = _parse_json_list(verdict_text)
    rel_count, score = 0, 0.0
    for i, v in enumerate(verdicts[:len(contexts)], 1):
        if v is True or v == "true" or v == 1:
            rel_count += 1
            score += rel_count / i
    return score / rel_count if rel_count else 0.0


async def eval_generation(kb: str = "clinical", limit: int = 0) -> dict:
    """生成层端到端评测（LLM-as-judge）：检索 → 生成答案 → 三项裁定
    （忠实度 Faithfulness / 切题度 Answer Relevancy / 上下文精度 Context Precision）。
    📌 均为「无参考答案」指标（RAGAS 同思路）；裁判=同一 LLM（temperature=0），
       有模型偏差、非金标准——数字当回归门禁看，不当绝对效果结论。"""
    from app.llm import get_llm  # 局部导入避免模块加载顺序问题

    eval_set = RAG_EVAL if kb == "clinical" else USAGE_EVAL
    if limit:
        eval_set = eval_set[:limit]
    retriever = PgVectorRetriever("eval-clinical" if kb == "clinical" else "eval-usage", kb)
    llm = get_llm()
    faith, relevancy, c_precision, details = [], [], [], []
    for q, _gold in eval_set:
        cands = retriever.hybrid_retrieve(q, top_k=RAG_TOP_K)
        reranked = retriever.rerank(q, cands)
        contexts = [c for _, _, c in reranked]
        ctx_text = "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        gen = await llm.chat(
            [
                Message(role="system", content="你是教学系统助手。仅依据检索结果回答并标注引用编号；检索结果没有的就明确说不知道。"),
                Message(role="user", content=f"检索结果：\n{ctx_text}\n\n问题：{q}"),
            ]
        )
        answer = gen.content or ""
        f = await judge_faithfulness(llm, answer, contexts)
        r = await judge_answer_relevancy(llm, q, answer)
        cp = await judge_context_precision(llm, q, contexts)
        faith.append(f)
        relevancy.append(r)
        c_precision.append(cp)
        details.append({"query": q, "answer": answer[:120], "faithfulness": f,
                        "relevancy": r, "context_precision": cp})
    n = len(eval_set)
    return {
        "kb": kb,
        "total": n,
        "faithfulness": round(sum(faith) / n, 4) if n else 0.0,
        "answer_relevancy": round(sum(relevancy) / n, 4) if n else 0.0,
        "context_precision": round(sum(c_precision) / n, 4) if n else 0.0,
        "details": details,
    }
