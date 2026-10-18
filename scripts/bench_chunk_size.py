# -*- coding: utf-8 -*-
"""为什么是 chunk=120？—— 长文档下的分块粒度实测（真实跑出来再定参数）

⚠️ 为什么必须有这一组实验：
    本院现有 5 篇临床语料**每篇只有约 73 字**，比 CHUNK_SIZE=120 还短，
    在短语料上跑 120/256/512 三档会得到完全一样的数字（切分逻辑压根没被触发），
    那样的"对比数据"是假的。所以这里构造**接近真实指南长度的长文档**（每篇 600-900 字、4 个段落），
    让分块粒度真正起作用。
    口径声明：语料为合成长文档（医学内容真实、结构模拟真实指南），判据为段落级；
    结论用于回答"怎么选 chunk size"，不能当作本院真实语料的效果指标。

判据：段落级 gold —— 召回的 chunk 归属于标注段落即命中（不同切分产出的 chunk 数不同，
    只能在段落级对齐，否则切得越碎天然越吃亏，不公平）。

跑法：OMP_NUM_THREADS=1 python scripts/bench_chunk_size.py
输出：bench_out/chunk_size_bench.json
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

from collections import Counter  # noqa: E402
import psycopg2  # noqa: E402
from app.rag import BM25, _rrf, _tokenize, get_embedder  # noqa: E402
from app.config import PGVECTOR_DSN  # noqa: E402

# ============ 合成长文档（4 篇 × 4 段，段=一个完整子主题）============
LONG_DOCS = [
    (
        "糖尿病酮症酸中毒诊疗规范",
        [
            "糖尿病酮症酸中毒是胰岛素绝对或相对不足引起的急性并发症，常见诱因包括感染、胰岛素中断、应激和手术，"
            "典型表现为多尿烦渴、恶心呕吐、呼吸深快伴烂苹果味呼气，血糖多超过十三点九毫摩尔每升，动脉血气提示代谢性酸中毒。",
            "诊断需同时满足高血糖、酮体阳性或升高、代谢性酸中毒三项，血酮体测定优于尿酮，"
            "静脉血气用于判断酸中毒程度，同时必须完善电解质、肾功能和尿常规检查以评估容量状态。",
            "治疗遵循补液、胰岛素、补钾三条主线，先快速建立静脉通路输注生理盐水恢复血容量，"
            "随后改用小剂量胰岛素持续静脉泵入，血糖下降至十三点九以下时改用含糖液继续胰岛素输注。",
            "补钾必须在胰岛素开始前评估血钾，尿量正常且血钾低于五点二即开始补钾，"
            "治疗中每小时复查血糖、每两到四小时复查电解质，防止低血糖与低钾导致的心律失常。",
        ],
    ),
    (
        "脓毒症与感染性休克集束化治疗",
        [
            "脓毒症定义为宿主对感染反应失调导致的危及生命的器官功能障碍，"
            "识别要点包括体温异常、心率增快、呼吸急促、意识改变以及乳酸升高，疑似病例应立即测量乳酸并留取血培养。",
            "确诊后一小时内完成集束化措施：留取血培养后立即使用广谱抗生素，"
            "三十分钟内启动晶体液复苏三十毫升每公斤，同时评估是否需要血管活性药物维持灌注。",
            "血流动力学目标是维持平均动脉压不低于六十五毫米汞柱，"
            "首选去甲肾上腺素作为一线升压药，乳酸升高者需要动态评估组织灌注并复查乳酸清除率。",
            "感染源控制是决定预后的关键环节，包括脓肿引流、坏死组织清创和导管拔除，"
            "抗感染方案应在病原学结果回报后七十二小时内完成降阶梯，避免广谱抗生素长期使用导致的二重感染。",
        ],
    ),
    (
        "成人心肺复苏操作要点",
        [
            "发现患者无反应且无正常呼吸应立即启动急救系统并取来自动体外除颤器，"
            "同时开始胸外按压，按压位置为胸骨下半段，深度五到六厘米，频率每分钟一百到一百二十次。",
            "按压与通气比例为三十比二，每次通气持续一秒并观察胸廓起伏，"
            "尽量减少按压中断，换人操作时间控制在五秒以内，以保证胸外按压分数最大化。",
            "除颤器到达后应立即分析心律，可除颤心律给予一次电击后马上恢复按压，"
            "不可除颤心律继续按压并每两分钟评估一次节律，肾上腺素每三到五分钟重复一次。",
            "复苏过程中需要寻找并处理可逆病因，包括低血容量、缺氧、酸中毒、低钾或高钾、低体温、"
            "张力性气胸、心包填塞、中毒和肺栓塞，自主循环恢复后进入目标温度管理与器官功能支持阶段。",
        ],
    ),
    (
        "高血压急症与围手术期血压管理",
        [
            "高血压急症指收缩压超过一百八十毫米汞柱或舒张压超过一百二十毫米汞柱并伴急性靶器官损害，"
            "常见表现包括胸痛、呼吸困难、意识改变、视力障碍和少尿，需与无症状的血压显著升高区分处理。",
            "静脉降压首选短效可控药物如乌拉地尔或尼卡地平，"
            "首小时平均动脉压下降幅度不超过治疗前的百分之二十五，随后二到六小时降至一百六十每百分之一百左右。",
            "降压过快会导致脑灌注不足和心肌缺血，主动脉夹层例外需在二十分钟内将收缩压降至一百二十以下，"
            "同时配合镇痛和心率控制以减少主动脉壁剪切力。",
            "围手术期血压管理要求术前评估靶器官损害程度，长期口服降压药应在手术当日继续使用，"
            "术后疼痛、缺氧和容量异常是血压波动的常见诱因，需要优先纠正可逆因素再追加降压药物。",
        ],
    ),
]

# 每段 3 个问题（同一知识点的不同问法）→ 16 段 × 3 = 48 题
PARA_QUESTIONS = [
    ["酮症酸中毒最常见的诱因有哪些", "DKA 患者呼气有什么典型气味", "什么情况容易诱发糖尿病酮症酸中毒"],
    ["酮症酸中毒的确诊需要满足哪三项", "为什么血酮测定优于尿酮", "怀疑酮症酸中毒要做哪些检查"],
    ["酮症酸中毒治疗的三条主线是什么", "胰岛素应该在什么时机开始使用", "血糖降到多少要改用含糖液"],
    ["什么时候开始补钾", "酮症酸中毒治疗中如何监测", "补钾不及时会有什么风险"],
    ["怎么识别脓毒症", "脓毒症的定义是什么", "疑似脓毒症第一步要做什么"],
    ["脓毒症确诊一小时内要完成哪些措施", "液体复苏量给多少", "什么时候需要用升压药"],
    ["脓毒症的平均动脉压目标是多少", "一线升压药首选哪个", "乳酸升高说明什么"],
    ["感染源控制包括哪些手段", "抗生素什么时候降阶梯", "长期使用广谱抗生素有什么问题"],
    ["心肺复苏按压深度和频率是多少", "按压位置在哪里", "发现人没反应第一步做什么"],
    ["按压通气的比例是多少", "怎么让按压不中断", "每次通气要持续多久"],
    ["除颤后下一步做什么", "不可除颤心律怎么处理", "肾上腺素多久重复一次"],
    ["心肺复苏要排查哪些可逆病因", "自主循环恢复后还要做什么", "低体温属于可逆病因吗"],
    ["什么是高血压急症", "高血压急症和单纯血压高怎么区分", "血压超过多少算高血压急症"],
    ["静脉降压首选什么药", "首小时血压能降多少", "降压的第一个小时目标是什么"],
    ["降压太快有什么风险", "主动脉夹层降压有什么不同", "升压оуправление例外情况怎么处理"],
    ["围手术期长期吃的降压药要不要停", "术后血压波动常见诱因有哪些", "术前为什么要评估靶器官损害"],
]
PARA_QUESTIONS[14][2] = "主动脉夹层降压的速度要求是多少"  # 修正上面手滑混入的异常字符


def build_flat():
    """返回 (docs_text, para_labels)：docs_text 每篇是 4 段拼成的长文档。"""
    docs, labels = [], []
    for title, paras in LONG_DOCS:
        docs.append("\n".join(paras))
        labels.extend(range(len(paras)))
    return docs, labels


def split_fixed(texts, labels, size, overlap):
    """定长滑窗切分：返回 [(chunk, 所属段落全局号)]，段落归属取 chunk 中心位置所在段落。"""
    out = []
    for d_i, t in enumerate(texts):
        paras = t.split("\n")
        bounds, pos = [], 0
        for p in paras:
            pos += len(p) + 1
            bounds.append(pos)
        step = max(size - overlap, 1)
        i = 0
        while i < len(t):
            center = min(i + size // 2, len(t) - 1)
            p_i = 0
            while p_i < len(bounds) - 1 and center >= bounds[p_i]:
                p_i += 1
            out.append((t[i:i + size], d_i * len(paras) + p_i))
            i += step
    return out


def split_sentence(texts, labels):
    """按句号切：自然语言边界。"""
    out = []
    for d_i, t in enumerate(texts):
        paras = t.split("\n")
        for p_i, p in enumerate(paras):
            for s in re.split(r"(?<=[。；！？])", p):
                s = s.strip()
                if len(s) >= 8:
                    out.append((s, d_i * len(paras) + p_i))
    return out


def split_recursive(texts, labels, size=120):
    """先按段落再按句号，超长才退化为定长。"""
    out = []
    for d_i, t in enumerate(texts):
        paras = t.split("\n")
        for p_i, p in enumerate(paras):
            if len(p) <= size:
                out.append((p, d_i * len(paras) + p_i))
                continue
            buf = ""
            for s in re.split(r"(?<=[。；！？])", p):
                if len(buf) + len(s) <= size:
                    buf += s
                else:
                    if buf:
                        out.append((buf, d_i * len(paras) + p_i))
                    buf = s
            if buf:
                out.append((buf, d_i * len(paras) + p_i))
    return out


CONFIGS = {
    "B0 按句切分(最碎)": lambda d, l: split_sentence(d, l),
    "B1 定长64/重叠16": lambda d, l: split_fixed(d, l, 64, 16),
    "B2 定长120/重叠30(现网)": lambda d, l: split_fixed(d, l, 120, 30),
    "B3 定长256/重叠50": lambda d, l: split_fixed(d, l, 256, 50),
    "B4 定长512/重叠100": lambda d, l: split_fixed(d, l, 512, 100),
    "B5 递归按段纽≥句(≤120)": lambda d, l: split_recursive(d, l, 120),
    "B6 按段落不切": lambda d, l: [(p, i) for i, (p) in enumerate(
        [p for t in d for p in t.split("\n")])],
}


def para_sentences(docs):
    """每个段落拆出句子列表，用于「多标签」判据与「覆盖率」计算。"""
    out = {}
    for d_i, t in enumerate(docs):
        paras = t.split("\n")
        for p_i, p in enumerate(paras):
            sents = [s.strip() for s in re.split(r"(?<=[。；])", p) if len(s.strip()) >= 8]
            out[d_i * len(paras) + p_i] = sents or [p]
    return out


def chunks_para_sets(texts, psent, threshold=0.7):
    """每个 chunk → 它覆盖的段落集合。

    ⚠️ 判据为什么这么定（踩过两轮坑才公平）：
       ① 按「chunk 中心落在哪段」打单标签 → 跨段落的粗粒度 chunk 标签是脏的，粗切分白吃亏；
       ② 要求「段落的完整句子出现在 chunk 里」→ 定长切分必然把句子切断，细切分白吃亏。
       最终口径：段落里每句话被 chunk 覆盖的字符比例 ≥70% 就认为这句话被覆盖，
                 一段里有任意一句被覆盖 → 该 chunk 归属这段（可同时归属多段）。
    """
    sets = []
    for c in texts:
        cc = Counter(c)
        s = set()
        for pid, sents in psent.items():
            for sent in sents:
                if sent in c:
                    covered = 1.0
                else:
                    sc = Counter(sent)
                    common = sum((cc & sc).values())
                    covered = common / max(len(sent), 1)
                if covered >= threshold:
                    s.add(pid)
                    break
        sets.append(s or {-1})
    return sets


def metrics(ranked, gold, ks=(1, 3, 5)):
    """ranked：每题一列 [set(段落...)]（多标签）；gold：[[段落]]。"""
    n = len(ranked)
    res = {"n": n}
    for k in ks:
        res[f"hit@{k}"] = round(sum(
            1.0 for r, g in zip(ranked, gold) if any(g[0] in s for s in r[:k])) / n, 4)
    mrr = 0.0
    for r, g in zip(ranked, gold):
        for pos, s in enumerate(r, 1):
            if g[0] in s:
                mrr += 1.0 / pos
                break
    res["mrr@10"] = round(mrr / n, 4)
    return res


def coverage(ranked_chunks, gold, psent):
    """覆盖度@3：gold 段落里的句子，有多大比例出现在召回的 top3 chunk 拼接文本里。

    📌     这是和精度成对出现的**反向指标**：切得越碎精度越高，但答案依据越容易被截断，
    喂给 LLM 的上下文残缺 → 生成质量下降。真实选型看的是这两条曲线的膝点。
    注意顺序：召回的 top3 chunk 是被打分打乱过的，不能直接字符串拼起来找句子
    （那样句子必然横跨 chunk 边界而找不到）。正确做法是取各 chunk 的字符并集
    （每个字符按出现次数的最大值合并），再看句子能被还原出多少。
    """
    vals = []
    for chs, g in zip(ranked_chunks, gold):
        merged = Counter()
        for ch in chs[:3]:
            merged = merged | Counter(ch)
        sents = psent.get(g[0], [])
        vals.append(sum(
            sum(min(merged[c], sc[c]) for c in sc) / max(len(sent), 1)
            for sc, sent in [(Counter(sent), sent) for sent in sents]
        ) / max(len(sents), 1))
    return round(sum(vals) / max(len(vals), 1), 4)


def vs(v):
    return "[" + ",".join(f"{x:.6f}" for x in v) + "]"


def main():
    docs, labels = build_flat()
    queries, golds = [], []
    flat_q = []
    for gi, qs in enumerate(PARA_QUESTIONS):
        for q in qs:
            flat_q.append((q, [gi]))
    queries = [q for q, _ in flat_q]
    golds = [g for _, g in flat_q]
    print(f"[data] {len(docs)} 篇长文档 / {len(set(labels))} 个段落 / 评测题 {len(queries)}", flush=True)

    ef = get_embedder()
    qvecs = {q: vs(ef.encode_one(q)) for q in dict.fromkeys(queries)}
    print(f"[embed] 查询向量化完成 {len(qvecs)} 条", flush=True)

    table = []
    for name, fn in CONFIGS.items():
        tenant = f"bench-long-{abs(hash(name)) % 10000}"
        chunks_with_para = fn(docs, labels)
        texts = [c for c, _ in chunks_with_para]
        para_of = [p for _, p in chunks_with_para]
        # ⚠️ 多标签判据：一个 chunk 若横跨相邻段落，允许同时归属两段（见 chunks_para_sets 注释）
        psent = para_sentences(docs)
        para_sets = chunks_para_sets(texts, psent)

        t0 = time.perf_counter()
        embs = [ef.encode_one(c) for c in texts]
        with psycopg2.connect(PGVECTOR_DSN, connect_timeout=5) as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM kb_chunks WHERE kb='long' AND tenant_id=%s", (tenant,))
                cur.executemany(
                    "INSERT INTO kb_chunks (kb,tenant_id,doc_idx,content,embedding) "
                    "VALUES ('long',%s,%s,%s,%s) ON CONFLICT DO NOTHING",
                    [(tenant, i, c, vs(embs[i])) for i, c in enumerate(texts)])
        build_s = time.perf_counter() - t0

        bm = BM25(texts)
        t0 = time.perf_counter()
        ranked_sets, ranked_texts = [], []
        for q in queries:
            with psycopg2.connect(PGVECTOR_DSN, connect_timeout=5) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT doc_idx FROM kb_chunks WHERE kb='long' AND tenant_id=%s "
                        "ORDER BY embedding <=> %s::vector LIMIT 20", (tenant, qvecs[q]))
                    v_idx = [r[0] for r in cur.fetchall()]
            b_idx = [i for s, i in sorted(bm.scores(q), key=lambda x: -x[0])[:20] if s > 0]
            fused = _rrf([v_idx, b_idx], k=60)
            cands = [(s, i, texts[i]) for i, s in fused][:9]
            qtok = set(_tokenize(q))
            cands.sort(key=lambda x: (len(qtok & set(_tokenize(x[2]))), x[0]), reverse=True)
            # 📌 多标签：每个 chunk 对应「它覆盖了哪些段落」的集合（跨段 chunk 可以同时覆盖两段）
            ranked_sets.append([para_sets[i] for _, i, _ in cands])
            ranked_texts.append([txt for _, _, txt in cands])
        search_s = time.perf_counter() - t0

        row = {"配置": name, "chunk数": len(texts),
               "平均长度": round(sum(len(c) for c in texts) / len(texts), 1),
               "灌库s": round(build_s, 1),
               "单题ms": round(search_s / len(queries) * 1000, 1)}
        row.update(metrics(ranked_sets, golds))
        row["覆盖度@3"] = coverage(ranked_texts, golds, psent)
        # 冗余度：top3 里同一段落重复出现的比例（切太碎的典型副作用：三条说的其实是一句话）
        row["top3重复率"] = round(sum(
            1 for rs in ranked_sets
            if len({p for s in rs[:3] for p in s}) < sum(len(s) for s in rs[:3])
        ) / len(ranked_sets), 4)
        table.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)

    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "chunk_size_bench.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, ensure_ascii=False, indent=2)
    print("\n=== 写入 bench_out/chunk_size_bench.json ===", flush=True)


if __name__ == "__main__":
    main()
