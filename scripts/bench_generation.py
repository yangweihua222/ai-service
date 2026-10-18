# -*- coding: utf-8 -*-
"""生成层 LLM-as-judge 实测：RAG 到底有没有减少幻觉？裁判换模型结论还稳吗？

实验：
  E1 三种答案来源对比：①带检索上下文(RAG) ②裸模型无上下文 ③故意给错的上下文(抗干扰)
     统一由 qwen-max 当裁判打「忠实度 Faithfulness」（断言里被上下文支撑的比例）
  E2 交叉验证：同一批答案分别让 qwen-max / deepseek-r1 打分，看两个裁判的一致性
     （面试考点：LLM-as-judge 有「裁判模型自身偏差」，不能把单一裁判的分当金标准）

跑法：OMP_NUM_THREADS=1 python scripts/bench_generation.py
输出：bench_out/generation_bench.json
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("GOVERNANCE_BACKEND", "memory")
os.environ.setdefault("MEMORY_BACKEND", "memory")

from app.llm import OpenAICompatLLM, Message  # noqa: E402
from app.rag import judge_faithfulness  # noqa: E402

GEN_MODEL = "qwen-max"
JUDGE_MODELS = ["qwen-max", "deepseek-r1"]

QUESTIONS = [
    ("糖尿病酮症酸中毒补液首选什么液体？", 0),
    ("脓毒症确诊后 1 小时内要完成哪两件事？", 1),
    ("成人心肺复苏的按压深度和频率是多少？", 2),
    ("高血压急症首小时降压幅度上限是多少？", 3),
    ("围手术期预防性抗生素什么时候给药，疗程多久？", 4),
    ("CPR 时按压通气比例是多少？", 2),
    ("脓毒症患者平均动脉压要维持在多少以上？", 1),
    ("DKA 用胰岛素的方式是什么？", 0),
    ("抗生素选药的基本原则是什么？", 4),
    ("高血压急症什么情况才需要静脉降压？", 3),
]

CONTEXT_POOL = [
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

SYS_RAG = ("你是临床实习教学助手。只能依据给定的【检索结果】回答，不得使用检索结果之外的知识；"
           "若检索结果不足以回答，就明确说『现有资料无法回答』。回答控制在 120 字内。")
SYS_BARE = "你是临床实习教学助手。用你的医学知识回答学员提问，控制在 120 字内。"


async def gen_answer(llm, q, contexts):
    if contexts is None:
        msgs = [Message(role="system", content=SYS_BARE), Message(role="user", content=q)]
    else:
        body = "\n".join(f"[{i}] {c}" for i, c in enumerate(contexts, 1))
        msgs = [Message(role="system", content=SYS_RAG),
                Message(role="user", content=f"【检索结果】\n{body}\n\n【问题】{q}")]
    r = await llm.chat(msgs, temperature=0.2)
    return (r.content or "").strip()


async def main():
    gen = OpenAICompatLLM(model=GEN_MODEL)
    rows = []

    print(f"===== E1 生成答案（模型={GEN_MODEL}）=====", flush=True)
    # ① RAG 正确上下文 ② 裸模型 ③ 错误上下文（取下一个文档的 chunk）
    for q, gold in QUESTIONS:
        variants = {
            "rag正确上下文": [CONTEXT_POOL[gold]],
            "裸模型无上下文": None,
            "错误上下文(干扰)": [CONTEXT_POOL[(gold + 2) % len(CONTEXT_POOL)]],
        }
        row = {"问题": q}
        for name, ctx in variants.items():
            a = await gen_answer(gen, q, ctx)
            row[name] = a
            await asyncio.sleep(0.2)
        rows.append(row)
        print(f"[gen] {q[:20]}… ok", flush=True)

    print("\n===== E2 忠实度评分（裁判=qwen-max）=====", flush=True)
    judge = OpenAICompatLLM(model=JUDGE_MODELS[0])
    scored = []
    for row in rows:
        d = {"问题": row["问题"]}
        for name, ctxs in [("rag正确上下文", True), ("裸模型无上下文", False), ("错误上下文(干扰)", "wrong")]:
            if ctxs is True:
                ctx = [row["rag正确上下文"]]
                ans = row["rag正确上下文"]
            elif ctxs is False:
                ctx = []
                ans = row["裸模型无上下文"]
            else:
                ctx = [row["错误上下文(干扰)"]]
                ans = row["错误上下文(干扰)"]
            f = await judge_faithfulness(judge, ans, ctx)
            d[f"{name}_忠实度"] = round(f, 3)
            d[f"{name}_字数"] = len(ans)
        scored.append(d)
        print(json.dumps(d, ensure_ascii=False), flush=True)

    avg = {}
    for name in ["rag正确上下文", "裸模型无上下文", "错误上下文(干扰)"]:
        vals = [s[f"{name}_忠实度"] for s in scored]
        lens = [s[f"{name}_字数"] for s in scored]
        avg[name] = {"平均忠实度": round(sum(vals) / len(vals), 4),
                     "平均字数": round(sum(lens) / len(lens), 1)}

    print("\n===== E3 交叉验证（换裁判 deepseek-r1，抽 4 题）=====", flush=True)
    judge2 = OpenAICompatLLM(model=JUDGE_MODELS[1])
    cross = []
    for src_row in rows[:4]:
        for name, ctx, ans_key in [("rag正确上下文", [src_row["rag正确上下文"]], "rag正确上下文"),
                                   ("裸模型无上下文", [], "裸模型无上下文")]:
            t0 = time.perf_counter()
            f = await judge_faithfulness(judge2, src_row[ans_key], ctx)
            cross.append({"问题": src_row["问题"][:20], "变体": name,
                          "deepseek_r1_忠实度": round(f, 3),
                          "耗时s": round(time.perf_counter() - t0, 1)})
            print(json.dumps(cross[-1], ensure_ascii=False), flush=True)

    # 一致性口径：两个裁判给同一份答案的打分差 <0.34（=1 条断言的粒度）即视为结论一致
    same = total = 0
    for i, c in enumerate(cross):
        v1 = scored[i // 2][f"{c['变体']}_忠实度"]
        c["qwen_max_忠实度"] = v1
        c["是否一致"] = abs(v1 - c["deepseek_r1_忠实度"]) < 0.34
        total += 1
        same += 1 if c["是否一致"] else 0

    out = {
        "生成模型": GEN_MODEL,
        "裁判模型": JUDGE_MODELS,
        "题数": len(QUESTIONS),
        "平均忠实度对比": avg,
        "逐题明细": scored,
        "交叉验证样本": cross,
        "交叉一致率": round(same / total, 4) if total else 0,
    }
    out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_out")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "generation_bench.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print("\n=== 写入 bench_out/generation_bench.json ===", flush=True)
    print(json.dumps(avg, ensure_ascii=False, indent=2), flush=True)
    print("交叉一致率:", out["交叉一致率"], flush=True)


if __name__ == "__main__":
    asyncio.run(main())
