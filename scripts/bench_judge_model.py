# -*- coding: utf-8 -*-
"""LLM-as-judge 裁判模型选型实测：谁的 JSON 遵循最稳 + 谁最便宜最快。

用法：OMP_NUM_THREADS=1 python scripts/bench_judge_model.py
输出：每个候选模型的 合法JSON率 / 平均耗时 / 平均 completion token / 判定一致性
判据：能原样输出「只含 JSON」的字符串，且能被 json.loads 解析（=能当裁判用）
"""
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 评测脚本不依赖 Redis / PostgreSQL：把治理与记忆降到进程内
os.environ.setdefault("GOVERNANCE_BACKEND", "memory")
os.environ.setdefault("MEMORY_BACKEND", "memory")

from app.llm import OpenAICompatLLM, Message  # noqa: E402

CANDIDATES = ["qwen-max", "glm-5", "qwen-mt-flash", "deepseek-r1"]

ANSWER = (
    "糖尿病酮症酸中毒（DKA）的处理：首先建立静脉通路并补液恢复血容量，首选生理盐水；"
    "胰岛素采用小剂量持续静脉输注；同时密切监测血钾防止低钾。"
    "胸外按压深度应为 5-6 厘米。"
)
CONTEXTS = [
    "糖尿病酮症酸中毒（DKA）诊疗要点：立即建立静脉通路补液恢复血容量，首选生理盐水；"
    "胰岛素采用小剂量持续静脉输注，纠正高血糖同时密切监测血钾，防止低钾。",
    "高血压急症处理：收缩压超过 180mmHg 且伴靶器官损害时需静脉降压，首小时降幅不宜超过 25%。",
]

SYS_CLAIM = (
    "你是评测裁判。从回答中提取全部事实性断言（观点/数字/结论），"
    "输出 JSON 字符串数组；没有断言输出 []。只输出 JSON。"
)
SYS_VERDICT = (
    "你是评测裁判。逐条判断每个断言是否能由检索结果支持。"
    "输出 JSON 布尔数组，与断言顺序一一对应（true=有依据 / false=无依据）。只输出 JSON。"
)
CLAIMS = [
    "DKA 需要立即建立静脉通路补液",
    "补液首选生理盐水",
    "胰岛素采用小剂量持续静脉输注",
    "需要密切监测血钾",
    "胸外按压深度应为 5-6 厘米",
]


def try_json(text: str):
    """严格判据：直接用原串 json.loads；容忍 ```json 围栏（工程上必须容忍的常见噪声）。"""
    t = text.strip()
    t = t.replace("```json", "").replace("```", "").strip()
    try:
        v = json.loads(t)
        return True, v
    except Exception:
        return False, None


async def probe(model: str, retries: int = 3) -> dict:
    llm = OpenAICompatLLM(model=model)
    claim_ok = verdict_ok = 0
    lat, tok = [], []
    parsed_claims = None
    for _ in range(retries):
        t0 = time.perf_counter()
        try:
            r1 = llm._request(
                [Message(role="system", content=SYS_CLAIM),
                 Message(role="user", content=ANSWER)],
                temperature=0.0,
            )
        except Exception as e:
            return {"model": model, "error": str(e)[:160]}
        dt = time.perf_counter() - t0
        lat.append(dt)
        tok.append(r1.usage.get("completion", 0))
        ok, v = try_json(r1.content or "")
        claim_ok += 1 if ok and isinstance(v, list) else 0
        if ok and isinstance(v, list):
            parsed_claims = v

        t0 = time.perf_counter()
        try:
            r2 = llm._request(
                [Message(role="system", content=SYS_VERDICT),
                 Message(role="user", content="断言：" + json.dumps(CLAIMS, ensure_ascii=False)
                         + "\n\n检索结果：\n" + "\n".join(f"[{i}] {c}" for i, c in enumerate(CONTEXTS, 1)))],
                temperature=0.0,
            )
        except Exception as e:
            return {"model": model, "error": str(e)[:160]}
        dt2 = time.perf_counter() - t0
        lat.append(dt2)
        tok.append(r2.usage.get("completion", 0))
        ok2, v2 = try_json(r2.content or "")
        verdict_ok += 1 if ok2 and isinstance(v2, list) else 0
    return {
        "model": model,
        "claim_json_ok": f"{claim_ok}/{retries}",
        "verdict_json_ok": f"{verdict_ok}/{retries}",
        "avg_latency_s": round(sum(lat) / len(lat), 2),
        "avg_completion_tok": round(sum(tok) / len(tok), 1),
        "claim_count": len(parsed_claims) if parsed_claims is not None else -1,
        "verdict_sample": v2 if isinstance(v2, list) else None,
    }


async def main():
    res = []
    for m in CANDIDATES:
        print(f"[probe] {m} ...", flush=True)
        r = await probe(m)
        res.append(r)
        print(json.dumps(r, ensure_ascii=False), flush=True)
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bench_out")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "judge_model_selection.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print("\n=== 结论 ===")
    for r in res:
        if "error" in r:
            print(f"{r['model']}: 不可用 —— {r['error']}")
        else:
            print(f"{r['model']}: 断言JSON {r['claim_json_ok']} / 判定JSON {r['verdict_json_ok']} "
                  f"/ 平均 {r['avg_latency_s']}s / {r['avg_completion_tok']} tok")


if __name__ == "__main__":
    asyncio.run(main())
