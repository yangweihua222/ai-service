# -*- coding: utf-8 -*-
"""RAG 评测一键跑：检索层四指标（零 token）+ 可选生成层三指标（LLM-as-judge，花 token）。

用法（在 ai-service 目录下）：
  python scripts/eval_metrics.py                # 只跑检索层四指标（两库）
  python scripts/eval_metrics.py --generation  # 加跑生成层三指标（真实调 LLM 裁判）
  python scripts/eval_metrics.py --generation --limit 2   # 生成层只跑前 2 题（省 token）
"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # 以 ai-service 为根

from app import rag  # noqa: E402

# ⚠️ 必须在 import 之后压：observability 模块在导入时会把 ai_service logger 设回 DEBUG
logging.getLogger("ai_service").setLevel(logging.WARNING)  # 关掉评测时的 JSON 日志刷屏


def show(title: str, res: dict) -> None:
    print(f"\n=== {title} ===")
    for k, v in res.items():
        if k == "details":
            print(f"  details: {len(v)} 题（top1 未命中：{[d['query'] for d in v if not d['top1_hit']] or '无'}）")
        else:
            print(f"  {k}: {v}")


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 评测：检索层四指标 + 生成层 LLM-as-judge")
    parser.add_argument("--generation", action="store_true", help="加跑生成层三指标（调 LLM，花 token）")
    parser.add_argument("--limit", type=int, default=0, help="生成层只跑前 N 题（省 token）")
    args = parser.parse_args()

    print("—— 检索层四指标（chunk 精确匹配）——")
    show("临床库", rag.eval_rag())
    show("使用说明库", rag.eval_usage())

    if args.generation:
        print("\n—— 生成层三指标（LLM-as-judge）——")
        show("临床库", _run(rag.eval_generation("clinical", args.limit)))
        show("使用说明库", _run(rag.eval_generation("usage", args.limit)))
    else:
        print("\n（生成层未跑：加 --generation 开启，会真实调 LLM 裁判）")


def _run(coro):
    import asyncio
    return asyncio.run(coro)


if __name__ == "__main__":
    main()
