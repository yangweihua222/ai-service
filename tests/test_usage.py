# -*- coding: utf-8 -*-
import asyncio

from app import rag


def test_usage_eval_retrieval_metrics():
    """使用说明检索评测：30 题标注集（操作手册型题库，题量刻意小于临床库）。"""
    res = rag.eval_usage()
    assert res["total"] == 30
    assert res["hit_rate"] == 1.0
    assert res["mrr"] >= 0.95


def test_usage_search_returns_citations():
    out = asyncio.run(rag.usage_search("怎么提交请假", "capital-medical"))
    assert "系统使用说明" in out
    assert "[1]" in out
