# -*- coding: utf-8 -*-
"""ai-service 启动入口（带 bge-m3 预热）。

为什么不用 `uvicorn app.main:app` 直接起：
  Windows + torch 下，bge-m3 若在 uvicorn 事件循环/startup 阶段才首次加载会 Segfault
  （pytest、独立脚本主线程同步加载均正常）。本脚本在 uvicorn.run() 之前、模块级主线程
  同步完成模型加载（类级单例只加载一次），再启动服务——请求进来时模型已就绪。

用法（ai-service 根目录）：
  python scripts/run_server.py
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# 模块级预热：主线程同步加载（uvicorn 事件循环内加载会 Segfault，见 main.startup_warmup 注释）
from app.config import EMBEDDING_MODE

if EMBEDDING_MODE == "bge":
    from app import rag

    print("[run_server] 预热 bge-m3（约 2.2GB，首次加载 10~30s）...")
    rag.get_embedder()._load()
    print("[run_server] bge-m3 预热完成")

import uvicorn

from app.main import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
