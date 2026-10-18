# AI 教学助手 Agent 服务（Python / FastAPI 微服务）
# 简历口径：独立 AI 编排层（ReAct + 9 工具），平台(Laravel+Vue) 经内部 API/SSE 调用本服务。
#
# 镜像设计取舍（对应备考库技术点30「技术选型与取舍」面试话术）：
# 1) 基础镜像用 python:3.11-slim —— torch/sentence-transformers 的 wheel 兼容性最稳（3.12/3.13 部分轮子缺失）。
# 2) bge-m3 模型权重(约 2.3GB)【不进镜像】，经 docker-compose 挂载 ./models 卷 —— 镜像小、模型可复用、换版本不用重打镜像。
# 3) .env（含 API Key）【不进镜像】，经 compose env_file 注入 —— 密钥不落镜像层，可审计。
# 4) teaching.db（Text2SQL 本地教学库）挂载卷持久化 —— 首次启动自动建表灌种子，重启不丢数据。

FROM python:3.11-slim

WORKDIR /app

# 仅 pip 装 Python 依赖：psycopg2-binary 自带 libpq，无需系统级 libpq-dev / gcc
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 业务代码（app/ 含 main/agent/rag/data/mcp_server 等；scripts/ 含启动与桩后端）
COPY app ./app
COPY scripts ./scripts
COPY README.md .

EXPOSE 8000

# 与本地一致的启动入口：模块级预热 bge-m3 再起服务（Windows 规避 torch 环境块 bug 的同一入口，Linux 下无碍、幂等）
# 若不想预热/用 mock embedding：docker compose run --rm -e EMBEDDING_MODE=mock ai-service python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
CMD ["python", "scripts/run_server.py"]
