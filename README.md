# AI 教学助手 Agent 服务（Python / FastAPI 微服务）

本服务是简历「项目一 · AI 教学助手（Agent 编排 + 工具调用）」的**可运行骨架**，
对应 Laravel+Vue 平台之外的独立 Python(FastAPI) AI 微服务。平台经内部 API / SSE 调用本服务。

## 特性
- **ReAct 编排**（规划 → 调用 → 观察 → 修正），可插拔 `Planner`（Mock 关键词路由 / 真实 LLM tool_calls）
- **9 个业务工具**：原 5 个（RAG 临床知识库 / 系统使用说明 RAG / Text2SQL 学员数据 / 轮转可行性检测 / 通知推送）+ **4 个真实后端工具**（提交请假 / 值班查询 / 缺勤查询 / 教学活动，调用 Laravel `/api/dify/tools/*`，小萱同源）
- **LLM 抽象层**：Mock（零依赖跑通）/ OpenAI 兼容（DeepSeek / 通义 / 智谱 复用）
- **多租户 RBAC**：租户隔离 + admin / teacher / student 三级权限；向量侧按 `tenant_id` 行级过滤（`WHERE tenant_id=?`，与业务数据同库，生产标准做法）
- **RAG 真向量库**：**pgvector**（PostgreSQL 向量扩展，`kb_chunks` 表 + HNSW ANN 索引）；**混合检索** = 向量召回 + BM25 关键词召回 → RRF 融合 → Rerank
- **正式 Embedding**：本地 **bge-m3** 模型（`sentence-transformers` 加载，1024 维，L2 归一化，医疗数据不出域；首次自动下载到 `./models/bge-m3`）
- **MCP Server**：基于官方 SDK 将 9 个工具封装为标准 MCP（`pip install mcp` 后真实可起），与 Agent 共享同一份工具定义
- **Text2SQL 真执行**：本地 SQLite 教学库（`./teaching.db`，首次自动建表灌种子数据）真跑 SQL；执行前过**安全闸门**（只读校验 + 租户强制过滤），租户值参数绑定防注入
- **评测集驱动**：RAG 4/4、使用说明 4/4、Text2SQL 5/5（**且每条真执行验证跑通**，与简历口径一致）

## 目录结构
```
ai-service/
├── app/
│   ├── main.py           # FastAPI 入口（SSE 流式 + 路由 + 评测）
│   ├── config.py         # 环境变量配置
│   ├── models.py         # 请求/响应/运行期上下文模型
│   ├── llm.py            # LLM 抽象层（Mock + OpenAI 兼容）
│   ├── agent.py          # ReAct 引擎 + Planner/Synthesizer + 防死循环 + 反思
│   ├── tools_registry.py # 工具注册表（JSON Schema + 权限）
│   ├── tools.py          # 4 个工具实现 + 注册
│   ├── rag.py            # RAG 检索 + 评测集
│   ├── data.py           # Text2SQL + 轮转分配引擎（贪心 + 局部搜索）
│   ├── memory.py         # 对话记忆（短期 + 长期压缩）
│   ├── rbac.py           # 多租户 RBAC + 鉴权依赖
│   ├── backend.py        # 真实后端客户端（调用 Laravel /api/dify/tools/*，小萱同源）
│   └── mcp_server.py     # MCP Server（可选, 需 pip install mcp）
├── scripts/
│   └── stub_backend.py   # 本地桩后端（模仿 Laravel 4 个工具接口，用于不碰生产库的验证）
├── tests/                # pytest：端到端 + 评测 + 鉴权 + 真实后端工具
├── requirements.txt
├── .env.example
└── README.md
```

## 快速开始
> ⚠️ 前置：**本地 PostgreSQL + pgvector 扩展**（RAG 真向量库依赖，见下方「pgvector 接入」）。
> Embedding 默认 bge-m3 正式模型：首次运行自动从 HuggingFace（hf-mirror 镜像）下载约 2GB 权重到 `./models/bge-m3`；
> 不想下载可用 `EMBEDDING_MODE=mock` 本地哈希兜底（仅开发调试，检索质量弱）。
> LLM 用 Mock 模式即可零成本跑通；RAG/Text2SQL 真执行需要本机 PG 可达。

```bash
cd ai-service
pip install -r requirements.txt
cp .env.example .env          # 填 PGVECTOR_DSN；LLM_MODE=mock 即可跑通
python scripts/run_server.py  # ⚠️ Windows 推荐入口：先模块级预热 bge-m3 再起服务（规避 Windows+torch 环境块 bug，见 app/rag.py _load 注释）
# 备选：python -m app.main    # 非 Windows / 不在意首次请求加载延迟时可用
```

### pgvector 接入（本地 PostgreSQL）
1. 安装 [PostgreSQL](https://www.postgresql.org/download/windows/)（建议 16+，安装时记住 postgres 密码）。
2. 启用扩展 + 建库（首次）：
   ```sql
   -- 用 pgAdmin / psql 连到 PostgreSQL
   CREATE EXTENSION IF NOT EXISTS vector;      -- pgvector 扩展（Windows 安装见下方备注）
   CREATE DATABASE ai_service;
   ```
3. `.env` 填连接串：`PGVECTOR_DSN=postgresql://postgres:你的密码@localhost:5432/ai_service`
4. 首次启动服务时自动建表 `kb_chunks`（含 HNSW 索引）并灌入种子语料，无需手动建表。

> Windows 装 pgvector：官方 [pgvector releases](https://github.com/pgvector/pgvector/releases) 下载
> `pgvector-<ver>-windows-x64.zip`，把 `lib/*.dll` 拷到 `C:\Program Files\PostgreSQL\<版本>\lib`、
> `share/extension/*` 拷到 `...\share\extension`，重启 PostgreSQL 后 `CREATE EXTENSION vector;`。

## Docker 部署（一条命令起服务 + PostgreSQL(pgvector)，免本机装 PG）
> 简历口径：AI 服务是**独立可部署的微服务**，`Dockerfile` + `docker-compose.yml` 把「AI 编排层 + 向量库」打包为两个容器，
> 平台(Laravel) 只需在 `.env` 里填 `BACKEND_BASE_URL` 反向调用。镜像设计取舍（模型不进镜像 / 密钥不进镜像）见 `Dockerfile` 顶部注释。

```bash
# 前置：本机装 Docker Desktop（Windows 开启 WSL2 后端）
cd ai-service
# 1) .env 调整：PGVECTOR_DSN 的 host 由 localhost 改为容器内服务名 pg，并补 POSTGRES_PASSWORD
#    PGVECTOR_DSN=postgresql://postgres:你的密码@pg:5432/ai_service
#    POSTGRES_PASSWORD=你的密码
# 2) 构建 + 启动（ai-service + pg 两个容器；首次自动下载 bge-m3 到 ./models，仅一次）
docker compose up -d --build
curl http://localhost:8000/health
```

- **挂载/持久化**：`./models`（bge-m3 权重，约 2.3GB，不进镜像）→ `/app/models`；命名卷 `ai_teaching_db`（Text2SQL 本地教学库）；`pgdata`（PostgreSQL 数据）。
- **不用真向量库 / 不想下模型**：`EMBEDDING_MODE=mock`（本地哈希兜底）+ 注释掉 `depends_on: pg` 即可纯 mock 跑通。
- **真 LLM**：`.env` 里 `LLM_MODE=openai` + `OPENAI_API_KEY`（通义/DeepSeek 兼容），容器内一并生效（`env_file` 注入）。

## 调用示例
```bash
curl http://localhost:8000/health

# 多工具编排：成绩预警 → 查数据 + 发通知
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"张三内科成绩低于80分，请预警并通知带教老师"}'

# 当前角色可见工具
curl http://localhost:8000/api/v1/tools

# 跑评测（RAG 4/4 / Text2SQL 5/5）
curl -X POST http://localhost:8000/api/v1/rag/eval
curl -X POST http://localhost:8000/api/v1/text2sql/eval
```

## 真实服务启动验证（bge 模式 · 2026-09-01 实测全绿）
`uvicorn app.main:app --host 0.0.0.0 --port 8000`（`EMBEDDING_MODE=bge` + pgvector）启动后实测：
- `GET /health` → `{"status":"ok"}`；`GET /api/v1/tools` 按租户返回工具清单
- `POST /api/v1/rag/eval` → **4/4（100%）**，首次含 bge-m3 模型加载约 20s，后续秒回
- `POST /api/v1/text2sql/eval` → **5/5 命中 + 5/5 真执行**（SQL 真实跑通；含考勤查询用例）
- `POST /api/v1/text2sql/tenant-isolation` → `isolated: true`（capital-medical 14 行 vs other-hospital 3 行）
- `POST /api/v1/usage/eval` → **4/4（100%）**
- `POST /api/v1/chat`（mock LLM）→ 综合回答带 RAG 引用；`/api/v1/chat/stream` → SSE 事件流（steps→token→[DONE]）正常

## 知识库文档导入（ingest · 2026-09-01 实测）
新增文档入库走 `scripts/ingest_doc.py`（与 RAG 检索链路同一套 chunk_text + bge-m3）：

```bash
# 示例：导入「学生实习大纲」到临床知识库（默认 kb=clinical, tenant=capital-medical）
python scripts/ingest_doc.py --file docs/学生实习大纲.md
# 导入后立即检索自检（看 top3 是否命中新文档）
python scripts/ingest_doc.py --file docs/学生实习大纲.md --verify "学生实习出科考核怎么算分？"
# 其他用法
python scripts/ingest_doc.py --text "直接传文本"                # 不走文件
python scripts/ingest_doc.py --file xxx.md --kb usage          # 导入"使用说明"知识库
python scripts/ingest_doc.py --file xxx.md --tenant other-hospital  # 其他租户（数据互相隔离）
```

- **原理**：文档 → `chunk_text(120字/30字重叠)` → `BgeEmbedder` 编码 1024 维 → `INSERT kb_chunks`（幂等：doc_idx 接续编号 + `ON CONFLICT DO NOTHING`，重复导入 0 新增）
- **知识域**：`kb` 字段区分知识库（clinical=临床知识库 / usage=使用说明），新文档默认并入 clinical，助手检索临床/教学问题即命中
- **多租户**：按 `tenant_id` 写入，行级隔离，换租户互不可见
- **实测（2026-09-01）**：`学生实习大纲.md`（712 字 → 8 chunk）导入后，问「学生实习大纲里实习目标是什么？」，RAG 引用 [1][2][3] 全部命中新大纲（实习目标/轮转/OSCE）；pgAdmin 连 `ai_service` → `kb_chunks` 表可按 `kb='clinical' AND tenant_id='capital-medical'` 过滤查看
- **注意**：若服务已在运行，新文档对**新进程/重启后**的检索立即可见；运行中进程无需重启即可在下次请求读到新数据（每次请求新建 Retriever 全量读库）。如遇行为不符，先确认进程加载的是否最新代码。

## 多租户 / RBAC
- dev 模式：默认上下文见 `.env`（`DEV_TENANT` / `DEV_USER` / `DEV_ROLE`）；可用请求头覆盖：
  `X-Tenant-Id`、`X-Role`(admin|teacher|student)、`X-User-Id`
- 角色权限：student 仅知识库；teacher/admin 全工具。工具执行前经 RBAC 二次校验。

## 切换真实 LLM
`.env` 设 `LLM_MODE=openai`，填 `OPENAI_API_KEY`（兼容 DeepSeek / 通义等：改 `OPENAI_API_BASE` / `OPENAI_MODEL`）。
- `OpenAICompatLLM` 已用标准库 `urllib` 实现 `/chat/completions` 调用与 `tool_calls` 解析，无需额外依赖。
- 真实多轮工具历史已在 `agent.py` 写入 `assistant(tool_calls)`，结构对齐 OpenAI 格式。

## MCP Server
```bash
pip install mcp
python -m app.mcp_server          # stdio 模式，供 MCP 客户端（Claude Desktop 等）调用
```
工具定义与 Agent 注册表共享（`registry`），改动一处两处生效。

## 对接真实后端（小萱同源）
小萱的本质是「Dify 编排 + Laravel 工具回调」。本项目用**自研 ReAct 当脑子**，复用同一套 Laravel 后端接口
（`medical-server-php/routes/api.php` 第 22–30 行，`/api/dify/tools/*` 是裸路由、无 auth 中间件、无需令牌），
业务建单（请假/审批人分配/站内信）全在 Laravel，AI 服务只负责「意图识别 → 选工具 → 调接口 → 汇总」。

- 后端客户端：`app/backend.py`（`BackendClient` 异步调用 `/dify/tools/{tool}`，取 Laravel 信封 `data` 字段）
- 真实工具：`app/tools.py` 中 `submit_leave` / `query_duty_schedule` / `query_absent_times` / `query_teaching_event`
- 身份：`user_id` 字段要「学号/工号(id_num)」，由请求头 `X-User-Num` 注入（生产由前端下发），缺省回退 `DEV_USER`

### 本地验证（不碰生产库、不需 Key）
```bash
# 终端1：起桩后端（模仿 Laravel 那 4 个接口）
python -m uvicorn scripts.stub_backend:app --port 9000
# 终端2：起 AI 服务，指向桩（mock 模式即可验证 HTTP 链路）
BACKEND_BASE_URL=http://127.0.0.1:9000/api LLM_MODE=mock python -m uvicorn app.main:app --port 8000
# 终端3：发一句话，观察是否真调通后端
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" -H "X-User-Num: 2021001" \
  -d '{"message":"帮我请个假，因为感冒，明天一天"}'
# → steps 出现 submit_leave，且桩后端日志打印 [STUB] submit_leave 收到调用
```

### 接真实服务器 + 真实 LLM（DeepSeek）
1. `.env`：`BACKEND_BASE_URL=https://你的域名/api`（你已部署、AI 服务可访问）
2. `.env`：`LLM_MODE=openai`、`OPENAI_API_BASE=https://api.deepseek.com/v1`、`OPENAI_API_KEY=你的key`、`OPENAI_MODEL=deepseek-chat`
3. 重启服务即可。真实 LLM 自己决定调哪个工具、填什么参数（比 mock 关键词路由更准）。

### 已知缺口
- **补卡**：后端 `OaProcess::TYPE_CHECKIN='补卡申请'` 已支持，但**未暴露成 `/api/dify/tools` 接口**（小萱当前也没独立补卡工具）。
  要做需在 Laravel 加一个补卡建单端点，再在 `app/tools.py` 加 `tool_submit_card_reissue`（复用 `backend.call_tool("submit_card_reissue", ...)`）。


## 评测集（简历口径）
- RAG：`app/rag.py::RAG_EVAL`，Top-1 命中 4/4
- Text2SQL：`app/data.py::TEXT2SQL_TESTSET`，关键子句匹配 5/5

## 与简历 / 备考库对齐点
| 简历表述 | 代码位置 |
|---|---|
| Agent 编排（ReAct）+ 工具调用 | `app/agent.py` + `app/tools.py` |
| RAG 临床知识库 | `app/rag.py` |
| Text2SQL 学员数据 | `app/data.py` |
| 轮转/排班可行性检测（贪心+局部搜索） | `app/data.py::check_rotation_feasibility` |
| 多租户 RBAC | `app/rbac.py` |
| MCP 标准化封装 | `app/mcp_server.py` |
| 评测体系（4/4） | `app/rag.py` / `app/data.py` 评测函数 |

**语义红线**：轮转 = 学员分配到各科室（贪心+局部搜索）；排班 = 考勤打卡班次（单独管理）。

## 扩展点（生产落地）
- RAG 换 Milvus（亿级/GPU 演进）：在 `rag.py` 增加新 Retriever 替换 `PgVectorRetriever` 即可（上层 `rag_search` / `usage_search` 只依赖 `retrieve` 接口，**换库不改业务代码**）
- Embedding 已用正式模型：默认 **bge-m3 本地模型**（`sentence-transformers` 加载，1024 维，数据不出域）；开发兜底 `EMBEDDING_MODE=mock`；换 OpenAI API 只需实现统一接口（`encode_one/__call__`）
- 流式生成：`/api/v1/chat/stream` 已支持 SSE
- JWT 鉴权：`rbac.get_request_context` 已预留解码点
- 工具并发执行：`agent.py` 可改为 `asyncio.gather`
- PG 连接池：`rag.py` 当前短连接（本地/教学量级够用），生产换 `psycopg_pool` 连接池

## 面试考点对照（重要）
- **`INTERVIEW_MAP.md`** —— 面试前打开它：每道面试题（技术点 1–30 + 实战题 ①–⑫）对应哪段代码的全局对照表 + 30 分钟速记读码顺序。
- 每个 `app/*.py` 文件顶部都有 `📌 面试考点` 块，关键函数旁有 `# 📌` 行内注释，编号与《技术点深挖备考库》一一对应。
- 备考库底稿：`../interview/技术点深挖备考库.md`（同一份编号体系）。

