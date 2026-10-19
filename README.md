# AI 教学助手（实习平台 AI 服务 · Python / FastAPI）

临床实习教学管理平台的 AI 层，为业务系统接入 AI 对话能力：学员和教师在平台对话框里用自然语言提问——查自己的轮转安排、值班、考勤、请假记录、考核成绩，或咨询临床知识——本服务负责意图识别与工具编排，把一句话转成一串具体的业务查询。查业务数据时回调平台 Laravel 工具网关并透传调用者 token，权限由业务后端按调用者身份判定（学生看不到教师专属数据）；RAG 知识检索在本服务内完成，按租户行级隔离。前端对话框通过 SSE 直连本服务，逐字流式输出，且每一步工具调用对用户可见。

## 特性

- **ReAct 编排**（规划 → 调用 → 观察 → 修正），可插拔 `Planner`：Mock 关键词路由（零成本跑通）/ 真实 LLM tool_calls
- **20 个注册工具**，按职责分组：
  - 实习平台 ×13：查轮转 / 值班 / 考勤 / 出勤统计 / 请假记录 / 教学活动 / 公告 / 考核 / 病历 / 待办汇总 / 提交请假 / 科室学生 / 待审批，统一回调 `/api/v1/ai/tools/call` 并透传用户 token
  - 知识检索 ×2：临床知识库 RAG / 系统使用说明 RAG
  - 通用后端 ×5：提交请假 / 值班查询 / 缺勤查询 / 教学活动（`/api/dify/tools/*`）+ 站内通知（进程内模拟投递）
- **LLM 抽象层**：Mock / OpenAI 兼容（通义 / DeepSeek / 智谱改 base 即复用），标准库 `urllib` 实现，无额外依赖
- **多租户 RBAC**：租户隔离 + admin / teacher / student 三级权限；向量侧按 `tenant_id` 行级过滤（`WHERE tenant_id=?`），工具执行前二次校验
- **RAG 真向量库**：pgvector（`kb_chunks` 表 + HNSW 索引）；**混合检索** = 向量召回 + BM25 → RRF 融合 → Rerank
- **正式 Embedding**：本地 bge-m3（`sentence-transformers` 加载，1024 维，L2 归一化，数据不出域；首次自动下载到 `./models/bge-m3`）
- **可观测性**：`/observability` 面板——请求审计、token 用量、工具调用统计、单次 trace 时间瀑布

## 架构

```mermaid
flowchart LR
    U["浏览器<br/>自带聊天页 / 实习系统悬浮球"] -->|"SSE /api/v1/chat/stream"| F["FastAPI :8001"]
    F --> AG["ReAct Agent<br/>规划 → 工具 → 观察 → 再决策"]
    AG -.->|"按角色过滤工具白名单"| RB["RBAC<br/>X-Role / 租户隔离"]
    AG --> RAG["query_knowledge_base<br/>临床知识库 RAG"]
    AG --> IT["internship_* 工具 ×13"]
    IT -->|"回调工具网关<br/>透传 X-Internship-Token"| LA["实习平台 Laravel API :8000"]
    RAG --> PG[("PostgreSQL + pgvector<br/>向量召回 + BM25 → RRF")]
    AG <--> LLM["LLM qwen3.8-max-0902<br/>enable_thinking=0"]
    F --> OBS["/observability<br/>审计 / token / 耗时瀑布"]
```

**关键设计**：Agent 不直接碰业务库。查实习数据一律走 `internship_*` 工具回调 Laravel 网关，由业务后端按调用者 token 判定权限（学生调教师工具 = 403）；RAG 检索只读，且在 SQL 层强制 `tenant_id` 过滤。

## 界面预览

完整对话录屏（实习平台悬浮球直连本服务，2 次工具调用 + 1 次 RAG 检索 + 观测面板反查）：[docs/demo/ai-assistant-demo.webm](docs/demo/ai-assistant-demo.webm)（浏览器可直接播放）

| 可观测面板（请求审计 + 工具调用统计） | 单次 trace 时间瀑布（定位慢在哪个阶段） |
|---|---|
| ![可观测面板](docs/screenshots/05-observability.png) | ![trace 瀑布](docs/screenshots/06-observability-trace.png) |

## 目录结构

```
ai-service/
├── app/
│   ├── main.py           # FastAPI 入口（SSE 流式 + 路由 + 评测接口）
│   ├── config.py         # 环境变量配置
│   ├── models.py         # 请求/响应/运行期上下文模型
│   ├── llm.py            # LLM 抽象层（Mock + OpenAI 兼容）
│   ├── agent.py          # ReAct 引擎 + Planner/Synthesizer + 防死循环 + 反思
│   ├── tools_registry.py # 工具注册表（JSON Schema + 权限）
│   ├── tools.py          # 通用工具实现（RAG / 轮转检测 / 通用后端工具）
│   ├── internship.py     # 实习平台 13 个工具（/api/v1/ai/tools/call，透传用户 token）
│   ├── rag.py            # RAG 检索 + 评测
│   ├── data.py           # 轮转分配引擎（贪心 + 局部搜索）
│   ├── memory.py         # 对话记忆（短期 + 长期压缩）
│   ├── rbac.py           # 多租户 RBAC + 鉴权依赖
│   ├── backend.py        # 通用后端客户端（/api/dify/tools/*）
│   └── observability.py  # 审计 / token 用量 / 耗时瀑布
├── scripts/
│   ├── run_server.py     # 推荐启动入口（先预热 bge-m3 再起服务）
│   ├── ingest_doc.py     # 知识库文档导入
│   └── stub_backend.py   # 本地桩后端（链路联调用）
├── tests/                # pytest：端到端 + 评测 + 鉴权 + 后端工具
├── requirements.txt
├── .env.example
└── README.md
```

## 快速开始

> 前置：**本地 PostgreSQL + pgvector 扩展**（RAG 向量库依赖，见下方「pgvector 接入」）。
> Embedding 默认 bge-m3：首次运行自动从 HuggingFace 镜像下载约 2GB 权重到 `./models/bge-m3`；
> 不想下载可用 `EMBEDDING_MODE=mock` 本地哈希兜底（仅开发调试）。
> LLM 用 Mock 模式即可零成本跑通。

```bash
cd ai-service
pip install -r requirements.txt
cp .env.example .env          # 填 PGVECTOR_DSN；LLM_MODE=mock 即可跑通
python scripts/run_server.py  # 推荐：先模块级预热 bge-m3 再起服务
# 备选：python -m app.main
```

### pgvector 接入（本地 PostgreSQL）

1. 安装 PostgreSQL（建议 16+）。
2. 启用扩展 + 建库（首次）：
   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   CREATE DATABASE ai_service;
   ```
3. `.env` 填连接串：`PGVECTOR_DSN=postgresql://postgres:你的密码@localhost:5432/ai_service`
4. 首次启动自动建表 `kb_chunks`（含 HNSW 索引）并灌入种子语料，无需手动建表。

## Docker 部署

`Dockerfile` + `docker-compose.yml` 把「AI 编排层 + 向量库」打包为两个容器；镜像设计取舍（模型不进镜像 / 密钥不进镜像）见 `Dockerfile` 顶部注释。

```bash
# 前置：本机装 Docker
cd ai-service
# .env 调整：PGVECTOR_DSN 的 host 由 localhost 改为容器服务名 pg，并补 POSTGRES_PASSWORD
#   PGVECTOR_DSN=postgresql://postgres:你的密码@pg:5432/ai_service
#   POSTGRES_PASSWORD=你的密码
docker compose up -d --build
curl http://localhost:8000/health
```

- **挂载/持久化**：`./models`（bge-m3 权重，不进镜像）→ `/app/models`；命名卷 `pgdata`（PostgreSQL 数据）。
- **不用真向量库**：`EMBEDDING_MODE=mock` + 注释掉 `depends_on: pg` 即可纯 mock 跑通。
- **真 LLM**：`.env` 设 `LLM_MODE=openai` + `OPENAI_API_KEY`，容器内经 `env_file` 注入生效。

## 调用示例

```bash
curl http://localhost:8000/health

# 多工具编排
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"帮我查一下本周的值班安排"}'

# 当前角色可见工具
curl http://localhost:8000/api/v1/tools

# 内置评测接口
curl -X POST http://localhost:8000/api/v1/rag/eval
```

## 接口自检

启动后（`EMBEDDING_MODE=bge` + pgvector）可逐项自检：

- `GET /health` → `{"status":"ok"}`；`GET /api/v1/tools` 按角色返回工具清单
- `POST /api/v1/rag/eval` → RAG 检索评测（首次含 bge-m3 加载约 20s，后续秒回）
- `POST /api/v1/chat` → 综合回答带 RAG 引用；`/api/v1/chat/stream` → SSE 事件流（steps → token → [DONE]）

## 知识库文档导入（ingest）

新增文档入库走 `scripts/ingest_doc.py`（与检索链路同一套 chunk + embedding）：

```bash
# 导入文档到临床知识库（默认 kb=clinical）
python scripts/ingest_doc.py --file docs/学生实习大纲.md
# 导入后立即检索自检
python scripts/ingest_doc.py --file docs/学生实习大纲.md --verify "学生实习出科考核怎么算分？"
# 其他用法
python scripts/ingest_doc.py --text "直接传文本"                    # 不走文件
python scripts/ingest_doc.py --file xxx.md --kb usage              # 导入"使用说明"知识库
python scripts/ingest_doc.py --file xxx.md --tenant other-hospital # 其他租户（数据互相隔离）
```

- **原理**：文档 → `chunk_text(120字/30字重叠)` → bge-m3 编码 1024 维 → `INSERT kb_chunks`（幂等：`ON CONFLICT DO NOTHING`，重复导入 0 新增）
- **知识域**：`kb` 字段区分知识库（clinical=临床知识库 / usage=使用说明）
- **多租户**：按 `tenant_id` 写入，行级隔离，换租户互不可见

## 多租户 / RBAC

- dev 模式默认上下文见 `.env`（`DEV_TENANT` / `DEV_USER` / `DEV_ROLE`）；可用请求头覆盖：
  `X-Tenant-Id`、`X-Role`(admin|teacher|student)、`X-User-Id`
- 角色决定可见工具集，工具执行前经 RBAC 二次校验。

## 切换真实 LLM

`.env` 设 `LLM_MODE=openai`，填 `OPENAI_API_KEY`（兼容通义 / DeepSeek 等：改 `OPENAI_API_BASE` / `OPENAI_MODEL`）。

- `OpenAICompatLLM` 用标准库 `urllib` 实现 `/chat/completions` 调用与 `tool_calls` 解析。
- 真实多轮工具历史在 `agent.py` 写入 `assistant(tool_calls)`，结构对齐 OpenAI 格式。
- 已验证可用模型：`qwen3.8-max-0902`（主）/ `qwen3.8-flash`（再决策步）/ `qwen3.8-27b` / `kimi-k3`
- **提速**：`LLM_ENABLE_THINKING=0` 关闭思维链输出，单轮 **19.7s → 7.1s**（`enable_thinking` 写在 `_build_payload`，流式与非流式共用同一份载荷）

## 对接实习平台

本服务作为实习教学管理平台的 AI 层：前端对话框直连本服务，本服务识别意图后回调 Laravel 工具网关取数据，RAG 检索留在本侧。

```
浏览器 :5173 ──SSE 对话──> ai-service :8001 ──工具回调(透传用户 token)──> Laravel :8000
```

### 协议

`POST /api/v1/chat/stream`（SSE），身份走请求头：

| 头 | 含义 |
|---|---|
| `X-Tenant-Id` | 租户隔离 |
| `X-User-Id` / `X-User-Num` | 用户标识 / 学号工号 |
| `X-Role` | admin / teacher / student，决定可见工具集 |
| `X-Internship-Token` | 平台 Sanctum token，回调时原样透传 |

响应是 SSE 事件流：`{type:"steps"}`（工具名 + 参数 + observation）→ `{type:"token"}` 逐字输出 → `[DONE]` 结束。前端气泡因此能看到每一步工具调用，不是黑盒。

### 13 个实习工具

查轮转 / 值班 / 考勤 / 出勤统计 / 请假记录 / 教学活动 / 公告 / 考核 / 病历 / 待办汇总 / **提交请假**，以及教师专属的**科室学生查询**、**待审批查询**。

权限由回调携带的 token 决定——学生调教师专属工具，Laravel 返回 403。工具返回给 LLM 的日期在网关侧统一格式化，避免 UTC ISO 串被模型误读。

### 运行配置

```ini
PORT=8001
LLM_MODE=openai
OPENAI_MODEL=qwen3.8-max-0902        # 首轮选型与综合
OPENAI_MODEL_FAST=qwen3.8-flash      # 再决策步
LLM_ENABLE_THINKING=0                # 关思维链提速
EMBEDDING_MODE=mock                  # 开发用本地哈希兜底；换 bge 即走真向量
MEMORY_BACKEND=memory                # 进程内记忆，重启丢上下文（切 redis 无需改代码）
```

> 注意：变量名是 `OPENAI_API_BASE`（不是 `OPENAI_BASE_URL`）。

### 本地链路联调（桩后端）

不依赖真实平台也可验证「意图识别 → 选工具 → 调接口 → 汇总」的完整 HTTP 链路：

```bash
# 终端1：起桩后端（模仿后端工具接口）
python -m uvicorn scripts.stub_backend:app --port 9000
# 终端2：起 AI 服务，指向桩
BACKEND_BASE_URL=http://127.0.0.1:9000/api LLM_MODE=mock python -m uvicorn app.main:app --port 8000
# 终端3：发一句话，观察是否真调通后端
curl -X POST http://127.0.0.1:8000/api/v1/chat \
  -H "Content-Type: application/json" -H "X-User-Num: 2021001" \
  -d '{"message":"帮我请个假，因为感冒，明天一天"}'
# → steps 出现 submit_leave，桩后端日志打印收到调用
```
