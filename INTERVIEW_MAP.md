# 面试考点 ↔ 代码位置 对照表（AI 教学助手微服务）

> 用途：面试前打开本文件，一眼看到「每道面试题对应 `ai-service` 里哪段代码」。
> 每个代码文件顶部也都有 `📌 面试考点` 块，打开文件第一眼即见。
> 备考库底稿：`../interview/技术点深挖备考库.md`（技术点 1–33 + 实战场景题 ①–⑫）。

## 一、总览：技术点 → 代码

| 备考库考点 | 面试问什么 | 看代码位置 |
|---|---|---|
| 技术点4 Function Calling / Tool Use | 工具四步闭环、JSON Schema、ReAct vs 纯 FC | `app/agent.py`(ReAct 循环) · `app/tools_registry.py`(Schema) · `app/tools.py`(实现) |
| 技术点5 MCP 协议 | MCP 是什么、与硬编码 API 区别、传输层 | `app/mcp_server.py` · `app/tools_registry.py`(共享 registry) |
| 技术点6 多 Agent（延展） | 为何本项目单 Agent 多工具 | `app/agent.py`(单 Agent 架构，注释已说明) |
| 技术点7 多租户架构 | 多院校隔离怎么落地 | `app/rbac.py::get_request_context` · `app/models.py::RequestContext` |
| 技术点8 RBAC 多角色 | 学员/带教/管理权限、数据权限 | `app/rbac.py::ROLE_PERMISSIONS` / `can_use` |
| 技术点9 多租户数据隔离 | 行级+向量库隔离、防串数据 | `app/rbac.py`(tenant 透传) · 各 tool 用 `ctx.tenant_id` · 向量侧 `app/rag.py::PgVectorRetriever`(SQL `WHERE tenant_id=?` 行级过滤) |
| 技术点10 文档分块 chunk+重叠 | 分块大小怎么定、重叠作用 | `app/rag.py::chunk_text` |
| 技术点11 Embedding 向量化 | 维度、中英文、增量 | `app/rag.py::BgeEmbedder`（本地 bge-m3 正式模型 1024 维；`get_embedder()` 工厂按 EMBEDDING_MODE 切换，mock 为开发兜底） |
| 技术点12 向量检索 | 真向量库、HNSW ANN、持久化 | `app/rag.py::PgVectorRetriever.retrieve`（pgvector `kb_chunks` 表 + HNSW 索引） |
| 技术点12-补 BM25 混合检索 | 为什么混合、RRF 怎么融合 | `app/rag.py::BM25` / `_rrf` / `hybrid_retrieve` |
| 技术点12-⑥ 向量库选型 | Chroma/pgvector/Milvus 区别、为什么这么选 | `app/rag.py::PgVectorRetriever`（开发/生产统一 pgvector）· `app/config.py::PGVECTOR_DSN` |
| 技术点13 Rerank 重排 | 为什么检索完还要 rerank | `app/rag.py::PgVectorRetriever.rerank` |
| 技术点14 引用溯源 | 引用怎么来的、防编造 | `app/rag.py::rag_search`（带 [1][2]） |
| 使用说明 RAG（双知识库） | 实习系统操作 FAQ 检索、与临床知识库并列 | `app/rag.py::usage_search` / `eval_usage` · `app/tools.py::tool_usage_guide` |
| 技术点15 Text2SQL | Schema 注入、防注入、边界红线 | `app/data.py::generate_sql` / `execute_sql` / `summarize_rotation` |
| 技术点15-补 Text2SQL 安全护栏 | 只读校验、租户强制过滤、不信任模型输出 | `app/data.py::assert_sql_safe`（落地点：`execute_sql` 执行前调用） |
| 技术点15-补 多租户 SQL 隔离 | 同 SQL 换租户只能看本院校数据 | `app/data.py::check_tenant_isolation` · 接口 `/api/v1/text2sql/tenant-isolation` |
| 技术点19 贪心+局部搜索 轮转 | 为什么不用精确算法、目标函数 | `app/data.py::greedy_init` / `objective` / `local_search` |
| 技术点29 双栈架构 | 为什么 Python 微服务而非 PHP 内嵌、SSE | `app/main.py::chat_stream` · README「与简历对齐」 |
| 技术点30 技术选型 | LLM/Embedding/向量库/MCP 取舍、自研 vs LangChain | `app/config.py` · `app/llm.py::get_llm` |
| 技术点33 知识库文档导入 | 文档怎么进向量库、幂等、要不要重启、多租户 | `scripts/ingest_doc.py`（分块→bge-m3→INSERT kb_chunks，doc_idx 接续+ON CONFLICT 幂等）· `app/rag.py::chunk_text/BgeEmbedder` |

## 二、实战场景题 → 代码

| 实战题 | 面试问什么 | 看代码位置 |
|---|---|---|
| ① RAG 检索正确性怎么验证 | 离线召回评测+引用校验+BadCase | `app/rag.py::eval_rag` / `rag_search` |
| ② RAG 检索不好怎么优化 | chunk/混合检索/Rerank/查询改写 | `app/rag.py`（各函数注释） |
| ③ 工具调用常见异常 | JSON错乱/幻觉工具/参数错/超时/超长 | `app/agent.py`(tool_calls 兜底) · `app/tools.py` |
| ④ 防死循环 | max_steps+重复检测+反思熔断 | `app/agent.py::AgentEngine.run`（循环+`recent` 检测） |
| ⑤ Token 成本优化 | 小模型分担、Router | `app/llm.py::OpenAICompatLLM`（接小模型点） |
| ⑥ 端到端延迟优化 | 并行/SSE/Router | `app/main.py::chat_stream`（SSE） |
| ⑦ 幂等性 | 写操作重复执行事故 | `app/tools.py::tool_notification`（幂等注释） |
| ⑧ 线上排查/降级/可观测 | 分层排查、指标、降级 | `app/main.py`(评测接口+RBAC) · `app/models.py::ToolStep` |
| ⑨ 反思机制 | 哪些模块加、防死循环 | `app/agent.py::run`（reflection 分支）+ `config.ENABLE_REFLECTION` |
| ⑩ 记忆系统/压缩 | 三层记忆、上下文膨胀 | `app/memory.py::MemoryStore` / `compact` |
| ⑪ 多租户 Agent 隔离 | 数据/权限/工具三层 | `app/rbac.py`(can_use+tenant 注入) |
| ⑫ 自研 ReAct vs LangGraph | 致命红线：生产自研 | `app/agent.py`(LLMPlanner/RulePlanner) · `app/mcp_server.py` |

## 三、读代码顺序（面试前 30 分钟速记）

1. `app/main.py` → 看双栈/SSE/评测接口入口（技术点29、实战①⑧）
2. `app/agent.py` → 看 ReAct 循环、防死循环、RBAC 校验、反思（技术点4、实战④⑨⑫）
3. `app/rag.py` → 看 RAG 五步：chunk→embed→retrieve→rerank→引用；以及双知识库之「系统使用说明」`usage_search`（技术点10–15、实战①②）
4. `app/data.py` → 看 Text2SQL + 轮转贪心+局部搜索（技术点15、19）
5. `app/rbac.py` → 看多租户+RBAC（技术点7/8/9、实战⑪）
6. `app/tools_registry.py` + `app/tools.py` → 看 9 工具 JSON Schema 与实现（含 4 真实后端工具 + 双知识库 RAG；技术点4、实战③⑦）
7. `app/mcp_server.py` → 看 MCP 真实封装（技术点5、实战⑫）
8. `scripts/ingest_doc.py` → 看文档入库链路（分块→bge-m3→pgvector 幂等插入；技术点33）

> 提示：所有文件顶部 `📌 面试考点` 块 + 关键函数行内 `# 📌` 注释，已与上述编号一一对应。
