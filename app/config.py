# -*- coding: utf-8 -*-
"""配置（环境变量驱动，支持 mock / openai 两种 LLM 模式）

📌 面试考点（对应《技术点深挖备考库》）：
  · 技术点30 技术选型与取舍（模型/向量库/Rerank/部署可按环境变量切换） → 见各 *_MODE / PGVECTOR_DSN
  · 实战④ 防死循环（max_steps 硬上限）                              → 见 MAX_STEPS
  · 实战⑨ 反思开关（enable_reflection）                            → 见 ENABLE_REFLECTION
  · 技术点16 服务治理（限流/熔断已进程内实现，见 app/governance.py；生产换 Redis）→ 见 LLM 治理配置块
"""
import os

from dotenv import load_dotenv

load_dotenv()  # 加载项目根 .env（不覆盖已存在的环境变量）；main/pytest/独立脚本任一入口都生效


def _env(key: str, default: str = "") -> str:
    return os.getenv(key, default)


# ============ LLM ============
LLM_MODE = _env("LLM_MODE", "mock").lower()          # mock | openai
# 📌 接 DeepSeek（用户选定）：把 base 指到 DeepSeek 即可，OpenAICompatLLM 通用
#    DeepSeek: https://api.deepseek.com/v1   模型: deepseek-chat / deepseek-reasoner
#    通义千问: https://dashscope.aliyuncs.com/compatible-mode/v1
#    智谱 GLM: https://open.bigmodel.cn/api/paas/v4
OPENAI_API_BASE = _env("OPENAI_API_BASE", "https://api.openai.com/v1")
OPENAI_API_KEY = _env("OPENAI_API_KEY", "")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4o-mini")
# 📌 模型路由（A' ①）：决策/路由类轻任务用快模型提速降本；综合生成仍用 OPENAI_MODEL。
#    不想启用路由时，把它设为与 OPENAI_MODEL 相同即可（等价单模型）。
OPENAI_MODEL_FAST = _env("OPENAI_MODEL_FAST", "qwen-turbo")
# 📌 思考链开关：qwen3 系列默认输出 reasoning_content（思维链），会显著拉长首字延迟。
#    本场景（工具选型 + 基于结构化结果的综合）不需要深度推理，关掉可提速并降低 token 消耗。
LLM_ENABLE_THINKING = _env("LLM_ENABLE_THINKING", "0").lower() in ("1", "true", "yes")

# ============ 真实后端（医院 Laravel 平台 / Dify 工具回调接口）============
# 📌 小萱同源：AI 服务当脑子，业务建单走 Laravel。填你服务器的 API 基址（含 /api）。
#    例：https://your-domain.com/api   或   http://120.x.x.x:8080/api
BACKEND_BASE_URL = _env("BACKEND_BASE_URL", "")
BACKEND_TIMEOUT = float(_env("BACKEND_TIMEOUT", "15"))
# 📌 P0 必修 (2026-09-08 code review): server-to-server HMAC 签名，防伪造。
#   协议：X-Timestamp: <unix-seconds> + X-Signature: hmac_sha256(secret, "{method}\n{path}\n{ts}\n{sha256(body)}")
#   服务端验签：ts 与 now 偏差 >BACKEND_HMAC_SKEW_SEC 默认拒绝 + 验签失败拒绝。
#   Laravel 端未实现验签前可设 BACKEND_HMAC_ENABLED=false 临时关闭（仍 WARNING）。
BACKEND_HMAC_ENABLED = _env("BACKEND_HMAC_ENABLED", "true").lower() == "true"
BACKEND_HMAC_SECRET = _env("BACKEND_HMAC_SECRET", "")
BACKEND_HMAC_SKEW_SEC = int(_env("BACKEND_HMAC_SKEW_SEC", "300"))  # 5 分钟时间窗，防重放

# ============ 真实平台直连（宣武实习管理平台，AES 加密协议）============
# 📌 协议见 app/xuanwu_client.py：Laravel Sanctum + 双层 base64 + AES-256-CBC。
#    配齐三项即启用「真实平台数据源」（优先于 BACKEND_BASE_URL 的明文契约）；
#    凭据只写本地 .env，不进代码库/简历/文档。
XUANWU_API_BASE = _env("XUANWU_API_BASE", "")   # 例：https://xuanwu.nobeliumedu.com/api
XUANWU_NUM = _env("XUANWU_NUM", "")             # 登录账号（手机号）
XUANWU_PASSWORD = _env("XUANWU_PASSWORD", "")
XUANWU_TIMEOUT = float(_env("XUANWU_TIMEOUT", "15"))

# ============ RAG / Embedding ============
RAG_TOP_K = int(_env("RAG_TOP_K", "3"))
CHUNK_SIZE = int(_env("CHUNK_SIZE", "120"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "30"))
# 📌 Embedding 模式：bge（默认，本地正式模型，医疗数据不出域）| mock（本地哈希，零依赖兜底）| openai（API 预留）
#   切换只改这一个配置：写入与查询必须同一模式，切换后需重建向量表（维度可能不同）。
EMBEDDING_MODE = _env("EMBEDDING_MODE", "bge").lower()
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "BAAI/bge-m3")
# 本地模型缓存路径：不存在则首次自动从 HuggingFace（镜像 hf-mirror.com）下载到该目录
EMBEDDING_MODEL_PATH = _env("EMBEDDING_MODEL_PATH", "./models/bge-m3")

# ============ 向量库（pgvector：PostgreSQL 向量扩展，本地/生产统一）============
# 📌 真向量库落地：pgvector 把向量当普通列存进 PostgreSQL（表 kb_chunks + HNSW ANN 索引）；
#   多租户按行级过滤 `WHERE tenant_id=?` 隔离（与业务数据同库，事务一致，可连 DBeaver/Navicat 查看）。
#   连接串示例：postgresql://postgres:你的密码@localhost:5432/ai_service
# 📌 维度与模型强绑定：bge-m3 = 1024 维；换模型（如 text-embedding-3-small 384 维）需改 EMBEDDING_DIM 并重建表
EMBEDDING_DIM = int(_env("EMBEDDING_DIM", "1024"))
PGVECTOR_DSN = _env("PGVECTOR_DSN", "")   # 例：postgresql://postgres:你的密码@localhost:5432/ai_service

# ============ Text2SQL 目标库（本地 SQLite 教学库，真执行 SQL）============
# 📌 为什么用 SQLite：Python 内置、零依赖、本地可跑；SQL  dialect 与 MySQL 基本一致，
#    生产把 TEACHING_DB_DIALECT 切到 mysql 并换驱动即可（执行层已隔离）。
TEACHING_DB_PATH = _env("TEACHING_DB_PATH", "./teaching.db")
TEACHING_DB_DIALECT = _env("TEACHING_DB_DIALECT", "sqlite").lower()   # sqlite | mysql

# ============ 实习管理系统（本地 Laravel：D:\实习系统\internship-server）============
# 📌 对接方式：AI 服务当脑子，业务数据全走 Laravel 的 AI 工具网关（白名单 + 用户 token 鉴权）。
#    网关入口：POST {INTERNSHIP_API_BASE}/ai/tools/call  body {"tool": 工具名, "arguments": {...}}
#    鉴权：透传前端下发的用户 Sanctum token（X-Internship-Token → Bearer），
#          数据权限 = token 所属用户（学生只能看自己的轮转/考勤，老师工具学生调会 403）。
INTERNSHIP_API_BASE = _env("INTERNSHIP_API_BASE", "")   # 例：http://127.0.0.1:8000/api/v1
INTERNSHIP_TIMEOUT = float(_env("INTERNSHIP_TIMEOUT", "15"))

# ============ 鉴权 ============
# 📌 P0 必修 (2026-09-08 code review): 生产忘配 → 任意身份伪造。
#   - AUTH_MODE 默认 dev 仅方便本地 pytest，main/mcp_server 启动时会 WARNING
#   - JWT_SECRET 默认空串；AUTH_MODE=jwt 时启动期强制 ≥32 字节，否则直接拒绝启动
#   - 真实部署：必须显式设 AUTH_MODE=jwt + JWT_SECRET=$(openssl rand -hex 32)
AUTH_MODE = _env("AUTH_MODE", "dev").lower()         # dev(固定 token) | jwt
JWT_SECRET = _env("JWT_SECRET", "")                  # ⚠️ 默认空串；jwt 模式长度需 ≥32 字节
DEV_TENANT = _env("DEV_TENANT", "capital-medical")
DEV_USER = _env("DEV_USER", "teacher-001")
DEV_ROLE = _env("DEV_ROLE", "teacher")               # admin | teacher | student


def validate_security_config() -> None:
    """启动期安全配置校验：fail-fast 防生产忘配（对应 code review P0-1）。

    校验项：
      ① AUTH_MODE 必须是 dev / jwt 之一
      ② AUTH_MODE=jwt 时，JWT_SECRET 必须 ≥32 字节（HS256 推荐 256 bit）
      ③ AUTH_MODE=dev 时仅 WARNING，不阻塞（demo/pytest 需要）

    调用方：app/main.py、app/mcp_server.py 入口；pytest 由 conftest 触发。
    """
    import logging
    log = logging.getLogger(__name__)

    if AUTH_MODE not in ("dev", "jwt"):
        raise RuntimeError(
            f"[security] AUTH_MODE={AUTH_MODE!r} 非法，必须是 'dev' 或 'jwt'。"
            " 设环境变量 AUTH_MODE=jwt 启用 JWT；dev 仅供本地 demo/pytest。"
        )

    if AUTH_MODE == "jwt":
        # HS256 推荐密钥 ≥32 字节（256 bit），短密钥有 InsecureKeyLengthWarning 并易被爆破
        if not JWT_SECRET or len(JWT_SECRET.encode("utf-8")) < 32:
            raise RuntimeError(
                f"[security] AUTH_MODE=jwt 但 JWT_SECRET 长度 {len(JWT_SECRET)} < 32 字节。"
                " 生成强密钥：python -c \"import secrets;print(secrets.token_hex(32))\""
            )
    else:
        # dev 模式：打印 WARNING 提醒生产不要用
        log.warning(
            "[security] AUTH_MODE=dev 启用，任何带 X-Role 头的请求都按所声明角色放行。"
            " ⚠️ 仅供本地 demo/pytest，生产请改 AUTH_MODE=jwt + JWT_SECRET（≥32 字节）。"
        )

# ============ Agent ============
MAX_STEPS = int(_env("MAX_STEPS", "6"))          # 📌 实战④ 防死循环：ReAct 最大步数硬上限
ENABLE_REFLECTION = _env("ENABLE_REFLECTION", "true").lower() == "true"  # 📌 实战⑨ 反思开关：按需开启，防延迟/ token 暴涨

# ============ Redis（生产级别：多副本一致 + Lua 原子化）============
# 📌 生产级别三件套：会话记忆 / 限流令牌桶 / 熔断状态都集中存储
#   - 2026-09-11 起**SQLite 后端已彻底下线**：单测/CI 一律用 backend=memory（进程内），生产用 redis
#   - 真生产（多 worker / 跨实例）必须切 backend=redis 并起 Redis 服务
#   - 启动期 fail-fast：选了 redis 后端但 REDIS_URL 不可达 → 进程直接退出（防静默退化）
REDIS_URL = _env("REDIS_URL", "")  # 例：redis://[:password]@127.0.0.1:6379/0
MEMORY_BACKEND = _env("MEMORY_BACKEND", "redis").lower()         # redis（生产）| memory（单测/降级）
GOVERNANCE_BACKEND = _env("GOVERNANCE_BACKEND", "memory").lower()  # memory | redis
# 记忆在 Redis 里的 TTL（秒）—— 默认 1h，过期由 Redis 服务端自动清理
MEMORY_REDIS_TTL_SECONDS = int(_env("MEMORY_REDIS_TTL_SECONDS", "3600"))


def validate_redis_config() -> None:
    """Redis 后端启动期校验（fail-fast）。

    校验项：
      ① MEMORY_BACKEND 必须是 redis/memory；GOVERNANCE_BACKEND 必须是 memory/redis
      ② 任一后端选了 redis → REDIS_URL 必须非空
      ③ 不强制 ping（连接失败在首次调用时感知，避免启动阻塞）

    调用方：app/main.py、app/mcp_server.py 入口
    """
    import logging
    log = logging.getLogger(__name__)

    valid_memory = {"redis", "memory"}
    valid_governance = {"memory", "redis"}
    if MEMORY_BACKEND not in valid_memory:
        raise RuntimeError(
            f"[redis] MEMORY_BACKEND={MEMORY_BACKEND!r} 非法，必须是 'redis' 或 'memory'"
        )
    if GOVERNANCE_BACKEND not in valid_governance:
        raise RuntimeError(
            f"[redis] GOVERNANCE_BACKEND={GOVERNANCE_BACKEND!r} 非法，必须是 'memory' 或 'redis'"
        )

    using_redis = (MEMORY_BACKEND == "redis" or GOVERNANCE_BACKEND == "redis")
    if using_redis and not REDIS_URL:
        raise RuntimeError(
            "[redis] MEMORY_BACKEND/GOVERNANCE_BACKEND 选了 redis 但 REDIS_URL 为空。"
            " 设 REDIS_URL=redis://127.0.0.1:6379/0 后重启，或把 MEMORY_BACKEND 改回 memory（进程内，重启丢记忆）"
        )
    if using_redis:
        log.info("[redis] 使用 Redis 后端 url=%s memory=%s governance=%s",
                 REDIS_URL.split("@")[-1], MEMORY_BACKEND, GOVERNANCE_BACKEND)


# ============ LLM 服务治理（限流 / 熔断，进程内实现）============
# 📌 限流：令牌桶控 QPS（burst 为桶容量，允许短时突发）；Token 预算按 60s 滚动窗口限 token 消耗
#    （模型成本跟 token 走，光控 QPS 拦不住长答案烧钱）。预算设 0 = 不启用。
LLM_RATELIMIT_QPS = float(_env("LLM_RATELIMIT_QPS", "2"))
LLM_RATELIMIT_BURST = int(_env("LLM_RATELIMIT_BURST", "4"))
LLM_TOKEN_BUDGET_PER_MIN = int(_env("LLM_TOKEN_BUDGET_PER_MIN", "200000"))
# 📌 熔断：滑动窗口内请求数 ≥ MIN_REQUESTS 且错误率 ≥ THRESHOLD → OPEN（快速失败）；
#    冷却 COOLDOWN 秒后 HALF_OPEN 放 PROBES 个探测，成功回 CLOSED / 失败回 OPEN。
LLM_CB_ERROR_THRESHOLD = float(_env("LLM_CB_ERROR_THRESHOLD", "0.5"))
LLM_CB_MIN_REQUESTS = int(_env("LLM_CB_MIN_REQUESTS", "5"))
LLM_CB_WINDOW = float(_env("LLM_CB_WINDOW", "60"))
LLM_CB_COOLDOWN = float(_env("LLM_CB_COOLDOWN", "30"))
LLM_CB_HALFOPEN_PROBES = int(_env("LLM_CB_HALFOPEN_PROBES", "2"))

# ============ 服务 ============
HOST = _env("HOST", "0.0.0.0")
PORT = int(_env("PORT", "8000"))
