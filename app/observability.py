# -*- coding: utf-8 -*-
"""可观测性模块：trace_id + 结构化日志 + JSONL 请求审计 + token/耗时统计 + OTel + Sentry + Prometheus。

📌 面试考点（对应《技术点深挖备考库》）：
  · 实战⑧ 可观测/降级 → 全链路 trace_id、结构化日志、请求审计、token 成本统计
  · 技术点（成本优化） → token 用量统计（prompt/completion 分开），支撑按量计费与成本复盘

设计原则（零后端也能跑，不破坏现有对外 API）：
  1. OTel / Sentry / Prometheus 均按环境变量开关；未配置时自动降级为本地日志 + JSONL + 内存指标。
  2. trace_id 用 contextvar 贯穿一次请求的 agent→llm→tool 全程，日志 / JSONL / OTel span 三方对齐。
  3. 全部为内部增强，不改动 /api/v1/chat 等对外契约。
"""
import json
import logging
import os
import time
import uuid
from datetime import datetime
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import List

# ============================================================
# trace_id：一次请求一个，贯穿 agent / llm / tool 全链路
# ============================================================
_trace_id: ContextVar[str] = ContextVar("observe_trace_id", default="")


def get_trace_id() -> str:
    """当前请求的 trace_id（由 main.py 中间件注入），无则空串。"""
    return _trace_id.get() or ""


def new_trace_id() -> str:
    return uuid.uuid4().hex  # 32 位 hex，恰好也是合法 trace id 形态


@contextmanager
def trace_scope(trace_id: str = ""):
    """为一次请求建立 trace_id 作用域（contextvar 在 asyncio task 内自动传播）。"""
    tid = trace_id or new_trace_id()
    token = _trace_id.set(tid)
    try:
        yield tid
    finally:
        _trace_id.reset(token)


# ============================================================
# 结构化日志（JSON Lines，落盘 logs/app.log + stdout）
# ============================================================
LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_PATH = LOG_DIR / "audit.jsonl"


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        obj = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": get_trace_id(),
            "msg": record.getMessage(),
        }
        # 支持 span 等附加字段（span() 里通过 extra 注入）
        for key in ("span", "duration_ms", "model", "tool", "tool_args"):
            if hasattr(record, key):
                obj[key] = getattr(record, key)
        # 业务 extra 字段（如 rag.search 的 vec_count/query/fused_topk 等）一并写入 JSON，
        # 供排查与面板读取；排除标准 logging 属性，避免噪声。
        _STD = {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module", "msecs",
            "message", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName", "taskName",
        }
        for k, v in record.__dict__.items():
            if k in _STD or k in obj:
                continue
            try:
                json.dumps(v)
                obj[k] = v
            except (TypeError, ValueError):
                obj[k] = str(v)
        if record.exc_info:
            obj["exc"] = self.formatException(record.exc_info)
        return json.dumps(obj, ensure_ascii=False)


def _init_logging() -> logging.Logger:
    logger = logging.getLogger("ai_service")
    if logger.handlers:  # 幂等：避免 uvicorn reload / 二次 import 重复加 handler
        return logger
    logger.setLevel(logging.DEBUG)
    fmt = JsonFormatter()
    fh = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger


log = _init_logging()


# ============================================================
# token 用量累计（contextvar，同一请求内多次 LLM 调用自动累加）
# ============================================================
_usage_ctx: ContextVar[dict] = ContextVar("observe_usage", default=None)


def reset_usage() -> None:
    _usage_ctx.set({"prompt": 0, "completion": 0, "total": 0, "llm_calls": 0})


def add_usage(prompt: int = 0, completion: int = 0, total: int = 0) -> None:
    u = _usage_ctx.get()
    if u is None:
        u = {"prompt": 0, "completion": 0, "total": 0, "llm_calls": 0}
        _usage_ctx.set(u)
    u["prompt"] += int(prompt or 0)
    u["completion"] += int(completion or 0)
    u["total"] += int(total or 0)
    u["llm_calls"] += 1


def get_usage() -> dict:
    return dict(_usage_ctx.get() or {"prompt": 0, "completion": 0, "total": 0, "llm_calls": 0})


# ============================================================
# Prometheus 指标（未安装时降级为 no-op）
# ============================================================
_PROM = False
try:
    from prometheus_client import Counter, Histogram

    METRIC_REQUESTS = Counter(
        "ai_requests_total", "AI 请求总数", ["tenant", "role", "status"]
    )
    METRIC_DURATION = Histogram(
        "ai_request_duration_seconds", "AI 请求耗时（秒）", ["tenant", "role"]
    )
    METRIC_TOKENS = Counter(
        "ai_llm_tokens_total", "LLM token 消耗", ["kind"]  # prompt | completion
    )
    METRIC_LLM_CALLS = Counter("ai_llm_calls_total", "LLM 调用次数")
    METRIC_TOOL_CALLS = Counter("ai_tool_calls_total", "工具调用次数", ["tool"])
    METRIC_TOOL_DURATION = Histogram("ai_tool_duration_seconds", "工具执行耗时（秒）", ["tool"])
    # RAG 检索专属指标：向量/BM25/RRF/重排各阶段耗时 + 召回条数分布
    METRIC_RAG_DURATION = Histogram("ai_rag_duration_seconds", "RAG 各阶段耗时（秒）", ["kb", "stage"])
    METRIC_RAG_RECALL = Histogram("ai_rag_recall_count", "RAG 各阶段召回条数", ["kb", "stage"])
    METRIC_RAG_HIT = Counter("ai_rag_top1_hit_total", "RAG Top-1 命中数（离线评测）", ["kb"])
    _PROM = True
except Exception:  # pragma: no cover - 依赖缺失时静默降级
    METRIC_RAG_DURATION = METRIC_RAG_RECALL = METRIC_RAG_HIT = None


def record_request(tenant: str, role: str, status: str, duration_s: float, usage: dict, tools: list) -> None:
    """一次请求结束时写入 Prometheus 指标（未启用则 no-op）。"""
    if not _PROM:
        return
    try:
        METRIC_REQUESTS.labels(tenant=tenant, role=role, status=status).inc()
        METRIC_DURATION.labels(tenant=tenant, role=role).observe(duration_s)
        METRIC_TOKENS.labels(kind="prompt").inc(int(usage.get("prompt", 0)))
        METRIC_TOKENS.labels(kind="completion").inc(int(usage.get("completion", 0)))
        METRIC_LLM_CALLS.inc(int(usage.get("llm_calls", 0)))
        for t in tools:
            METRIC_TOOL_CALLS.labels(tool=t).inc()
    except Exception:  # 指标绝不反噬业务
        log.exception("record metrics failed")


def record_rag(stage: str, kb: str, duration_s: float, count: int = None) -> None:
    """RAG 检索各阶段指标（未安装 prometheus_client 时 no-op）。

    stage: total(混合检索总耗时) / bm25_recall / fused / ...；count 为对应召回条数。
    """
    if not _PROM or METRIC_RAG_DURATION is None:
        return
    try:
        METRIC_RAG_DURATION.labels(kb=kb, stage=stage).observe(float(duration_s or 0))
        if count is not None:
            METRIC_RAG_RECALL.labels(kb=kb, stage=stage).observe(int(count))
    except Exception:
        log.exception("record rag metrics failed")


def record_tool_duration(tool: str, duration_s: float) -> None:
    if not _PROM:
        return
    try:
        METRIC_TOOL_DURATION.labels(tool=tool).observe(duration_s)
    except Exception:
        pass


# ============================================================
# OpenTelemetry（未配 OTLP endpoint 时降级为日志）
# ============================================================
_tracer = None
try:
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
    from opentelemetry.trace import Status, StatusCode

    _OTEL = True
except Exception:  # pragma: no cover
    _OTEL = False


def _init_otel():
    if not _OTEL:
        return None
    endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip()
    if not endpoint:
        return None
    try:
        provider = TracerProvider(
            resource=Resource.create({"service.name": "ai-teaching-assistant"})
        )
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=endpoint)))
        otel_trace.set_tracer_provider(provider)
        return otel_trace.get_tracer("ai-service")
    except Exception:
        log.exception("otel init failed, fallback to log-only tracing")
        return None


_tracer = _init_otel()


@contextmanager
def span(name: str, **attrs):
    """统一的埋点入口：OTel 可用时建真实 span，否则降级为结构化日志（含耗时）。

    用法：with span("llm.chat", model="qwen-plus"): ...
    """
    t0 = time.perf_counter()
    if _tracer is not None:
        with _tracer.start_as_current_span(name) as s:
            for k, v in attrs.items():
                s.set_attribute(k, v)
            try:
                yield s
            except Exception as e:
                s.set_status(Status(StatusCode.ERROR))
                s.record_exception(e)
                raise
            finally:
                s.set_attribute("duration_ms", round((time.perf_counter() - t0) * 1000, 2))
    else:
        # 降级：无 OTLP 后端，仅记一条结构化日志（debug 级，避免刷屏）
        try:
            yield None
        finally:
            log.debug(
                "span",
                extra={
                    "span": name,
                    "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                    **attrs,
                },
            )


# ============================================================
# Sentry（未配 DSN 时降级为本地 error 日志）
# ============================================================
_sentry = False


def _init_sentry() -> None:
    global _sentry
    dsn = os.getenv("SENTRY_DSN", "").strip()
    if not dsn:
        return
    try:
        import sentry_sdk

        sentry_sdk.init(
            dsn=dsn,
            traces_sample_rate=float(os.getenv("SENTRY_TRACES_SAMPLE_RATE", "0.2")),
            environment=os.getenv("ENV", "local"),
        )
        _sentry = True
    except Exception:
        log.exception("sentry init failed, fallback to local log")


_init_sentry()


def capture_exception(exc: BaseException) -> None:
    if _sentry:
        try:
            import sentry_sdk

            sentry_sdk.capture_exception(exc)
        except Exception:
            pass
    log.error("unhandled_exception", exc_info=exc)


# ============================================================
# JSONL 请求审计：每轮一行，排查线上问题的主抓手
# ============================================================
def audit(entry: dict) -> None:
    """追加一条审计记录到 logs/audit.jsonl。

    entry 字段约定：trace_id / ts / user / tenant / role / question /
                    answer_len / tools / usage / duration_ms / status / error
    """
    entry.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S"))
    entry.setdefault("trace_id", get_trace_id())
    try:
        with open(AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        log.exception("audit write failed")


def read_audit(last_n: int = 100) -> List[dict]:
    """读取最近 last_n 条审计记录（按时间升序），供 /observability 面板渲染。

    读取失败（文件不存在/损坏行）时降级为空列表，绝不反噬业务。
    """
    if not AUDIT_PATH.exists():
        return []
    try:
        lines = AUDIT_PATH.read_text(encoding="utf-8").splitlines()
    except Exception:
        log.exception("audit read failed")
        return []
    out: List[dict] = []
    for line in lines[-last_n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def get_trace_spans(trace_id: str) -> dict:
    """解析 logs/app.log，按 trace_id 提取逐环节 span，返回时间瀑布数据。

    供 /observability 面板的「详情」抽屉渲染：每个 span 给出 label / 类型 / 相对起始偏移 / 耗时，
    前端据此画出与排查时一致的时间瀑布（谁慢、卡在哪一阶段一目了然）。

    返回结构：{"trace_id", "total_ms", "stages":[{"label","type","offset_ms","duration_ms","items"}]}
      · items：该阶段的明细（仅 RAG 各阶段有），如召回命中的文档原文与得分，供面板展开查看。
        来源是 span 名为 "<stage>.detail" 的结构化日志，本身不参与瀑布绘制。
    解析失败或该 trace 无 span 时返回空 stages（不反噬业务）。
    """
    path = LOG_DIR / "app.log"
    if not path.exists() or not trace_id:
        return {"trace_id": trace_id, "total_ms": 0, "stages": []}
    raw: List[dict] = []
    detail_map: dict = {}  # {"rag.vector": [items...]} 召回明细，挂到同名阶段上
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("trace_id") != trace_id:
                continue
            span = rec.get("span")
            if span in (None, "request_done"):  # request_done 是整请求容器，不进瀑布
                continue
            if span.endswith(".detail"):  # 明细载荷（召回内容），只挂载不单独成条
                detail_map[span[: -len(".detail")]] = rec.get("items") or []
                continue
            dur = float(rec.get("duration_ms", 0) or 0)
            ts = rec.get("ts", "")
            start_unix = 0.0
            try:
                start_unix = datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S").timestamp() - dur / 1000.0
            except Exception:
                start_unix = 0.0
            model = rec.get("model", "") or ""
            tool = rec.get("tool", "") or ""
            if span.startswith("llm."):
                label = f"LLM 推理 ({model})" if model else "LLM 推理"
            elif span == "tool.execute":
                label = f"工具: {tool}"
            elif span.startswith("rag."):
                _rag_map = {
                    "rag.vector": "向量召回",
                    "rag.bm25": "BM25 召回",
                    "rag.rrf": "RRF 融合",
                    "rag.rerank": "Rerank 精排",
                    "rag.search": "RAG 检索总览",
                }
                label = _rag_map.get(span, span)
            else:
                label = str(span)
            raw.append({
                "label": label, "type": span, "model": model, "tool": tool,
                "start_unix": start_unix, "duration_ms": dur,
            })
    except Exception:
        log.exception("get_trace_spans failed")
        return {"trace_id": trace_id, "total_ms": 0, "stages": []}

    if not raw:
        return {"trace_id": trace_id, "total_ms": 0, "stages": []}

    # 明细日志排在 span 日志之后，必须等整个文件扫完再挂载，否则 map 还是空的
    for s in raw:
        s["items"] = detail_map.get(s["type"], [])

    # 按起始时间排序，再给 LLM 调用按出现顺序标注 决策 / 再决策 / 综合
    raw.sort(key=lambda s: s["start_unix"])
    llm_stages = [s for s in raw if s["type"].startswith("llm.")]
    n_llm = len(llm_stages)
    li = 0
    for s in raw:
        if s["type"].startswith("llm."):
            li += 1
            if n_llm == 1:
                s["label"] = f"LLM 生成 ({s['model']})" if s["model"] else "LLM 生成"
            elif li == 1:
                s["label"] = f"LLM 决策 ({s['model']})" if s["model"] else "LLM 决策"
            elif li == n_llm:
                s["label"] = f"LLM 综合生成 ({s['model']})" if s["model"] else "LLM 综合生成"
            else:
                s["label"] = f"LLM 再决策 ({s['model']})" if s["model"] else "LLM 再决策"

    min_start = min(s["start_unix"] for s in raw)
    stages = []
    for s in raw:
        offset_ms = round((s["start_unix"] - min_start) * 1000, 1)
        stages.append({
            "label": s["label"],
            "type": s["type"],
            "offset_ms": offset_ms,
            "duration_ms": round(s["duration_ms"], 1),
            "items": s.get("items", []),
        })
    total_ms = round(max((s["offset_ms"] + s["duration_ms"]) for s in stages), 1)
    return {"trace_id": trace_id, "total_ms": total_ms, "stages": stages}
