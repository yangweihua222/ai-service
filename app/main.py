# -*- coding: utf-8 -*-
"""FastAPI 入口：SSE 流式对话 + 工具列表 + RAG/Text2SQL 评测 + 健康。
平台（Laravel+Vue）经内部 API / SSE 调用本服务。

📌 面试考点（对应《技术点深挖备考库》）：
  · 技术点29 双栈架构（PHP 平台经内部 API + SSE 调 Python AI 服务） → 见 chat_stream()（SSE 流式）
  · 技术点4 Agent 编排入口（平台只转发鉴权，不碰 AI 逻辑）         → 见 /api/v1/chat
  · 实战① 评测驱动（RAG/Text2SQL 评测接口，回归不掉点才合并）       → 见 /api/v1/rag/eval / text2sql/eval
  · 实战⑧ 可观测/降级（每轮 steps 返回 + 评测接口带 RBAC 校验）     → 见 rag_eval()/text2sql_eval() 的 can_use 校验
"""
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# 确保从 ai-service 根目录运行时可导入 app 包
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import Depends, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.agent import AgentEngine, LLMPlanner, RulePlanner, Synthesizer
from app.config import (
    BACKEND_BASE_URL,
    BACKEND_HMAC_ENABLED,
    BACKEND_HMAC_SECRET,
    EMBEDDING_MODE,
    ENABLE_REFLECTION,
    HOST,
    LLM_MODE,
    MAX_STEPS,
    MEMORY_BACKEND,
    OPENAI_MODEL_FAST,
    PORT,
    REDIS_URL,
    validate_security_config,
    validate_redis_config,
)
from app.data import eval_text2sql, check_tenant_isolation
from app.llm import get_llm
from app.memory import MemoryStore
from app.models import ChatRequest, ChatResponse
from app import rag, observability
from app.rbac import can_use, get_request_context
from app.tools import register_tools
from app.tools_registry import registry

# 📌 P0 必修 (2026-09-08 code review): 启动期 fail-fast 校验鉴权配置
#   AUTH_MODE=jwt 但 JWT_SECRET 太短 → 直接拒绝启动（防生产忘配）
#   AUTH_MODE=dev → WARNING 不阻塞（demo/pytest 需要）
validate_security_config()
# 📌 启动期校验 Redis 后端配置（生产级别接入）
#   选了 backend=redis 但 REDIS_URL 空 → 拒绝启动（防静默退化到内存）
validate_redis_config()

# 📌 P0 必修 (2026-09-08 code review): BackendClient HMAC 签名开关校验
#   启用时 BACKEND_HMAC_SECRET 必须 ≥32 字节；Laravel 端未实现验签前可设 BACKEND_HMAC_ENABLED=false
if BACKEND_HMAC_ENABLED and BACKEND_BASE_URL:
    if not BACKEND_HMAC_SECRET or len(BACKEND_HMAC_SECRET.encode("utf-8")) < 32:
        import logging
        logging.getLogger(__name__).warning(
            "[security] BACKEND_HMAC_ENABLED=true 但 BACKEND_HMAC_SECRET 长度 < 32 字节。"
            " 已自动降级为 False（不签名直连）。Laravel 端验签未实现前可显式设 BACKEND_HMAC_ENABLED=false。"
        )

# Prometheus 暴露格式（未安装时降级为 503，不阻断业务）
try:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    _PROM_AVAILABLE = True
except Exception:  # pragma: no cover - 依赖缺失时静默降级
    _PROM_AVAILABLE = False
    generate_latest = CONTENT_TYPE_LATEST = None

register_tools()

llm = get_llm()
# 📌 模型路由（A' ①）：决策步用快模型 qwen-turbo 提速降本，综合生成仍用主模型 qwen-plus
llm_fast = get_llm(OPENAI_MODEL_FAST)
planner = LLMPlanner(llm, llm_fast) if LLM_MODE == "openai" else RulePlanner()
synthesizer = Synthesizer(llm)
# 📌 P1 修复（2026-09-11）：MemoryStore 必须显式从 env 拿 backend + redis_url
#   governance 是模块级单例自动从 env 读 → backend=redis 已生效
#   memory 这条链没有模块级单例，默认走 sqlite → 必须显式传，否则 .env 切了也走 SQLite
memory = MemoryStore(backend=MEMORY_BACKEND, redis_url=REDIS_URL)
agent = AgentEngine(llm, planner, synthesizer, registry, memory, MAX_STEPS, ENABLE_REFLECTION)

@asynccontextmanager
async def lifespan(_: FastAPI):
    # 📌 Windows + torch 稳定性：bge-m3 若在请求时首次加载，多线程下会 Segfault（pytest 单进程无感）；
    #    启动阶段（主线程、事件循环启动期）预热，类级单例 _shared_model 保证只加载一次，请求进来即用。
    #    （scripts/run_server.py 已在 uvicorn.run 之前模块级预热，此处为双保险，幂等。）
    if EMBEDDING_MODE == "bge":
        rag.get_embedder()._load()
        print("[startup] bge-m3 embedding 预热完成（进程内只加载一次）")
    yield


app = FastAPI(title="AI 教学助手 Agent 服务", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ============================================================
# 📌 实战⑧ 可观测：全链路 trace_id 中间件
#   每个请求生成（或复用上游 X-Trace-Id 透传）一个 trace_id，贯穿 agent→llm→tool 全程，
#   日志 / JSONL 审计 / OTel span 三方对齐；响应头回写 X-Trace-Id 便于前端/网关关联。
# ============================================================
@app.middleware("http")
async def trace_middleware(request: Request, call_next):
    tid = request.headers.get("X-Trace-Id") or observability.new_trace_id()
    with observability.trace_scope(tid):
        response = await call_next(request)
        response.headers["X-Trace-Id"] = tid
        return response


@app.get("/health")
async def health():
    return {"status": "ok", "llm_mode": LLM_MODE}


# ============================================================
# 📌 实战⑧ 可观测：Prometheus 指标端点 + 本地观测面板
# ============================================================
@app.get("/metrics")
async def metrics():
    if not _PROM_AVAILABLE:
        return Response("prometheus_client 未安装", status_code=503)
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/observability", response_class=HTMLResponse)
async def observability_page():
    """本地观测面板：渲染最近请求审计 + 汇总指标，排查线上问题的主抓手。"""
    entries = observability.read_audit(last_n=100)
    total = len(entries)
    ok = sum(1 for e in entries if e.get("status") != "error")
    err = total - ok
    prompt = sum(int(e.get("usage", {}).get("prompt", 0) or 0) for e in entries)
    completion = sum(int(e.get("usage", {}).get("completion", 0) or 0) for e in entries)
    total_tokens = prompt + completion
    durations = [float(e.get("duration_ms", 0) or 0) for e in entries if e.get("duration_ms")]
    avg_ms = round(sum(durations) / len(durations), 1) if durations else 0
    tool_counter: dict = {}
    for e in entries:
        for t in (e.get("tools") or []):
            tool_counter[t] = tool_counter.get(t, 0) + 1

    rows = []
    for e in reversed(entries[-50:]):
        q = (e.get("question") or "").replace("<", "&lt;")
        tools = ", ".join(e.get("tools") or []) or "-"
        u = e.get("usage", {}) or {}
        tid = e.get("trace_id", "")
        rows.append(
            f"<tr>"
            f"<td title='{tid}'>{str(tid)[:8]}</td>"
            f"<td>{e.get('ts','')}</td>"
            f"<td>{e.get('role','')}/{e.get('tenant','')}</td>"
            f"<td class='q'>{q}</td>"
            f"<td>{tools}</td>"
            f"<td>{int(u.get('prompt',0) or 0)}/{int(u.get('completion',0) or 0)}</td>"
            f"<td>{e.get('duration_ms','')}</td>"
            f"<td class='{'' if e.get('status')!='error' else 'err'}'>{e.get('status','')}</td>"
            f"<td><button class='det' data-trace='{tid}' onclick='loadTrace(this)'>详情</button></td>"
            f"</tr>"
            f"<tr class='drow' id='d_{tid}' style='display:none'>"
            f"<td colspan='9'><div class='detail' data-trace='{tid}'></div></td></tr>"
        )

    tool_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in
        sorted(tool_counter.items(), key=lambda kv: -kv[1])
    )

    html = f"""<!doctype html><html lang="zh"><head><meta charset="utf-8">
<title>AI 教学助手 · 可观测面板</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,'Microsoft YaHei',sans-serif;margin:0;padding:24px;background:#f5f6f8;color:#1f2329}}
 h1{{font-size:20px;margin:0 0 4px}} .sub{{color:#8a8f99;font-size:13px;margin-bottom:20px}}
 .cards{{display:flex;gap:14px;flex-wrap:wrap;margin-bottom:24px}}
 .card{{background:#fff;border:1px solid #e6e8eb;border-radius:10px;padding:16px 20px;min-width:130px}}
 .card .n{{font-size:24px;font-weight:700}} .card .l{{font-size:12px;color:#8a8f99;margin-top:4px}}
 .sec{{background:#fff;border:1px solid #e6e8eb;border-radius:10px;padding:16px;margin-bottom:24px;overflow:auto}}
 table{{border-collapse:collapse;width:100%;font-size:13px}}
 th,td{{border-bottom:1px solid #eee;padding:7px 9px;text-align:left;vertical-align:top}}
 th{{background:#fafbfc;color:#5a6068;font-weight:600}}
 td.q{{max-width:320px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
 td.err,.err{{color:#e54545;font-weight:600}}
 .ok{{color:#2ea121}}
 .det{{cursor:pointer;background:#4c6ef5;color:#fff;border:none;border-radius:5px;padding:3px 9px;font-size:12px}}
 .det:hover{{background:#3b5bdb}}
 .drow td{{background:#fafbfc}}
 .detail{{padding:6px 2px}}
 .rt{{cursor:pointer;color:#4c6ef5;font-size:12px;user-select:none;margin:0 0 4px 6px;display:inline-block}}
 .rt:hover{{text-decoration:underline}}
 .rec{{border-left:3px solid #2f9e44;background:#f6fbf7;border-radius:5px;padding:6px 9px;margin:5px 0 5px 6px;font-size:12px;line-height:1.6}}
 .rec .h{{color:#5a6068;font-size:11px;margin-bottom:2px}}
 .rec .h b{{color:#1f2329}}
 .rec .h code{{background:#eef1f4;border-radius:3px;padding:0 4px}}
 .rec .t{{color:#1f2329;white-space:pre-wrap;word-break:break-word}}
</style></head><body>
<h1>AI 教学助手 · 可观测面板</h1>
<div class="sub">全链路 trace_id · 请求审计 · token 成本 · 工具调用统计（数据来自 logs/audit.jsonl） · 点「详情」看该次请求的时间瀑布，定位慢在哪个阶段</div>
<div class="cards">
 <div class="card"><div class="n">{total}</div><div class="l">最近请求数</div></div>
 <div class="card"><div class="n ok">{ok}</div><div class="l">成功</div></div>
 <div class="card"><div class="n" style="color:#e54545">{err}</div><div class="l">失败</div></div>
 <div class="card"><div class="n">{total_tokens}</div><div class="l">累计 token</div></div>
 <div class="card"><div class="n">{prompt}/{completion}</div><div class="l">prompt/completion</div></div>
 <div class="card"><div class="n">{avg_ms}ms</div><div class="l">平均耗时</div></div>
</div>
<div class="sec">
 <h3 style="margin:4px 0 12px">工具调用统计</h3>
 <table><tr><th>工具</th><th>调用次数</th></tr>{tool_rows or '<tr><td colspan=2>暂无</td></tr>'}</table>
</div>
<div class="sec">
 <h3 style="margin:4px 0 12px">最近请求明细（点「详情」展开时间瀑布）</h3>
 <table>
  <tr><th>trace</th><th>时间</th><th>角色/租户</th><th>用户问题</th><th>工具</th><th>token(p/c)</th><th>耗时ms</th><th>状态</th><th>详情</th></tr>
  {''.join(rows) or '<tr><td colspan=9>暂无审计记录，先发几条消息</td></tr>'}
 </table>
</div>
"""

    WATERFALL_JS = """
<script>
function loadTrace(btn){
  var tid = btn.getAttribute('data-trace');
  var row = document.getElementById('d_'+tid);
  var box = row ? row.querySelector('.detail') : null;
  if(!row||!box) return;
  if(row.style.display==='table-row'){ row.style.display='none'; return; }
  if(box.dataset.loaded){ row.style.display='table-row'; return; }
  btn.textContent='加载中…';
  fetch('/observability/trace/'+encodeURIComponent(tid)).then(function(r){return r.json();}).then(function(d){
    box.innerHTML = renderWaterfall(d);
    box.dataset.loaded='1';
    row.style.display='table-row';
    btn.textContent='详情';
  }).catch(function(){
    box.innerHTML='<span style="color:#e54545">加载失败，请确认服务已重启</span>';
    row.style.display='table-row';
    btn.textContent='详情';
  });
}
function escHtml(s){
  return String(s==null?'':s).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}
function toggleRecall(el){
  var box = el.nextElementSibling;
  if(!box) return;
  var open = box.style.display !== 'none';
  box.style.display = open ? 'none' : 'block';
  el.textContent = (open ? '▸ ' : '▾ ') + el.getAttribute('data-label');
}
var SCORE_LABEL = {'rag.vector':'相似度','rag.bm25':'BM25 得分','rag.rrf':'RRF 融合分','rag.rerank':'精排分'};
function renderItems(s){
  if(!s.items || !s.items.length) return '';
  var sLabel = SCORE_LABEL[s.type] || '得分';
  var label = '查看召回内容（' + s.items.length + ' 条）';
  var inner = s.items.map(function(it){
    return '<div class="rec">'
      + '<div class="h"><b>#'+escHtml(it.rank)+'</b> · 文档块 <code>doc_idx='+escHtml(it.doc_idx)+'</code> · '+sLabel+' <code>'+escHtml(it.score)+'</code></div>'
      + '<div class="t">'+escHtml(it.text)+'</div>'
      + '</div>';
  }).join('');
  return '<div><div class="rt" data-label="'+label+'" onclick="toggleRecall(this)">▸ '+label+'</div>'
    + '<div style="display:none">'+inner+'</div></div>';
}
function renderWaterfall(d){
  if(!d.stages || !d.stages.length){
    return '<div style="color:#8a8f99;padding:8px">无详细 span 数据（app.log 中可能已滚动或该 trace 未记录逐环节耗时）。</div>';
  }
  var total = d.total_ms || 1;
  var colors = {'llm.stream_chat':'#4c6ef5','llm.chat':'#4c6ef5','tool.execute':'#f59f00','rag.vector':'#2f9e44','rag.bm25':'#2f9e44','rag.rrf':'#2f9e44','rag.rerank':'#2f9e44','rag.search':'#2f9e44'};
  var html = '<div style="padding:6px 2px">'
    + '<div style="font-size:12px;color:#8a8f99;margin-bottom:10px">trace_id: <code>'+escHtml(d.trace_id)+'</code> · 总耗时 <b>'+total+'ms</b> · 彩色条=各阶段耗时占比，一眼看出慢在哪；RAG 各阶段可展开看实际召回内容</div>';
  d.stages.forEach(function(s){
    var left = (s.offset_ms/total*100).toFixed(2);
    var width = (s.duration_ms/total*100).toFixed(2);
    var color = colors[s.type]||'#868e96';
    var pct = (s.duration_ms/total*100).toFixed(1);
    html += '<div style="display:flex;align-items:center;margin:6px 0;font-size:12px">'
      + '<div style="width:210px;flex:none;color:#1f2329;overflow:hidden;text-overflow:ellipsis;white-space:nowrap" title="'+escHtml(s.label)+'">'+escHtml(s.label)+'</div>'
      + '<div style="position:relative;flex:1;height:22px;background:#f0f1f3;border-radius:4px;margin:0 10px">'
      + '<div style="position:absolute;left:'+left+'%;width:'+width+'%;height:100%;background:'+color+';border-radius:4px" title="'+s.duration_ms+'ms ('+pct+'%)"></div></div>'
      + '<div style="width:140px;flex:none;text-align:right;color:#5a6068">'+s.duration_ms+'ms · '+pct+'%</div>'
      + '</div>';
    html += renderItems(s);
  });
  html += '</div>';
  return html;
}
</script>
"""
    html = html + WATERFALL_JS + "\n</body></html>"
    return HTMLResponse(html)


@app.get("/observability/trace/{trace_id}")
async def observability_trace(trace_id: str):
    """返回某次请求（trace_id）的逐环节耗时瀑布数据，供面板「详情」抽屉渲染。"""
    return JSONResponse(observability.get_trace_spans(trace_id))


@app.get("/api/v1/tools")
async def list_tools(ctx=Depends(get_request_context)):
    tools = registry.for_role(ctx.role)
    return {
        "tenant_id": ctx.tenant_id,
        "role": ctx.role.value,
        "tools": [{"name": t.name, "description": t.description} for t in tools],
    }


@app.get("/api/v1/governance")
async def governance_status():
    """LLM 服务治理实时状态：限流令牌 / Token 预算 / 熔断器三态（可观测端点）。"""
    from app.governance import llm_governance
    return llm_governance.snapshot()


@app.post("/api/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, ctx=Depends(get_request_context)):
    answer, steps = await agent.run(req.message, ctx, req.session_id or "")
    return ChatResponse(
        answer=answer,
        steps=steps,
        tenant_id=ctx.tenant_id,
        session_id=req.session_id or ctx.user_id,
        model=LLM_MODE,
    )


@app.post("/api/v1/chat/stream")
async def chat_stream(req: ChatRequest, ctx=Depends(get_request_context)):
    # 📌 技术点29 双栈架构：SSE 流式把回答逐 token 吐回前端(打字机)，平台只转发鉴权不碰 AI 逻辑
    #    （run_stream 为真正的 LLM token 级流式：决策/综合边生成边吐出，非生成完再切分）
    async def gen():
        async for ev in agent.run_stream(req.message, ctx, req.session_id or ""):
            if ev["type"] == "steps":
                yield "data: " + json.dumps(
                    {"type": "steps", "steps": [s.model_dump() for s in ev["steps"]]},
                    ensure_ascii=False,
                ) + "\n\n"
            else:  # token
                yield "data: " + json.dumps(
                    {"type": "token", "text": ev["text"]}, ensure_ascii=False
                ) + "\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@app.post("/api/v1/rag/eval")
async def rag_eval(ctx=Depends(get_request_context)):
    # 📌 实战① 评测驱动：离线召回评测接口；实战⑧ 接口带 RBAC 校验(仅知识库权限者可调)
    if not can_use(ctx.role, "query_knowledge_base"):
        raise HTTPException(status_code=403, detail="无权限")
    return rag.eval_rag()


@app.post("/api/v1/rag/eval/generation")
async def rag_eval_generation(ctx=Depends(get_request_context)):
    # 📌 生成层评测（LLM-as-judge）：检索→生成→忠实度/切题度/上下文精度 三项裁定；同 RBAC
    if not can_use(ctx.role, "query_knowledge_base"):
        raise HTTPException(status_code=403, detail="无权限")
    return await rag.eval_generation("clinical")


@app.post("/api/v1/text2sql/eval")
async def text2sql_eval(ctx=Depends(get_request_context)):
    # 返回 SQL 结构命中率(hit) + 真实执行通过率(executed)：不仅"生成对"，还要"跑得通"
    if not can_use(ctx.role, "query_teaching_data"):
        raise HTTPException(status_code=403, detail="无权限")
    return eval_text2sql(ctx.tenant_id)


@app.post("/api/v1/text2sql/tenant-isolation")
async def text2sql_isolation(ctx=Depends(get_request_context)):
    # 📌 多租户隔离的可跑证据：同一条 SQL 换租户只能看到本院校数据
    if not can_use(ctx.role, "query_teaching_data"):
        raise HTTPException(status_code=403, detail="无权限")
    return check_tenant_isolation()


@app.post("/api/v1/usage/eval")
async def usage_eval(ctx=Depends(get_request_context)):
    # 📌 与 rag/text2sql 评测对称：使用说明检索离线评测，RBAC 校验
    if not can_use(ctx.role, "query_usage_guide"):
        raise HTTPException(status_code=403, detail="无权限")
    return rag.eval_usage()


@app.post("/api/v1/usage/eval/generation")
async def usage_eval_generation(ctx=Depends(get_request_context)):
    # 📌 使用说明库生成层评测（LLM-as-judge），与临床库对称；RBAC 校验
    if not can_use(ctx.role, "query_usage_guide"):
        raise HTTPException(status_code=403, detail="无权限")
    return await rag.eval_generation("usage")


# 📌 本地问答页：挂载 static/ 作为根路径（http://localhost:8000/ 直接打开问答台）。
#    放在所有 API 路由之后注册，FastAPI 按顺序匹配，/api/* 与 /health 优先，静态目录作 fallback。
_STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
if _STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=False)
