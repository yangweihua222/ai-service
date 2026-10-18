# -*- coding: utf-8 -*-
"""本地桩后端：模仿医院 Laravel 平台的 /api/dify/tools/* 接口。

用途：在本地验证「ai-service → 真实 HTTP → 后端工具」链路，无需连你生产服务器、无需 API Key。
返回格式对齐 Laravel：Helper::SuccessJsonResponse($data) → {"code":0,"msg":"success","data":<payload>}

运行：
    uvicorn scripts.stub_backend:app --port 9000
然后让 ai-service 指向它：
    BACKEND_BASE_URL=http://127.0.0.1:9000/api uvicorn app.main:app --port 8000
"""
import json
import sys
from pathlib import Path

from fastapi import FastAPI, Request

# 允许脚本直接 `python scripts/stub_backend.py` 或 `uvicorn scripts.stub_backend:app`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app = FastAPI(title="Stub Laravel Backend (dify tools)")


def ok(data):
    return {"code": 0, "msg": "success", "data": data}


@app.post("/api/dify/tools/submit_leave")
async def submit_leave(req: Request):
    body = await req.json()
    print(f"[STUB] submit_leave 收到调用 -> {json.dumps(body, ensure_ascii=False)}", flush=True)
    return ok({"result": "请假成功", "received": body})


@app.post("/api/dify/tools/get_duty_schedule")
async def get_duty_schedule(req: Request):
    body = await req.json()
    print(f"[STUB] get_duty_schedule 收到调用 -> {json.dumps(body, ensure_ascii=False)}", flush=True)
    return ok({"值班日期": ["2026-09-03", "2026-09-05"], "值班科室": ["骨科", "急诊科"]})


@app.post("/api/dify/tools/get_absent_times")
async def get_absent_times(req: Request):
    body = await req.json()
    print(f"[STUB] get_absent_times 收到调用 -> {json.dumps(body, ensure_ascii=False)}", flush=True)
    return ok({"count": 0})


@app.post("/api/dify/tools/get_teaching_event")
async def get_teaching_event(req: Request):
    body = await req.json()
    print(f"[STUB] get_teaching_event 收到调用 -> {json.dumps(body, ensure_ascii=False)}", flush=True)
    return ok([{"title": "教学查房", "start_time": "2026-09-03 09:00"}])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=9000)
