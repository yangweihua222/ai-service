# -*- coding: utf-8 -*-
"""生成 Grafana 看板 JSON（ai-service 指标看板）。

覆盖 3 个 RAG 核心指标 + 请求/延迟/Token/工具调用，与 observability.py 中的
Prometheus 指标命名严格对应，可直接被 Grafana provisioning 加载。
"""
import json
import os

PANELS = [
    {
        "id": 1, "title": "RAG 检索总耗时 P95 (clinical)", "type": "timeseries",
        "datasource": {"type": "prometheus", "uid": "prom"},
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0},
        "fieldConfig": {"defaults": {"unit": "s"}, "overrides": []},
        "targets": [{
            "expr": 'histogram_quantile(0.95, sum(rate(ai_rag_duration_seconds_bucket{kb="clinical",stage="total"}[5m])) by (le))',
            "legendFormat": "P95"
        }],
    },
    {
        "id": 2, "title": "RAG 各阶段平均耗时 (ms)", "type": "barchart",
        "datasource": {"type": "prometheus", "uid": "prom"},
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0},
        "fieldConfig": {"defaults": {"unit": "ms"}, "overrides": []},
        "targets": [{
            "expr": '((sum(rate(ai_rag_duration_seconds_sum{kb="clinical"}[5m])) by (stage)) / (sum(rate(ai_rag_duration_seconds_count{kb="clinical"}[5m])) by (stage))) * 1000',
            "legendFormat": "{{stage}}"
        }],
    },
    {
        "id": 3, "title": "RAG 平均召回条数 (clinical)", "type": "stat",
        "datasource": {"type": "prometheus", "uid": "prom"},
        "gridPos": {"h": 8, "w": 8, "x": 0, "y": 8},
        "fieldConfig": {"defaults": {"unit": "short"}, "overrides": []},
        "targets": [{
            "expr": 'sum(rate(ai_rag_recall_count_sum{kb="clinical"}[5m])) / sum(rate(ai_rag_recall_count_count{kb="clinical"}[5m]))',
            "legendFormat": "召回条数"
        }],
    },
    {
        "id": 4, "title": "RAG Top1 命中速率 (clinical)", "type": "stat",
        "datasource": {"type": "prometheus", "uid": "prom"},
        "gridPos": {"h": 8, "w": 8, "x": 8, "y": 8},
        "fieldConfig": {"defaults": {"unit": "ops"}, "overrides": []},
        "targets": [{
            "expr": 'sum(rate(ai_rag_top1_hit_total{kb="clinical"}[5m]))',
            "legendFormat": "命中/s"
        }],
    },
    {
        "id": 5, "title": "工具调用次数 (by tool)", "type": "barchart",
        "datasource": {"type": "prometheus", "uid": "prom"},
        "gridPos": {"h": 8, "w": 8, "x": 16, "y": 8},
        "fieldConfig": {"defaults": {"unit": "short"}, "overrides": []},
        "targets": [{
            "expr": 'sum(ai_tool_calls_total) by (tool)',
            "legendFormat": "{{tool}}"
        }],
    },
    {
        "id": 6, "title": "请求 QPS", "type": "timeseries",
        "datasource": {"type": "prometheus", "uid": "prom"},
        "gridPos": {"h": 8, "w": 12, "x": 0, "y": 16},
        "fieldConfig": {"defaults": {"unit": "ops"}, "overrides": []},
        "targets": [{
            "expr": 'sum(rate(ai_requests_total[5m]))',
            "legendFormat": "QPS"
        }],
    },
    {
        "id": 7, "title": "请求延迟 P95", "type": "timeseries",
        "datasource": {"type": "prometheus", "uid": "prom"},
        "gridPos": {"h": 8, "w": 12, "x": 12, "y": 16},
        "fieldConfig": {"defaults": {"unit": "s"}, "overrides": []},
        "targets": [{
            "expr": 'histogram_quantile(0.95, sum(rate(ai_request_duration_seconds_bucket[5m])) by (le))',
            "legendFormat": "P95"
        }],
    },
    {
        "id": 8, "title": "Token 消耗速率 (prompt/completion)", "type": "timeseries",
        "datasource": {"type": "prometheus", "uid": "prom"},
        "gridPos": {"h": 8, "w": 24, "x": 0, "y": 24},
        "fieldConfig": {"defaults": {"unit": "short"}, "overrides": []},
        "targets": [{
            "expr": 'sum(rate(ai_llm_tokens_total[5m])) by (kind)',
            "legendFormat": "{{kind}}"
        }],
    },
]

DASH = {
    "annotations": {"list": []},
    "editable": True,
    "graphTooltip": 0,
    "schemaVersion": 39,
    "tags": ["ai-service", "rag", "observability"],
    "templating": {"list": []},
    "time": {"from": "now-1h", "to": "now"},
    "timezone": "browser",
    "refresh": "10s",
    "title": "AI 教学助手 · 指标看板",
    "uid": "ai-service-overview",
    "version": 1,
    "panels": PANELS,
}

out_dir = os.path.join(os.path.dirname(__file__), "grafana", "provisioning", "dashboards")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "ai-service.json")
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(DASH, f, ensure_ascii=False, indent=2)
print("dashboard written ->", out_path)
