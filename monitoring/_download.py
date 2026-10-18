# -*- coding: utf-8 -*-
"""下载 Prometheus / Grafana Windows 原生二进制（无 Docker 时本地拉起监控栈用）。"""
import os
import urllib.request

URLS = {
    "prometheus": "https://github.com/prometheus/prometheus/releases/download/v3.14.0/prometheus-3.14.0.windows-amd64.zip",
    "grafana": "https://dl.grafana.com/oss/release/grafana-13.2.1.windows-amd64.zip",
}

OUT = os.path.join(os.path.dirname(__file__), "bin")
os.makedirs(OUT, exist_ok=True)

for name, url in URLS.items():
    fn = os.path.join(OUT, url.split("/")[-1])
    print(f"[download] {name} <- {url}", flush=True)
    try:
        urllib.request.urlretrieve(url, fn)
        print(f"[done] {name} size={os.path.getsize(fn)} path={fn}", flush=True)
    except Exception as e:
        print(f"[error] {name}: {e!r}", flush=True)
print("[all-done]", flush=True)
