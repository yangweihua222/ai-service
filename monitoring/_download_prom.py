# -*- coding: utf-8 -*-
"""从镜像源断点续传下载 Prometheus（GitHub 直连被重置，走 gh-proxy.com）。"""
import os, time, urllib.request

GH = "https://github.com/prometheus/prometheus/releases/download/v3.14.0/prometheus-3.14.0.windows-amd64.zip"
MIRRORS = ["https://gh-proxy.com/", "https://ghfast.top/"]
OUT = os.path.join(os.path.dirname(__file__), "bin")
fn = os.path.join(OUT, GH.split("/")[-1])

for attempt in range(1, 31):
    url = MIRRORS[(attempt - 1) % len(MIRRORS)] + GH
    try:
        have = os.path.getsize(fn) if os.path.exists(fn) else 0
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        r = urllib.request.urlopen(req, timeout=60)
        total_hdr = r.headers.get("Content-Range")
        total = int(total_hdr.split("/")[-1]) if total_hdr else None
        if r.status == 200 and have:
            # 服务器不支持 Range 且已有部分文件：重头下
            have = 0
        with open(fn, "wb" if have == 0 else "ab") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        size = os.path.getsize(fn)
        print(f"[attempt {attempt}] done size={size}/{total}", flush=True)
        if total is None or size >= total:
            print("[all-done]", flush=True)
            break
    except Exception as e:
        print(f"[attempt {attempt}] {MIRRORS[(attempt-1)%len(MIRRORS)]} err: {e!r}", flush=True)
        time.sleep(5)
