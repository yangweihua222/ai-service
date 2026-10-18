# -*- coding: utf-8 -*-
"""断点续传 + 重试下载 Grafana Windows 二进制（467MB，服务器支持 Range）。"""
import os, time, urllib.request

URL = "https://dl.grafana.com/oss/release/grafana-13.2.1.windows-amd64.zip"
OUT = os.path.join(os.path.dirname(__file__), "bin")
fn = os.path.join(OUT, URL.split("/")[-1])
TOTAL = 467496351

for attempt in range(1, 21):
    try:
        have = os.path.getsize(fn) if os.path.exists(fn) else 0
        if have >= TOTAL:
            print(f"[attempt {attempt}] already complete size={have}", flush=True)
            break
        req = urllib.request.Request(URL, headers={"User-Agent": "Mozilla/5.0"})
        if have:
            req.add_header("Range", f"bytes={have}-")
        r = urllib.request.urlopen(req, timeout=60)
        with open(fn, "ab" if have else "wb") as f:
            while True:
                chunk = r.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        size = os.path.getsize(fn)
        print(f"[attempt {attempt}] done size={size}/{TOTAL}", flush=True)
        if size >= TOTAL:
            print("[all-done]", flush=True)
            break
    except Exception as e:
        print(f"[attempt {attempt}] err: {e!r}", flush=True)
        time.sleep(5)
