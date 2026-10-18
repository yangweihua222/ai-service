#!/usr/bin/env bash
# 本地无 Docker 时，用原生二进制拉起监控栈（Prometheus + Grafana）。
# 用法（ai-service 目录）：bash monitoring/start_native.sh
# 前置：monitoring/bin/prometheus/prometheus.exe 与 monitoring/bin/grafana/bin/grafana.exe 已解压好
# 注意：① 先启动 8000 AI 服务（scripts/run_server.py）；② Grafana 13 用 `grafana.exe server` 子命令；
#       ③ dashboards.yml 里 path 已写死绝对路径，挪目录需同步改。
set -e
cd "$(dirname "$0")/.."            # 切到 ai-service 根
BIN="monitoring/bin"
ROOT="$(pwd -W 2>/dev/null || pwd)"

echo "[start] prometheus (scraping :8000, web :9090) ..."
"$BIN/prometheus/prometheus.exe" --config.file="$BIN/../prometheus.yml" \
  --storage.tsdb.path="$BIN/promdata" --web.listen-address=127.0.0.1:9090 \
  > monitoring/prometheus.log 2>&1 &
echo "  prometheus pid $!"

echo "[start] grafana (web :3000, admin/admin) ..."
GF_PATHS_PROVISIONING="$ROOT/monitoring/grafana/provisioning" \
GF_SECURITY_ADMIN_USER=admin GF_SECURITY_ADMIN_PASSWORD=admin \
  "$BIN/grafana/bin/grafana.exe" server --homepath "$ROOT/$BIN/grafana" \
  > monitoring/grafana.log 2>&1 &
echo "  grafana pid $!"

echo "[done]"
echo "  Grafana:    http://localhost:3000   (admin / admin)"
echo "  Dashboard:  http://localhost:3000/d/ai-service-overview"
echo "  Prometheus: http://localhost:9090"
