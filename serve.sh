#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
if [ -f .env ]; then set -a; . ./.env; set +a; fi
WEB=${DDD_WEB_PORT:-47520}
MCP=${DDD_MCP_PORT:-47521}
mkdir -p logs
for p in "$WEB" "$MCP"; do
  pids=$(lsof -ti tcp:"$p" -sTCP:LISTEN || true)
  if [ -n "$pids" ]; then kill $pids || true; fi
done
sleep 1
source .venv/bin/activate
nohup python -m ddd.mcp_server > logs/mcp.log 2>&1 &
nohup python -m uvicorn ddd.app:app --host 127.0.0.1 --port "$WEB" > logs/web.log 2>&1 &
echo "Starting ddd (first start can take ~20s while Python compiles RDKit and scikit-learn) ..."
ok=0
for i in $(seq 1 45); do
  if curl -fs "http://127.0.0.1:$WEB/healthz" > /dev/null; then ok=1; break; fi
  if ! pgrep -f "uvicorn ddd.app:app" > /dev/null; then break; fi
  sleep 1
done
if [ "$ok" = 1 ]; then
  echo "Web  : http://127.0.0.1:$WEB"
else
  echo "Web failed to start. Last lines of logs/web.log:"
  tail -25 logs/web.log
fi
echo "MCP  : http://127.0.0.1:$MCP/mcp  (log: logs/mcp.log)"
