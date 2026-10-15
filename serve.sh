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
sleep 3
if curl -fs "http://127.0.0.1:$WEB/api/info" > /dev/null; then echo "Web  : http://127.0.0.1:$WEB"; else echo "Web failed to start, see logs/web.log"; fi
echo "MCP  : http://127.0.0.1:$MCP/mcp  (log: logs/mcp.log)"
