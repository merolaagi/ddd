#!/usr/bin/env bash
# ddd process supervisor for macOS (launchd).
# Puts the web and MCP processes under launchd with KeepAlive so the OS
# restarts them the instant they crash, and starts them on login/boot.
#
#   ./service.sh install     set up + start the supervised services (one-time)
#   ./service.sh restart      restart both services now
#   ./service.sh stop         stop both (stays stopped until `install` again)
#   ./service.sh status       show whether each is running + listening
#   ./service.sh logs         tail recent logs
#   ./service.sh uninstall    remove the services entirely
set -euo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
[ -f .env ] && { set -a; . ./.env; set +a; }
WEB_PORT="${DDD_WEB_PORT:-47520}"
MCP_PORT="${DDD_MCP_PORT:-47521}"
PY="$ROOT/.venv/bin/python"
LA="$HOME/Library/LaunchAgents"
DOMAIN="gui/$(id -u)"
WEB_LABEL="com.fueldeskpro.ddd-web"
MCP_LABEL="com.fueldeskpro.ddd-mcp"

_plist() {
  local label="$1" cmd="$2"
  mkdir -p "$LA" "$ROOT/logs"
  cat > "$LA/$label.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key><string>$label</string>
  <key>ProgramArguments</key>
  <array>
    <string>/bin/bash</string>
    <string>-lc</string>
    <string>cd '$ROOT' &amp;&amp; set -a; [ -f .env ] &amp;&amp; . ./.env; set +a; exec $cmd</string>
  </array>
  <key>WorkingDirectory</key><string>$ROOT</string>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>10</integer>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>$ROOT/logs/$label.out</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/$label.err</string>
</dict>
</plist>
PLIST
}

_boot() {
  local label="$1"
  launchctl bootout "$DOMAIN/$label" 2>/dev/null || true
  launchctl bootstrap "$DOMAIN" "$LA/$label.plist" 2>/dev/null || launchctl load -w "$LA/$label.plist"
  launchctl enable "$DOMAIN/$label" 2>/dev/null || true
}

_wait_up() {
  echo "Waiting for startup (first start compiles RDKit, ~20s)..."
  for i in $(seq 1 45); do
    curl -fs "http://127.0.0.1:$WEB_PORT/healthz" >/dev/null 2>&1 && return 0
    sleep 1
  done
  return 1
}

cmd_install() {
  for p in "$WEB_PORT" "$MCP_PORT"; do
    pids=$(lsof -ti tcp:"$p" -sTCP:LISTEN 2>/dev/null || true)
    [ -n "$pids" ] && kill $pids 2>/dev/null || true
  done
  sleep 1
  _plist "$MCP_LABEL" "$PY -m ddd.mcp_server"
  _plist "$WEB_LABEL" "$PY -m uvicorn ddd.app:app --host 127.0.0.1 --port $WEB_PORT"
  _boot "$MCP_LABEL"
  _boot "$WEB_LABEL"
  echo "Installed. ddd is now supervised by launchd (auto-restarts on crash, starts on login)."
  _wait_up || true
  cmd_status
}

cmd_restart() {
  launchctl kickstart -k "$DOMAIN/$MCP_LABEL" 2>/dev/null || _boot "$MCP_LABEL"
  launchctl kickstart -k "$DOMAIN/$WEB_LABEL" 2>/dev/null || _boot "$WEB_LABEL"
  echo "Restarted both services."
  _wait_up || true
  cmd_status
}

cmd_stop() {
  launchctl bootout "$DOMAIN/$WEB_LABEL" 2>/dev/null || true
  launchctl bootout "$DOMAIN/$MCP_LABEL" 2>/dev/null || true
  echo "Stopped. Run './service.sh install' to bring them back."
}

cmd_status() {
  for label in "$WEB_LABEL" "$MCP_LABEL"; do
    if launchctl print "$DOMAIN/$label" >/dev/null 2>&1; then
      echo "$label: registered"
    else
      echo "$label: NOT registered"
    fi
  done
  lsof -ti tcp:"$WEB_PORT" -sTCP:LISTEN >/dev/null 2>&1 && echo "web  : listening on $WEB_PORT" || echo "web  : DOWN"
  lsof -ti tcp:"$MCP_PORT" -sTCP:LISTEN >/dev/null 2>&1 && echo "mcp  : listening on $MCP_PORT" || echo "mcp  : DOWN"
  curl -fs "http://127.0.0.1:$WEB_PORT/healthz" >/dev/null 2>&1 && echo "health: ok" || echo "health: not ready"
}

cmd_logs() {
  for f in logs/$WEB_LABEL.err logs/web.log logs/$MCP_LABEL.err logs/mcp.log; do
    [ -f "$f" ] && { echo "=== $f ==="; tail -15 "$f"; }
  done
}

cmd_uninstall() {
  cmd_stop
  rm -f "$LA/$WEB_LABEL.plist" "$LA/$MCP_LABEL.plist"
  echo "Uninstalled. serve.sh will go back to its plain nohup start."
}

case "${1:-status}" in
  install) cmd_install ;;
  restart) cmd_restart ;;
  stop) cmd_stop ;;
  status) cmd_status ;;
  logs) cmd_logs ;;
  uninstall) cmd_uninstall ;;
  *) echo "usage: ./service.sh {install|restart|stop|status|logs|uninstall}"; exit 1 ;;
esac
