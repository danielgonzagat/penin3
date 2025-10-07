#!/usr/bin/env bash
set -euo pipefail

ORCH_SCRIPT="/tmp/unified_agi_orchestrator.py"
OUT_FILE="/root/unified_agi_orchestrator.out"
PID_FILE="/tmp/unified_agi_orchestrator.pid"

log() { echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*"; }

status() {
  if pgrep -f "$ORCH_SCRIPT" >/dev/null 2>&1; then
    log "Orchestrator RUNNING (pids: $(pgrep -f "$ORCH_SCRIPT" | tr '\n' ' '))"
  else
    log "Orchestrator NOT RUNNING"
  fi
}

stop_orchestrator() {
  if pgrep -f "$ORCH_SCRIPT" >/dev/null 2>&1; then
    local pids
    pids=$(pgrep -f "$ORCH_SCRIPT" | tr '\n' ' ')
    log "Stopping orchestrator: $pids"
    kill -TERM $pids || true
    for i in {1..20}; do
      sleep 0.5
      if ! pgrep -f "$ORCH_SCRIPT" >/dev/null 2>&1; then
        log "Stopped gracefully"
        break
      fi
    done
    if pgrep -f "$ORCH_SCRIPT" >/dev/null 2>&1; then
      log "Force killing orchestrator"
      kill -KILL $pids || true
    fi
  else
    log "No orchestrator process found"
  fi
}

start_orchestrator() {
  log "Starting orchestrator"
  nohup python3 "$ORCH_SCRIPT" > "$OUT_FILE" 2>&1 &
  echo $! > "$PID_FILE"
  sleep 1
  status
  log "Tail output: $OUT_FILE (use: tail -n 100 -f $OUT_FILE)"
}

case "${1:-}" in
  status)
    status ;;
  stop)
    stop_orchestrator ;;
  start)
    start_orchestrator ;;
  restart|"")
    stop_orchestrator
    start_orchestrator ;;
  *)
    echo "Usage: $0 [status|stop|start|restart]" ; exit 1 ;;
 esac
