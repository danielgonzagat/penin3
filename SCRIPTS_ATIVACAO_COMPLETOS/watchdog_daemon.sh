#!/bin/bash
# 🐕 WATCHDOG DAEMON - Restart automático de processos críticos

WATCHDOG_LOG="/root/watchdog.log"
DAEMONS=(
    "brain_daemon_real_env.py:/root/UNIFIED_BRAIN:/root/brain_watchdog_restart.log"
)

log_watchdog() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$WATCHDOG_LOG"
}

log_watchdog "🐕 Watchdog iniciado"

while true; do
    for daemon_spec in "${DAEMONS[@]}"; do
        IFS=':' read -r daemon_name daemon_dir daemon_log <<< "$daemon_spec"
        
        if ! pgrep -f "$daemon_name" > /dev/null; then
            log_watchdog "⚠️ $daemon_name MORREU! Reiniciando..."
            
            cd "$daemon_dir" || continue
            nohup python3 -u "$daemon_name" >> "$daemon_log" 2>&1 &
            
            sleep 5
            
            if pgrep -f "$daemon_name" > /dev/null; then
                log_watchdog "✅ $daemon_name REINICIADO"
            else
                log_watchdog "❌ FALHA ao reiniciar $daemon_name"
            fi
        fi
    done
    
    sleep 30
done
