#!/bin/bash
# 🏥 HEALTH MONITOR - FASE 1
# Monitora Brain Daemon e Darwin Evolution

LOG="/root/health_monitor.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

log "════════════════════════════════════════════════════════"
log "🏥 HEALTH MONITOR INICIADO"
log "════════════════════════════════════════════════════════"

# Standardize Darwinacci WORM paths for any subprocesses
export DARWINACCI_WORM_PATH="/root/darwinacci_omega/data/worm.csv"
export DARWINACCI_WORM_HEAD="/root/darwinacci_omega/data/worm_head.txt"

# Single instance lock
LOCK="/tmp/health_monitor.lock"
exec 9>"$LOCK"
if ! flock -n 9; then
  log "⚠️ Another health monitor instance running. Exiting."
  exit 0
fi

while true; do
    # === Check Brain Daemon ===
    if ! pgrep -f "brain_daemon_real_env.py" > /dev/null; then
        log "⚠️ Brain Daemon DOWN! Reiniciando..."
        cd /root/UNIFIED_BRAIN
        python3 -u brain_daemon_real_env.py > /root/brain_AUTO_$(date +%s).log 2>&1 &
        sleep 10
        
        if pgrep -f "brain_daemon_real_env.py" > /dev/null; then
            log "✅ Brain Daemon REINICIADO"
        else
            log "❌ FALHA ao reiniciar Brain Daemon!"
        fi
    else
        # Check for errors
        ERRORS=$(grep -c "ERROR.*Inference tensors" /root/UNIFIED_BRAIN/logs/unified_brain.log 2>/dev/null || echo "0")
        if [ "$ERRORS" -gt 0 ]; then
            log "⚠️ Inference tensor error detectado! Reiniciando Brain..."
            pkill -9 -f brain_daemon_real_env.py
            sleep 3
            cd /root/UNIFIED_BRAIN
            python3 -u brain_daemon_real_env.py > /root/brain_FIXED_$(date +%s).log 2>&1 &
        else
            log "✅ Brain Daemon OK (no inference errors)"
        fi
    fi
    
    # === Check Darwin Evolution ===
    # Delegated to darwin_continuity_guardian.sh to avoid duplication/races
    :
    
    # === Check System Load ===
    LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    LOAD_INT=$(echo "$LOAD" | cut -d'.' -f1)
    
    if [ "$LOAD_INT" -gt 200 ]; then
        log "⚠️ Load HIGH: $LOAD (killing excess processes...)"
        pkill -f "monitor_continuo"
        pkill -f "watchdog_daemon"
    else
        log "✅ Load OK: $LOAD"
    fi
    
    # === Status Summary ===
    BRAIN_COUNT=$(pgrep -f "brain_daemon_real_env.py" | wc -l)
    DARWIN_COUNT=$(pgrep -f "darwin_runner.py" | wc -l)
    
    log "────────────────────────────────────────────────────────"
    log "📊 STATUS: Brain=$BRAIN_COUNT Darwin=$DARWIN_COUNT Load=$LOAD"
    log "────────────────────────────────────────────────────────"
    
    # Randomized sleep (5min ±30s)
    J=$(( (RANDOM % 61) - 30 ))
    S=$((300 + J))
    sleep "$S"
done
