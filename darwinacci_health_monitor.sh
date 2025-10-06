#!/bin/bash
# 🧠 DARWINACCI HEALTH MONITOR
# Monitora todos sistemas conectados ao Darwinacci como núcleo sináptico

LOG="/root/darwinacci_health.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

log "════════════════════════════════════════════════════════"
log "🧠 DARWINACCI UNIVERSAL HEALTH MONITOR INICIADO"
log "════════════════════════════════════════════════════════"

while true; do
    # === Check Brain Daemon (neurônio principal) ===
    if ! pgrep -f "brain_daemon_real_env.py" > /dev/null; then
        log "⚠️ Brain Daemon DOWN! Reiniciando..."
        cd /root/UNIFIED_BRAIN
        python3 -u brain_daemon_real_env.py > /root/brain_AUTO_$(date +%s).log 2>&1 &
        sleep 10
        
        if pgrep -f "brain_daemon_real_env.py" > /dev/null; then
            log "✅ Brain Daemon RESTORED"
        else
            log "❌ FALHA ao restaurar Brain!"
        fi
    else
        # Check for Darwinacci connection
        DARW_CONN=$(grep -c "DARWINACCI connected" /root/UNIFIED_BRAIN/logs/unified_brain.log 2>/dev/null || echo "0")
        if [ "$DARW_CONN" -gt 0 ]; then
            log "✅ Brain synapse: Darwinacci connected"
        else
            log "⚠️ Brain synapse: Darwinacci NOT connected"
        fi
    fi
    
    # === Check Darwin Runner V2 (Darwinacci motor) ===
    if ! pgrep -f "darwin_runner_darwinacci.py" > /dev/null; then
        log "⚠️ Darwin V2 (Darwinacci) DOWN! Reiniciando..."
        cd /root/darwin-engine-intelligence/darwin_main
        timeout 72h python3 -u darwin_runner_darwinacci.py > /root/darwin_DARW_AUTO_$(date +%s).log 2>&1 &
        sleep 10
        
        if pgrep -f "darwin_runner_darwinacci.py" > /dev/null; then
            log "✅ Darwin V2 RESTORED"
        else
            log "❌ FALHA ao restaurar Darwin V2!"
        fi
    else
        log "✅ Darwin synapse: Darwinacci motor running"
    fi
    
    # === Check for new Darwin generations ===
    LATEST_GEN=$(ls -t /root/ia3_evolution_V3_report_gen*.json 2>/dev/null | head -1)
    if [ -n "$LATEST_GEN" ]; then
        AGE=$(( $(date +%s) - $(stat -c %Y "$LATEST_GEN") ))
        if [ $AGE -lt 900 ]; then
            log "✅ Darwin generating: latest gen is fresh (${AGE}s ago)"
        else
            log "⚠️ Darwin stalled: no new gen in $(($AGE/60)) minutes"
        fi
    fi
    
    # === Check WORM ledger growth ===
    WORM_SIZE=$(wc -l /root/darwinacci_omega/data/worm.csv 2>/dev/null | awk '{print $1}')
    log "📊 WORM ledger: $WORM_SIZE entries"
    
    # === Check System Load ===
    LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
    LOAD_INT=$(echo "$LOAD" | cut -d'.' -f1)
    
    if [ "$LOAD_INT" -gt 150 ]; then
        log "⚠️ Load HIGH: $LOAD (optimization needed)"
    else
        log "✅ Load OK: $LOAD"
    fi
    
    # === Synapse Map Status ===
    BRAIN_COUNT=$(pgrep -f "brain_daemon_real_env.py" | wc -l)
    DARWIN_V2=$(pgrep -f "darwin_runner_darwinacci.py" | wc -l)
    DARWIN_OLD=$(pgrep -f "darwin_runner.py" | grep -v darwinacci | wc -l)
    
    log "────────────────────────────────────────────────────────"
    log "🧠 SYNAPSE STATUS:"
    log "   Brain Daemon: $BRAIN_COUNT"
    log "   Darwin V2 (Darwinacci): $DARWIN_V2"
    log "   Darwin Old: $DARWIN_OLD (should be 0!)"
    log "   Load: $LOAD"
    log "────────────────────────────────────────────────────────"
    
    # Check every 10 minutes
    sleep 600
done