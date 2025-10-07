#!/bin/bash
# 🐕 WATCHDOG - Auto-restart crashed daemons
# BLOCO 2 - TAREFA 24

# Daemon specs: "script_name:log_file"
DAEMONS=(
    "brain_daemon_real_env.py:/root/brain_test.log"
    "EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py:/root/system_connector.out"
    "META_LEARNER_REALTIME.py:/root/meta_learner.out"
    "CROSS_POLLINATION_AUTO.py:/root/cross_pollination.out"
    "DYNAMIC_FITNESS_ENGINE.py:/root/dynamic_fitness.out"
    "V7_DARWIN_REALTIME_BRIDGE.py:/root/v7_darwin_bridge.out"
)

LOG="/root/watchdog.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"
}

log "🐕 Watchdog started (PID $$)"

while true; do
    for daemon_spec in "${DAEMONS[@]}"; do
        IFS=':' read -r daemon logfile <<< "$daemon_spec"
        
        # Check if running
        if ! pgrep -f "$daemon" > /dev/null 2>&1; then
            log "⚠️ $daemon crashed, restarting..."
            
            # Find full path
            full_path="/root/$daemon"
            if [ ! -f "$full_path" ]; then
                full_path="/root/UNIFIED_BRAIN/$daemon"
            fi
            
            if [ -f "$full_path" ]; then
                # Restart
                nohup python3 -u "$full_path" > "$logfile" 2>&1 &
                PID=$!
                log "✅ $daemon restarted (PID $PID)"
                
                # Wait a bit to see if it stays alive
                sleep 5
                if ps -p $PID > /dev/null 2>&1; then
                    log "   Process stable"
                else
                    log "   ❌ Process died immediately - check logs: $logfile"
                fi
            else
                log "   ❌ File not found: $full_path"
            fi
        fi
    done
    
    # Check every 30s
    sleep 30
done
