#!/bin/bash
# 🧬 DARWIN CONTINUITY GUARDIAN
# Garante que Darwin Evolution roda continuamente (7 dias+)

LOG_FILE="/root/darwin_guardian.log"
DLOCK="/tmp/darwin_guardian.lock"
DARWIN_RUNNER="/root/darwin-engine-intelligence/darwin_main/darwin_runner.py"
DARWIN_LOG="/root/darwin_evolution_continuous.log"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

log "═══════════════════════════════════════════════════════════"
log "🧬 DARWIN CONTINUITY GUARDIAN INICIADO"
log "═══════════════════════════════════════════════════════════"

# Standardize Darwinacci WORM paths for any subprocesses
export DARWINACCI_WORM_PATH="/root/darwinacci_omega/data/worm.csv"
export DARWINACCI_WORM_HEAD="/root/darwinacci_omega/data/worm_head.txt"

# Ensure single instance using flock
exec 9>"$DLOCK"
if ! flock -n 9; then
  log "⚠️ Another guardian instance is running. Exiting."
  exit 0
fi

while true; do
    # Check if Darwin is running (avoid duplicate managers: stop health_monitor's Darwin control)
    if ! pgrep -f "darwin_runner.py" > /dev/null; then
        log "⚠️ Darwin NÃO está rodando! Iniciando..."
        
        cd /root/darwin-engine-intelligence/darwin_main
        nohup python3 -u darwin_runner.py >> "$DARWIN_LOG" 2>&1 &
        
        sleep 10
        
        if pgrep -f "darwin_runner.py" > /dev/null; then
            log "✅ Darwin REINICIADO com sucesso"
        else
            log "❌ FALHA ao reiniciar Darwin!"
        fi
    else
        log "✅ Darwin rodando normalmente"
    fi
    
    # Randomized sleep to avoid sync with other monitors (5min ±30s)
    J=$(( (RANDOM % 61) - 30 ))
    S=$((300 + J))
    sleep "$S"
done
