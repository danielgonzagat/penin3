#!/bin/bash
# MONITORAMENTO 24/7 F2 STRATEGY
# Mantém F2 Strategy ativo e monitora funcionamento

LOG_FILE="/root/f2_monitor.log"
F2_SCRIPT="/root/neuronios-clone_python/f2_strategy/original/penin_f2_daemon.py"

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a $LOG_FILE
}

check_f2_status() {
    if [ -f /tmp/f2_strategy_pid ]; then
        F2_PID=$(cat /tmp/f2_strategy_pid | cut -d' ' -f2)
        if kill -0 $F2_PID 2>/dev/null; then
            return 0  # F2 está rodando
        fi
    fi
    return 1  # F2 não está rodando
}

start_f2() {
    log_message "🚀 Iniciando F2 Strategy..."
    cd /root/neuronios-clone_python
    nohup python3 $F2_SCRIPT > /dev/null 2>&1 &
    F2_PID=$!
    echo "PID: $F2_PID" > /tmp/f2_strategy_pid
    log_message "✅ F2 Strategy iniciado com PID: $F2_PID"
}

monitor_f2() {
    while true; do
        if check_f2_status; then
            F2_PID=$(cat /tmp/f2_strategy_pid | cut -d' ' -f2)
            
            # Verificar se está gerando estratégias
            RECENT_STRATEGIES=$(sqlite3 /root/penin_f2_strategy.db "SELECT COUNT(*) FROM strategies WHERE timestamp > $(date -d '2 minutes ago' +%s)" 2>/dev/null || echo "0")
            
            if [ $RECENT_STRATEGIES -gt 0 ]; then
                log_message "✅ F2 Strategy ativo (PID: $F2_PID) - $RECENT_STRATEGIES estratégias recentes"
            else
                log_message "⚠️ F2 Strategy ativo mas sem estratégias recentes"
            fi
        else
            log_message "❌ F2 Strategy parou - reiniciando..."
            start_f2
        fi
        
        sleep 60  # Verificar a cada minuto
    done
}

# Inicializar monitoramento
log_message "🔄 Iniciando monitoramento 24/7 do F2 Strategy"

if ! check_f2_status; then
    start_f2
fi

monitor_f2
