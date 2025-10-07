#!/bin/bash
# 🔄 MONITOR CONTÍNUO 24/7 - Roda SOZINHO sem precisar do Assistant
# Este script monitora, detecta problemas e toma ações automáticas

LOG_FILE="/root/monitor_continuo.log"
ALERT_FILE="/root/ALERTAS_CRITICOS.txt"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

alerta_critico() {
    echo "=== ALERTA CRÍTICO ===" >> "$ALERT_FILE"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" >> "$ALERT_FILE"
    echo "===================" >> "$ALERT_FILE"
    log "🚨 ALERTA CRÍTICO: $1"
}

log "════════════════════════════════════════════════"
log "🤖 MONITOR CONTÍNUO INICIADO"
log "════════════════════════════════════════════════"

CICLO=0

while true; do
    CICLO=$((CICLO + 1))
    
    # === CHECK 1: Brain daemon está rodando? ===
    BRAIN_COUNT=$(pgrep -f "brain_daemon_real_env.py" | wc -l)
    
    if [ "$BRAIN_COUNT" -eq 0 ]; then
        alerta_critico "Brain daemon MORREU! Tentando restart..."
        cd /root/UNIFIED_BRAIN
        nohup python3 -u brain_daemon_real_env.py >> /root/brain_auto_restart.log 2>&1 &
        sleep 10
        
        # Verificar se restart funcionou
        if pgrep -f "brain_daemon_real_env.py" > /dev/null; then
            log "✅ Brain daemon REINICIADO com sucesso"
        else
            alerta_critico "❌ FALHA AO REINICIAR brain daemon!"
        fi
    elif [ "$BRAIN_COUNT" -gt 2 ]; then
        alerta_critico "⚠️ MÚLTIPLOS daemons ($BRAIN_COUNT)! Possível leak."
    else
        log "✅ Brain daemon OK ($BRAIN_COUNT processos)"
    fi
    
    # === CHECK 2: CPU/RAM não estão em overload? ===
    CPU_PCT=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    RAM_PCT=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
    
    if (( $(echo "$CPU_PCT > 95.0" | bc -l) )); then
        alerta_critico "CPU overload: ${CPU_PCT}%"
    fi
    
    if [ "$RAM_PCT" -gt 90 ]; then
        alerta_critico "RAM crítica: ${RAM_PCT}%"
    fi
    
    log "📊 Recursos: CPU=${CPU_PCT}%, RAM=${RAM_PCT}%"
    
    # === CHECK 3: Últimos logs têm erros críticos? ===
    if [ -f /root/brain_v3_FIXED.log ]; then
        ERROS_RECENTES=$(tail -100 /root/brain_v3_FIXED.log | grep -c "ERROR")
        if [ "$ERROS_RECENTES" -gt 10 ]; then
            alerta_critico "Muitos erros recentes: $ERROS_RECENTES nos últimos 100 logs"
        fi
    fi
    
    # === CHECK 4: Disco não está cheio? ===
    DISCO_PCT=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
    if [ "$DISCO_PCT" -gt 90 ]; then
        alerta_critico "Disco cheio: ${DISCO_PCT}%"
    fi
    
    # === CHECK 5: Sinais de emergência? ===
    if [ -f /root/intelligence_system/data/emergence_surprises.db ]; then
        SURPRISES=$(sqlite3 /root/intelligence_system/data/emergence_surprises.db "SELECT COUNT(*) FROM surprises WHERE sigma > 5.0" 2>/dev/null || echo "0")
        if [ "$SURPRISES" -gt 10 ]; then
            log "🌟 EMERGÊNCIA DETECTADA: $SURPRISES surprises > 5σ!"
            echo "🌟 POSSÍVEL INTELIGÊNCIA EMERGENTE: $SURPRISES high-sigma surprises" >> "$ALERT_FILE"
        fi
    fi
    
    # === STATUS A CADA 10 CICLOS ===
    if [ $((CICLO % 10)) -eq 0 ]; then
        log "════════════════════════════════════════════════"
        log "📊 STATUS (Ciclo $CICLO):"
        log "   Brain: $BRAIN_COUNT processos"
        log "   CPU: ${CPU_PCT}%"
        log "   RAM: ${RAM_PCT}%"
        log "   Disco: ${DISCO_PCT}%"
        log "════════════════════════════════════════════════"
    fi
    
    # Aguardar 60 segundos antes do próximo check
    sleep 60
done
