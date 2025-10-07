#!/bin/bash
# 🌟 MONITOR DE EMERGÊNCIA - Detecta sinais de inteligência real

EMERGENCE_LOG="/root/emergence_signals.log"

log_emergence() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$EMERGENCE_LOG"
}

log_emergence "═══════════════════════════════════════════════════════════"
log_emergence "🌟 MONITOR DE EMERGÊNCIA INICIADO"
log_emergence "═══════════════════════════════════════════════════════════"

while true; do
    # === SINAL 1: High-sigma surprises ===
    SURPRISES_9S=$(sqlite3 /root/intelligence_system/data/emergence_surprises.db \
        "SELECT COUNT(*) FROM surprises WHERE sigma > 9.0" 2>/dev/null || echo "0")
    
    if [ "$SURPRISES_9S" -gt 0 ]; then
        log_emergence "🚨 EMERGÊNCIA FORTE: $SURPRISES_9S surprises >9σ detectadas!"
        echo "INTELIGÊNCIA EMERGENTE POSSÍVEL" >> /root/ALERTAS_CRITICOS.txt
        
        # Get details
        sqlite3 /root/intelligence_system/data/emergence_surprises.db \
            "SELECT metric_name, sigma, actual_value FROM surprises WHERE sigma > 9.0 ORDER BY sigma DESC LIMIT 3" \
            >> "$EMERGENCE_LOG" 2>/dev/null
    fi
    
    # === SINAL 2: Aprendizado acelerado ===
    BRAIN_METRICS=$(sqlite3 /root/intelligence_system/data/intelligence.db \
        "SELECT episode, energy FROM brain_metrics ORDER BY timestamp DESC LIMIT 5" 2>/dev/null)
    
    if [ -n "$BRAIN_METRICS" ]; then
        # Get latest energy
        LATEST_ENERGY=$(echo "$BRAIN_METRICS" | sed -n '1p' | awk -F'|' '{print $2}')
        
        if (( $(echo "$LATEST_ENERGY > 0.25" | bc -l 2>/dev/null || echo "0") )); then
            log_emergence "📈 Aprendizado forte: energy=$LATEST_ENERGY (>25% de max)"
        fi
    fi
    
    # === SINAL 3: Darwin fitness crescente ===
    LATEST_DARWIN_GEN=$(ls -t /root/ia3_evolution_V3_report_gen*.json 2>/dev/null | sed -n '1p')
    
    if [ -n "$LATEST_DARWIN_GEN" ]; then
        AVG_FITNESS=$(python3 -c "import json; print(json.load(open('$LATEST_DARWIN_GEN'))['current_population'])" 2>/dev/null || echo "0")
        
        if [ "$AVG_FITNESS" -gt 18000 ]; then
            log_emergence "🧬 Darwin evoluindo: population=$AVG_FITNESS (>18K neurons!)"
        fi
    fi
    
    # === SINAL 4: CartPole solved? ===
    if [ -f /root/brain_FIXED_TELEMETRY_v2.log ]; then
        SOLVED=$(grep "reward=" /root/brain_FIXED_TELEMETRY_v2.log | tail -20 | \
                 grep -c "reward=2[0-9][0-9]\.0\|reward=1[5-9][0-9]\.0")
        
        if [ "$SOLVED" -gt 10 ]; then
            log_emergence "🎊 POSSÍVEL SOLUÇÃO: CartPole consistently >150!"
            echo "CARTPOLE POSSIVELMENTE SOLVED" >> /root/ALERTAS_CRITICOS.txt
        fi
    fi
    
    # === STATUS a cada 30 ciclos ===
    CICLO=$((CICLO + 1))
    if [ $((CICLO % 30)) -eq 0 ]; then
        log_emergence "────────────────────────────────────────────────────────────"
        log_emergence "📊 STATUS (Ciclo $CICLO):"
        log_emergence "   Surprises >9σ: $SURPRISES_9S"
        log_emergence "   Latest energy: ${LATEST_ENERGY:-N/A}"
        log_emergence "   Darwin pop: ${AVG_FITNESS:-N/A}"
        log_emergence "────────────────────────────────────────────────────────────"
    fi
    
    # Randomized sleep (2min ±15s)
    J=$(( (RANDOM % 31) - 15 ))
    S=$((120 + J))
    sleep "$S"  # Check every ~2 minutes
done
