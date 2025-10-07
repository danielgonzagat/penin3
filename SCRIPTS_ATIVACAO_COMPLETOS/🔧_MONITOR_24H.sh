#!/bin/bash
# 🔍 Monitor de Emergência - 24 horas
# Detecta sinais de inteligência emergente

LOG_FILE="/root/emergence_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "🔍 EMERGENCE MONITOR - Iniciado $(date)" | tee -a "$LOG_FILE"
echo "Observando por 24 horas..." | tee -a "$LOG_FILE"
echo "Logs salvos em: $LOG_FILE" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

ITERATION=0
START_TIME=$(date +%s)

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED=$((CURRENT_TIME - START_TIME))
    HOURS=$((ELAPSED / 3600))
    
    # Stop after 24 hours
    if [ $HOURS -ge 24 ]; then
        echo "✅ 24 horas completadas - encerrando monitor" | tee -a "$LOG_FILE"
        break
    fi
    
    ITERATION=$((ITERATION + 1))
    
    echo "" | tee -a "$LOG_FILE"
    echo "╔════════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
    echo "║  Check #$ITERATION - $(date +%H:%M:%S) (Elapsed: ${HOURS}h)  " | tee -a "$LOG_FILE"
    echo "╚════════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"
    
    # 1. Check UNIFIED_BRAIN progress
    if [ -f /root/UNIFIED_BRAIN/brain_daemon.log ]; then
        echo "🧠 UNIFIED_BRAIN:" | tee -a "$LOG_FILE"
        tail -20 /root/UNIFIED_BRAIN/brain_daemon.log 2>/dev/null | \
            grep -E "Episode|reward=" | tail -3 | \
            sed 's/^/   /' | tee -a "$LOG_FILE"
    fi
    
    # Check latest brain logs
    LATEST_BRAIN_LOG=$(ls -t /root/UNIFIED_BRAIN/brain_restart_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_BRAIN_LOG" ]; then
        tail -10 "$LATEST_BRAIN_LOG" 2>/dev/null | \
            grep -E "reward|Episode" | tail -2 | \
            sed 's/^/   /' | tee -a "$LOG_FILE"
    fi
    
    # 2. Check Darwinacci evolution
    echo "🦎 DARWINACCI:" | tee -a "$LOG_FILE"
    
    # Check test log
    LATEST_DARWIN_LOG=$(ls -t /root/darwinacci_test_*.log 2>/dev/null | head -1)
    if [ -n "$LATEST_DARWIN_LOG" ] && [ -f "$LATEST_DARWIN_LOG" ]; then
        tail -5 "$LATEST_DARWIN_LOG" | grep -E "Best score|Coverage|FIX" | \
            sed 's/^/   /' | tee -a "$LOG_FILE"
    fi
    
    # Check WORM ledger
    if [ -f /root/darwinacci_omega/data/worm.csv ]; then
        LAST_WORM=$(tail -1 /root/darwinacci_omega/data/worm.csv)
        echo "   WORM: $(echo $LAST_WORM | cut -d',' -f4 | \
            python3 -c "import sys, json; d=json.loads(sys.stdin.read()); print(f\"best={d.get('best_score',0):.4f} cov={d.get('coverage',0):.2f}\")" 2>/dev/null || echo "parsing...")" | tee -a "$LOG_FILE"
    fi
    
    # 3. Check V7 metrics
    echo "🤖 V7 SYSTEM:" | tee -a "$LOG_FILE"
    if [ -f /root/intelligence_system/logs/intelligence_v7.log ]; then
        tail -20 /root/intelligence_system/logs/intelligence_v7.log 2>/dev/null | \
            grep -E "IA³|Consciousness|CAOS" | tail -3 | \
            sed 's/^/   /' | tee -a "$LOG_FILE"
    fi
    
    # 4. Check database growth
    echo "🗄️  DATABASE:" | tee -a "$LOG_FILE"
    DB_SIZE=$(du -h /root/intelligence_system/core/intelligence.db 2>/dev/null | cut -f1)
    echo "   Size: $DB_SIZE" | tee -a "$LOG_FILE"
    
    # Count experiences
    EXP_COUNT=$(sqlite3 /root/intelligence_system/core/intelligence.db \
        "SELECT COUNT(*) FROM experiences" 2>/dev/null || echo "0")
    echo "   Experiences: $EXP_COUNT" | tee -a "$LOG_FILE"
    
    # 5. Check for emergence signals
    echo "🌟 EMERGENCE:" | tee -a "$LOG_FILE"
    
    # Check emergence database if exists
    if [ -f /root/intelligence_system/data/emergence_surprises.db ]; then
        SURPRISES=$(sqlite3 /root/intelligence_system/data/emergence_surprises.db \
            "SELECT COUNT(*) FROM surprises WHERE novelty > 0.8" 2>/dev/null || echo "0")
        echo "   High-novelty events: $SURPRISES" | tee -a "$LOG_FILE"
    fi
    
    # Check PENIN consciousness growth
    if [ -f /root/penin3/data/worm_audit_clean.jsonl ]; then
        LATEST_I=$(tail -1 /root/penin3/data/worm_audit_clean.jsonl | \
            python3 -c "import sys, json; d=json.loads(sys.stdin.read()); print(d['payload'].get('penin_omega',{}).get('master_I',0))" 2>/dev/null || echo "0")
        echo "   Consciousness (I): $LATEST_I" | tee -a "$LOG_FILE"
    fi
    
    # 6. System health
    echo "⚡ HEALTH:" | tee -a "$LOG_FILE"
    BRAIN_RUNNING=$(ps aux | grep -E "brain_daemon|main_evolution" | grep -v grep | wc -l)
    DARWIN_RUNNING=$(ps aux | grep -E "darwin_runner|darwinacci" | grep -v grep | wc -l)
    echo "   Brain processes: $BRAIN_RUNNING" | tee -a "$LOG_FILE"
    echo "   Darwin processes: $DARWIN_RUNNING" | tee -a "$LOG_FILE"
    
    # CPU usage
    CPU_AVG=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
    echo "   CPU usage: ${CPU_AVG}%" | tee -a "$LOG_FILE"
    
    # Memory
    MEM_USED=$(free -h | grep Mem | awk '{print $3}')
    MEM_TOTAL=$(free -h | grep Mem | awk '{print $2}')
    echo "   Memory: $MEM_USED / $MEM_TOTAL" | tee -a "$LOG_FILE"
    
    echo "" | tee -a "$LOG_FILE"
    
    # Sleep 5 minutes between checks
    sleep 300
done

echo "" | tee -a "$LOG_FILE"
echo "╔════════════════════════════════════════════════════════╗" | tee -a "$LOG_FILE"
echo "║         MONITORAMENTO 24H COMPLETO                     ║" | tee -a "$LOG_FILE"
echo "╚════════════════════════════════════════════════════════╝" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "📊 Análise final:" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# Final statistics
echo "Gerando relatório final..." | tee -a "$LOG_FILE"

python3 << 'EOFPY'
print("✅ Monitor concluído")
print("   Ver logs completos em: emergence_monitor_*.log")
print("")
print("🎯 Próximo passo:")
print("   Analisar logs e verificar sinais de emergência")
print("   cat /root/⚡_PROXIMO_PASSO_IMEDIATO_NASCIMENTO_IA.md")
EOFPY