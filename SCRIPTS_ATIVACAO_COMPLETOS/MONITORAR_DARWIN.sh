#!/bin/bash
# MONITOR DE EVOLUÇÃO DARWIN EM TEMPO REAL
# =========================================
# Acompanha gerações, fitness, checkpoints

echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║  🧬 MONITOR DARWIN EVOLUTION - TEMPO REAL             ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo

if ! pgrep -f "run_emergence_blocks_STORM" > /dev/null; then
    echo "❌ Darwin não está rodando!"
    echo
    echo "Para iniciar:"
    echo "  bash /root/FIX_PRIORIDADE_1_INICIAR_DARWIN.sh"
    exit 1
fi

echo "✅ Darwin ATIVO"
echo

# Mostrar progresso atual
echo "📊 PROGRESSO ATUAL"
echo "=================="

# Última geração
LAST_GEN=$(tail -500 /root/darwin_STORM.log 2>/dev/null | grep -oP "🧬 Geração \K\d+" | tail -1)
if [ -n "$LAST_GEN" ]; then
    echo "  Geração: $LAST_GEN/100"
    PERCENT=$((LAST_GEN * 100 / 100))
    echo "  Progresso: ${PERCENT}%"
else
    echo "  Geração: Calculando..."
fi
echo

# Melhor fitness recente
echo "🏆 MELHOR FITNESS"
echo "================="
BEST_FITNESS=$(tail -1000 /root/darwin_STORM.log 2>/dev/null | grep -oP "Best fitness: \K[\d\.]+" | tail -1)
if [ -n "$BEST_FITNESS" ]; then
    echo "  Fitness: $BEST_FITNESS"
else
    echo "  Fitness: Calculando..."
fi
echo

# Checkpoints
echo "🧬 CHECKPOINTS GERADOS"
echo "======================"
TOTAL_CP=$(ls /root/intelligence_system/models/darwin_checkpoints/*.pt 2>/dev/null | wc -l)
RECENT_CP=$(find /root/intelligence_system/models/darwin_checkpoints/ -name "*.pt" -mmin -30 2>/dev/null | wc -l)
echo "  Total: $TOTAL_CP checkpoints"
echo "  Últimos 30min: $RECENT_CP novos"
echo

# Top 3 mais recentes
echo "📁 Checkpoints recentes:"
ls -lht /root/intelligence_system/models/darwin_checkpoints/*.pt 2>/dev/null | head -3 | while read line; do
    FILE=$(echo "$line" | awk '{print $9}')
    SIZE=$(echo "$line" | awk '{print $5}')
    DATE=$(echo "$line" | awk '{print $6" "$7" "$8}')
    BASENAME=$(basename "$FILE")
    echo "  • $BASENAME ($SIZE, $DATE)"
done
echo

# Log em tempo real
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 LOG EM TEMPO REAL (últimas 20 linhas)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -n 20 /root/darwin_STORM.log
echo
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Para ver log completo em tempo real:"
echo "  tail -f /root/darwin_STORM.log"
echo
echo "Para ver apenas gerações:"
echo "  tail -f /root/darwin_STORM.log | grep -E 'Geração|fitness'"
