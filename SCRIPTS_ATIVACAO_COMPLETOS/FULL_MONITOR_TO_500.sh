#!/bin/bash
# Monitor completo até completar 500 ciclos
LOG=/root/500_cycles_output.log
EVIDENCE=/root/HOTPATCH_EVIDENCE.log

echo "═══════════════════════════════════════════════════════════════" | tee -a $EVIDENCE
echo "🔍 MONITOR COMPLETO - Até Ciclo 1560" | tee -a $EVIDENCE
echo "Início: $(date)" | tee -a $EVIDENCE
echo "═══════════════════════════════════════════════════════════════" | tee -a $EVIDENCE
echo "" | tee -a $EVIDENCE

# Ciclo atual
CURRENT=$(tail -100 $LOG | grep "CYCLE" | tail -1 | grep -oP "CYCLE \K\d+")
echo "📍 Ciclo atual: $CURRENT" | tee -a $EVIDENCE
echo "🎯 Ciclo alvo: 1560" | tee -a $EVIDENCE
echo "⏳ Faltam: $((1560 - CURRENT)) ciclos" | tee -a $EVIDENCE
echo "" | tee -a $EVIDENCE

# Captura TUDO relevante
tail -f $LOG 2>&1 | grep --line-buffered -E "🔥|🫀|♻️|🔁|🧠 optimizer|🔴|📊 API|HOTPATCH|CYCLE|reloaded:|swapped|applied:|unknown_op:|Provider.*DOWN|circuit breaker" | while read line; do
    echo "$line" | tee -a $EVIDENCE
    
    # Verifica se completou
    if echo "$line" | grep -q "CYCLE 1560"; then
        echo "" | tee -a $EVIDENCE
        echo "🎉 COMPLETOU 500 CICLOS!" | tee -a $EVIDENCE
        echo "═══════════════════════════════════════════════════════════════" | tee -a $EVIDENCE
        break
    fi
done
