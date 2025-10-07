#!/bin/bash
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     🔬 AUDITORIA REAL - SISTEMA V7.0 EM EXECUÇÃO              ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# 1. Estado do Processo
echo "1️⃣ PROCESSO:"
PID=$(cat /root/500_cycles.pid 2>/dev/null)
if [ -z "$PID" ]; then
    echo "   ❌ PID não encontrado"
    exit 1
fi

if ps -p $PID > /dev/null 2>&1; then
    echo "   ✅ VIVO - PID: $PID"
    ps -p $PID -o pid,comm,%cpu,%mem,time,etime | tail -1 | awk '{print "   CPU:", $3"% | MEM:", $4"% | Time:", $5, "| Uptime:", $6}'
else
    echo "   ❌ MORTO"
    exit 1
fi

echo ""

# 2. Logs Atualizando?
echo "2️⃣ LOGS:"
LOG=/root/500_cycles_output.log
if [ -f "$LOG" ]; then
    LAST_MOD=$(stat -c %Y "$LOG")
    NOW=$(date +%s)
    AGE=$((NOW - LAST_MOD))
    
    echo "   Arquivo: $LOG"
    echo "   Tamanho: $(du -h "$LOG" | cut -f1)"
    echo "   Última atualização: ${AGE}s atrás"
    
    if [ $AGE -lt 60 ]; then
        echo "   ✅ ATUALIZANDO (< 60s)"
    else
        echo "   ⚠️  PARADO há ${AGE}s"
    fi
else
    echo "   ❌ Log não existe"
fi

echo ""

# 3. Ciclo Atual
echo "3️⃣ PROGRESSO:"
CURRENT_CYCLE=$(tail -100 "$LOG" 2>/dev/null | grep -oP "CYCLE \K\d+" | tail -1)
if [ ! -z "$CURRENT_CYCLE" ]; then
    echo "   Ciclo atual: $CURRENT_CYCLE"
    echo "   Meta: ~1560 (500 ciclos desde 1060)"
    REMAINING=$((1560 - CURRENT_CYCLE))
    echo "   Restantes: $REMAINING"
    echo "   Progresso: $((100 * (500 - REMAINING) / 500))%"
else
    echo "   ❌ Não detectado"
fi

echo ""

# 4. Métricas Recentes
echo "4️⃣ MÉTRICAS (últimos 10 ciclos):"
tail -500 "$LOG" 2>/dev/null | grep -E "IA³ Score:|Test:|Last:|Avg\(100\):" | tail -10 | while read line; do
    echo "   $line"
done

echo ""

# 5. Erros/Warnings Recentes
echo "5️⃣ PROBLEMAS (últimos 100 linhas):"
ERROR_COUNT=$(tail -100 "$LOG" 2>/dev/null | grep -c "ERROR")
WARN_COUNT=$(tail -100 "$LOG" 2>/dev/null | grep -c "WARNING")
echo "   Errors: $ERROR_COUNT"
echo "   Warnings: $WARN_COUNT"

if [ $WARN_COUNT -gt 0 ]; then
    echo ""
    echo "   Top 5 warnings:"
    tail -500 "$LOG" 2>/dev/null | grep "WARNING" | cut -d'-' -f4- | sort | uniq -c | sort -rn | head -5 | sed 's/^/   /'
fi

echo ""

# 6. APIs Funcionando?
echo "6️⃣ APIs:"
API_SUCCESS=$(tail -200 "$LOG" 2>/dev/null | grep -c "✅ Consulted")
API_FAIL=$(tail -200 "$LOG" 2>/dev/null | grep -c "LiteLLM call failed")
echo "   Sucessos: $API_SUCCESS"
echo "   Falhas: $API_FAIL"

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      FIM DA AUDITORIA                          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
