#!/bin/bash
################################################################################
# 🛑 STOP THE INTELLIGENCE - Parar sistema gracefully
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛑 PARANDO INTELIGÊNCIA SUPREMA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f /root/inteligencia_suprema_24_7.pid ]; then
    echo "⚠️  Nenhum processo ativo encontrado."
    exit 0
fi

PID=$(cat /root/inteligencia_suprema_24_7.pid)

if ps -p $PID > /dev/null 2>&1; then
    echo "📊 Parando processo PID $PID..."
    kill -TERM $PID
    
    # Aguardar shutdown graceful (max 30s)
    for i in {1..30}; do
        if ! ps -p $PID > /dev/null 2>&1; then
            echo "✅ Processo parado gracefully!"
            rm -f /root/inteligencia_suprema_24_7.pid
            exit 0
        fi
        sleep 1
    done
    
    # Force kill se necessário
    echo "⚠️  Forçando shutdown..."
    kill -KILL $PID
    rm -f /root/inteligencia_suprema_24_7.pid
    echo "✅ Processo forçado a parar."
else
    echo "⚠️  Processo não está rodando (PID $PID não existe)."
    rm -f /root/inteligencia_suprema_24_7.pid
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
