#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🛑 STOP INTELIGÊNCIA DEFINITIVA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛑 PARANDO INTELIGÊNCIA DEFINITIVA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "/root/inteligencia_definitiva.pid" ]; then
    echo "⚠️  Sistema não está rodando"
    exit 1
fi

PID=$(cat /root/inteligencia_definitiva.pid)

if ! ps -p $PID > /dev/null 2>&1; then
    echo "⚠️  Processo $PID não existe"
    rm /root/inteligencia_definitiva.pid
    exit 1
fi

echo "🛑 Parando processo $PID..."
kill $PID

for i in {1..10}; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "✅ Sistema parado!"
        rm /root/inteligencia_definitiva.pid
        
        # Estatísticas finais
        if [ -f "/root/inteligencia_definitiva.db" ]; then
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "📊 ESTATÍSTICAS FINAIS:"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            
            sqlite3 /root/inteligencia_definitiva.db << EOF
SELECT 'Total Ciclos:', COUNT(*) FROM cycles;
SELECT 'Melhor MNIST:', ROUND(MAX(test_accuracy), 2) || '%' FROM mnist_metrics;
SELECT 'Melhor CartPole:', ROUND(MAX(avg_reward_100), 2) FROM cartpole_metrics;
SELECT 'Total Sucessos:', COUNT(*) FROM successes;
SELECT 'Total Erros:', COUNT(*) FROM errors;
EOF
            
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        fi
        
        echo ""
        echo "💾 Dados preservados em /root/inteligencia_definitiva.db"
        echo "🚀 Para reiniciar: ./START_DEFINITIVA.sh"
        
        exit 0
    fi
    sleep 1
done

echo "⚠️  Forçando parada..."
kill -9 $PID
rm /root/inteligencia_definitiva.pid
echo "✅ Sistema forçadamente parado"
