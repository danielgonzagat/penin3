#!/bin/bash
echo "🛑 Parando sistema..."

if [ ! -f "/root/sistema_definitivo.pid" ]; then
    echo "⚠️  Não está rodando"
    exit 1
fi

PID=$(cat /root/sistema_definitivo.pid)
kill $PID 2>/dev/null

sleep 3

if ! ps -p $PID > /dev/null 2>&1; then
    rm /root/sistema_definitivo.pid
    echo "✅ Sistema parado"
    
    if [ -f "/root/sistema_definitivo.db" ]; then
        echo ""
        echo "📊 Stats finais:"
        sqlite3 /root/sistema_definitivo.db << EOF
SELECT 'Ciclos:', COUNT(*) FROM cycles;
SELECT 'MNIST:', ROUND(MAX(test_accuracy), 2) || '%' FROM mnist_metrics;
SELECT 'CartPole:', ROUND(MAX(avg_reward_100), 2) FROM cartpole_metrics;
SELECT 'Sucessos:', COUNT(*) FROM successes;
EOF
    fi
else
    kill -9 $PID
    rm /root/sistema_definitivo.pid
    echo "✅ Forçado"
fi
