#!/bin/bash
echo "🛑 Parando sistema..."

if [ ! -f "/root/sistema_real_simples.pid" ]; then
    echo "⚠️  Não está rodando"
    exit 1
fi

PID=$(cat /root/sistema_real_simples.pid)

kill $PID 2>/dev/null

for i in {1..10}; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "✅ Parado!"
        rm /root/sistema_real_simples.pid
        
        # Stats finais
        if [ -f "/root/sistema_real_simples.db" ]; then
            echo ""
            sqlite3 /root/sistema_real_simples.db << EOF
SELECT 'Ciclos:', MAX(cycle) FROM cycles;
SELECT 'MNIST:', ROUND(MAX(mnist_acc), 2) || '%' FROM cycles;
SELECT 'CartPole:', ROUND(MAX(cart_reward), 2) FROM cycles;
EOF
        fi
        
        exit 0
    fi
    sleep 1
done

kill -9 $PID 2>/dev/null
rm /root/sistema_real_simples.pid
echo "✅ Forçadamente parado"
