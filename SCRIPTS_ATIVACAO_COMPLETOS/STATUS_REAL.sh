#!/bin/bash
echo "📊 STATUS DO SISTEMA"
echo ""

if [ ! -f "/root/sistema_real_simples.pid" ]; then
    echo "❌ NÃO ESTÁ RODANDO"
    echo "Inicie com: ./START_REAL.sh"
    exit 1
fi

PID=$(cat /root/sistema_real_simples.pid)

if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ PID $PID não existe"
    rm /root/sistema_real_simples.pid
    exit 1
fi

echo "✅ RODANDO (PID: $PID)"
echo ""

# Processo
ps -p $PID -o pid,etime,pcpu,pmem,cmd

echo ""

# Database
if [ -f "/root/sistema_real_simples.db" ]; then
    echo "📊 MÉTRICAS:"
    sqlite3 /root/sistema_real_simples.db << EOF
.mode column
.headers on

SELECT 
    MAX(cycle) as Ciclos,
    ROUND(MAX(mnist_acc), 2) as 'MNIST%',
    ROUND(MAX(cart_reward), 2) as CartPole
FROM cycles;
EOF
    
    echo ""
    echo "🧠 MNIST (últimas 5):"
    sqlite3 /root/sistema_real_simples.db << EOF
SELECT ROUND(value, 2) || '%' 
FROM metrics 
WHERE component='mnist' AND metric='test_acc' 
ORDER BY id DESC LIMIT 5;
EOF
    
    echo ""
    echo "🎮 CartPole (últimas 5):"
    sqlite3 /root/sistema_real_simples.db << EOF
SELECT ROUND(value, 1) 
FROM metrics 
WHERE component='cartpole' AND metric='avg_100' 
ORDER BY id DESC LIMIT 5;
EOF
fi

echo ""
echo "📝 Últimas linhas:"
tail -10 /root/sistema_real_simples.log
