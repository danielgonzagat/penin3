#!/bin/bash
echo "📊 STATUS SISTEMA DEFINITIVO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ ! -f "/root/sistema_definitivo.pid" ]; then
    echo "❌ NÃO RODANDO"
    exit 1
fi

PID=$(cat /root/sistema_definitivo.pid)

if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ PID INVÁLIDO"
    rm /root/sistema_definitivo.pid
    exit 1
fi

echo "✅ RODANDO (PID: $PID)"
echo ""

ps -p $PID -o pid,etime,pcpu,pmem,cmd
echo ""

if [ -f "/root/sistema_definitivo.db" ]; then
    echo "📊 ESTATÍSTICAS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    sqlite3 /root/sistema_definitivo.db << EOF
.mode column
.headers on
SELECT 
    (SELECT COUNT(*) FROM cycles) as Ciclos,
    (SELECT ROUND(MAX(test_accuracy), 2) FROM mnist_metrics) as 'MNIST%',
    (SELECT ROUND(MAX(avg_reward_100), 2) FROM cartpole_metrics) as CartPole,
    (SELECT COUNT(*) FROM successes) as Sucessos,
    (SELECT COUNT(*) FROM errors) as Erros;
EOF
    
    echo ""
    echo "🧠 MNIST (Top 3):"
    sqlite3 /root/sistema_definitivo.db "SELECT cycle, ROUND(test_accuracy,2) FROM mnist_metrics ORDER BY cycle DESC LIMIT 3"
    
    echo ""
    echo "🎮 CartPole (Top 3):"
    sqlite3 /root/sistema_definitivo.db "SELECT cycle, ROUND(avg_reward_100,2) FROM cartpole_metrics ORDER BY cycle DESC LIMIT 3"
fi

echo ""
echo "📝 LOG:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -10 /root/sistema_definitivo.log
