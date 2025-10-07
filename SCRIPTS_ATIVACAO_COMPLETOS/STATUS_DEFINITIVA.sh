#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 STATUS INTELIGÊNCIA DEFINITIVA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STATUS - INTELIGÊNCIA UNIFICADA DEFINITIVA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f "/root/inteligencia_definitiva.pid" ]; then
    echo "❌ SISTEMA NÃO ESTÁ RODANDO"
    echo ""
    echo "Para iniciar: ./START_DEFINITIVA.sh"
    exit 1
fi

PID=$(cat /root/inteligencia_definitiva.pid)

if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ PROCESSO $PID NÃO EXISTE"
    rm /root/inteligencia_definitiva.pid
    exit 1
fi

echo "✅ SISTEMA RODANDO"
echo ""

# Processo info
echo "📋 PROCESSO:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ps -p $PID -o pid,ppid,etime,pcpu,pmem,cmd
echo ""

# Database stats
if [ -f "/root/inteligencia_definitiva.db" ]; then
    echo "📊 ESTATÍSTICAS GERAIS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    sqlite3 /root/inteligencia_definitiva.db << EOF
.mode column
.headers on

SELECT 
    (SELECT COUNT(*) FROM cycles) as 'Ciclos',
    (SELECT ROUND(MAX(test_accuracy), 2) FROM mnist_metrics) as 'MNIST%',
    (SELECT ROUND(MAX(avg_reward_100), 2) FROM cartpole_metrics) as 'CartPole',
    (SELECT COUNT(*) FROM successes) as 'Sucessos',
    (SELECT COUNT(*) FROM errors) as 'Erros';
EOF
    
    echo ""
    echo "🧠 MNIST (Top 5):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    sqlite3 /root/inteligencia_definitiva.db << EOF
.mode column
.headers on

SELECT 
    cycle as 'Ciclo',
    ROUND(test_accuracy, 2) as 'Test%',
    godelian_interventions as 'Interv.'
FROM mnist_metrics 
ORDER BY cycle DESC 
LIMIT 5;
EOF
    
    echo ""
    echo "🎮 CARTPOLE (Top 5):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    sqlite3 /root/inteligencia_definitiva.db << EOF
.mode column
.headers on

SELECT 
    cycle as 'Ciclo',
    ROUND(reward, 2) as 'Reward',
    ROUND(avg_reward_100, 2) as 'Avg100'
FROM cartpole_metrics 
ORDER BY cycle DESC 
LIMIT 5;
EOF
    
    echo ""
    echo "📡 APIs:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    sqlite3 /root/inteligencia_definitiva.db << EOF
.mode column
.headers on

SELECT 
    api_name as 'API',
    COUNT(*) as 'Consultas'
FROM api_responses 
GROUP BY api_name;
EOF
    
    echo ""
    echo "🏆 SUCESSOS RECENTES:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    sqlite3 /root/inteligencia_definitiva.db << EOF
.mode column
.headers on

SELECT 
    component as 'Componente',
    achievement as 'Conquista',
    ROUND(metric_value, 2) as 'Valor'
FROM successes 
ORDER BY id DESC 
LIMIT 5;
EOF
    
    echo ""
fi

# Arquivos
echo "💾 ARQUIVOS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ls -lh /root/inteligencia_definitiva.* 2>/dev/null | awk '{print $9, "-", $5}'
echo ""

# Últimas linhas
echo "📝 ÚLTIMAS LINHAS DO LOG:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -15 /root/inteligencia_definitiva.log
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "💡 Comandos úteis:"
echo "  ./STOP_DEFINITIVA.sh     - Parar sistema"
echo "  tail -f /root/inteligencia_definitiva.log - Ver logs ao vivo"
