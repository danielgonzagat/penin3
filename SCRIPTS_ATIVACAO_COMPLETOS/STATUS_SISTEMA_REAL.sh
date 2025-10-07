#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 📊 STATUS SISTEMA REAL 24/7
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STATUS SISTEMA REAL 24/7"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Verifica se está rodando
if [ ! -f "/root/sistema_real_24_7.pid" ]; then
    echo "❌ SISTEMA NÃO ESTÁ RODANDO"
    echo ""
    echo "Para iniciar: ./START_SISTEMA_REAL.sh"
    exit 1
fi

PID=$(cat /root/sistema_real_24_7.pid)

if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ SISTEMA NÃO ESTÁ RODANDO (PID inválido: $PID)"
    echo ""
    echo "Limpando PID file..."
    rm /root/sistema_real_24_7.pid
    echo "Para iniciar: ./START_SISTEMA_REAL.sh"
    exit 1
fi

# Sistema está rodando
echo "✅ SISTEMA RODANDO"
echo ""

# Informações do processo
echo "📋 PROCESSO:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ps -p $PID -o pid,ppid,etime,pcpu,pmem,cmd
echo ""

# Estatísticas do database
if [ -f "/root/sistema_real_24_7.db" ]; then
    echo "📊 ESTATÍSTICAS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    sqlite3 /root/sistema_real_24_7.db << EOF
.mode column
.headers on

SELECT 
    (SELECT COUNT(*) FROM cycles) as 'Total Ciclos',
    (SELECT ROUND(test_accuracy, 2) FROM mnist_metrics ORDER BY cycle DESC LIMIT 1) as 'MNIST Acc%',
    (SELECT ROUND(reward, 2) FROM cartpole_metrics ORDER BY cycle DESC LIMIT 1) as 'CartPole Reward',
    (SELECT COUNT(*) FROM errors) as 'Erros',
    (SELECT COUNT(DISTINCT api_name) FROM api_responses) as 'APIs Usadas';
EOF
    echo ""
    
    # Últimas métricas MNIST
    echo "🧠 MNIST (Últimas 5 epochs):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    sqlite3 /root/sistema_real_24_7.db << EOF
.mode column
.headers on

SELECT 
    cycle as 'Ciclo',
    ROUND(train_accuracy, 2) as 'Train Acc%',
    ROUND(test_accuracy, 2) as 'Test Acc%',
    ROUND(test_loss, 4) as 'Loss'
FROM mnist_metrics 
ORDER BY cycle DESC 
LIMIT 5;
EOF
    echo ""
    
    # Últimas métricas CartPole
    echo "🎮 CARTPOLE (Últimos 5 episódios):"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    sqlite3 /root/sistema_real_24_7.db << EOF
.mode column
.headers on

SELECT 
    cycle as 'Ciclo',
    ROUND(reward, 2) as 'Reward',
    steps as 'Steps',
    ROUND(epsilon, 3) as 'Epsilon'
FROM cartpole_metrics 
ORDER BY cycle DESC 
LIMIT 5;
EOF
    echo ""
    
    # APIs consultadas
    echo "📡 APIs CONSULTADAS:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    sqlite3 /root/sistema_real_24_7.db << EOF
.mode column
.headers on

SELECT 
    api_name as 'API',
    COUNT(*) as 'Consultas',
    MAX(cycle) as 'Último Ciclo'
FROM api_responses 
GROUP BY api_name
ORDER BY COUNT(*) DESC;
EOF
    echo ""
fi

# Tamanho dos arquivos
echo "💾 ARQUIVOS:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
ls -lh /root/sistema_real_24_7.* 2>/dev/null | awk '{print $9, "-", $5}'
echo ""

# Últimas linhas do log
echo "📝 ÚLTIMAS LINHAS DO LOG:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
tail -10 /root/sistema_real_24_7.log
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "💡 COMANDOS ÚTEIS:"
echo "  Parar:      ./STOP_SISTEMA_REAL.sh"
echo "  Ver logs:   tail -f /root/sistema_real_24_7.log"
echo "  Database:   sqlite3 /root/sistema_real_24_7.db"
