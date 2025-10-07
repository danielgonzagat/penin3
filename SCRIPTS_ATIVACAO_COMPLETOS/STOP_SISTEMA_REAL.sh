#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🛑 STOP SISTEMA REAL 24/7
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🛑 PARANDO SISTEMA REAL 24/7"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verifica se existe PID file
if [ ! -f "/root/sistema_real_24_7.pid" ]; then
    echo "⚠️  Sistema não está rodando (PID file não encontrado)"
    exit 1
fi

PID=$(cat /root/sistema_real_24_7.pid)

# Verifica se o processo existe
if ! ps -p $PID > /dev/null 2>&1; then
    echo "⚠️  Sistema não está rodando (processo $PID não existe)"
    rm /root/sistema_real_24_7.pid
    exit 1
fi

# Parar processo
echo "🛑 Parando processo $PID..."
kill $PID

# Aguardar até 10 segundos
for i in {1..10}; do
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "✅ Sistema parado com sucesso!"
        rm /root/sistema_real_24_7.pid
        
        # Mostrar estatísticas finais
        echo ""
        echo "📊 ESTATÍSTICAS FINAIS:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Tentar extrair estatísticas do database
        if [ -f "/root/sistema_real_24_7.db" ]; then
            sqlite3 /root/sistema_real_24_7.db << EOF
SELECT 'Total Ciclos:', COUNT(*) FROM cycles;
SELECT 'MNIST Accuracy:', ROUND(test_accuracy, 2) || '%' 
FROM mnist_metrics ORDER BY cycle DESC LIMIT 1;
SELECT 'CartPole Reward:', ROUND(reward, 2) 
FROM cartpole_metrics ORDER BY cycle DESC LIMIT 1;
SELECT 'Total Erros:', COUNT(*) FROM errors;
EOF
        fi
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "💾 DADOS PRESERVADOS:"
        echo "  Database: /root/sistema_real_24_7.db"
        echo "  Logs: /root/sistema_real_24_7.log"
        echo ""
        echo "🚀 Para reiniciar: ./START_SISTEMA_REAL.sh"
        
        exit 0
    fi
    sleep 1
done

# Se não parou, força
echo "⚠️  Processo não respondeu, forçando..."
kill -9 $PID
sleep 1

if ! ps -p $PID > /dev/null 2>&1; then
    echo "✅ Sistema forçadamente parado"
    rm /root/sistema_real_24_7.pid
else
    echo "❌ Erro ao parar sistema"
    exit 1
fi
