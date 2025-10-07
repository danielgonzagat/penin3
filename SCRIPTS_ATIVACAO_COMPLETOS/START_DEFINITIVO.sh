#!/bin/bash
echo "🚀 Iniciando SISTEMA DEFINITIVO REAL..."

if [ -f "/root/sistema_definitivo.pid" ]; then
    PID=$(cat /root/sistema_definitivo.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Já rodando (PID: $PID)"
        exit 1
    fi
fi

nohup python3 /root/SISTEMA_DEFINITIVO_REAL.py > /root/sistema_definitivo.out 2>&1 &

sleep 5

if [ -f "/root/sistema_definitivo.pid" ]; then
    PID=$(cat /root/sistema_definitivo.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ SISTEMA INICIADO (PID: $PID)"
        echo ""
        echo "📊 COMPONENTES:"
        echo "  ✅ MNIST (PyTorch + Gödelian)"
        echo "  ✅ CartPole (PPO CleanRL)"
        echo "  ✅ 6 APIs Frontier"
        echo "  ✅ Memória Completa"
        echo ""
        echo "💡 Comandos:"
        echo "  ./STATUS_DEFINITIVO.sh - Ver status"
        echo "  ./STOP_DEFINITIVO.sh - Parar"
        echo "  tail -f /root/sistema_definitivo.log - Logs"
        echo ""
        tail -15 /root/sistema_definitivo.log
    else
        echo "❌ Erro ao iniciar"
        cat /root/sistema_definitivo.out
    fi
else
    echo "❌ PID não criado"
    cat /root/sistema_definitivo.out
fi
