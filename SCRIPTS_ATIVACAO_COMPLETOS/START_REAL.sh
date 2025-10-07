#!/bin/bash
echo "🚀 INICIANDO SISTEMA REAL SIMPLES..."

# Verificar se já roda
if [ -f "/root/sistema_real_simples.pid" ]; then
    PID=$(cat /root/sistema_real_simples.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Já rodando (PID: $PID)"
        exit 1
    fi
fi

# Instalar deps
pip install -q torch torchvision gymnasium openai mistralai anthropic google-generativeai 2>/dev/null || true

# Iniciar
nohup python3 /root/SISTEMA_REAL_SIMPLES.py > /root/sistema_real_simples.out 2>&1 &

sleep 5

# Verificar
if [ -f "/root/sistema_real_simples.pid" ]; then
    PID=$(cat /root/sistema_real_simples.pid)
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ SISTEMA INICIADO!"
        echo ""
        echo "PID: $PID"
        echo "Log: tail -f /root/sistema_real_simples.log"
        echo "DB: /root/sistema_real_simples.db"
        echo ""
        echo "Parar: ./STOP_REAL.sh"
        echo "Status: ./STATUS_REAL.sh"
        echo ""
        
        echo "📝 PRIMEIRAS LINHAS:"
        tail -20 /root/sistema_real_simples.log
    else
        echo "❌ Erro ao iniciar"
        cat /root/sistema_real_simples.out
        exit 1
    fi
else
    echo "❌ PID não criado"
    cat /root/sistema_real_simples.out
    exit 1
fi
