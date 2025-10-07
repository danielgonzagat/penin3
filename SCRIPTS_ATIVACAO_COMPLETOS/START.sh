#!/bin/bash
echo "🚀 INICIANDO..."

if [ -f "/root/sistema_funcional.pid" ]; then
    PID=$(cat /root/sistema_funcional.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Já rodando: $PID"
        exit 1
    fi
fi

# Iniciar
PYTHONSTARTUP="" nohup python3 -u /root/SISTEMA_FUNCIONAL_CLEAN.py > /root/sistema_funcional.out 2>&1 &

sleep 5

if [ -f "/root/sistema_funcional.pid" ]; then
    PID=$(cat /root/sistema_funcional.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ RODANDO: PID $PID"
        echo ""
        echo "Log: tail -f /root/sistema_funcional.log"
        echo "Parar: ./STOP.sh"
        echo ""
        tail -15 /root/sistema_funcional.log
    else
        echo "❌ Falhou"
        cat /root/sistema_funcional.out
    fi
else
    echo "❌ PID não criado"
    cat /root/sistema_funcional.out
fi
