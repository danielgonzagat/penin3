#!/bin/bash
#
# Script para parar CUBIC FARM 24/7 graciosamente
#

PID_FILE="/root/cubic_24_7.pid"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "⏹️ Parando CUBIC FARM 24/7 (PID: $PID)..."
        
        # Enviar SIGTERM para shutdown gracioso
        kill -TERM $PID
        
        # Aguardar até 10 segundos
        for i in {1..10}; do
            if ! ps -p $PID > /dev/null 2>&1; then
                echo "✅ Sistema parado com sucesso"
                rm -f "$PID_FILE"
                exit 0
            fi
            sleep 1
        done
        
        # Se ainda estiver rodando, forçar
        echo "⚠️ Forçando parada..."
        kill -9 $PID
        rm -f "$PID_FILE"
        echo "✅ Sistema parado forçadamente"
    else
        echo "Sistema não está rodando"
        rm -f "$PID_FILE"
    fi
else
    echo "Arquivo PID não encontrado. Sistema provavelmente não está rodando."
fi