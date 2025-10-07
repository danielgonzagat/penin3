#!/bin/bash
# TEIS 24/7 Stop Script

echo "🛑 Parando TEIS Ultimate Real 24/7..."
echo "=================================="

# Função para parar processos
stop_process() {
    local process_name=$1
    local display_name=$2
    
    if pgrep -f "$process_name" > /dev/null; then
        echo "🛑 Parando $display_name..."
        pkill -f "$process_name"
        
        # Aguardar até 10 segundos
        for i in {1..10}; do
            if ! pgrep -f "$process_name" > /dev/null; then
                echo "✅ $display_name parado"
                return 0
            fi
            sleep 1
        done
        
        # Forçar parada se ainda estiver rodando
        echo "⚠️ Forçando parada de $display_name..."
        pkill -9 -f "$process_name"
        sleep 1
        
        if ! pgrep -f "$process_name" > /dev/null; then
            echo "✅ $display_name parado (forçado)"
        else
            echo "❌ Falha ao parar $display_name"
            return 1
        fi
    else
        echo "ℹ️ $display_name não está rodando"
    fi
}

# Parar TEIS primeiro
stop_process "teis_ultimate_real_deterministic.py" "TEIS"

# Aguardar um pouco
sleep 2

# Parar daemon
stop_process "teis_daemon_24_7.py" "TEIS Daemon"

echo ""
echo "✅ TEIS 24/7 parado completamente"
echo "📊 Verifique estatísticas em: /root/teis_daemon_stats.json"
echo "📝 Logs disponíveis em: /root/teis_daemon.log"
