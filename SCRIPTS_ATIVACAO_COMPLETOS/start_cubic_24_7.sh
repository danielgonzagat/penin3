#!/bin/bash
#
# Script para iniciar e manter CUBIC FARM 24/7 rodando
#

SCRIPT_DIR="/root"
PYTHON_SCRIPT="cubic_farm_24_7.py"
LOG_FILE="/root/cubic_24_7_logs/startup.log"
PID_FILE="/root/cubic_24_7.pid"

# Criar diretórios necessários
mkdir -p /root/cubic_24_7_logs
mkdir -p /root/cubic_24_7_checkpoints

# Função para verificar se já está rodando
check_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            echo "CUBIC FARM 24/7 já está rodando (PID: $PID)"
            return 0
        fi
    fi
    return 1
}

# Função para iniciar
start_system() {
    if check_running; then
        exit 0
    fi
    
    echo "🚀 Iniciando CUBIC FARM 24/7..."
    echo "[$(date)] Iniciando sistema" >> "$LOG_FILE"
    
    cd "$SCRIPT_DIR"
    
    # Iniciar em background com nohup
    nohup python3 "$PYTHON_SCRIPT" >> "$LOG_FILE" 2>&1 &
    
    # Salvar PID
    echo $! > "$PID_FILE"
    
    sleep 2
    
    if check_running; then
        echo "✅ CUBIC FARM 24/7 iniciado com sucesso (PID: $(cat $PID_FILE))"
        echo "📊 Logs em: /root/cubic_24_7_logs/"
        echo "💾 Checkpoints em: /root/cubic_24_7_checkpoints/"
        echo ""
        echo "Comandos úteis:"
        echo "  Ver status: python3 cubic_farm_24_7.py status"
        echo "  Parar: ./stop_cubic_24_7.sh"
        echo "  Ver logs: tail -f /root/cubic_24_7_logs/cubic_farm_24_7.log"
    else
        echo "❌ Falha ao iniciar sistema"
        exit 1
    fi
}

# Função para monitorar e reiniciar se necessário
monitor_and_restart() {
    while true; do
        if ! check_running; then
            echo "[$(date)] Sistema parou, reiniciando..." >> "$LOG_FILE"
            start_system
        fi
        sleep 60  # Verificar a cada minuto
    done
}

# Processar argumentos
case "$1" in
    monitor)
        echo "📡 Modo monitor: verificando a cada minuto"
        monitor_and_restart
        ;;
    *)
        start_system
        ;;
esac