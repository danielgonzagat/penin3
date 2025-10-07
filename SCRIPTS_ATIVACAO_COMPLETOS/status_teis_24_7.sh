#!/bin/bash
# TEIS 24/7 Status Script

echo "📊 Status do TEIS Ultimate Real 24/7"
echo "=================================="

# Verificar daemon
if pgrep -f "teis_daemon_24_7.py" > /dev/null; then
    DAEMON_PID=$(pgrep -f "teis_daemon_24_7.py")
    echo "✅ Daemon ativo (PID: $DAEMON_PID)"
    
    # Verificar TEIS
    if pgrep -f "teis_ultimate_real_deterministic.py" > /dev/null; then
        TEIS_PID=$(pgrep -f "teis_ultimate_real_deterministic.py")
        echo "✅ TEIS ativo (PID: $TEIS_PID)"
        
        # Estatísticas de processo
        echo ""
        echo "📈 Estatísticas do Processo:"
        ps -p $TEIS_PID -o pid,ppid,cmd,%cpu,%mem,etime
        
        # Uso de memória
        MEM_USAGE=$(ps -p $TEIS_PID -o pmem= | tr -d ' ')
        CPU_USAGE=$(ps -p $TEIS_PID -o pcpu= | tr -d ' ')
        echo "Memória: ${MEM_USAGE}% | CPU: ${CPU_USAGE}%"
        
    else
        echo "❌ TEIS não está ativo"
    fi
else
    echo "❌ Daemon não está ativo"
fi

# Verificar arquivos de estatísticas
echo ""
echo "📊 Estatísticas do Sistema:"
if [ -f "/root/teis_daemon_stats.json" ]; then
    echo "✅ Arquivo de estatísticas encontrado"
    # Mostrar algumas estatísticas básicas
    RESTARTS=$(grep -o '"restarts": [0-9]*' /root/teis_daemon_stats.json | cut -d' ' -f2)
    echo "Reinicializações: $RESTARTS"
else
    echo "⚠️ Arquivo de estatísticas não encontrado"
fi

# Verificar logs recentes
echo ""
echo "📝 Logs Recentes:"
if [ -f "/root/teis_daemon.log" ]; then
    tail -5 /root/teis_daemon.log
else
    echo "⚠️ Arquivo de log não encontrado"
fi

echo ""
echo "💡 Comandos:"
echo "  Iniciar: ./start_teis_24_7.sh"
echo "  Parar: ./stop_teis_24_7.sh"
echo "  Logs completos: tail -f /root/teis_daemon.log"
