#!/bin/bash
# TEIS 24/7 Startup Script

echo "🚀 Iniciando TEIS Ultimate Real em modo 24/7..."
echo "Data/Hora: $(date)"
echo "=========================================="

# Verificar se já está rodando
if pgrep -f "teis_daemon_24_7.py" > /dev/null; then
    echo "⚠️ TEIS Daemon já está rodando!"
    echo "Para parar: ./stop_teis_24_7.sh"
    exit 1
fi

# Verificar se há TEIS rodando diretamente
if pgrep -f "teis_ultimate_real_deterministic.py" > /dev/null; then
    echo "⚠️ TEIS já está rodando diretamente!"
    echo "Pare o processo primeiro ou use o daemon."
    exit 1
fi

# Criar diretório de logs se não existir
mkdir -p /root/teis_logs

# Iniciar daemon em background
echo "🔄 Iniciando daemon..."
nohup /root/teis_daemon_24_7.py > /root/teis_daemon.out 2>&1 &

# Aguardar um pouco
sleep 3

# Verificar se iniciou
if pgrep -f "teis_daemon_24_7.py" > /dev/null; then
    echo "✅ TEIS Daemon iniciado com sucesso!"
    echo "📊 Monitore os logs em: /root/teis_daemon.log"
    echo "📈 Estatísticas em: /root/teis_daemon_stats.json"
    echo ""
    echo "Comandos úteis:"
    echo "  Ver status: ./status_teis_24_7.sh"
    echo "  Parar: ./stop_teis_24_7.sh"
    echo "  Logs: tail -f /root/teis_daemon.log"
else
    echo "❌ Falha ao iniciar TEIS Daemon!"
    echo "Verifique: /root/teis_daemon.out"
    exit 1
fi
