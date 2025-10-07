#!/bin/bash
# Monitor contínuo do treinamento

echo "🔬 MONITOR DE TREINAMENTO ATIVO"
echo "================================"
echo "PID: $(cat /root/brain_daemon.pid 2>/dev/null || echo 'N/A')"
echo "Log: /root/brain_daemon_v3_improved.log"
echo ""
echo "Pressione Ctrl+C para parar o monitor (daemon continua rodando)"
echo ""
echo "Aguardando episódios..."
echo ""

tail -f /root/brain_daemon_v3_improved.log 2>/dev/null | grep --line-buffered -E "Ep [0-9]+:|scheduler|🎊|Phase.*hooks"