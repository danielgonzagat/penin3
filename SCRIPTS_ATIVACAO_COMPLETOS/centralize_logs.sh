#!/bin/bash
# Script para centralizar logs
echo "📋 Centralizando logs..."

# Criar diretório se não existir
mkdir -p /root/centralized_logs

# Coletar logs principais
tail -f /root/v7_runner_daemon.log /root/emergence_unified.log /root/system_connector.log > /root/centralized_logs/combined.log 2>&1 &
echo $! > /root/centralize_logs.pid

echo "✅ Logs centralizados em /root/centralized_logs/combined.log (PID: $(cat /root/centralize_logs.pid))"
