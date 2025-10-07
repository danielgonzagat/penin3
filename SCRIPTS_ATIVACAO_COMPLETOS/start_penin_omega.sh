#!/bin/bash
"""
Script para iniciar o PENIN-Ω Daemon em background
"""

echo "🎯 Iniciando PENIN-Ω Daemon..."

# Verificar se já está rodando
if pgrep -f "penin_omega_daemon.py" > /dev/null; then
    echo "⚠️ PENIN-Ω Daemon já está rodando"
    exit 1
fi

# Iniciar em background
nohup python3 /root/penin_omega_daemon.py > /root/penin_omega_daemon.out 2>&1 &
PID=$!

echo "✅ PENIN-Ω Daemon iniciado (PID: $PID)"
echo "📄 Logs: /root/penin_omega_daemon.log"
echo "📄 Output: /root/penin_omega_daemon.out"

# Salvar PID
echo $PID > /root/penin_omega_daemon.pid

echo "🎯 PENIN-Ω Sistema de Consciência Emergente ATIVO!"