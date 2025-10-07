#!/bin/bash
# IA³ STARTUP SCRIPT - Inicia todos os sistemas

echo "╔══════════════════════════════════════════════════════════╗"
echo "║                   IA³ SYSTEM STARTUP                      ║"
echo "║         Inteligência Artificial ao Cubo Real              ║"
echo "╚══════════════════════════════════════════════════════════╝"

# Limpa processos antigos
echo "🧹 Limpando processos antigos..."
pkill -f "python3.*ia3" 2>/dev/null
pkill -f "python3.*teis" 2>/dev/null
pkill -f "python3.*neural" 2>/dev/null
pkill -f "python3.*api_daemon" 2>/dev/null
sleep 2

# Cria diretórios necessários
echo "📁 Criando diretórios..."
mkdir -p /root/ia3_checkpoints
mkdir -p /root/ia3_data
mkdir -p /root/ia3_logs

# Inicia sistema principal IA³
echo "🚀 Iniciando IA³ Unified System..."
nohup python3 /root/ia3_unified_real.py > /root/ia3_logs/unified.log 2>&1 &
echo "   PID: $!"
sleep 3

# Inicia connector
echo "🔗 Iniciando IA³ Connector..."
nohup python3 /root/ia3_connector.py > /root/ia3_logs/connector.log 2>&1 &
echo "   PID: $!"
sleep 2

# Inicia monitor em foreground
echo "📊 Iniciando IA³ Monitor..."
echo ""
echo "Sistema IA³ iniciado! Abrindo monitor..."
echo "Pressione Ctrl+C para parar o monitor (sistemas continuam rodando)"
echo ""
sleep 2

python3 /root/ia3_monitor.py
