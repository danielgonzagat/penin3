#!/bin/bash
# V7.0 Startup - Clean mode (sem async hooks)

cd /root/intelligence_system

echo "🧹 Limpando processos antigos..."
pkill -9 -f "system_v[67]" 2>/dev/null
sleep 2

echo "🚀 Iniciando V7.0 ULTIMATE (clean mode)..."

# Desativar incompletude temporariamente
export INCOMPLETUDE_DISABLE=1

# Iniciar V7 em background
nohup python3 -u core/system_v7_ultimate.py > logs/v7_output.log 2>&1 &

PID=$!
echo "✅ V7.0 iniciado! PID: $PID"
echo "📊 Log: tail -f logs/v7_output.log"

sleep 3
ps aux | grep $PID | grep -v grep && echo "✅ Processo confirmado rodando"
