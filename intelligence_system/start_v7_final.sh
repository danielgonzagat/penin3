#!/bin/bash
# V7.0 Final Startup - Clean

cd /root/intelligence_system

echo "🧹 Limpando processos antigos..."
pkill -9 -f "system_v7" 2>/dev/null
sleep 3

echo "🚀 Iniciando V7.0 FINAL..."
python3 -W ignore::RuntimeWarning -u core/system_v7_ultimate.py > logs/v7_final.log 2>&1 &

PID=$!
echo "✅ V7.0 iniciado! PID: $PID"

sleep 5
if ps -p $PID > /dev/null; then
    echo "✅ Processo confirmado rodando"
    echo "📊 Logs: tail -f logs/v7_final.log"
else
    echo "❌ Processo não está rodando!"
fi
