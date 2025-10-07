#!/bin/bash
echo "🔄 Parando daemon antigo..."
pkill -f "python3.*brain_daemon_real_env.py"
sleep 3
echo "✅ Daemon parado"

echo "🧪 Verificando sintaxe..."
python3 -m py_compile /root/UNIFIED_BRAIN/brain_daemon_real_env.py
if [ $? -ne 0 ]; then
    echo "❌ ERRO DE SINTAXE!"
    exit 1
fi
echo "✅ Sintaxe OK"

echo "🚀 Iniciando daemon novo..."
cd /root/UNIFIED_BRAIN
nohup python3 -u brain_daemon_real_env.py > /root/brain_v3_restart.log 2>&1 &
sleep 10

echo "📋 Últimas linhas do log:"
tail -30 /root/brain_v3_restart.log

echo ""
echo "✅ Daemon reiniciado!"
echo "Monitor: tail -f /root/brain_v3_restart.log"
