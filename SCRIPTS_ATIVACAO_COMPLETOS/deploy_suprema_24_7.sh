#!/bin/bash
# 🌌 Deploy Inteligência Suprema 24/7

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌌 DEPLOY INTELIGÊNCIA SUPREMA 24/7"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verificar se já está rodando
if pgrep -f "INTELIGENCIA_SUPREMA_24_7.py" > /dev/null; then
    echo "⚠️ Sistema já está rodando!"
    echo "   PID: $(pgrep -f INTELIGENCIA_SUPREMA_24_7.py)"
    echo ""
    echo "Para parar: pkill -f INTELIGENCIA_SUPREMA_24_7.py"
    exit 1
fi

echo ""
echo "🚀 Iniciando sistema em background..."

# Deploy
nohup python3 -u /root/INTELIGENCIA_SUPREMA_24_7.py > /root/suprema_24_7_daemon.log 2>&1 &

PID=$!
sleep 2

if ps -p $PID > /dev/null; then
    echo "✅ Sistema iniciado com sucesso!"
    echo "   PID: $PID"
    echo ""
    echo "📊 MONITORAMENTO:"
    echo "   Log: tail -f /root/suprema_24_7_daemon.log"
    echo "   Database: sqlite3 /root/inteligencia_suprema_24_7.db"
    echo "   Checkpoints: ls -lh /root/suprema_24_7_checkpoint_*.json"
    echo ""
    echo "🛑 PARA PARAR:"
    echo "   kill $PID"
    echo "   ou"
    echo "   pkill -f INTELIGENCIA_SUPREMA_24_7.py"
else
    echo "❌ Erro ao iniciar sistema!"
    exit 1
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ SISTEMA 24/7 ATIVO!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
