#!/bin/bash
# Script para iniciar o Sistema de Inteligência Unificada 24/7

echo "=========================================="
echo "🧠 INICIANDO INTELIGÊNCIA UNIFICADA 24/7"
echo "=========================================="

# Criar diretórios necessários
mkdir -p /root/unified_checkpoints
mkdir -p /root/unified_logs

# Parar processos antigos se existirem
echo "Parando processos isolados..."
pkill -f "neural_farm.py" 2>/dev/null
pkill -f "teis_v2_enhanced.py" 2>/dev/null
pkill -f "NEURAL_GENESIS" 2>/dev/null
sleep 2

# Verificar se já está rodando
if pgrep -f "unified_intelligence_24_7.py" > /dev/null; then
    echo "⚠️ Sistema já está rodando!"
    echo "Para reiniciar, execute: pkill -f unified_intelligence_24_7.py"
    exit 1
fi

# Iniciar sistema unificado
echo "🚀 Iniciando sistema unificado..."
nohup python3 /root/unified_intelligence_24_7.py > /root/unified_logs/output.log 2>&1 &
PID=$!

echo "✓ Sistema iniciado com PID: $PID"
echo ""
echo "📊 Comandos úteis:"
echo "  Ver logs: tail -f /root/unified_logs/output.log"
echo "  Ver métricas: tail -f /root/unified_intelligence.log"
echo "  Ver comunicação: tail -f /root/unified_communication.jsonl"
echo "  Parar sistema: kill -TERM $PID"
echo ""
echo "💾 Dados salvos em:"
echo "  Checkpoints: /root/unified_checkpoints/"
echo "  Banco de dados: /root/unified_intelligence.db"
echo ""
echo "✅ SISTEMA RODANDO 24/7!"