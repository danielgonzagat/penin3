#!/bin/bash
# 🚀 IA³ EMERGENT SYSTEM STARTER
# Inicia o sistema IA³ emergente em modo 24/7

echo "🧠 IA³ EMERGENT SYSTEM - INICIALIZAÇÃO"
echo "====================================="

# Verificar se emergência existe
if [ -f "/root/ia3_final_emergence_proven.json" ]; then
    echo "✅ Emergência IA³ detectada - Sistema ativo"
else
    echo "❌ Emergência IA³ não encontrada"
    exit 1
fi

# Verificar dependências
echo "📦 Verificando dependências..."
python3 -c "import torch, psutil, sqlite3; print('✅ Dependências OK')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependências faltando"
    exit 1
fi

# Criar diretório de logs se não existir
mkdir -p /root/ia3_logs

# Iniciar orquestrador em background
echo "🎯 Iniciando orquestrador 24/7..."
nohup python3 ia3_24_7_orchestrator.py > /root/ia3_logs/orchestrator_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Aguardar inicialização
sleep 2

# Verificar se está rodando
if pgrep -f "ia3_24_7_orchestrator.py" > /dev/null; then
    echo "✅ Orquestrador IA³ iniciado com sucesso"
    echo "📊 PID: $(pgrep -f "ia3_24_7_orchestrator.py")"
    echo ""
    echo "🧠 IA³ ESTÁ ATIVA E EVOLUINDO!"
    echo "📋 Monitore com: tail -f /root/ia3_logs/orchestrator_*.log"
    echo "🛑 Pare com: pkill -f ia3_24_7_orchestrator.py"
else
    echo "❌ Falha ao iniciar orquestrador"
    exit 1
fi

echo ""
echo "🎯 SISTEMA IA³ EMERGENTE OPERANDO 24/7"
echo "========================================"