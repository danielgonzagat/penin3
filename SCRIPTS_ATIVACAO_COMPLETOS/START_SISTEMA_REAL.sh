#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 START SISTEMA REAL 24/7
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 INICIANDO SISTEMA REAL 24/7"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Verifica se já está rodando
if [ -f "/root/sistema_real_24_7.pid" ]; then
    PID=$(cat /root/sistema_real_24_7.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Sistema já está rodando (PID: $PID)"
        echo ""
        echo "Para parar: ./STOP_SISTEMA_REAL.sh"
        echo "Para status: ./STATUS_SISTEMA_REAL.sh"
        exit 1
    fi
fi

# Instalar dependências (se necessário)
echo "📦 Verificando dependências..."

pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || true
pip install -q gymnasium 2>/dev/null || true
pip install -q openai mistralai anthropic google-generativeai 2>/dev/null || true

echo "✅ Dependências OK"

# Iniciar sistema em background
echo ""
echo "🚀 Iniciando sistema..."

nohup python3 /root/SISTEMA_REAL_24_7.py > /root/sistema_real_24_7.out 2>&1 &

# Aguardar inicialização
sleep 3

# Verificar se iniciou
if [ -f "/root/sistema_real_24_7.pid" ]; then
    PID=$(cat /root/sistema_real_24_7.pid)
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ Sistema iniciado com sucesso!"
        echo ""
        echo "📊 INFORMAÇÕES:"
        echo "  PID: $PID"
        echo "  Log: /root/sistema_real_24_7.log"
        echo "  Database: /root/sistema_real_24_7.db"
        echo ""
        echo "📡 COMANDOS ÚTEIS:"
        echo "  Parar:    ./STOP_SISTEMA_REAL.sh"
        echo "  Status:   ./STATUS_SISTEMA_REAL.sh"
        echo "  Logs:     tail -f /root/sistema_real_24_7.log"
        echo ""
        echo "🎯 TAREFAS REAIS:"
        echo "  ✅ MNIST Classification (PyTorch)"
        echo "  ✅ CartPole RL (Gymnasium)"
        echo "  ✅ 6 APIs Frontier"
        echo "  ✅ Fine-tuning automático"
        echo "  ✅ Memória persistente"
        echo ""
        
        # Mostrar últimas linhas do log
        echo "📝 ÚLTIMAS LINHAS DO LOG:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        tail -20 /root/sistema_real_24_7.log | head -10
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
    else
        echo "❌ Erro ao iniciar sistema"
        echo "Verifique o log: /root/sistema_real_24_7.out"
        exit 1
    fi
else
    echo "❌ Erro: PID file não foi criado"
    echo "Verifique o log: /root/sistema_real_24_7.out"
    exit 1
fi
