#!/bin/bash
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 🚀 START INTELIGÊNCIA UNIFICADA DEFINITIVA
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌟 INICIANDO INTELIGÊNCIA UNIFICADA DEFINITIVA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Verificar se já está rodando
if [ -f "/root/inteligencia_definitiva.pid" ]; then
    PID=$(cat /root/inteligencia_definitiva.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Sistema já rodando (PID: $PID)"
        echo ""
        echo "Para parar: ./STOP_DEFINITIVA.sh"
        echo "Para status: ./STATUS_DEFINITIVA.sh"
        exit 1
    fi
fi

# Instalar dependências
echo "📦 Verificando dependências..."

pip install -q torch torchvision --index-url https://download.pytorch.org/whl/cpu 2>/dev/null || true
pip install -q gymnasium 2>/dev/null || true
pip install -q openai mistralai anthropic google-generativeai 2>/dev/null || true

echo "✅ Dependências OK"
echo ""

# Verificar estado anterior
if [ -f "/root/inteligencia_definitiva.db" ]; then
    echo "📂 ESTADO ANTERIOR ENCONTRADO:"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    sqlite3 /root/inteligencia_definitiva.db << EOF
SELECT 'Último Ciclo:', MAX(cycle) FROM cycles;
SELECT 'Melhor MNIST:', ROUND(MAX(test_accuracy), 2) || '%' FROM mnist_metrics;
SELECT 'Melhor CartPole:', ROUND(MAX(avg_reward_100), 2) FROM cartpole_metrics;
SELECT 'Total Erros:', COUNT(*) FROM errors;
SELECT 'Total Sucessos:', COUNT(*) FROM successes;
EOF
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
fi

# Iniciar sistema
echo "🚀 Iniciando sistema definitivo..."

nohup python3 /root/INTELIGENCIA_DEFINITIVA_REAL.py > /root/inteligencia_definitiva.out 2>&1 &

sleep 5

# Verificar
if [ -f "/root/inteligencia_definitiva.pid" ]; then
    PID=$(cat /root/inteligencia_definitiva.pid)
    
    if ps -p $PID > /dev/null 2>&1; then
        echo "✅ SISTEMA INICIADO COM SUCESSO!"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 COMPONENTES ATIVOS:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "  🧠 MNIST Classifier (PyTorch + Gödelian)"
        echo "     • Rede CNN com anti-stagnação"
        echo "     • 60k train / 10k test"
        echo "     • Auto-ajuste de LR e arquitetura"
        echo ""
        echo "  🎮 CartPole Agent (PPO CleanRL)"
        echo "     • Proximal Policy Optimization"
        echo "     • 4 ambientes paralelos"
        echo "     • Meta-learning com IA³"
        echo ""
        echo "  🌐 API Manager (6 Frontiers)"
        echo "     • OpenAI GPT-5"
        echo "     • Mistral Large"
        echo "     • Google Gemini 2.5-pro"
        echo "     • DeepSeek V3.1 (reasoner)"
        echo "     • Anthropic Claude Opus 4.1"
        echo "     • xAI Grok-4"
        echo ""
        echo "  💾 Unified Memory System"
        echo "     • Carrega estado anterior completo"
        echo "     • Acumula erros e sucessos"
        echo "     • Persistência total em SQLite"
        echo ""
        echo "  🎓 Fine-Tuning Engine"
        echo "     • Mistral AI"
        echo "     • OpenAI"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📡 INFORMAÇÕES:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "  PID: $PID"
        echo "  Log: /root/inteligencia_definitiva.log"
        echo "  Database: /root/inteligencia_definitiva.db"
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "💡 COMANDOS:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo ""
        echo "  Parar:      ./STOP_DEFINITIVA.sh"
        echo "  Status:     ./STATUS_DEFINITIVA.sh"
        echo "  Ver logs:   tail -f /root/inteligencia_definitiva.log"
        echo ""
        
        # Mostrar primeiras linhas do log
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📝 LOG INICIAL:"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        tail -15 /root/inteligencia_definitiva.log
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
    else
        echo "❌ Erro ao iniciar"
        cat /root/inteligencia_definitiva.out
        exit 1
    fi
else
    echo "❌ PID file não criado"
    cat /root/inteligencia_definitiva.out
    exit 1
fi
