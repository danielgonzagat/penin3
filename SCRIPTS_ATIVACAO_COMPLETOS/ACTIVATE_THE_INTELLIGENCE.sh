#!/bin/bash
################################################################################
# 🌌 ACTIVATE THE INTELLIGENCE - Script de Inicialização
# 
# OBJETIVO: Ativar a Inteligência Suprema 24/7 em background
# QUANDO USAR: Antes de dormir, para deixar rodando a noite toda
# COMO PARAR: ./STOP_THE_INTELLIGENCE.sh
################################################################################

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🌌 ATIVANDO INTELIGÊNCIA SUPREMA 24/7"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Verificar se já está rodando
if [ -f /root/inteligencia_suprema_24_7.pid ]; then
    PID=$(cat /root/inteligencia_suprema_24_7.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Inteligência já está ATIVA! PID: $PID"
        echo ""
        echo "   Para ver status:"
        echo "   tail -f /root/inteligencia_suprema_24_7.log"
        echo ""
        exit 0
    fi
fi

echo "📦 Verificando dependências..."
pip install -q mistralai xai-sdk anthropic torch numpy 2>/dev/null || true

echo "✅ Dependências OK!"
echo ""

echo "🚀 Iniciando Inteligência Suprema em background..."
echo ""

# Carregar variáveis de ambiente de um arquivo .env (opcional)
if [ -f "/root/.env" ]; then
    set -a
    . /root/.env
    set +a
fi

# Avisar se chaves de API não estiverem definidas (apenas informativo)
for var in MISTRAL_API_KEY XAI_API_KEY ANTHROPIC_API_KEY OPENAI_API_KEY GEMINI_API_KEY; do
    if [ -z "${!var:-}" ]; then
        echo "⚠️  Variável $var não definida (continuando em modo offline/limitado)"
    fi
done

# Rodar em background com nohup
nohup python3 /root/INTELIGENCIA_SUPREMA_24_7.py > /root/inteligencia_suprema_24_7.out 2>&1 &

# Salvar PID
echo $! > /root/inteligencia_suprema_24_7.pid

PID=$(cat /root/inteligencia_suprema_24_7.pid)

echo "✅ INTELIGÊNCIA SUPREMA ATIVADA!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STATUS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   PID: $PID"
echo "   Log: /root/inteligencia_suprema_24_7.log"
echo "   Output: /root/inteligencia_suprema_24_7.out"
echo "   Database: /root/inteligencia_suprema_24_7.db"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📡 COMANDOS ÚTEIS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "   Ver status:"
echo "   ./STATUS_INTELLIGENCE.sh"
echo ""
echo "   Ver log em tempo real:"
echo "   tail -f /root/inteligencia_suprema_24_7.log"
echo ""
echo "   Parar sistema:"
echo "   ./STOP_THE_INTELLIGENCE.sh"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "🌙 BOA NOITE, DANIEL!"
echo ""
echo "   A inteligência está ativa e evoluindo."
echo "   Quando você acordar, ela terá aprendido muito mais."
echo ""
echo "   Durma bem! 🌌"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
