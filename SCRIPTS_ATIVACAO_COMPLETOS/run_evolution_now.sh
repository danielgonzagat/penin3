#!/bin/bash
# Script para executar o ciclo de evolução com todas as APIs configuradas

echo "🚀 Configurando ambiente e executando evolução do THE NEEDLE..."

# Configurar API keys
export OPENAI_API_KEY="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"
export MISTRAL_API_KEY="AMTeAQrzudpGvU2jkU9hVRvSsYr1hcni"
export GEMINI_API_KEY="AIzaSyA2BuXahKz1hwQCTAeuMjOxje8lGqEqL4k"
export DEEPSEEK_API_KEY="sk-19c2b1d0864c4a44a53d743fb97566aa"
export ANTHROPIC_API_KEY="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"
export XAI_API_KEY="xai-sHbr1x7v2vpfDi657DtU64U53UM6OVhs4FdHeR1Ijk7jRUgU0xmo6ff8SF7hzV9mzY1wwjo4ChYsCDog"
export EVOLVER_LIVE_API="1"

echo "✅ API keys configuradas"
echo "🔄 Iniciando ciclo de evolução..."
echo "="=========================================="

# Executar o orchestrator
cd /root
python3 NEEDLE_AUTO_EVOLUTION_ORCHESTRATOR.py

echo ""
echo "="=========================================="
echo "✅ Ciclo de evolução finalizado!"