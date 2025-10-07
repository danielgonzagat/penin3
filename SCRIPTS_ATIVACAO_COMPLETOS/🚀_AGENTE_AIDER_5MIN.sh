#!/bin/bash
echo "🚀 AGENTE AUTÔNOMO EM 5 MINUTOS"
echo "=============================="
pip install -q aider-chat
export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY /root/.env | cut -d'=' -f2)
cd /root && git add -A && git commit -m "SAFE: Pre-agent $(date)" && git tag "safe-$(date +%s)"
echo "✅ Pronto! Execute:"
echo "   cd /root/intelligence_system"
echo "   aider --model claude-3-5-sonnet-20241022"
