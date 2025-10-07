#!/bin/bash
# Monitor passivo - apenas observa o Qwen agindo sozinho

echo "👁️ OBSERVANDO QWEN EMERGENTE - SEM INTERFERÊNCIA"
echo "=================================================="
echo ""

while true; do
    clear
    echo "🌟 QWEN AUTÔNOMO - MONITORAMENTO PASSIVO"
    echo "=========================================="
    echo ""
    echo "📊 STATUS DO PROCESSO:"
    ps aux | grep qwen_complete_system.py | grep -v grep | head -1
    echo ""
    echo "📝 ÚLTIMAS 30 LINHAS DO LOG:"
    echo "----------------------------"
    tail -30 /root/qwen_autonomous_pure.log 2>/dev/null || echo "Aguardando logs..."
    echo ""
    echo "⏰ $(date)"
    echo "🔄 Atualizando em 10 segundos... (Ctrl+C para sair)"
    sleep 10
done
