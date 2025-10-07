#!/bin/bash
# Monitor do Sistema NEEDLE IA³ 24/7

echo "========================================"
echo "🔍 MONITOR NEEDLE IA³ 24/7 SYSTEM"
echo "========================================"
echo ""

# Verificar processo
echo "📊 STATUS DO PROCESSO:"
if ps aux | grep -v grep | grep -q "needle_ia3_ultimate_fixed.py"; then
    echo "✅ Sistema RODANDO"
    ps aux | grep "needle_ia3_ultimate_fixed.py" | grep -v grep | awk '{print "   PID:", $2, "| CPU:", $3"% | MEM:", $4"% | Runtime:", $10}'
else
    echo "❌ Sistema PARADO"
fi
echo ""

# Última geração
echo "🧬 ÚLTIMA GERAÇÃO:"
if [ -f "/root/needle_ia3_active.log" ]; then
    tail -1 /root/needle_ia3_active.log | grep "Gen" | awk '{print "   "$3, $4, "| Fitness:", $8, "| Best:", $10}'
fi
echo ""

# Checkpoints
echo "💾 CHECKPOINTS SALVOS:"
ls -lah /root/needle_ia3_checkpoints/*.pt 2>/dev/null | wc -l | awk '{print "   Total:", $1, "checkpoints"}'
if [ -d "/root/needle_ia3_checkpoints" ]; then
    latest=$(ls -t /root/needle_ia3_checkpoints/*.pt 2>/dev/null | head -1)
    if [ -n "$latest" ]; then
        echo "   Último: $(basename $latest)"
    fi
fi
echo ""

# Métricas recentes
echo "📈 MÉTRICAS RECENTES:"
tail -5 /root/needle_ia3_active.log 2>/dev/null | grep "Gen" | while read line; do
    echo "   $line" | cut -d' ' -f3-
done
echo ""

# Uso de recursos
echo "💻 USO DE RECURSOS:"
echo "   CPU Total: $(ps aux | grep -E "needle|fazenda|teis|darwin|ia3" | grep -v grep | awk '{sum+=$3} END {print sum"%"}')"
echo "   Processos IA ativos: $(ps aux | grep -E "needle|fazenda|teis|darwin|ia3" | grep -v grep | wc -l)"
echo ""

echo "========================================"
echo "Para parar o sistema: kill $(ps aux | grep needle_ia3_ultimate_fixed.py | grep -v grep | awk '{print $2}' | head -1) 2>/dev/null"
echo "Para ver logs: tail -f /root/needle_ia3_active.log"
echo "========================================" 