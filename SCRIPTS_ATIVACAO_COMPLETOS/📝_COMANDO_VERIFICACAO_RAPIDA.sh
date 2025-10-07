#!/bin/bash
# VERIFICAÇÃO RÁPIDA - Use quando voltar!

echo "════════════════════════════════════════════════════════"
echo "🔍 VERIFICAÇÃO RÁPIDA DO SISTEMA"
echo "Data: $(date)"
echo "════════════════════════════════════════════════════════"
echo ""

# Episode e reward
echo "📊 Brain 254 neurônios:"
tail -5 /root/brain_254_FINAL_*.log 2>/dev/null | grep "Ep [0-9]*:" | tail -1 || echo "  ⚠️ Sem dados"
echo ""

# Surprises
echo "🎯 Surprises detectadas:"
sqlite3 /root/intelligence_system/data/emergence_surprises.db \
  "SELECT COUNT(*) as total FROM surprises" 2>/dev/null || echo "  0"
echo ""

# Checkpoints
echo "💾 Checkpoints salvos:"
ls /root/UNIFIED_BRAIN/checkpoints/*.pt 2>/dev/null | wc -l || echo "  0"
echo ""

# Load
echo "⚡ Sistema:"
uptime | awk '{print "  Load:", $10, $11, $12}'
echo ""

# Darwin V2
echo "🧬 Darwin V2:"
tail -2 /root/darwin_V2_REAL_MNIST_*.log 2>/dev/null | grep "acc=" | tail -1 || echo "  ⚠️ Sem dados"
echo ""

echo "════════════════════════════════════════════════════════"
echo "✅ Para detalhes: cat /root/📖_GUIA_RAPIDO_VERIFICACAO.md"
echo "════════════════════════════════════════════════════════"
