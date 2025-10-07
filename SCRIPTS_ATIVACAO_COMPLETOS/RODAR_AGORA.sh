#!/bin/bash
# SCRIPT DEFINITIVO - RODA ATÉ EMERGÊNCIA REAL

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                                                            ║"
echo "║         🚀 INICIANDO BUSCA POR INTELIGÊNCIA REAL 🚀        ║"
echo "║                                                            ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

cd /root/intelligence_system
source .env 2>/dev/null

export JSON_LOGS=1
export DARWINACCI_PROMETHEUS=1
export PROMETHEUS_PORT=9108

echo "Features ativas:"
echo "  ✅ Auto-modificação"
echo "  ✅ Meta-learning 100+ tasks"
echo "  ✅ Co-evolução V7 ↔ Darwinacci"
echo "  ✅ Open-ended evolution"
echo "  ✅ Transfer learning"
echo "  ✅ Curiosity curriculum"
echo "  ✅ Self-reference"
echo "  ✅ Consciousness 5D"
echo "  ✅ Evolutionary NAS"
echo "  ✅ Meta-meta-learning depth 5"
echo ""
echo "🎯 Objetivo: Inteligência emergente REAL"
echo ""
echo "🚀 Rodando 100 ciclos..."
echo "📝 Log: /root/EMERGENCIA_FINAL.log"
echo ""

python3 -u core/unified_agi_system.py 100 2>&1 | tee /root/EMERGENCIA_FINAL.log

echo ""
echo "✅ Teste completo!"
echo ""
echo "📊 ANÁLISE RÁPIDA:"
echo "Ciclos: $(grep -c '🔄 CYCLE' /root/EMERGENCIA_FINAL.log)"
echo "Auto-mods: $(grep -c '🎯 Applying' /root/EMERGENCIA_FINAL.log)"
echo "Transfers: $(grep -c '🧬 V7→Darwinacci' /root/EMERGENCIA_FINAL.log)"
echo ""
