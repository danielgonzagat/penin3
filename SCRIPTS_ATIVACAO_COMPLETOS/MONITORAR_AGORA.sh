#!/bin/bash
# Monitora UNIFIED_BRAIN em tempo real

echo "════════════════════════════════════════════════════════════════════════════════"
echo "👁️  MONITORANDO UNIFIED_BRAIN EM TEMPO REAL"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "PID: 2024297"
echo "Log: /root/UNIFIED_BRAIN/brain_fixed.log"
echo ""
echo "Pressione Ctrl+C para parar o monitoramento"
echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo ""

tail -f /root/UNIFIED_BRAIN/brain_fixed.log | grep --line-buffered "Ep \|Episode\|reward\|best\|SUCESSO"
