#!/bin/bash
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "🔍 MONITOR CONTÍNUO - Capturando Todas as Evidências"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "PID: 1286736"
echo "Início: $(date '+%Y-%m-%d %H:%M:%S')"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Monitor principal
tail -f /root/500_cycles_output.log 2>&1 | grep --line-buffered -E "🔥|🫀|🔁|♻️|HOTPATCH|CYCLE|swap.*optimizer|optimizer-manager|reloaded:|applied:|unknown_op:|Provider.*DOWN|API Summary"

