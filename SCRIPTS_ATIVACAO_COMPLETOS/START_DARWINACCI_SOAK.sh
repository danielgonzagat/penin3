#!/bin/bash
# Quick launcher for Darwinacci + V7 + PENIN³ soak test

echo "🌟 DARWINACCI SOAK TEST LAUNCHER"
echo "================================"
echo ""

CYCLES=${1:-200}

echo "Configuration:"
echo "  Cycles: ${CYCLES}"
echo "  APIs: 4/6 active (Mistral, Gemini, Anthropic, Grok)"
echo "  Darwinacci: Active"
echo "  JSON Logs: Enabled"
echo "  Prometheus: :9108"
echo ""

read -p "Press ENTER to start or Ctrl+C to cancel..."

cd /root
bash intelligence_system/scripts/run_soak_200.sh

echo ""
echo "✅ Soak test started!"
echo ""
echo "Monitor with:"
echo "  curl http://localhost:9108/metrics"
echo "  tail -f intelligence_system/data/soak_*/run.log"
