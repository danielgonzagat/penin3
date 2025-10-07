#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

echo "⚠️  RESETTING system for fresh evolution observation..."
rm -f data/intelligence.db || true
rm -f models/ppo_cartpole_v7.pth || true
rm -f models/meta_learner.pth || true

stamp=$(date +%Y%m%d_%H%M%S)
out="/root/test_100_real_${stamp}.log"

echo "🚀 Starting 100 cycles (this may take ~4h). Logs: ${out}"
nohup python3 test_100_cycles_real.py 100 > "${out}" 2>&1 &
echo $! > /root/test_100.pid

echo "✅ Started. PID: $(cat /root/test_100.pid)"
echo "Monitor with: tail -f ${out}"
echo "Export metrics after run with:"
echo "  python3 tools/export_unified_metrics.py data/unified_metrics_${stamp}.csv"
