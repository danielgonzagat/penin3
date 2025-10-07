#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP=$(date +%Y%m%d_%H%M%S)
BASELINE_LOG="/root/ab_baseline_${STAMP}.log"
TREAT_LOG="/root/ab_treatment_${STAMP}.log"

# Baseline: V7 solo (no PENIN³)
echo "🚀 Starting baseline V7-only 100 cycles... (log: ${BASELINE_LOG})"
nohup python3 - <<'PY' > "$BASELINE_LOG" 2>&1 &
import logging, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO)
from core.system_v7_ultimate import IntelligenceSystemV7

v7 = IntelligenceSystemV7()
for i in range(100):
    v7.run_cycle()
    if (i+1) % 10 == 0:
        print(f"Cycle {i+1}/100: MNIST={v7.best['mnist']:.1f}%, CartPole={v7.best['cartpole']:.1f}")
status = v7.get_system_status()
print(f"FINAL: IA3={status['ia3_score_calculated']:.1f}% | MNIST={status['best_mnist']:.1f}% | CartPole={status['best_cartpole']:.1f}")
PY

BASE_PID=$!
echo $BASE_PID > /root/ab_baseline.pid

# Treatment: Unified (V7 + PENIN³)
echo "🚀 Starting treatment Unified 100 cycles... (log: ${TREAT_LOG})"
nohup python3 test_100_cycles_real.py 100 > "$TREAT_LOG" 2>&1 &
TREAT_PID=$!
echo $TREAT_PID > /root/ab_treatment.pid

cat <<EOM
✅ A/B started
  Baseline PID: $(cat /root/ab_baseline.pid)  Log: ${BASELINE_LOG}
  Treatment PID: $(cat /root/ab_treatment.pid) Log: ${TREAT_LOG}
Monitor with:
  tail -f ${BASELINE_LOG}
  tail -f ${TREAT_LOG}
EOM
