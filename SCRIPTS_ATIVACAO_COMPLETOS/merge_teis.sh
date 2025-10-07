#!/usr/bin/env bash
set -euo pipefail
LOG="/root/teis_logs/merge_run_$(date +%F_%H%M%S).log"
echo "== TEIS FULL MERGE ==" | tee -a "$LOG"
python3 /root/teis_tools/merge_all_checkpoints.py 2>&1 | tee -a "$LOG"
echo "OK. Veja ponteiro: /root/teis_checkpoints/FINAL_MERGED_LATEST.json" | tee -a "$LOG"
