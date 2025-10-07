#!/usr/bin/env bash
set -euo pipefail

ROUNDS_PER_BATCH=${ROUNDS_PER_BATCH:-6}
BATCHES=${BATCHES:-5}
POP=${POP:-12}
GEN=${GEN:-2}
EPOCHS=${EPOCHS:-8}
LR=${LR:-0.002}
GRID=${GRID:-5}

for i in $(seq 1 "$BATCHES"); do
  echo "=== BATCH $i/$BATCHES ===";
  python3 CUBIC_FARM_PHASE2.py --rounds "$ROUNDS_PER_BATCH" --generations "$GEN" --population "$POP" --train-epochs "$EPOCHS" --initial-lr "$LR" --grid-size "$GRID";
  python3 aggregate_phase2_reports.py;
  sleep 1;
done

echo "Done. Aggregate at cubic_farm_phase2_reports/phase2_aggregate.json"