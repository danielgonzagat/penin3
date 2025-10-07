#!/bin/bash
# Awaken com limites seguros
export UBRAIN_ACTIVE_NEURONS=8
export UBRAIN_TOP_K=8
export UBRAIN_NUM_STEPS=1
export ENABLE_GODEL=1
export ENABLE_NEEDLE_META=1

cd /root/UNIFIED_BRAIN

echo "=== AWAKEN SAFE MODE ==="
echo "Limites: 8 neurons, top_k=8, steps=1"
echo "Hooks: Gödel + Needle enabled"
echo ""

# Timeout de 10 minutos para evitar travamento
timeout 600 python3 awaken_all_systems.py 2>&1 | tee /root/awaken_safe.log

EXIT_CODE=$?
if [ $EXIT_CODE -eq 124 ]; then
    echo "TIMEOUT: Awaken excedeu 10 minutos"
    exit 1
elif [ $EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: Awaken completo"
    exit 0
else
    echo "ERROR: Exit code $EXIT_CODE"
    exit $EXIT_CODE
fi