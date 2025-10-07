#!/usr/bin/env bash
set -euo pipefail
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] SPAWN: id=$NEURON_ID reason=$REASON" >> /root/ia3_darwin_hooks.log
echo "✅ Neurônio $NEURON_ID logado"
