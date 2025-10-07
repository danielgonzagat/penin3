#!/bin/bash
# Start Unified AGI System as daemon

CYCLES=${1:-100}
LOG_FILE="/root/intelligence_system/data/unified_agi.log"

echo "🚀 Starting Unified AGI System (${CYCLES} cycles)"
echo "   Log: ${LOG_FILE}"

cd /root

# Export env if .env exists
if [ -f "/root/intelligence_system/.env" ]; then
    set -a
    source /root/intelligence_system/.env
    set +a
    echo "✅ Environment loaded from .env"
fi

# Start in background
nohup python3 intelligence_system/core/unified_agi_system.py ${CYCLES} \
    > "${LOG_FILE}" 2>&1 &

PID=$!
echo ${PID} > /root/intelligence_system/data/unified_agi.pid

echo "✅ Started with PID ${PID}"
echo "   Monitor: tail -f ${LOG_FILE}"
echo "   Stop: bash intelligence_system/scripts/stop_unified.sh"