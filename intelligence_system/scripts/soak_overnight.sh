#!/bin/bash
# Run overnight soak test with full monitoring

CYCLES=${1:-500}
LOG_DIR="/root/intelligence_system/data/soak_$(date +%Y%m%d_%H%M%S)"

mkdir -p "${LOG_DIR}"

echo "🌙 Starting overnight soak test"
echo "   Cycles: ${CYCLES}"
echo "   Log dir: ${LOG_DIR}"

cd /root

# Load environment
if [ -f "/root/intelligence_system/.env" ]; then
    set -a
    source /root/intelligence_system/.env
    set +a
fi

# Start system
nohup python3 intelligence_system/core/unified_agi_system.py ${CYCLES} \
    > "${LOG_DIR}/unified_agi.log" 2>&1 &

PID=$!
echo ${PID} > "${LOG_DIR}/soak.pid"

echo "✅ Started with PID ${PID}"
echo ""
echo "📊 Monitor with:"
echo "   tail -f ${LOG_DIR}/unified_agi.log"
echo "   curl http://localhost:9108/metrics"
echo "   cat intelligence_system/data/exports/timeline_metrics.csv | tail -20"
echo ""
echo "🛑 Stop with:"
echo "   kill ${PID}"