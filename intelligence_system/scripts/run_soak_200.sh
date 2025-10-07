#!/bin/bash
# Run 200-cycle soak test with monitoring

echo "🔬 Starting 200-cycle soak test"

cd /root

# Ensure .env is loaded
if [ -f "/root/intelligence_system/.env" ]; then
    set -a
    source /root/intelligence_system/.env
    set +a
fi

# Create soak directory
SOAK_DIR="/root/intelligence_system/data/soak_200_$(date +%Y%m%d_%H%M%S)"
mkdir -p "${SOAK_DIR}"

echo "📁 Soak directory: ${SOAK_DIR}"

# Run with timeout (2 hours max)
timeout 7200 python3 intelligence_system/core/unified_agi_system.py 200 \
    > "${SOAK_DIR}/run.log" 2>&1 &

PID=$!
echo ${PID} > "${SOAK_DIR}/soak.pid"

echo "✅ Started PID ${PID}"
echo ""
echo "📊 Monitor:"
echo "   tail -f ${SOAK_DIR}/run.log"
echo "   curl http://localhost:9108/metrics"
echo ""
echo "🛑 Stop:"
echo "   kill ${PID}"
echo ""
echo "📈 Results will be in:"
echo "   ${SOAK_DIR}/run.log"
echo "   intelligence_system/data/exports/timeline_metrics.csv"
echo "   intelligence_system/data/unified_worm.jsonl"