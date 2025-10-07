#!/bin/bash
# Status Intelligence System V3.0

if [ ! -f v3.pid ]; then
    echo "❌ V3 not running (no PID file)"
    exit 1
fi

PID=$(cat v3.pid)

if ps -p $PID > /dev/null 2>&1; then
    echo "✅ V3.0 RUNNING (PID: $PID)"
    ps -p $PID -o pid,etime,vsz,pmem,comm | tail -1
    echo ""
    echo "Recent logs:"
    tail -20 logs/v3_output.log 2>/dev/null || tail -20 logs/intelligence_v3.log 2>/dev/null || echo "No logs yet"
else
    echo "❌ V3 NOT RUNNING (PID $PID not found)"
    echo ""
    echo "Last logs:"
    tail -20 logs/v3_output.log 2>/dev/null || tail -20 logs/intelligence_v3.log 2>/dev/null
    exit 1
fi
