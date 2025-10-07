#!/bin/bash
if [ -f system.pid ]; then
    PID=$(cat system.pid)
    if ps -p $PID > /dev/null; then
        echo "✅ RUNNING (PID: $PID)"
        ps -p $PID -o pid,etime,%cpu,%mem,cmd --no-headers
        echo ""
        tail -20 logs/intelligence.log | grep -E "(CYCLE|MNIST|CartPole|RECORD|API)"
    else
        echo "❌ NOT RUNNING (stale PID)"
        rm system.pid
    fi
else
    echo "⚠️  NOT RUNNING (no PID file)"
fi
