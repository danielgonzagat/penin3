#!/bin/bash
# Start Intelligence System V3.0 - MEGA-MERGE Edition

echo "🚀 Starting Intelligence System V3.0..."

# Kill old version if running
if [ -f intelligence.pid ]; then
    OLD_PID=$(cat intelligence.pid)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "⏹️  Stopping old system (PID: $OLD_PID)"
        kill $OLD_PID
        sleep 2
    fi
fi

# Start V3
nohup python3 -u core/system_v3_integrated.py > logs/v3_output.log 2>&1 &
V3_PID=$!

echo $V3_PID > v3.pid
echo "✅ V3.0 started (PID: $V3_PID)"
echo "📝 Logs: logs/v3_output.log"
echo "📊 Watch: tail -f logs/v3_output.log"
