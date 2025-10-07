#!/bin/bash
# Clean start V6 with all patches

echo "🧹 Cleaning up..."
pkill -9 -f system_v6 2>/dev/null
sleep 2

echo "🚀 Starting V6 (clean)..."
cd /root/intelligence_system
nohup python3 -u core/system_v6_ia3_complete.py > logs/v6_output.log 2>&1 &
V6_PID=$!

sleep 3

if ps -p $V6_PID > /dev/null; then
    echo "✅ V6 started successfully (PID: $V6_PID)"
else
    echo "❌ V6 failed to start"
    tail -50 logs/v6_output.log
    exit 1
fi

