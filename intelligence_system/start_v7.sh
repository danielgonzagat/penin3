#!/bin/bash
# Start Intelligence System V7.0 - ULTIMATE MERGE

echo "🔥 Starting Intelligence System V7.0 - ULTIMATE MERGE"

# Stop old versions
pkill -9 -f "system_v[0-9]" 2>/dev/null
sleep 2

# Start V7.0
cd /root/intelligence_system
nohup python3 -u core/system_v7_ultimate.py > logs/v7_output.log 2>&1 &
V7_PID=$!

sleep 3

if ps -p $V7_PID > /dev/null 2>&1; then
    echo "✅ V7.0 started successfully!"
    echo "   PID: $V7_PID"
    echo "📄 Logs: tail -f logs/v7_output.log"
    echo "📊 Monitor: sqlite3 data/intelligence.db 'SELECT * FROM cycles ORDER BY cycle DESC LIMIT 5;'"
else
    echo "❌ V7.0 failed to start"
    tail -30 logs/v7_output.log
    exit 1
fi

