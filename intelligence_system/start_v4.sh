#!/bin/bash
# Start Intelligence System V4.0

echo "🔥 Starting Intelligence System V4.0 - FULL MERGE"

# Stop old version if running
if pgrep -f "system_v3_integrated.py" > /dev/null; then
    echo "⏹️  Stopping V3.0..."
    pkill -f "system_v3_integrated.py"
    sleep 2
fi

# Start V4.0
nohup python3 core/system_v4_full_merge.py > logs/v4_output.log 2>&1 &

sleep 2

if pgrep -f "system_v4_full_merge.py" > /dev/null; then
    echo "✅ V4.0 started successfully!"
    echo "📄 Logs: tail -f logs/v4_output.log"
else
    echo "❌ Failed to start V4.0"
    echo "📄 Check logs: cat logs/v4_output.log"
fi

