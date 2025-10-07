#!/bin/bash
# Start Intelligence System V6.0 - IA³ COMPLETE

echo "🔥 Starting Intelligence System V6.0 - IA³ COMPLETE"

# Stop old versions
for ver in v3 v4 v5; do
    if pgrep -f "system_${ver}" > /dev/null; then
        echo "⏹️  Stopping ${ver}..."
        pkill -f "system_${ver}"
        sleep 2
    fi
done

# Start V6.0
nohup python3 core/system_v6_ia3_complete.py > logs/v6_output.log 2>&1 &

sleep 3

if pgrep -f "system_v6_ia3_complete.py" > /dev/null; then
    PID=$(pgrep -f "system_v6_ia3_complete.py")
    echo "✅ V6.0 started successfully!"
    echo "   PID: $PID"
    echo "📊 Status: ./status_v6.sh"
    echo "📄 Logs: tail -f logs/v6_output.log"
else
    echo "❌ Failed to start V6.0"
    tail -20 logs/v6_output.log
    exit 1
fi

