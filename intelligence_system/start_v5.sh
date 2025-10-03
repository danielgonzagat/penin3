#!/bin/bash
# Start Intelligence System V5.0 - EXTRACTED MERGE

echo "🔥 Starting Intelligence System V5.0 - EXTRACTED MERGE"

# Stop old versions
for ver in v3 v4; do
    if pgrep -f "system_${ver}" > /dev/null; then
        echo "⏹️  Stopping ${ver}..."
        pkill -f "system_${ver}"
        sleep 2
    fi
done

# Start V5.0
nohup python3 core/system_v5_extracted_merge.py > logs/v5_output.log 2>&1 &

sleep 3

if pgrep -f "system_v5_extracted_merge.py" > /dev/null; then
    PID=$(pgrep -f "system_v5_extracted_merge.py")
    echo "✅ V5.0 started successfully!"
    echo "   PID: $PID"
    echo "📄 Logs: tail -f logs/v5_output.log"
    echo "📊 Status: ./status_v5.sh"
else
    echo "❌ Failed to start V5.0"
    echo "📄 Check logs: cat logs/v5_output.log"
fi

