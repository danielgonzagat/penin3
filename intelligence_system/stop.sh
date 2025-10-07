#!/bin/bash
if [ -f system.pid ]; then
    PID=$(cat system.pid)
    kill $PID 2>/dev/null
    rm system.pid
    echo "✅ System stopped (PID: $PID)"
else
    echo "⚠️  No PID file found"
fi
