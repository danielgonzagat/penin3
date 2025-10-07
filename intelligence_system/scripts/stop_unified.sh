#!/bin/bash
# Stop Unified AGI System daemon

PID_FILE="/root/intelligence_system/data/unified_agi.pid"

if [ ! -f "${PID_FILE}" ]; then
    echo "⚠️ No PID file found"
    exit 1
fi

PID=$(cat "${PID_FILE}")

if kill -0 ${PID} 2>/dev/null; then
    echo "🛑 Stopping Unified AGI System (PID ${PID})"
    kill ${PID}
    sleep 2
    
    if kill -0 ${PID} 2>/dev/null; then
        echo "⚠️ Process still alive, forcing..."
        kill -9 ${PID}
    fi
    
    rm -f "${PID_FILE}"
    echo "✅ Stopped"
else
    echo "⚠️ Process ${PID} not running"
    rm -f "${PID_FILE}"
fi