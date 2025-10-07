#!/bin/bash
# 🚀 START BRAIN 24/7
# Script para iniciar o cérebro como daemon

echo "================================"
echo "🧠 STARTING BRAIN DAEMON 24/7"
echo "================================"
echo ""

cd /root/UNIFIED_BRAIN

# Check if already running
if [ -f brain_daemon.pid ]; then
    PID=$(cat brain_daemon.pid)
    if ps -p $PID > /dev/null 2>&1; then
        echo "⚠️  Brain daemon already running (PID: $PID)"
        exit 1
    else
        rm brain_daemon.pid
    fi
fi

# Start daemon in background
nohup python3 brain_daemon_real_env.py > v3_final.log 2>&1 &
PID=$!

# Save PID
echo $PID > brain_v3_final.pid

echo "✅ Brain daemon started!"
echo "   PID: $PID"
echo "   Log: /root/UNIFIED_BRAIN/v3_final.log"
echo ""
echo "Commands:"
echo "   • View log: tail -f /root/UNIFIED_BRAIN/v3_final.log"
echo "   • Stop: kill $PID"
echo "   • Status: ps -p $PID"
echo ""
echo "🎊 Brain is now running 24/7!"
