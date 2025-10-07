#!/bin/bash
# 🔥 START REAL INTELLIGENCE
# Inicia sistema com ambiente REAL = emergência real

echo "================================"
echo "🔥 STARTING REAL INTELLIGENCE"
echo "================================"
echo ""

cd /root/UNIFIED_BRAIN

# Stop old daemon if running
if [ -f brain_daemon.pid ]; then
    OLD_PID=$(cat brain_daemon.pid)
    kill $OLD_PID 2>/dev/null
    sleep 2
    rm brain_daemon.pid
fi

# Start REAL environment brain
echo "Starting brain with REAL environment (CartPole)..."
nohup python3 brain_daemon_real_env.py > real_brain.log 2>&1 &
PID=$!

echo $PID > brain_real.pid

echo "✅ Real Intelligence started!"
echo "   PID: $PID"
echo "   Log: /root/UNIFIED_BRAIN/real_brain.log"
echo ""
echo "🔥 This brain will LEARN FOR REAL:"
echo "   • Real environment (CartPole)"
echo "   • Real feedback loop"
echo "   • Real consequences"
echo "   • Real learning progress"
echo ""
echo "Commands:"
echo "   • View learning: tail -f /root/UNIFIED_BRAIN/real_brain.log"
echo "   • Stop: kill $PID"
echo ""
echo "🎯 Watch it learn to balance!"
