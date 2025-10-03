#!/bin/bash
cd /root/intelligence_system
nohup python3 -u core/system.py > logs/system.out 2>&1 &
echo $! > system.pid
echo "✅ System started (PID: $(cat system.pid))"
