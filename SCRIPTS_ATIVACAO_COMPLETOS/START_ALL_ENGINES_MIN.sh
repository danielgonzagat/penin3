#!/bin/bash
set -e

# Minimal start for all engines and catalysts
# Avoids interactive prompts and complex heredocs

# 0) Disable Incompletude Infinita if present
pkill -f incompletude_daemon 2>/dev/null || true
pkill -f ".incompletude_daemon.py" 2>/dev/null || true

# 1) Meta-Learner (corrected)
pkill -f "META_LEARNER_REALTIME.py" 2>/dev/null || true
nohup python3 -u /root/META_LEARNER_REALTIME.py > /root/meta_learner_CORRIGIDO_FINAL.log 2>&1 &

# 2) System Connector (fixed schema)
pkill -f "EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py" 2>/dev/null || true
nohup python3 -u /root/EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py 100 60 > /root/system_connector_CORRIGIDO.log 2>&1 &

# 3) Darwin STORM evolution runner
pkill -f "run_emergence_blocks_STORM.py" 2>/dev/null || true
nohup python3 -u /root/run_emergence_blocks_STORM.py > /root/darwin_STORM.log 2>&1 &

# 4) Cross-Pollination AUTO (fixed)
pkill -f "CROSS_POLLINATION_AUTO_FIXED.py" 2>/dev/null || true
nohup python3 -u /root/CROSS_POLLINATION_AUTO_FIXED.py > /root/cross_pollination_AUTO.log 2>&1 &

# 5) Self-Reflection (fixed, applies changes)
pkill -f "SELF_REFLECTION_ENGINE_FIXED.py" 2>/dev/null || true
nohup python3 -u /root/SELF_REFLECTION_ENGINE_FIXED.py > /root/self_reflection_ENGINE.log 2>&1 &

# 6) Dynamic Fitness Engine
pkill -f "DYNAMIC_FITNESS_ENGINE.py" 2>/dev/null || true
nohup python3 -u /root/DYNAMIC_FITNESS_ENGINE.py > /root/dynamic_fitness_ENGINE.log 2>&1 &

# 7) V7 ↔ Darwin Realtime Bridge
pkill -f "V7_DARWIN_REALTIME_BRIDGE.py" 2>/dev/null || true
nohup python3 -u /root/V7_DARWIN_REALTIME_BRIDGE.py > /root/v7_darwin_BRIDGE.log 2>&1 &

# Optional: non-interactive emergency hotpatches (commented by default)
# nohup python3 -u /root/EMERGENCY_HOTPATCH_2_DARWIN_LLAMA_LOOP.py > /root/darwin_llama_LOOP.log 2>&1 &
# nohup python3 -u /root/EMERGENCY_HOTPATCH_3_INJECT_MASSIVE_NEURONS.py > /root/massive_injection.log 2>&1 &

sleep 2
echo "✅ All engines launched (check logs under /root/*.log)"
