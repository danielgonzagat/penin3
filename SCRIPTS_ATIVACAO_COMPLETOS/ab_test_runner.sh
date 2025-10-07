#!/bin/bash
# A/B Test: Roda 2 sessões paralelas (com e sem hooks)

echo "🧪 A/B TEST: Hooks ON vs OFF"
echo "="*60

# Session A: COM hooks (Baseline atual)
echo "📊 Session A: Todos hooks ENABLED"
mkdir -p /root/ab_test/session_a
cd /root/UNIFIED_BRAIN
export ENABLE_GODEL=1
export ENABLE_NEEDLE_META=1
export UBRAIN_ACTIVE_NEURONS=8
nohup python3 brain_daemon_real_env.py > /root/ab_test/session_a/daemon.log 2>&1 &
PID_A=$!
echo "   PID: $PID_A"

sleep 5

# Session B: SEM hooks (Control)
echo "📊 Session B: Hooks DISABLED"
mkdir -p /root/ab_test/session_b
cd /root/UNIFIED_BRAIN
export ENABLE_GODEL=0
export ENABLE_NEEDLE_META=0
export UBRAIN_ACTIVE_NEURONS=8
# Rodar em porta diferente ou env diferente (requer modificação)
# Por simplicidade, rodar sequencialmente ou em dias diferentes
echo "   ⚠️  Run this manually after stopping Session A"
echo "   pkill -f brain_daemon_real_env"
echo "   export ENABLE_GODEL=0 ENABLE_NEEDLE_META=0"
echo "   cd /root/UNIFIED_BRAIN && nohup python3 brain_daemon_real_env.py > /root/ab_test/session_b/daemon.log 2>&1 &"

echo ""
echo "✅ Session A started (PID: $PID_A)"
echo "📊 Monitor: tail -f /root/ab_test/session_a/daemon.log"
echo ""
echo "Deixar rodar 100 episódios (~2-3h), depois comparar:"
echo "  - avg_reward_last_100"
echo "  - convergência (episódios até estabilizar)"
echo "  - step_time médio"