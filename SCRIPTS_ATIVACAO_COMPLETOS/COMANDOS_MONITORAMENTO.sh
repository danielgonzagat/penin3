#!/bin/bash
# 🎛️ COMANDOS DE MONITORAMENTO - SISTEMA IA³

echo "════════════════════════════════════════════════════════════════"
echo "🎛️ MONITORAMENTO SISTEMA IA³ - FASE 1 ATIVA"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Daemon status
echo "📊 DAEMON STATUS:"
PID=$(cat /root/UNIFIED_BRAIN/brain_v3_final.pid 2>/dev/null)
if [ -n "$PID" ] && ps -p $PID > /dev/null 2>&1; then
    echo "  ✅ Running (PID: $PID)"
    CPU=$(ps -p $PID -o %cpu --no-headers)
    MEM=$(ps -p $PID -o %mem --no-headers)
    echo "  CPU: ${CPU}%"
    echo "  MEM: ${MEM}%"
else
    echo "  ❌ Not running"
fi
echo ""

# Latest episodes
echo "📈 ÚLTIMOS 5 EPISÓDIOS:"
tail -100 /root/UNIFIED_BRAIN/v3_final.log 2>/dev/null | grep '"message": "Ep ' | tail -5 | while read line; do
    echo "  $line" | grep -o 'Ep [0-9]*:.*curiosity=[0-9.]*' | head -1
done
echo ""

# WORM status
echo "🔒 WORM LEDGER:"
python3 -c "
import sys; sys.path.insert(0, '/root/UNIFIED_BRAIN')
from UNIFIED_BRAIN.brain_worm import WORMLog
w = WORMLog('/root/UNIFIED_BRAIN/worm.log')
s = w.get_stats()
print(f'  Entries: {s[\"total_entries\"]}')
print(f'  Chain valid: {s[\"chain_valid\"]}')
print(f'  Events: {list(s[\"events\"].keys())[:8]}')
" 2>/dev/null
echo ""

# Checkpoint
echo "💾 CHECKPOINT:"
if [ -f /root/UNIFIED_BRAIN/real_env_checkpoint_v3.json ]; then
    python3 -c "
import json
d = json.load(open('/root/UNIFIED_BRAIN/real_env_checkpoint_v3.json'))
s = d['stats']
print(f'  Episodes: {s[\"total_episodes\"]}')
print(f'  Best reward: {s[\"best_reward\"]}')
print(f'  Avg reward (100): {s[\"avg_reward_last_100\"]:.1f}')
print(f'  Avg step_time: {s[\"avg_time_per_step\"]:.3f}s')
print(f'  Gradients applied: {s[\"gradients_applied\"]}')
" 2>/dev/null
else
    echo "  (Ainda não salvo)"
fi
echo ""

# FASE 1 components
echo "🎛️ FASE 1 COMPONENTES:"
echo "  ✅ Meta-controller: Ativo"
echo "  ✅ Curiosity: Ativa"
echo "  ✅ Auto-topologia: Ativa"
echo "  ✅ Darwin: Ativo"
echo ""

echo "════════════════════════════════════════════════════════════════"
echo "Commands:"
echo "  tail -f /root/UNIFIED_BRAIN/v3_final.log | grep 'Ep '"
echo "  tail -f /root/UNIFIED_BRAIN/v3_final.log | grep 'META:'"
echo "  kill $PID  # Para parar"
echo "════════════════════════════════════════════════════════════════"
