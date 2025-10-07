#!/bin/bash
# Status rápido do sistema

clear
echo "╔════════════════════════════════════════════════════════════╗"
echo "║         🧠 UNIFIED BRAIN - STATUS RÁPIDO                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# PID
PID=$(cat /root/brain_daemon.pid 2>/dev/null)
if [ -z "$PID" ] || ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ Daemon: NÃO RODANDO"
    exit 1
else
    echo "✅ Daemon: RODANDO (PID: $PID)"
fi

# Checkpoint
if [ -f /root/UNIFIED_BRAIN/real_env_checkpoint_v3.json ]; then
    echo ""
    echo "📊 CHECKPOINT ATUAL:"
    python3 << 'EOF'
import json
from pathlib import Path
ckpt = json.loads(Path('/root/UNIFIED_BRAIN/real_env_checkpoint_v3.json').read_text())
stats = ckpt['stats']
print(f"   Episode: {ckpt['episode']}")
print(f"   Best reward: {stats['best_reward']:.1f}")
print(f"   Avg (100): {stats['avg_reward_last_100']:.1f}")
print(f"   Steps: {stats['total_steps']}")
print(f"   Gradients: {stats['gradients_applied']}")
print(f"   Avg step time: {stats['avg_time_per_step']:.3f}s")
EOF
fi

# Últimos episódios
echo ""
echo "📈 ÚLTIMOS 5 EPISÓDIOS:"
tail -200 /root/brain_daemon_v3_improved.log 2>/dev/null | grep "Ep [0-9]*:" | tail -5 | while read line; do
    echo "   $line"
done

# Scheduler
echo ""
echo "🎯 SCHEDULER (últimas decisões):"
tail -200 /root/brain_daemon_v3_improved.log 2>/dev/null | grep "scheduler" | tail -3 | while read line; do
    echo "   $line"
done

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "Comandos úteis:"
echo "  ./monitor_training.sh    - Monitor em tempo real"
echo "  python3 /root/metrics_analyzer.py - Análise detalhada"
echo "  tail -f /root/brain_daemon_v3_improved.log - Log completo"
echo "═══════════════════════════════════════════════════════════"