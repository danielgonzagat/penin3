#!/bin/bash
# Health Check Script for Intelligence System

echo "╔══════════════════════════════════════╗"
echo "║  HEALTH CHECK - Intelligence System  ║"
echo "╚══════════════════════════════════════╝"
echo ""

# 1. Processos
echo "📊 PROCESSOS:"
PROCS=$(pgrep -af "brain_daemon|EMERGENCE|META|CROSS|DYNAMIC|V7_DARWIN|llama-server" 2>/dev/null | wc -l)
echo "  Ativos: $PROCS"

# 2. Brain checkpoint
echo ""
echo "🧠 BRAIN V3:"
if [ -f /root/UNIFIED_BRAIN/real_env_checkpoint_v3.json ]; then
    python3 << 'EOF'
import json
try:
    data = json.load(open('/root/UNIFIED_BRAIN/real_env_checkpoint_v3.json'))
    print(f"  Episode: {data.get('episode', 0)}")
    print(f"  Best reward: {data.get('best_reward', 0)}")
    print(f"  Avg last 100: {data['stats'].get('avg_reward_last_100', 0)}")
except Exception as e:
    print(f"  ⚠️ Error: {e}")
EOF
else
    echo "  ⚠️ Checkpoint não encontrado"
fi

# 3. Surprises
echo ""
echo "🎲 EMERGÊNCIA:"
if [ -f /root/emergence_surprises.db ]; then
    SURPRISES=$(sqlite3 /root/emergence_surprises.db "SELECT COUNT(*) FROM surprises" 2>/dev/null || echo "0")
    echo "  Surprises detectadas: $SURPRISES"
else
    echo "  ⚠️ DB não encontrado"
fi

# 4. Llama
echo ""
echo "🦙 LLAMA:"
curl -s -m 2 http://localhost:8001/health 2>/dev/null && echo "  ✅ Respondendo (8001)" || \
curl -s -m 2 http://localhost:8080/health 2>/dev/null && echo "  ✅ Respondendo (8080)" || \
echo "  ❌ Offline"

# 5. Disk space
echo ""
echo "💾 DISCO:"
df -h / | tail -1 | awk '{print "  Usado: "$3" / "$2" ("$5")"}'

# 6. CPU/RAM
echo ""
echo "⚡ RECURSOS:"
top -bn1 | grep "Cpu(s)" | awk '{print "  CPU: "$2}' 2>/dev/null || echo "  CPU: N/A"
free -h | grep Mem | awk '{print "  RAM: "$3" / "$2}' 2>/dev/null || echo "  RAM: N/A"

echo ""
echo "═══════════════════════════════════════"
echo "Run: bash /root/health_check.sh"
