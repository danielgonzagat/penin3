#!/bin/bash
# 🚀 RESTART BRAIN DAEMON WITH ALL IMPROVEMENTS
# Aplica Darwin, Synthesis, MAML, PENIN3, Recursive

set -e

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 RESTARTING BRAIN DAEMON WITH ALL IMPROVEMENTS"
echo "════════════════════════════════════════════════════════════════════════════════"
echo

# ═══════════════════════════════════════════════════════════════════════════════
# 1. BACKUP ATUAL
# ═══════════════════════════════════════════════════════════════════════════════
echo "💾 Creating backup..."
BACKUP_DIR="/root/BACKUPS_AUTONOMOS/restart_$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Salvar dashboard e checkpoint atuais
cp /root/UNIFIED_BRAIN/dashboard.txt $BACKUP_DIR/ 2>/dev/null || true
cp /root/UNIFIED_BRAIN/real_env_checkpoint_v3.json $BACKUP_DIR/ 2>/dev/null || true
cp /root/UNIFIED_BRAIN/cerebrum_genome.json $BACKUP_DIR/ 2>/dev/null || true

echo "   ✅ Backup saved to: $BACKUP_DIR"
echo

# ═══════════════════════════════════════════════════════════════════════════════
# 2. PARAR DAEMON ATUAL (gracefully)
# ═══════════════════════════════════════════════════════════════════════════════
echo "🛑 Stopping current brain daemon..."

if pgrep -f "brain_daemon_real_env.py" > /dev/null; then
    CURRENT_PID=$(pgrep -f "brain_daemon_real_env.py" | head -1)
    echo "   Current PID: $CURRENT_PID"
    
    # Try graceful shutdown first
    kill -SIGTERM $CURRENT_PID 2>/dev/null || true
    
    # Wait up to 10 seconds
    for i in {1..10}; do
        if ! ps -p $CURRENT_PID > /dev/null 2>&1; then
            echo "   ✅ Stopped gracefully"
            break
        fi
        sleep 1
    done
    
    # Force if still running
    if ps -p $CURRENT_PID > /dev/null 2>&1; then
        echo "   ⚠️  Forcing stop..."
        kill -9 $CURRENT_PID 2>/dev/null || true
        sleep 2
    fi
else
    echo "   ℹ️  No daemon running"
fi

echo

# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXPORT ENVIRONMENT VARIABLES
# ═══════════════════════════════════════════════════════════════════════════════
echo "⚙️  Setting environment..."

export ENABLE_INCOMPLETENESS_HOOK=0
export UBRAIN_SYNTHESIS=1
export UBRAIN_DARWIN=1
export UBRAIN_IA3_CALC=1
export UBRAIN_MAML=1
export UBRAIN_PENIN3=1
export UBRAIN_RECURSIVE=0  # Daemon separado

echo "   ✅ Environment configured"
echo

# ═══════════════════════════════════════════════════════════════════════════════
# 4. START NEW DAEMON WITH ALL IMPROVEMENTS
# ═══════════════════════════════════════════════════════════════════════════════
echo "🚀 Starting brain daemon with improvements..."

cd /root/UNIFIED_BRAIN

nohup python3 -u brain_daemon_real_env.py \
    > /root/brain_full_improvements.log 2>&1 &

NEW_PID=$!
echo $NEW_PID > /root/brain_unified.pid

echo "   ✅ New daemon started"
echo "   PID: $NEW_PID"
echo "   Log: /root/brain_full_improvements.log"
echo

# ═══════════════════════════════════════════════════════════════════════════════
# 5. WAIT FOR INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════
echo "⏳ Waiting for initialization (15 seconds)..."
sleep 15

# ═══════════════════════════════════════════════════════════════════════════════
# 6. VERIFY ACTIVATIONS
# ═══════════════════════════════════════════════════════════════════════════════
echo "🔍 Verifying activations..."
echo

if ps -p $NEW_PID > /dev/null 2>&1; then
    echo "   ✅ Daemon is running (PID $NEW_PID)"
    
    # Check for activation messages
    echo
    echo "   Checking activation logs:"
    
    if grep -q "Darwin.*CONNECT\|Darwin.*ACTIVE" /root/brain_full_improvements.log 2>/dev/null; then
        echo "   ✅ Darwin Evolution: ACTIVE"
    else
        echo "   ⚠️  Darwin Evolution: Not detected in logs yet"
    fi
    
    if grep -q "Synthesis.*ACTIVE\|synthesis_enabled.*True" /root/brain_full_improvements.log 2>/dev/null; then
        echo "   ✅ Module Synthesis: ACTIVE"
    else
        echo "   ⚠️  Module Synthesis: Not detected in logs yet"
    fi
    
    if grep -q "MAML" /root/brain_full_improvements.log 2>/dev/null; then
        echo "   ✅ MAML: Mentioned in logs"
    else
        echo "   ⚠️  MAML: Not detected in logs yet"
    fi
    
    echo
    echo "   📊 Latest log entries:"
    tail -10 /root/brain_full_improvements.log
    
else
    echo "   ❌ Daemon failed to start!"
    echo "   Check logs: tail -50 /root/brain_full_improvements.log"
    exit 1
fi

echo
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ RESTART COMPLETE!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo
echo "📊 Monitor learning:"
echo "   tail -f /root/brain_full_improvements.log | grep 'NEW BEST'"
echo
echo "🧬 Monitor Darwin:"
echo "   tail -f /root/brain_full_improvements.log | grep 'EVOLVED'"
echo
echo "✨ Monitor Synthesis:"
echo "   tail -f /root/brain_full_improvements.log | grep 'NEW NEURON'"
echo
echo "════════════════════════════════════════════════════════════════════════════════"