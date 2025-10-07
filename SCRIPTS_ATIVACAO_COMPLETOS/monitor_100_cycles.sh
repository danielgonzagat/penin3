#!/bin/bash

echo "════════════════════════════════════════════════════════════"
echo "📊 MONITORING 100-CYCLE AUDIT"
echo "════════════════════════════════════════════════════════════"
echo ""

LOG_FILE="/root/audit_100_cycles_full.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ Log file not found: $LOG_FILE"
    exit 1
fi

# Check if test is still running
if pgrep -f "test_100_cycles_audit.py" > /dev/null; then
    echo "✅ Test is RUNNING"
else
    echo "⚠️ Test appears to be stopped"
fi

echo ""

# Count completed cycles
CYCLE_COUNT=$(grep -c "V7 cycle" "$LOG_FILE")
echo "Cycles completed: $CYCLE_COUNT / 100"

# Latest metrics
echo ""
echo "Latest V7 metrics:"
grep "V7 cycle" "$LOG_FILE" | tail -1

# Latest synergy execution
echo ""
echo "Latest synergy execution:"
grep "TOTAL AMPLIFICATION" "$LOG_FILE" | tail -1

# Consciousness evolution
echo ""
echo "Consciousness evolution:"
grep "consciousness:" "$LOG_FILE" | tail -5

echo ""
echo "════════════════════════════════════════════════════════════"
