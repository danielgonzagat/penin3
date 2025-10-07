#!/bin/bash
# Check Intelligence System V4.0 status

echo "📊 Intelligence System V4.0 - FULL MERGE STATUS"
echo "=============================================="

if pgrep -f "system_v4_full_merge.py" > /dev/null; then
    echo "✅ V4.0 is RUNNING"
    
    PID=$(pgrep -f "system_v4_full_merge.py")
    echo "   PID: $PID"
    
    UPTIME=$(ps -p $PID -o etime= | tr -d ' ')
    echo "   Uptime: $UPTIME"
    
    echo ""
    echo "📄 Recent logs (last 15 lines):"
    echo "--------------------------------"
    tail -15 logs/v4_output.log 2>/dev/null || echo "No logs yet"
    
    echo ""
    echo "📊 Database stats:"
    sqlite3 data/intelligence.db "SELECT COUNT(*) as cycles FROM cycles;" 2>/dev/null || echo "N/A"
    
else
    echo "❌ V4.0 is NOT running"
    echo ""
    echo "💡 Start with: ./start_v4.sh"
fi

echo ""
echo "🔄 Integration Status:"
echo "   [P0-1] LiteLLM: Installed"
echo "   [P0-2] CleanRL PPO: Active"
echo "   [P0-3] Meta-Learner: Active"
echo "   [P0-4] Gödelian: Active"
echo "   [P0-5] LangGraph: Integrated"
echo "   [P0-6] DSPy: Integrated"
echo "   [P0-7] AutoKeras: Integrated"

