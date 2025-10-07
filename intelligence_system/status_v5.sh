#!/bin/bash
# Check Intelligence System V5.0 status

echo "📊 Intelligence System V5.0 - EXTRACTED MERGE STATUS"
echo "================================================================"

if pgrep -f "system_v5_extracted_merge.py" > /dev/null; then
    echo "✅ V5.0 is RUNNING"
    
    PID=$(pgrep -f "system_v5_extracted_merge.py")
    echo "   PID: $PID"
    
    UPTIME=$(ps -p $PID -o etime= | tr -d ' ')
    echo "   Uptime: $UPTIME"
    
    echo ""
    echo "📄 Recent logs (last 20 lines):"
    echo "----------------------------------------------------------------"
    tail -20 logs/v5_output.log 2>/dev/null || echo "No logs yet"
    
else
    echo "❌ V5.0 is NOT running"
    echo ""
    echo "💡 Start with: ./start_v5.sh"
fi

echo ""
echo "🔧 Components:"
echo "   V4.0: PPO, LiteLLM, Meta-Learner, Gödelian, LangGraph, DSPy"
echo "   V5.0: Neural Evolution, Self-Modification, Neuronal Farm"

