#!/bin/bash
# Status Intelligence System V6.0

echo "📊 Intelligence System V6.0 - IA³ COMPLETE STATUS"
echo "================================================================"

if pgrep -f "system_v6_ia3_complete.py" > /dev/null; then
    echo "✅ V6.0 is RUNNING"
    
    PID=$(pgrep -f "system_v6_ia3_complete.py")
    echo "   PID: $PID"
    
    UPTIME=$(ps -p $PID -o etime= 2>/dev/null | tr -d ' ')
    echo "   Uptime: $UPTIME"
    
    echo ""
    echo "📄 Recent logs:"
    echo "----------------------------------------------------------------"
    tail -15 logs/v6_output.log 2>/dev/null || echo "No logs yet"
    
    echo ""
    echo "📊 Database stats:"
    sqlite3 data/intelligence.db "SELECT COUNT(*) as cycles, MAX(mnist_accuracy) as best_mnist, MAX(cartpole_avg_reward) as best_cartpole FROM cycles;" 2>/dev/null
    
else
    echo "❌ V6.0 is NOT running"
    echo ""
    echo "💡 Start with: ./start_v6.sh"
fi

echo ""
echo "🔧 V6.0 Components (13):"
echo "   V4.0: PPO, LiteLLM, Meta, Gödelian, LangGraph, DSPy, AutoKeras"
echo "   V5.0: Neural Evo, Self-Mod, Neuronal Farm"
echo "   V6.0: Code Validator, Advanced Evo, Multi-Coordinator"

