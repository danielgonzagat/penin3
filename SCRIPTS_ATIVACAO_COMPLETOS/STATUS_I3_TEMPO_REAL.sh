#!/bin/bash
# Dashboard em tempo real do Sistema I³

clear
while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════════════════╗"
    echo "║                                                                      ║"
    echo "║   📊 SISTEMA I³ - DASHBOARD TEMPO REAL                              ║"
    echo "║   $(date '+%Y-%m-%d %H:%M:%S')                                              ║"
    echo "║                                                                      ║"
    echo "╚══════════════════════════════════════════════════════════════════════╝"
    echo ""
    
    # I³ Score
    if [ -f /root/i3_score_final.txt ]; then
        cat /root/i3_score_final.txt
    fi
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🟢 PROCESSOS CORE"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    ps aux | grep -E "META_LEARNER_REALTIME|run_emergence_blocks_STORM|SYSTEM_CONNECTOR" | grep -v grep | awk '{printf "  %-30s PID: %-8s CPU: %5s%%\n", $11, $2, $3}' | head -10
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "🤖 SISTEMAS I³"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    ps aux | grep -E "AUTO_VALIDATOR|AUTO_REPAIR|SELF_MODIFICATION|ETERNAL_LOOP" | grep -v grep | awk '{printf "  %-30s PID: %-8s CPU: %5s%%\n", $11, $2, $3}' | head -10
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📁 AUTO-GERADOS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    echo -n "  Módulos Python: "
    ls /root/auto_generated_modules/*.py 2>/dev/null | wc -l
    
    echo -n "  Arquiteturas: "
    ls /root/evolved_architectures/checkpoint_*.json 2>/dev/null | wc -l
    
    echo -n "  Conectores: "
    ls /root/applied_architecture/CONNECTOR_*.py 2>/dev/null | wc -l
    
    echo -n "  Checkpoints Darwin: "
    ls /root/intelligence_system/models/darwin_checkpoints/*.pt 2>/dev/null | wc -l
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "💾 RECURSOS"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    df -h / | grep -v Filesystem | awk '{printf "  Disco: %s usado (%s livre)\n", $5, $4}'
    free -h | grep Mem | awk '{printf "  RAM: %s usado (%s livre)\n", $3, $7}'
    uptime | awk -F'load average:' '{printf "  Load: %s\n", $2}'
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "📝 ÚLTIMAS ATIVIDADES (logs)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    
    echo ""
    echo "Auto-Validator (última linha):"
    tail -n 1 /root/auto_validator_daemon.log 2>/dev/null | sed 's/^/  /'
    
    echo ""
    echo "Auto-Repair (última linha):"
    tail -n 1 /root/auto_repair_daemon.log 2>/dev/null | sed 's/^/  /'
    
    echo ""
    echo "Meta-Learner (última linha):"
    tail -n 1 /root/meta_learner_CORRIGIDO_FINAL.log 2>/dev/null | sed 's/^/  /'
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "Atualiza em 10s... (Ctrl+C para sair)"
    echo ""
    
    sleep 10
done
