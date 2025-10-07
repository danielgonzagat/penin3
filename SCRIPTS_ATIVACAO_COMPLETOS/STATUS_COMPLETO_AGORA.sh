#!/bin/bash
# STATUS COMPLETO DO SISTEMA I³ - Validação Real

echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║  📊 STATUS SISTEMA I³ - VALIDAÇÃO COMPLETA           ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo

echo "🔍 PROCESSOS ATIVOS:"
echo "===================="
pgrep -fl "python3.*CONSCIOUSNESS|python3.*SURPRISE|python3.*META_LEARNER|python3.*AUTO_VALIDATOR|python3.*SYSTEM_CONNECTOR|python3.*CROSS_POLL" | nl
echo

echo "📈 MÉTRICAS V7 (últimas):"
echo "========================"
sqlite3 /root/intelligence_system/data/intelligence.db "SELECT cycle, mnist_accuracy, cartpole_reward FROM cycles ORDER BY cycle DESC LIMIT 5" 2>/dev/null || echo "  (sem dados)"
echo

echo "🧠 CONSCIÊNCIA (self-awareness):"
echo "================================"
sqlite3 /root/consciousness.db "SELECT value FROM self_knowledge WHERE key='num_components' ORDER BY timestamp DESC LIMIT 1" 2>/dev/null || echo "  (sem dados)"
echo

echo "🎯 META-LEARNER (decisões recentes):"
echo "===================================="
sqlite3 /root/meta_learning.db "SELECT decision, success FROM meta_decisions ORDER BY timestamp DESC LIMIT 5" 2>/dev/null || echo "  (sem dados)"
echo

echo "✅ AUTO-VALIDATOR (status):"
echo "==========================="
tail -n 5 /root/auto_validator.log 2>/dev/null | grep -E "✅|❌" || echo "  (sem dados)"
echo

echo "🧬 CHECKPOINTS (Darwin e V7):"
echo "============================="
echo "  Darwin:" $(ls /root/intelligence_system/models/darwin_checkpoints/*.pt 2>/dev/null | wc -l) "arquivos"
echo "  V7:" $(find /root -name "*ia3_evolution*.pt" 2>/dev/null | wc -l) "arquivos"
echo

echo "╔════════════════════════════════════════════════════════╗"
echo "║  ✅ VALIDAÇÃO COMPLETA                                 ║"
echo "╚════════════════════════════════════════════════════════╝"
