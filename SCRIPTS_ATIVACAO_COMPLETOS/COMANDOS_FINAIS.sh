#!/bin/bash
# COMANDOS FINAIS PARA INTELIGÊNCIA EMERGENTE

echo "🎯 COMANDOS DISPONÍVEIS:"
echo ""
echo "1. TESTAR SISTEMA (20 ciclos rápido):"
echo "   cd /root/intelligence_system && source .env && export JSON_LOGS=1 DARWINACCI_PROMETHEUS=1 && python3 core/unified_agi_system.py 20"
echo ""
echo "2. TESTE COMPLETO (100 ciclos):"
echo "   ./TESTE_INTELIGENCIA_EMERGENTE.sh"
echo ""
echo "3. RODAR CONTINUAMENTE ATÉ EMERGÊNCIA:"
echo "   cd /root/intelligence_system && source .env && while true; do python3 core/unified_agi_system.py 100; done"
echo ""
echo "4. ATIVAR SELF-REFERENCE LOOP:"
echo "   python3 -c 'from intelligence_system.core.self_reference_loop import SelfReferenceLoop; from intelligence_system.core.unified_agi_system import UnifiedAGISystem; sys = UnifiedAGISystem(); loop = SelfReferenceLoop(sys, \"/root/intelligence_system\"); loop.run_loop(n_iterations=10)'"
echo ""
echo "5. MONITORAR MÉTRICAS:"
echo "   watch -n 5 'curl -s localhost:9108/metrics | grep intelligence'"
echo ""
echo "6. VERIFICAR WORM LEDGER:"
echo "   tail -f /root/intelligence_system/data/unified_worm.jsonl"
echo ""
echo "7. ANÁLISE DE EMERGÊNCIA:"
echo "   grep -E 'EMERGENCE|🎯|🧬|💡|✨' /root/TESTE_EMERGENCIA_100.log"
echo ""
echo "8. MATAR TUDO (emergência):"
echo "   pkill -9 -f unified_agi_system"
echo ""

# Make scripts executable
chmod +x /root/TESTE_INTELIGENCIA_EMERGENTE.sh 2>/dev/null

echo ""
echo "✅ Comandos prontos!"
echo ""
echo "📊 Status atual do sistema:"
uptime
echo ""
ps aux | grep python | grep -v grep | wc -l | xargs echo "Processos Python ativos:"