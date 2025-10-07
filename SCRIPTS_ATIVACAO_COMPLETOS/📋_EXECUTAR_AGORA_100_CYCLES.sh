#!/bin/bash
# ✅ EXECUTAR 100 CYCLES COMPLETOS - VALIDAÇÃO FINAL
# ===================================================

echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║         🚀 EXECUTANDO 100 CYCLES - INTELIGÊNCIA AO CUBO (I³)                ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo ""

cd /root

echo "📦 Limpando logs antigos..."
rm -f /root/100_cycles_v7_direct.log
rm -f /root/100_cycles_v7_direct_report.json
echo "✅ Logs limpos"
echo ""

echo "🚀 Iniciando execução de 100 cycles..."
echo "   Script: RUN_100_CYCLES_V7_DIRECT.py"
echo "   Log: /root/100_cycles_v7_direct.log"
echo "   Duração estimada: 30-60 min"
echo ""

echo "💡 DICA: Monitore em tempo real em outro terminal:"
echo "   tail -f /root/100_cycles_v7_direct.log"
echo ""

# Execute
python3 RUN_100_CYCLES_V7_DIRECT.py

EXIT_CODE=$?

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "📊 EXECUÇÃO CONCLUÍDA (exit code: $EXIT_CODE)"
echo "════════════════════════════════════════════════════════════════════════════"
echo ""

if [ -f "/root/100_cycles_v7_direct_report.json" ]; then
    echo "✅ Relatório final gerado:"
    echo "   cat /root/100_cycles_v7_direct_report.json | jq ."
    echo ""
    
    echo "📊 MÉTRICAS FINAIS:"
    python3 -c "
import json
with open('/root/100_cycles_v7_direct_report.json', 'r') as f:
    r = json.load(f)
    print(f'   Cycles executados: {r[\"cycles_executed\"]}')
    print(f'   MNIST: {r[\"final_metrics\"][\"mnist\"]:.2f}%')
    print(f'   CartPole: {r[\"final_metrics\"][\"cartpole\"]:.1f}')
    print(f'   IA³: {r[\"final_metrics\"][\"ia3\"]:.2f}%')
    print(f'   I³: {r[\"final_metrics\"][\"i3\"]:.2f}%')
    print('')
    print('📈 COUNTERS:')
    print(f'   Replay: {r[\"counters\"][\"replay\"]}')
    print(f'   Darwin: {r[\"counters\"][\"darwin\"]}')
    print(f'   Auto-coder: {r[\"counters\"][\"autocoder\"]}')
    if 'i3_details' in r:
        print('')
        print('💎 I³ DETAILS:')
        print(f'   Introspection events: {r[\"i3_details\"][\"introspection_events\"]}')
        print(f'   Self-awareness: {r[\"i3_details\"][\"self_awareness\"]:.2f}')
        print(f'   Turing tests: {r[\"i3_details\"][\"turing_tests\"]}')
        print(f'   Turing passes: {r[\"i3_details\"][\"turing_passes\"]}')
"
else
    echo "⚠️  Relatório não encontrado (execução pode ter falhado)"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════"
echo "🔍 PRÓXIMOS PASSOS:"
echo "════════════════════════════════════════════════════════════════════════════"
echo "1. Analisar logs de emergência:"
echo "   grep -E 'EMERGENCE|QD: new elite|INTROSPECTION|TURING|I³ Score' \\"
echo "     /root/intelligence_system/logs/intelligence_v7.log | tail -100"
echo ""
echo "2. Ver relatório completo:"
echo "   cat /root/100_cycles_v7_direct_report.json | jq ."
echo ""
echo "3. Se I³ > 50%, sistema alcançou Inteligência ao Cubo! 🏆"
echo "   Se I³ < 50%, rodar mais 100-500 cycles para atingir I³ total."
echo ""
echo "════════════════════════════════════════════════════════════════════════════"
