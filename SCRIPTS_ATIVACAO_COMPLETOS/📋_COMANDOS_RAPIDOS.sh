#!/bin/bash
# 📋 COMANDOS RÁPIDOS PARA MONITORAMENTO

echo "═══════════════════════════════════════════════════════════"
echo "📋 COMANDOS RÁPIDOS - COPIE E COLE"
echo "═══════════════════════════════════════════════════════════"
echo ""

echo "🚀 1. INICIAR SISTEMA:"
echo "   bash /root/🎯_EXECUTAR_AGORA_EMERGENCE.sh"
echo ""

echo "👀 2. MONITORAR LOG PRINCIPAL:"
echo "   tail -f /root/massive_replay.log"
echo ""

echo "📊 3. MONITORAR WORM (eventos críticos):"
echo "   tail -f /root/massive_replay_output/massive_replay_worm.jsonl | jq '.'"
echo ""

echo "📈 4. VERIFICAR PROGRESSO RÁPIDO:"
echo "   tail -5 /root/massive_replay_output/massive_replay_worm.jsonl | jq '.'"
echo ""

echo "🔍 5. VERIFICAR EMERGÊNCIA (após 6h):"
echo "   bash /root/🔍_VERIFICAR_EMERGENCE.sh"
echo ""

echo "🛑 6. PARAR SISTEMA:"
echo "   kill \$(cat /root/massive_replay.pid)"
echo ""

echo "📊 7. VER ESTATÍSTICAS DO DATABASE:"
echo "   sqlite3 /root/intelligence_system/data/intelligence.db \\"
echo "     \"SELECT COUNT(*) FROM gate_evals WHERE uplift > 0;\""
echo ""

echo "🧬 8. VER DARWINACCI PROGRESS:"
echo "   ls -lh /root/darwinacci_omega/checkpoints/*.pkl | wc -l"
echo ""

echo "🔥 9. VER SURPRESAS ESTATÍSTICAS:"
echo "   sqlite3 /root/intelligence_system/data/emergence_surprises.db \\"
echo "     \"SELECT * FROM surprises ORDER BY z_score DESC LIMIT 5;\""
echo ""

echo "✅ 10. STATUS COMPLETO:"
echo "   bash /root/🔍_VERIFICAR_EMERGENCE.sh"
echo ""
