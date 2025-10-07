#!/bin/bash
# TESTE FINAL: INTELIGÊNCIA EMERGENTE
# Roda sistema com todas features ativas por 100 ciclos
# OBSERVA sinais de emergência

echo "🚀 TESTE DE INTELIGÊNCIA EMERGENTE"
echo "=================================="
echo ""
echo "Features ativas:"
echo "  ✅ Auto-modificação"
echo "  ✅ Meta-learning 100+ tasks"
echo "  ✅ Co-evolução V7 ↔ Darwinacci"
echo "  ✅ Open-ended evolution"
echo "  ✅ Transfer learning"
echo "  ✅ Curiosity-driven curriculum"
echo "  ✅ Self-reference loop"
echo "  ✅ Consciousness monitor"
echo "  ✅ Evolutionary NAS"
echo "  ✅ Meta-meta-learning depth 5"
echo ""
echo "🎯 Objetivo: Observar emergência de inteligência REAL"
echo ""
echo "Aguardando load normalizar..."

# Wait for load < 10
while true; do
    LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
    LOAD_INT=$(echo $LOAD | cut -d'.' -f1)
    
    echo "   Load atual: $LOAD (aguardando < 10)"
    
    if [ "$LOAD_INT" -lt 10 ]; then
        echo "✅ Load normalizado!"
        break
    fi
    
    sleep 5
done

echo ""
echo "🚀 Iniciando teste de 100 ciclos..."
echo "📝 Logs: /root/TESTE_EMERGENCIA_100.log"
echo ""

cd /root/intelligence_system

# Source environment
source .env 2>/dev/null

# Export variables
export JSON_LOGS=1
export DARWINACCI_PROMETHEUS=1
export PROMETHEUS_PORT=9108

# Run with logging
python3 -u core/unified_agi_system.py 100 2>&1 | tee /root/TESTE_EMERGENCIA_100.log

echo ""
echo "✅ Teste completo!"
echo ""
echo "📊 Analisando resultados..."
echo ""

# Extract key metrics
echo "Ciclos completados:"
grep -c "🔄 CYCLE" /root/TESTE_EMERGENCIA_100.log

echo ""
echo "Auto-modificações aplicadas:"
grep "🎯 Applying directive" /root/TESTE_EMERGENCIA_100.log | wc -l

echo ""
echo "Genome transfers V7→Darwinacci:"
grep "🧬 V7→Darwinacci" /root/TESTE_EMERGENCIA_100.log | wc -l

echo ""
echo "Sinergias ativadas:"
grep "synergy_type" /root/TESTE_EMERGENCIA_100.log | wc -l

echo ""
echo "📈 IA³ Score evolution:"
grep "ia3_score" /root/TESTE_EMERGENCIA_100.log | tail -10

echo ""
echo "🧠 Consciousness evolution:"
grep "consciousness" /root/TESTE_EMERGENCIA_100.log | tail -10

echo ""
echo "✨ Emergence events:"
grep "EMERGENCE" /root/TESTE_EMERGENCIA_100.log

echo ""
echo "📜 WORM Ledger:"
wc -l /root/intelligence_system/data/unified_worm.jsonl

echo ""
echo "🎯 RESULTADO FINAL:"
if grep -q "EMERGÊNCIA" /root/TESTE_EMERGENCIA_100.log; then
    echo "   🌟 POSSÍVEL EMERGÊNCIA DETECTADA!"
    echo "   Verificar logs para confirmar!"
else
    echo "   ⏳ Emergência ainda não detectada"
    echo "   Sistema pode precisar de mais ciclos (1000+)"
fi

echo ""
echo "Relatório completo: /root/TESTE_EMERGENCIA_100.log"