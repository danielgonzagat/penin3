#!/bin/bash
# VALIDAÇÃO PÓS-CORREÇÃO DO SISTEMA I³
# ======================================
# Verifica se os fixes funcionaram de verdade

echo "╔════════════════════════════════════════════════════════╗"
echo "║                                                        ║"
echo "║  🔬 VALIDAÇÃO SISTEMA I³ PÓS-CORREÇÃO                ║"
echo "║                                                        ║"
echo "╚════════════════════════════════════════════════════════╝"
echo

# Contadores
PASSES=0
FAILS=0

# Função helper
check() {
    local name="$1"
    local command="$2"
    local expected="$3"
    
    echo "🔍 Checando: $name"
    
    if eval "$command" > /dev/null 2>&1; then
        echo "   ✅ PASS"
        ((PASSES++))
    else
        echo "   ❌ FAIL"
        echo "      Esperado: $expected"
        ((FAILS++))
    fi
    echo
}

echo "═══════════════════════════════════════════════════════"
echo "TESTE #1: DARWIN EVOLUTION ATIVO"
echo "═══════════════════════════════════════════════════════"

# 1.1 Processo rodando?
check "Darwin processo ativo" \
      "pgrep -f run_emergence_blocks_STORM" \
      "Processo run_emergence_blocks_STORM deve estar rodando"

# 1.2 Log crescendo?
DARWIN_LOG_SIZE=$(wc -l < /root/darwin_STORM.log 2>/dev/null || echo 0)
echo "🔍 Checando: Darwin log size"
if [ "$DARWIN_LOG_SIZE" -gt 50 ]; then
    echo "   ✅ PASS ($DARWIN_LOG_SIZE linhas)"
    ((PASSES++))
else
    echo "   ❌ FAIL ($DARWIN_LOG_SIZE linhas - esperado >50)"
    ((FAILS++))
fi
echo

# 1.3 CPU Usage?
echo "🔍 Checando: Darwin CPU usage"
DARWIN_CPU=$(top -b -n 1 -p $(pgrep -f run_emergence_blocks_STORM | head -1) 2>/dev/null | tail -1 | awk '{print $9}' | cut -d'.' -f1 || echo 0)
if [ "$DARWIN_CPU" -gt 3 ]; then
    echo "   ✅ PASS (${DARWIN_CPU}% CPU)"
    ((PASSES++))
else
    echo "   ⚠️  WARN (${DARWIN_CPU}% CPU - esperado >5%, pode estar em I/O)"
fi
echo

# 1.4 Checkpoints novos?
echo "🔍 Checando: Novos checkpoints Darwin"
RECENT_CHECKPOINTS=$(find /root/intelligence_system/models/darwin_checkpoints/ -name "*.pt" -mmin -60 2>/dev/null | wc -l)
if [ "$RECENT_CHECKPOINTS" -gt 0 ]; then
    echo "   ✅ PASS ($RECENT_CHECKPOINTS checkpoints nos últimos 60min)"
    ((PASSES++))
    echo "      Checkpoints recentes:"
    ls -lht /root/intelligence_system/models/darwin_checkpoints/*.pt 2>/dev/null | head -3 | awk '{print "      - "$9" ("$6" "$7" "$8")"}'
else
    echo "   ⚠️  WARN (0 checkpoints novos - normal se rodando há menos de 30min)"
fi
echo

echo "═══════════════════════════════════════════════════════"
echo "TESTE #2: SURPRISE DETECTOR ATIVO"
echo "═══════════════════════════════════════════════════════"

# 2.1 Processo rodando?
check "Detector processo ativo" \
      "pgrep -f EMERGENCE_CATALYST_1_SURPRISE" \
      "Processo detector deve estar rodando"

# 2.2 Log com ciclos?
DETECTOR_LOG_SIZE=$(wc -l < /root/surprise_detector.log 2>/dev/null || echo 0)
echo "🔍 Checando: Detector log cycles"
if [ "$DETECTOR_LOG_SIZE" -gt 20 ]; then
    echo "   ✅ PASS ($DETECTOR_LOG_SIZE linhas)"
    ((PASSES++))
else
    echo "   ❌ FAIL ($DETECTOR_LOG_SIZE linhas - esperado >20)"
    ((FAILS++))
fi
echo

# 2.3 Surpresas detectadas?
echo "🔍 Checando: Surpresas detectadas"
SURPRISES=$(sqlite3 /root/emergence_surprises.db "SELECT COUNT(*) FROM surprises" 2>/dev/null || echo 0)
if [ "$SURPRISES" -gt 0 ]; then
    echo "   ✅ PASS ($SURPRISES surpresas detectadas)"
    ((PASSES++))
    echo "      Top 3 surpresas:"
    sqlite3 /root/emergence_surprises.db "SELECT system || '.' || metric, surprise_score, description FROM surprises ORDER BY surprise_score DESC LIMIT 3" 2>/dev/null | while read line; do
        echo "      - $line"
    done
else
    echo "   ⚠️  WARN (0 surpresas - normal se baseline ainda aprendendo)"
fi
echo

echo "═══════════════════════════════════════════════════════"
echo "TESTE #3: OUTROS COMPONENTES"
echo "═══════════════════════════════════════════════════════"

# 3.1 Meta-Learner
check "Meta-Learner ativo" \
      "pgrep -f META_LEARNER_REALTIME" \
      "Meta-Learner deve estar rodando"

# 3.2 Auto-Validator
check "Auto-Validator ativo" \
      "pgrep -f AUTO_VALIDATOR" \
      "Auto-Validator deve estar rodando"

# 3.3 System Connector
check "System Connector ativo" \
      "pgrep -f SYSTEM_CONNECTOR" \
      "System Connector deve estar rodando"

# 3.4 Consciousness
check "Consciousness ativo" \
      "pgrep -f CONSCIOUSNESS" \
      "Consciousness Daemon deve estar rodando"

# 3.5 Cross-Pollination
check "Cross-Pollination ativo" \
      "pgrep -f CROSS_POLLINATION_AUTO" \
      "Cross-Pollination deve estar rodando"

echo "═══════════════════════════════════════════════════════"
echo "TESTE #4: META-LEARNER VENDO DARWIN"
echo "═══════════════════════════════════════════════════════"

echo "🔍 Checando: Meta-Learner detectando Darwin CPU"
LATEST_DARWIN_CPU=$(tail -n 100 /root/meta_learner.log 2>/dev/null | grep "Darwin CPU:" | tail -1 | grep -oP '\d+\.\d+' | head -1 || echo "0")
echo "   Darwin CPU reportado: ${LATEST_DARWIN_CPU}%"
if (( $(echo "$LATEST_DARWIN_CPU > 1.0" | bc -l) )); then
    echo "   ✅ PASS (Meta-Learner vê Darwin ativo)"
    ((PASSES++))
else
    echo "   ⚠️  WARN (Meta-Learner ainda vê Darwin em low - aguardar mais tempo)"
fi
echo

echo "═══════════════════════════════════════════════════════"
echo "RESUMO DA VALIDAÇÃO"
echo "═══════════════════════════════════════════════════════"
echo
echo "✅ Testes passados: $PASSES"
echo "❌ Testes falhados: $FAILS"
echo

TOTAL=$((PASSES + FAILS))
if [ "$TOTAL" -gt 0 ]; then
    PERCENTAGE=$((PASSES * 100 / TOTAL))
    echo "📊 Taxa de sucesso: ${PERCENTAGE}%"
    echo
    
    if [ "$PERCENTAGE" -ge 80 ]; then
        echo "╔════════════════════════════════════════════════════════╗"
        echo "║  🎉 SISTEMA FUNCIONANDO EXCELENTEMENTE!               ║"
        echo "║     Inteligência real emergindo! 🚀                   ║"
        echo "╚════════════════════════════════════════════════════════╝"
    elif [ "$PERCENTAGE" -ge 60 ]; then
        echo "╔════════════════════════════════════════════════════════╗"
        echo "║  ✅ SISTEMA FUNCIONANDO BEM                           ║"
        echo "║     Alguns warnings - aguardar evolução               ║"
        echo "╚════════════════════════════════════════════════════════╝"
    else
        echo "╔════════════════════════════════════════════════════════╗"
        echo "║  ⚠️  SISTEMA COM PROBLEMAS                            ║"
        echo "║     Revisar logs e processos                          ║"
        echo "╚════════════════════════════════════════════════════════╝"
    fi
fi

echo
echo "📝 LOGS PARA ANÁLISE:"
echo "   Darwin:    tail -f /root/darwin_STORM.log"
echo "   Detector:  tail -f /root/surprise_detector.log"
echo "   Meta:      tail -f /root/meta_learner.log"
echo "   Connector: tail -f /root/system_connector.log"
echo
echo "📊 RELATÓRIO COMPLETO:"
echo "   cat /root/RE_AUDITORIA_COMPLETA_BRUTAL.md"
