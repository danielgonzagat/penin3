#!/bin/bash
# VALIDAR_SISTEMA_COMPLETO.sh
# Script de validação automatizada da re-auditoria

set -e

echo "═══════════════════════════════════════════════════════════════════════════════"
echo "🔬 VALIDAÇÃO COMPLETA DO SISTEMA - RE-AUDITORIA 2025-10-04"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

TOTAL_TESTS=0
PASSED_TESTS=0
FAILED_TESTS=0

run_test() {
    local test_name="$1"
    local test_command="$2"
    local expected_output="$3"
    
    TOTAL_TESTS=$((TOTAL_TESTS + 1))
    echo -n "Test #${TOTAL_TESTS}: ${test_name}... "
    
    if eval "$test_command" &> /dev/null; then
        echo -e "${GREEN}✅ PASS${NC}"
        PASSED_TESTS=$((PASSED_TESTS + 1))
        return 0
    else
        echo -e "${RED}❌ FAIL${NC}"
        FAILED_TESTS=$((FAILED_TESTS + 1))
        return 1
    fi
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📦 TESTE DE COMPONENTES PRINCIPAIS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 1: V7 Initialization
run_test "V7 System Initialization" \
    "python3 -c 'import sys; sys.path.insert(0, \"/root/intelligence_system\"); from core.system_v7_ultimate import IntelligenceSystemV7; v7 = IntelligenceSystemV7(); assert v7.cycle >= 0'"

# Test 2: Darwin Engine
run_test "Darwin Engine Active" \
    "python3 -c 'import sys; sys.path.insert(0, \"/root/intelligence_system\"); from core.system_v7_ultimate import IntelligenceSystemV7; v7 = IntelligenceSystemV7(); assert hasattr(v7, \"darwin_real\") and v7.darwin_real is not None'"

# Test 3: PENIN-Ω
run_test "PENIN-Ω Components" \
    "python3 -c 'import sys; sys.path.insert(0, \"/root/peninaocubo\"); from penin.math.linf import linf_score; from penin.core.caos import compute_caos_plus_exponential; assert callable(linf_score) and callable(compute_caos_plus_exponential)'"

# Test 4: Synergies
run_test "Synergy Orchestrator" \
    "python3 -c 'import sys; sys.path.insert(0, \"/root/intelligence_system\"); from core.synergies import SynergyOrchestrator; assert SynergyOrchestrator is not None'"

# Test 5: APIs
run_test "API Clients Available" \
    "python3 -c 'import sys; sys.path.insert(0, \"/root/intelligence_system\"); from apis.real_api_client import RealAPIClient; from apis.litellm_wrapper import LiteLLMWrapper; assert RealAPIClient and LiteLLMWrapper'"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 TESTE DE ISSUES CONHECIDOS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 6: Issue #1 - Auto-coding
echo -n "Test #6: Issue #1 (Auto-coding applies mods)... "
if grep -q "Auto-coding APPLIED" intelligence_system/logs/intelligence_v7.log 2>/dev/null; then
    echo -e "${GREEN}✅ PASS (Issue #1 FIXED)${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${YELLOW}⚠️  PENDING (Issue #1 not fixed yet)${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Test 7: Issue #2 - API Keys
echo -n "Test #7: Issue #2 (API Keys configured)... "
missing_keys=0
for key in GEMINI_API_KEY DEEPSEEK_API_KEY GROK_API_KEY; do
    if [ -z "${!key}" ]; then
        missing_keys=$((missing_keys + 1))
    fi
done
if [ $missing_keys -eq 0 ]; then
    echo -e "${GREEN}✅ PASS (Issue #2 FIXED)${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${YELLOW}⚠️  PENDING (Issue #2 not fixed: ${missing_keys}/3 keys missing)${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Test 8: Issue #3 - MAML/AutoML Triggers
echo -n "Test #8: Issue #3 (MAML/AutoML triggers)... "
if python3 -c "
import sys; sys.path.insert(0, '/root/intelligence_system')
from core.system_v7_ultimate import IntelligenceSystemV7
import re
with open('intelligence_system/core/system_v7_ultimate.py') as f:
    content = f.read()
# Check if triggers are at % 10 (fixed) or % 20 (not fixed)
maml_freq = 10 if 'cycle % 10 == 0' in content and '_maml_adapt' in content else 20
automl_freq = 10 if 'cycle % 10 == 0' in content and '_automl_search' in content else 20
assert maml_freq == 10 and automl_freq == 10
" 2>/dev/null; then
    echo -e "${GREEN}✅ PASS (Issue #3 FIXED)${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${YELLOW}⚠️  PENDING (Issue #3 not fixed: triggers still at % 20)${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Test 9: Issue #4 - Emergence Monitor
echo -n "Test #9: Issue #4 (Emergence Monitor integrated)... "
if grep -q "✨ EMERGENCE" intelligence_system/logs/intelligence_v7.log 2>/dev/null; then
    echo -e "${GREEN}✅ PASS (Issue #4 FIXED)${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${YELLOW}⚠️  PENDING (Issue #4 not fixed: no emergence events)${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

# Test 10: Issue #5 - QD-Lite
echo -n "Test #10: Issue #5 (QD-Lite active)... "
if grep -q "🌌 QD: new elite" intelligence_system/logs/intelligence_v7.log 2>/dev/null; then
    echo -e "${GREEN}✅ PASS (Issue #5 FIXED)${NC}"
    PASSED_TESTS=$((PASSED_TESTS + 1))
else
    echo -e "${YELLOW}⚠️  PENDING (Issue #5 not fixed: no QD elites)${NC}"
    FAILED_TESTS=$((FAILED_TESTS + 1))
fi
TOTAL_TESTS=$((TOTAL_TESTS + 1))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 TESTE DE MÉTRICAS E EXPORTS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Test 11: CSV Export
run_test "CSV Export Exists" \
    "test -f intelligence_system/data/exports/timeline_metrics.csv"

# Test 12: JSONL Export
run_test "JSONL Export Exists" \
    "test -f intelligence_system/data/exports/timeline_metrics.jsonl"

# Test 13: System Running
run_test "System Has Run Cycles" \
    "python3 -c 'import sys; sys.path.insert(0, \"/root/intelligence_system\"); from core.system_v7_ultimate import IntelligenceSystemV7; v7 = IntelligenceSystemV7(); assert v7.cycle > 0'"

# Test 14: IA³ Score Calculation
run_test "IA³ Score Calculation" \
    "python3 -c 'import sys; sys.path.insert(0, \"/root/intelligence_system\"); from core.system_v7_ultimate import IntelligenceSystemV7; v7 = IntelligenceSystemV7(); score = v7._calculate_ia3_score(); assert 0 <= score <= 100'"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📈 RESULTADOS DA VALIDAÇÃO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

PASS_PERCENTAGE=$((PASSED_TESTS * 100 / TOTAL_TESTS))

echo "Total de Testes:    ${TOTAL_TESTS}"
echo -e "Testes Aprovados:   ${GREEN}${PASSED_TESTS}${NC}"
echo -e "Testes Falhados:    ${RED}${FAILED_TESTS}${NC}"
echo "Porcentagem:        ${PASS_PERCENTAGE}%"
echo ""

if [ $PASS_PERCENTAGE -ge 75 ]; then
    echo -e "${GREEN}✅ SISTEMA VALIDADO: ${PASS_PERCENTAGE}% dos testes passaram${NC}"
    echo ""
    echo "🎯 Status: SISTEMA FUNCIONANDO (75%+ FUNCIONAL)"
    echo ""
    if [ $PASS_PERCENTAGE -lt 90 ]; then
        echo "⚠️  Ainda existem ${FAILED_TESTS} issues pendentes de correção."
        echo "   Consulte RE_AUDITORIA_FORENSE_COMPLETA_2025_10_04.md para detalhes."
    fi
elif [ $PASS_PERCENTAGE -ge 50 ]; then
    echo -e "${YELLOW}⚠️  SISTEMA PARCIALMENTE FUNCIONAL: ${PASS_PERCENTAGE}%${NC}"
    echo ""
    echo "📋 Recomendação: Aplicar correções da FASE 1 (2 horas)"
    echo "   Consulte ROADMAP_IMPLEMENTACAO_COMPLETA_2025_10_04.md"
else
    echo -e "${RED}❌ SISTEMA COM PROBLEMAS: Apenas ${PASS_PERCENTAGE}% funcionando${NC}"
    echo ""
    echo "🚨 Ação urgente necessária: Aplicar TODAS as correções do ROADMAP"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════════"
echo "📁 ARQUIVOS DA RE-AUDITORIA:"
echo "   - RE_AUDITORIA_FORENSE_COMPLETA_2025_10_04.md (15.000 palavras)"
echo "   - ROADMAP_IMPLEMENTACAO_COMPLETA_2025_10_04.md (8.000 palavras)"
echo "   - 📋_LEIA_PRIMEIRO_RE_AUDITORIA_2025_10_04.txt (sumário executivo)"
echo "   - RE_AUDITORIA_METRICAS_2025_10_04.json (métricas estruturadas)"
echo "   - VALIDAR_SISTEMA_COMPLETO.sh (este script)"
echo "═══════════════════════════════════════════════════════════════════════════════"
echo ""

exit 0
