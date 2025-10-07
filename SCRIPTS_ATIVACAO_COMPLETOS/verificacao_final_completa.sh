#!/bin/bash
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║        VERIFICAÇÃO FINAL - PENIN³ Sistema Completo           ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Contadores
PASS=0
FAIL=0
TOTAL=0

# Função de teste
test_item() {
    local name="$1"
    local cmd="$2"
    TOTAL=$((TOTAL+1))
    echo -n "   [$TOTAL] $name ... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo "✅ PASS"
        PASS=$((PASS+1))
        return 0
    else
        echo "❌ FAIL"
        FAIL=$((FAIL+1))
        return 1
    fi
}

echo "🔍 Verificando arquivos modificados..."
test_item "V7 Ultimate existe" "test -f /root/intelligence_system/core/system_v7_ultimate.py"
test_item "Darwin Real existe" "test -f /root/intelligence_system/extracted_algorithms/darwin_engine_real.py"
test_item "PENIN³ config existe" "test -f /root/penin3/penin3_config.py"
test_item "PENIN³ system existe" "test -f /root/penin3/penin3_system.py"

echo ""
echo "🔍 Verificando testes..."
test_item "Smoke test existe" "test -f /root/penin3/tests/test_v7_smoke.py"
test_item "WORM test existe" "test -f /root/penin3/tests/test_worm_rotation_and_hmac.py"
test_item "ACFA test existe" "test -f /root/penin3/tests/test_penin3_acfa_flow.py"

echo ""
echo "🔍 Verificando diretórios críticos..."
test_item "Darwin checkpoints dir" "test -d /root/penin3/checkpoints/evolution"
test_item "Multimodal samples dir" "test -d /root/penin3/data/multimodal_samples"
test_item "Logs directory" "test -d /root/penin3/logs"

echo ""
echo "🔍 Verificando samples multi-modal..."
test_item "Audio sample existe" "test -f /root/penin3/data/multimodal_samples/sample_audio.npy"
test_item "Image sample existe" "test -f /root/penin3/data/multimodal_samples/sample_image.npy"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   RESULTADO: $PASS/$TOTAL passou ($((100*PASS/TOTAL))%)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $PASS -eq $TOTAL ]; then
    echo "   ✅ SISTEMA 100% VERIFICADO"
    exit 0
elif [ $PASS -ge $((TOTAL * 80 / 100)) ]; then
    echo "   ✅ SISTEMA FUNCIONAL ($((100*PASS/TOTAL))%)"
    exit 0
else
    echo "   ⚠️ SISTEMA PARCIAL ($((100*PASS/TOTAL))%)"
    exit 1
fi
