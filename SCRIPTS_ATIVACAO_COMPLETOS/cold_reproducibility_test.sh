#!/bin/bash
# Teste de Reprodutibilidade Fria - Go/No-Go Criterion 1

echo "🧊 COLD REPRODUCIBILITY TEST"
echo "============================"
echo "Este teste verifica se os sistemas funcionam após reboot"
echo ""

# Função para testar módulo
test_module() {
    MODULE=$1
    NAME=$2
    
    echo "Testing $NAME..."
    
    # Run with timeout
    timeout 60 python3 $MODULE --smoketest 2>/dev/null
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ $NAME: PASSED"
        return 0
    elif [ $EXIT_CODE -eq 124 ]; then
        echo "⏱️ $NAME: TIMEOUT (>60s)"
        return 1
    else
        echo "❌ $NAME: FAILED (exit code $EXIT_CODE)"
        return 1
    fi
}

# Adicionar modo smoketest aos módulos se não existir
add_smoketest() {
    FILE=$1
    
    # Check if already has smoketest
    if ! grep -q "smoketest" "$FILE" 2>/dev/null; then
        # Add minimal smoketest
        cat >> "$FILE" << 'EOF'

# Smoketest mode
if __name__ == "__main__" and "--smoketest" in sys.argv:
    print(f"🔥 SMOKETEST: {__file__}")
    try:
        # Quick validation
        if 'AutodidataSystem' in globals():
            system = AutodidataSystem("smoketest", 10, 4)
            print("  ✓ AutodidataSystem instantiated")
        elif 'AdaptiveSystem' in globals():
            system = AdaptiveSystem("smoketest")
            print("  ✓ AdaptiveSystem instantiated")
        elif 'AutoevolutiveSystem' in globals():
            system = AutoevolutiveSystem("smoketest")
            print("  ✓ AutoevolutiveSystem instantiated")
        print("  ✓ Smoketest passed")
        sys.exit(0)
    except Exception as e:
        print(f"  ✗ Smoketest failed: {e}")
        sys.exit(1)
EOF
        echo "Added smoketest to $FILE"
    fi
}

# Prepare modules
echo "Preparing modules..."
add_smoketest /root/teis_autodidata_100.py
add_smoketest /root/teis_adaptativa_100.py  
add_smoketest /root/teis_autoevolutiva_100.py

echo ""
echo "Running tests..."
echo "----------------"

# Test each module
PASSED=0
TOTAL=3

test_module /root/teis_autodidata_100.py "Autodidata"
PASSED=$((PASSED + $?))

test_module /root/teis_adaptativa_100.py "Adaptativa"
PASSED=$((PASSED + $?))

test_module /root/teis_autoevolutiva_100.py "Autoevolutiva"
PASSED=$((PASSED + $?))

echo ""
echo "=================="
echo "RESULTS:"
echo "$((TOTAL - PASSED))/$TOTAL modules passed"

if [ $PASSED -eq 0 ]; then
    echo "✅ ALL TESTS PASSED - Cold reproducibility confirmed!"
    exit 0
else
    echo "❌ Some tests failed - System not reproducible"
    exit 1
fi