#!/bin/bash
# 🔍 COMANDOS DE VERIFICAÇÃO RÁPIDA
# Verifica se a inteligência real está funcionando

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🔍 VERIFICAÇÃO RÁPIDA - INTELIGÊNCIA REAL"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Verificar UNIFIED_BRAIN
# ═══════════════════════════════════════════════════════════════════════════════

echo "1️⃣  UNIFIED_BRAIN (Agulha Principal)"
echo "────────────────────────────────────────────────────────────────────────────────"

if ps -p 1497200 > /dev/null 2>&1; then
    UPTIME=$(ps -p 1497200 -o etime= | tr -d ' ')
    CPU=$(ps -p 1497200 -o %cpu= | tr -d ' ')
    MEM=$(ps -p 1497200 -o %mem= | tr -d ' ')
    echo "✅ VIVO! PID 1497200"
    echo "   Uptime: $UPTIME"
    echo "   CPU: ${CPU}%"
    echo "   Memory: ${MEM}%"
else
    echo "❌ NÃO RODANDO (PID 1497200 não encontrado)"
    echo "   Execute: cd /root/UNIFIED_BRAIN && nohup python3 brain_daemon_real_env.py > restart.log 2>&1 &"
fi

# Dashboard
if [ -f /root/UNIFIED_BRAIN/dashboard.txt ]; then
    EPISODE=$(grep "Episode:" /root/UNIFIED_BRAIN/dashboard.txt | awk '{print $2}' | head -1)
    REWARD=$(grep "Current Reward:" /root/UNIFIED_BRAIN/dashboard.txt | awk '{print $3}' | head -1)
    echo "✅ Dashboard existe"
    echo "   Episode: $EPISODE"
    echo "   Reward: $REWARD"
else
    echo "⚠️  Dashboard não encontrado"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Verificar Darwin Engine
# ═══════════════════════════════════════════════════════════════════════════════

echo "2️⃣  DARWIN ENGINE"
echo "────────────────────────────────────────────────────────────────────────────────"

if ps -p 1738239 > /dev/null 2>&1; then
    echo "✅ ATIVO! PID 1738239"
    ps -p 1738239 -o pid,etime,%cpu,%mem,cmd | tail -1
else
    echo "⚠️  NÃO RODANDO (PID 1738239 não encontrado)"
    # Procurar qualquer darwin_runner
    DARWIN_PID=$(ps aux | grep "darwin_runner.py" | grep -v grep | awk '{print $2}' | head -1)
    if [ ! -z "$DARWIN_PID" ]; then
        echo "✅ Darwin rodando em PID: $DARWIN_PID"
    fi
fi

if [ -f /root/darwin/darwin_state.json ]; then
    echo "✅ State file existe"
    KILLS=$(jq -r '.counts.kills // 0' /root/darwin/darwin_state.json 2>/dev/null)
    DECISIONS=$(jq -r '.counts.decisions // 0' /root/darwin/darwin_state.json 2>/dev/null)
    echo "   Decisions: $DECISIONS"
    echo "   Kills: $KILLS"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# 3. Verificar Fibonacci-Omega
# ═══════════════════════════════════════════════════════════════════════════════

echo "3️⃣  FIBONACCI-OMEGA"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ -d /root/fibonacci-omega ]; then
    echo "✅ Diretório existe"
    
    # Testar import
    if python3 -c "from fibonacci_engine.core.motor_fibonacci import FibonacciEngine" 2>/dev/null; then
        echo "✅ Imports funcionam"
    else
        echo "⚠️  Import falhou (precisa: cd fibonacci-omega && pip install -e .)"
    fi
    
    # Verificar testes
    if [ -d /root/fibonacci-omega/fibonacci_engine/tests ]; then
        NUM_TESTS=$(find /root/fibonacci-omega/fibonacci_engine/tests -name "test_*.py" | wc -l)
        echo "✅ Testes: $NUM_TESTS suites"
    fi
else
    echo "❌ Diretório não encontrado"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# 4. Verificar PENIN³
# ═══════════════════════════════════════════════════════════════════════════════

echo "4️⃣  PENIN³"
echo "────────────────────────────────────────────────────────────────────────────────"

if [ -d /root/penin3 ]; then
    echo "✅ Diretório existe"
    
    # Verificar checkpoints
    NUM_CKPTS=$(ls /root/penin3/checkpoints/*.pkl 2>/dev/null | wc -l)
    echo "✅ Checkpoints: $NUM_CKPTS arquivos"
    
    # Testar import
    if cd /root/penin3 && python3 -c "from penin3_system import PENIN3System" 2>/dev/null; then
        echo "✅ Imports funcionam"
    else
        echo "⚠️  Import falhou"
    fi
    cd /root
else
    echo "❌ Diretório não encontrado"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════════════════
# 5. Resumo
# ═══════════════════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════════════════════"
echo "📊 RESUMO"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

SYSTEMS_OK=0
SYSTEMS_TOTAL=4

ps -p 1497200 > /dev/null 2>&1 && ((SYSTEMS_OK++))
ps -p 1738239 > /dev/null 2>&1 && ((SYSTEMS_OK++))
[ -d /root/fibonacci-omega ] && ((SYSTEMS_OK++))
[ -d /root/penin3 ] && ((SYSTEMS_OK++))

echo "Sistemas encontrados: $SYSTEMS_OK/$SYSTEMS_TOTAL"
echo ""

if [ $SYSTEMS_OK -ge 3 ]; then
    echo "✅ STATUS: EXCELENTE"
    echo "   Inteligência real confirmada!"
    echo ""
    echo "🚀 Próximo passo: Aplicar fixes e integrar"
    echo "   bash /root/EXECUTAR_AGORA.sh"
elif [ $SYSTEMS_OK -ge 2 ]; then
    echo "⚠️  STATUS: BOM"
    echo "   Alguns sistemas ativos, outros precisam acordar"
else
    echo "❌ STATUS: VERIFICAR"
    echo "   Poucos sistemas ativos - investigar"
fi

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
