#!/bin/bash
# FIX PRIORIDADE #1: INICIAR DARWIN EVOLUTION
# ============================================
# PROBLEMA: Darwin não está rodando (CPU 0.1%, sem checkpoints novos)
# SOLUÇÃO: Iniciar run_emergence_blocks_STORM.py

set -e

echo "🔥 FIX #1: INICIANDO DARWIN COM MUTATION STORM"
echo "=============================================="
echo

# 1. Verificar se Darwin já está rodando
if pgrep -f "run_emergence_blocks_STORM" > /dev/null; then
    echo "⚠️  Darwin já está rodando!"
    pgrep -fl run_emergence_blocks_STORM
    echo
    echo "Se quiser reiniciar, execute:"
    echo "  pkill -f run_emergence_blocks_STORM"
    echo "  bash $0"
    exit 0
fi

# 2. Iniciar Darwin em background
echo "1️⃣ Iniciando Darwin evolution..."
nohup python3 /root/run_emergence_blocks_STORM.py > /root/darwin_STORM.log 2>&1 &
DARWIN_PID=$!
echo "   PID: $DARWIN_PID"
echo

# 3. Aguardar inicio
echo "2️⃣ Aguardando Darwin iniciar..."
sleep 5

# 4. Verificar se está rodando
if pgrep -f "run_emergence_blocks_STORM" > /dev/null; then
    echo "✅ Darwin ATIVO!"
    pgrep -fl run_emergence_blocks_STORM
    echo
else
    echo "❌ Darwin falhou ao iniciar!"
    echo
    echo "Log de erro:"
    tail -n 50 /root/darwin_STORM.log
    exit 1
fi

# 5. Mostrar log inicial
echo "3️⃣ Log Darwin (primeiras linhas):"
echo "================================"
tail -n 30 /root/darwin_STORM.log
echo

# 6. Instruções
echo "╔════════════════════════════════════════╗"
echo "║  ✅ DARWIN INICIADO COM SUCESSO!      ║"
echo "╚════════════════════════════════════════╝"
echo
echo "📊 Para acompanhar evolução:"
echo "   tail -f /root/darwin_STORM.log"
echo
echo "🔍 Verificar CPU Darwin (deve subir para 10-30%):"
echo "   top -p \$(pgrep -f run_emergence_blocks_STORM)"
echo
echo "🧬 Verificar novos checkpoints (após 30min):"
echo "   ls -lht /root/intelligence_system/models/darwin_checkpoints/*.pt | head -10"
echo
echo "🎯 PRÓXIMO PASSO: bash /root/FIX_PRIORIDADE_2_SURPRISE_DETECTOR.sh"
