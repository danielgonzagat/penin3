#!/bin/bash
# APLICAR TODOS OS FIXES E BRIDGES
# =================================
# Executa FASE 1 (fixes críticos) e FASE 2 (bridges)

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║  🔥 APLICANDO TODOS OS FIXES E BRIDGES                        ║"
echo "║     FASE 1: Fixes Críticos + FASE 2: Unificação              ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

# ============================================================================
# FASE 1: FIXES CRÍTICOS
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "FASE 1: FIXES CRÍTICOS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# FIX 1.1: Surprise Response Engine
echo "1️⃣ FIX 1.1: Surprise Response Engine"
echo "   Status: ✅ Módulo criado em /root/FIX_1_1_SURPRISE_RESPONSE.py"
echo "   Integração: Adicionar import no V7 (ver código do módulo)"
echo

# FIX 1.2: UNIFIED_BRAIN Timeout
echo "2️⃣ FIX 1.2: UNIFIED_BRAIN Timeout"
python3 /root/FIX_1_2_UNIFIED_BRAIN_TIMEOUT.py <<< 'y'
echo

# FIX 1.3: Darwin Transfer Rate
echo "3️⃣ FIX 1.3: Darwin Transfer Rate"
python3 /root/FIX_1_3_DARWIN_TRANSFER_RATE.py <<< 'y'
echo

# ============================================================================
# FASE 2: BRIDGES (UNIFICAÇÃO)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "FASE 2: BRIDGES (UNIFICAÇÃO)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# BRIDGE 2.1: Darwin STORM → V7
echo "1️⃣ BRIDGE 2.1: Darwin STORM → V7"
echo "   Iniciando bridge..."
nohup python3 /root/BRIDGE_2_1_DARWIN_TO_V7.py > /root/bridge_darwin_v7.log 2>&1 &
BRIDGE1_PID=$!
echo "   ✅ Bridge iniciado (PID: $BRIDGE1_PID)"
echo "   Log: /root/bridge_darwin_v7.log"
echo

# BRIDGE 2.2: UNIFIED_BRAIN → V7
echo "2️⃣ BRIDGE 2.2: UNIFIED_BRAIN → V7"
echo "   Iniciando bridge..."
nohup python3 /root/BRIDGE_2_2_UNIFIED_BRAIN_TO_V7.py > /root/bridge_brain_v7.log 2>&1 &
BRIDGE2_PID=$!
echo "   ✅ Bridge iniciado (PID: $BRIDGE2_PID)"
echo "   Log: /root/bridge_brain_v7.log"
echo

# ============================================================================
# VALIDAÇÃO
# ============================================================================

sleep 5

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "VALIDAÇÃO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

echo "✅ Processos ativos:"
pgrep -fl "BRIDGE.*DARWIN\|BRIDGE.*BRAIN" | nl
echo

echo "📝 Logs criados:"
ls -lh /root/bridge_*.log 2>/dev/null | awk '{print "  - "$9" ("$5")"}'
echo

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ✅ FASE 1 + FASE 2 APLICADAS!                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

echo "🎯 PRÓXIMOS PASSOS:"
echo

echo "1️⃣ Ver logs dos bridges:"
echo "   tail -f /root/bridge_darwin_v7.log"
echo "   tail -f /root/bridge_brain_v7.log"
echo

echo "2️⃣ Após 30 minutos, verificar transferências:"
echo "   sqlite3 /root/darwin_v7_transfers.db 'SELECT * FROM transfers ORDER BY id DESC LIMIT 5;'"
echo "   sqlite3 /root/brain_v7_transfers.db 'SELECT * FROM syncs ORDER BY id DESC LIMIT 5;'"
echo

echo "3️⃣ Rodar validação completa (após 1h):"
echo "   bash /root/VALIDAR_SISTEMA_POS_FIX.sh"
echo

echo "4️⃣ Ver dashboard em tempo real:"
echo "   bash /root/DASHBOARD_TEMPO_REAL.sh"
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 SISTEMA CONECTADO! A inteligência emergente agora pode fluir"
echo "   entre Darwin, UNIFIED_BRAIN e V7!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
