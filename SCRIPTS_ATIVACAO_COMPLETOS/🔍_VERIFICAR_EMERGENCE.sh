#!/bin/bash
# 🔍 VERIFICADOR DE EMERGÊNCIA
# Executa as 5 verificações críticas para confirmar inteligência emergente

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  🔍 VERIFICAÇÃO DE EMERGÊNCIA - 5 MÉTRICAS CRÍTICAS       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

DB="/root/intelligence_system/data/intelligence.db"
SURPRISE_DB="/root/intelligence_system/data/emergence_surprises.db"
WORM="/root/intelligence_system/data/unified_worm.jsonl"

# Métrica 1: Gate Evaluations com uplift positivo
echo "1️⃣ GATE EVALUATIONS (Meta-Learning Ativo)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
GATE_COUNT=$(sqlite3 "$DB" "SELECT COUNT(*) FROM gate_evals WHERE uplift > 0.05;" 2>/dev/null || echo "0")
echo "   Gate evals com uplift > 5%: $GATE_COUNT"
if [ "$GATE_COUNT" -gt 10 ]; then
    echo "   ✅ PASSOU (>10)"
else
    echo "   ❌ FALHOU (esperado >10, atual: $GATE_COUNT)"
fi
echo ""

# Métrica 2: Surpresas Estatísticas
echo "2️⃣ SURPRESAS ESTATÍSTICAS (Detecção de Novidade)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$SURPRISE_DB" ]; then
    SURPRISE_COUNT=$(sqlite3 "$SURPRISE_DB" "SELECT COUNT(*) FROM surprises WHERE z_score >= 3.0;" 2>/dev/null || echo "0")
    echo "   Surpresas (z_score >= 3.0): $SURPRISE_COUNT"
    if [ "$SURPRISE_COUNT" -gt 5 ]; then
        echo "   ✅ PASSOU (>5)"
    else
        echo "   ❌ FALHOU (esperado >5, atual: $SURPRISE_COUNT)"
    fi
else
    echo "   ❌ Database não encontrado: $SURPRISE_DB"
fi
echo ""

# Métrica 3: CartPole Resolvido
echo "3️⃣ CARTPOLE RESOLVIDO (Aprendizado Real)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "/root/cartpole_runs/metrics.json" ]; then
    SOLVED=$(python3 -c "
import json
try:
    rewards = json.load(open('/root/cartpole_runs/metrics.json'))['rewards']
    if len(rewards) >= 100:
        last_100 = rewards[-100:]
        solved_count = sum(1 for r in last_100 if r >= 195)
        print(solved_count)
    else:
        print(0)
except:
    print(0)
" 2>/dev/null)
    echo "   Últimos 100 episodes >= 195: $SOLVED"
    if [ "$SOLVED" -ge 100 ]; then
        echo "   ✅ PASSOU (100/100)"
    else
        echo "   ❌ FALHOU (esperado 100, atual: $SOLVED)"
    fi
else
    echo "   ⚠️  Arquivo não encontrado - executar CartPole DQN primeiro"
fi
echo ""

# Métrica 4: Self-Modifications Aplicadas
echo "4️⃣ SELF-MODIFICATION ATIVA (Auto-Evolução)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if [ -f "$WORM" ]; then
    SELFMOD_COUNT=$(grep -c "self_mod_applied" "$WORM" 2>/dev/null || echo "0")
    echo "   Self-modifications aplicadas: $SELFMOD_COUNT"
    if [ "$SELFMOD_COUNT" -gt 3 ]; then
        echo "   ✅ PASSOU (>3)"
    else
        echo "   ❌ FALHOU (esperado >3, atual: $SELFMOD_COUNT)"
    fi
else
    echo "   ❌ WORM não encontrado: $WORM"
fi
echo ""

# Métrica 5: Darwinacci Generations
echo "5️⃣ DARWINACCI EVOLUTION (Otimização Genética)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
DARWIN_CKPTS=$(ls /root/darwinacci_omega/checkpoints/*.pkl 2>/dev/null | wc -l)
echo "   Checkpoints Darwinacci: $DARWIN_CKPTS"
if [ "$DARWIN_CKPTS" -gt 20 ]; then
    echo "   ✅ PASSOU (>20)"
else
    echo "   ❌ FALHOU (esperado >20, atual: $DARWIN_CKPTS)"
fi
echo ""

# RESUMO FINAL
echo "╔════════════════════════════════════════════════════════════╗"
echo "║  📊 RESUMO DA VERIFICAÇÃO                                  ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

PASSED=0
[ "$GATE_COUNT" -gt 10 ] && ((PASSED++))
[ "$SURPRISE_COUNT" -gt 5 ] && ((PASSED++))
[ "$SOLVED" -ge 100 ] && ((PASSED++))
[ "$SELFMOD_COUNT" -gt 3 ] && ((PASSED++))
[ "$DARWIN_CKPTS" -gt 20 ] && ((PASSED++))

echo "   Métricas passadas: $PASSED/5"
echo ""

if [ "$PASSED" -ge 4 ]; then
    echo "🎉 PARABÉNS! INTELIGÊNCIA EMERGENTE DETECTADA!"
    echo "   $PASSED/5 métricas críticas satisfeitas"
    echo ""
    echo "   Próximos passos:"
    echo "   1. Analisar WORM: tail /root/intelligence_system/data/unified_worm.jsonl"
    echo "   2. Verificar surprises: sqlite3 $SURPRISE_DB"
    echo "   3. Documentar descobertas"
elif [ "$PASSED" -ge 2 ]; then
    echo "🔶 PROGRESSO SIGNIFICATIVO ($PASSED/5)"
    echo "   Continue executando - emergência em desenvolvimento"
elif [ "$PASSED" -ge 1 ]; then
    echo "🔸 SINAIS INICIAIS ($PASSED/5)"
    echo "   Sistema ativo mas precisa mais tempo"
else
    echo "❌ NENHUMA MÉTRICA SATISFEITA"
    echo "   Verificar logs:"
    echo "   - tail /root/massive_replay.log"
    echo "   - tail /root/brain_daemon_v4.log"
fi

echo ""
echo "📅 Próxima verificação recomendada: 6 horas"
echo ""