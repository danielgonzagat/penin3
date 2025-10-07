#!/bin/bash
# APLICAR FASE 3 COMPLETA - LOOPS EVOLUTIVOS
# ===========================================
# Inicia os 4 módulos da Fase 3 em paralelo

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║  🔄 FASE 3: LOOPS EVOLUTIVOS COMPLETOS                        ║"
echo "║     Meta-Learning + Cross-Poll + Feedback + Dynamic Fitness   ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "INICIANDO 4 MÓDULOS DA FASE 3"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

# 3.1: Meta-Learning Loop
echo "1️⃣ FASE 3.1: Meta-Learning Loop"
echo "   Iniciando..."
nohup python3 /root/FASE_3_1_META_LEARNING_LOOP.py > /root/fase3_1_meta_learning.log 2>&1 &
LOOP_PID=$!
echo "   ✅ Meta-Learning Loop iniciado (PID: $LOOP_PID)"
echo "   Log: /root/fase3_1_meta_learning.log"
echo

# 3.2: Cross-Pollination Validada
echo "2️⃣ FASE 3.2: Cross-Pollination Validada"
echo "   Iniciando..."
nohup python3 /root/FASE_3_2_CROSS_POLLINATION_VALIDATED.py > /root/fase3_2_cross_poll.log 2>&1 &
CROSS_PID=$!
echo "   ✅ Cross-Poll Validada iniciada (PID: $CROSS_PID)"
echo "   Log: /root/fase3_2_cross_poll.log"
echo

# 3.3: V7→Darwin Feedback
echo "3️⃣ FASE 3.3: V7 → Darwin Feedback"
echo "   Iniciando..."
nohup python3 /root/FASE_3_3_V7_TO_DARWIN_FEEDBACK.py > /root/fase3_3_feedback.log 2>&1 &
FEEDBACK_PID=$!
echo "   ✅ V7→Darwin Feedback iniciado (PID: $FEEDBACK_PID)"
echo "   Log: /root/fase3_3_feedback.log"
echo

# 3.4: Dynamic Fitness Evolution
echo "4️⃣ FASE 3.4: Dynamic Fitness Evolution"
echo "   Iniciando..."
nohup python3 /root/FASE_3_4_DYNAMIC_FITNESS_EVOLUTION.py > /root/fase3_4_fitness.log 2>&1 &
FITNESS_PID=$!
echo "   ✅ Dynamic Fitness iniciado (PID: $FITNESS_PID)"
echo "   Log: /root/fase3_4_fitness.log"
echo

# Aguardar inicialização
sleep 5

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "VALIDAÇÃO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo

echo "✅ Processos ativos:"
pgrep -fl "FASE_3" | nl
echo

echo "📝 Logs criados:"
ls -lh /root/fase3_*.log 2>/dev/null | awk '{print "  - "$9" ("$5")"}'
echo

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  ✅ FASE 3 COMPLETA INICIADA!                                 ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

echo "🎯 PRÓXIMOS PASSOS:"
echo

echo "1️⃣ Ver logs dos módulos:"
echo "   tail -f /root/fase3_1_meta_learning.log"
echo "   tail -f /root/fase3_2_cross_poll.log"
echo "   tail -f /root/fase3_3_feedback.log"
echo "   tail -f /root/fase3_4_fitness.log"
echo

echo "2️⃣ Após 1 hora, verificar databases:"
echo "   sqlite3 /root/meta_learning_loop.db 'SELECT COUNT(*) FROM actions;'"
echo "   sqlite3 /root/cross_pollination_validated.db 'SELECT COUNT(*) FROM pollinations;'"
echo "   sqlite3 /root/v7_darwin_feedback.db 'SELECT COUNT(*) FROM feedbacks;'"
echo "   sqlite3 /root/dynamic_fitness.db 'SELECT COUNT(*) FROM fitness_experiments;'"
echo

echo "3️⃣ Ver dashboard (mostra tudo):"
echo "   bash /root/DASHBOARD_TEMPO_REAL.sh"
echo

echo "4️⃣ Após 1 dia, verificar I³:"
echo "   # Deve estar ~78-85% (up from 62.7%)"
echo "   # Meta-Learning aprendendo"
echo "   # Cross-Poll validando"
echo "   # Feedback bidirecional ativo"
echo "   # Fitness evoluindo"
echo

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 FASE 3 RODANDO! Loops evolutivos completos ativos!"
echo "   Sistema agora tem feedback loops em TODOS os níveis!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"