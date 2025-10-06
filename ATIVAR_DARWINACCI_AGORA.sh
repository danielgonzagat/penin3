#!/bin/bash
# 🚀 ATIVAR DARWINACCI + UNIFIED_BRAIN INTEGRAÇÃO

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 ATIVANDO DARWINACCI-Ω + UNIFIED_BRAIN"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Esta integração conecta o motor evolutivo Darwinacci ao UNIFIED_BRAIN"
echo "para evoluir hiperparâmetros automaticamente."
echo ""

# 1. Iniciar Darwinacci Hyperparameter Evolver
echo "1️⃣ Iniciando Darwinacci Hyperparameter Evolver..."
chmod +x /root/DARWINACCI_QUICK_INTEGRATION.py
nohup python3 /root/DARWINACCI_QUICK_INTEGRATION.py > /root/darwinacci_evolver.log 2>&1 &
EVOLVER_PID=$!
echo "   ✅ Evolver rodando (PID $EVOLVER_PID)"
echo ""

# 2. Verificar processos
echo "2️⃣ Verificando processos autônomos..."
sleep 3
PROCS=$(ps aux | grep python3 | grep -E '(WATCHDOG|MONITOR|brain_daemon|DARWINACCI)' | grep -v grep | wc -l)
echo "   ✅ $PROCS processos ativos"
echo ""

# 3. Status
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ SISTEMA INTEGRADO ATIVO"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Processos rodando:"
echo "  1. UNIFIED_BRAIN Daemon (aprendizado contínuo)"
echo "  2. Autonomous Watchdog (mantém tudo vivo)"
echo "  3. Emergence Monitor (detecta sinais)"
echo "  4. Darwinacci Evolver (evolve hiperparâmetros) ← NOVO!"
echo ""
echo "Monitoramento:"
echo "  # Darwinacci evolution events"
echo "  tail -f /root/darwinacci_evolver.log"
echo ""
echo "  # Hyperparameters being applied"
echo "  watch -n 10 'cat /root/UNIFIED_BRAIN/runtime_suggestions.json 2>/dev/null'"
echo ""
echo "  # Overall status"
echo "  tail -f /root/watchdog.log"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "💎 DARWINACCI AGORA CONECTADO AO SISTEMA PRINCIPAL"
echo "════════════════════════════════════════════════════════════════════════════════"

