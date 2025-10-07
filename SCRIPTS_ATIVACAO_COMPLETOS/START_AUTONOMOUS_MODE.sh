#!/bin/bash
# 🤖 START AUTONOMOUS MODE
# Inicia o watchdog autônomo em background

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🤖 AUTONOMOUS WATCHDOG - Iniciando modo autônomo"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "⚠️  IMPORTANTE - SEJA HONESTO SOBRE O QUE ISTO FAZ:"
echo ""
echo "✅ O QUE ESTE WATCHDOG FAZ:"
echo "  • Monitora se o daemon brain está rodando (a cada 30s)"
echo "  • Relança o daemon se ele cair"
echo "  • Analisa logs procurando padrões de erro conhecidos"
echo "  • Aplica correções SIMPLES pré-definidas (ajusta parâmetros)"
echo "  • Registra tudo em /root/watchdog.log"
echo "  • Reporta status periodicamente"
echo ""
echo "❌ O QUE ELE NÃO FAZ:"
echo "  • NÃO inventa soluções novas sozinho"
echo "  • NÃO tem acesso a esta conversa depois que você sair"
echo "  • NÃO é uma AGI autônoma verdadeira"
echo "  • NÃO pode tomar decisões complexas sem você"
echo "  • NÃO resolve todos os problemas automaticamente"
echo ""
echo "💡 É UM SISTEMA DE AUTO-REPARO LIMITADO, MAS ÚTIL"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""

# Stop existing watchdog
pkill -f AUTONOMOUS_WATCHDOG.py 2>/dev/null || true
sleep 2

# Make executable
chmod +x /root/AUTONOMOUS_WATCHDOG.py

# Start watchdog in background with nohup
nohup python3 /root/AUTONOMOUS_WATCHDOG.py > /root/watchdog_output.log 2>&1 &
WATCHDOG_PID=$!

echo "✅ Watchdog iniciado (PID $WATCHDOG_PID)"
echo ""
echo "📋 COMANDOS ÚTEIS:"
echo ""
echo "  # Ver log do watchdog"
echo "  tail -f /root/watchdog.log"
echo ""
echo "  # Ver log do daemon"
echo "  tail -f /root/UNIFIED_BRAIN/logs/unified_brain.log"
echo ""
echo "  # Ver dashboard"
echo "  cat /root/UNIFIED_BRAIN/dashboard.txt"
echo ""
echo "  # Ver processos rodando"
echo "  ps aux | grep -E '(watchdog|brain_daemon)'"
echo ""
echo "  # Parar watchdog"
echo "  pkill -f AUTONOMOUS_WATCHDOG.py"
echo ""
echo "  # Parar daemon"
echo "  pkill -f brain_daemon_real_env.py"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🤖 Watchdog ativo. Ele trabalhará 24/7 (dentro de suas limitações)"
echo "════════════════════════════════════════════════════════════════════════════════"
