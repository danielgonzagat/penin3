#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# WATCH INTELLIGENCE - Monitoramento Contínuo em Tempo Real
# Executa: ./WATCH_INTELLIGENCE.sh
# ═══════════════════════════════════════════════════════════════

while true; do
    clear
    echo "╔═══════════════════════════════════════════════════════════════════════╗"
    echo "║           🧠 INTELLIGENCE SYSTEM - MONITORAMENTO TEMPO REAL          ║"
    echo "║                     Press Ctrl+C to exit                             ║"
    echo "╚═══════════════════════════════════════════════════════════════════════╝"
    echo ""
    date "+%Y-%m-%d %H:%M:%S"
    echo ""
    
    # Dashboard
    python3 /root/SISTEMA_STATUS_DASHBOARD.py 2>/dev/null || echo "⚠️  Dashboard error"
    
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "📋 LOGS RECENTES:"
    echo ""
    
    # Intelligence Nexus
    echo "🧠 Intelligence Nexus (últimas 3 linhas):"
    tail -3 /root/intelligence_nexus.log 2>/dev/null | sed 's/^/   /' || echo "   (sem logs)"
    echo ""
    
    # Darwinacci
    echo "🧬 Darwinacci (últimas 2 linhas):"
    tail -2 /root/darwinacci_daemon.log 2>/dev/null | grep "Darwinacci Gen" | tail -1 | sed 's/^/   /' || echo "   (sem logs)"
    echo ""
    
    # Auto-Modifier
    echo "🔧 Auto-Modifier (última linha):"
    tail -1 /root/auto_modifier.log 2>/dev/null | sed 's/^/   /' || echo "   (sem logs ainda)"
    echo ""
    
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "🎯 COMANDOS RÁPIDOS:"
    echo "   Ver logs completos:  tail -f /root/intelligence_nexus.log"
    echo "   Forçar ciclo Nexus:  python3 /root/INTELLIGENCE_NEXUS.py --once"
    echo "   Chat com Qwen:       python3 /root/qwen_chat.py"
    echo ""
    
    sleep 10
done
