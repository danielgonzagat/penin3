#!/bin/bash
# 📊 STATUS RÁPIDO - Para você verificar a qualquer momento

echo "════════════════════════════════════════════════════════════"
echo "📊 STATUS DO SISTEMA - $(date '+%Y-%m-%d %H:%M:%S')"
echo "════════════════════════════════════════════════════════════"
echo ""

# Brain Daemon
BRAIN_COUNT=$(pgrep -f "brain_daemon_real_env.py" | wc -l)
if [ "$BRAIN_COUNT" -gt 0 ]; then
    echo "🧠 Brain Daemon: ✅ ATIVO ($BRAIN_COUNT processos)"
    
    # Último episódio
    if [ -f /root/brain_v3_FIXED.log ]; then
        ULTIMO_EP=$(grep "Ep [0-9]" /root/brain_v3_FIXED.log | tail -1)
        echo "   $ULTIMO_EP"
    fi
else
    echo "🧠 Brain Daemon: ❌ MORTO"
fi

echo ""

# Recursos
CPU_PCT=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
RAM_PCT=$(free | grep Mem | awk '{print int($3/$2 * 100)}')
DISCO_PCT=$(df / | tail -1 | awk '{print $5}')

echo "⚡ Recursos:"
echo "   CPU: ${CPU_PCT}%"
echo "   RAM: ${RAM_PCT}%"
echo "   Disco: ${DISCO_PCT}"
echo ""

# Surprises
if [ -f /root/intelligence_system/data/emergence_surprises.db ]; then
    SURPRISES_5S=$(sqlite3 /root/intelligence_system/data/emergence_surprises.db "SELECT COUNT(*) FROM surprises WHERE sigma > 5.0" 2>/dev/null || echo "0")
    SURPRISES_9S=$(sqlite3 /root/intelligence_system/data/emergence_surprises.db "SELECT COUNT(*) FROM surprises WHERE sigma > 9.0" 2>/dev/null || echo "0")
    
    echo "🌟 Emergência:"
    echo "   Surprises >5σ: $SURPRISES_5S"
    echo "   Surprises >9σ: $SURPRISES_9S"
    
    if [ "$SURPRISES_9S" -gt 0 ]; then
        echo "   ⚡ SINAIS DE INTELIGÊNCIA EMERGENTE DETECTADOS!"
    fi
fi

echo ""

# Alertas críticos
if [ -f /root/ALERTAS_CRITICOS.txt ]; then
    NUM_ALERTAS=$(wc -l < /root/ALERTAS_CRITICOS.txt)
    if [ "$NUM_ALERTAS" -gt 0 ]; then
        echo "🚨 ALERTAS CRÍTICOS: $NUM_ALERTAS"
        echo "   Ver: cat /root/ALERTAS_CRITICOS.txt"
    else
        echo "✅ Sem alertas críticos"
    fi
else
    echo "✅ Sem alertas críticos"
fi

echo ""

# Monitor contínuo
if pgrep -f "monitor_continuo.sh" > /dev/null; then
    echo "🔄 Monitor contínuo: ✅ ATIVO"
else
    echo "🔄 Monitor contínuo: ⚠️ NÃO ATIVO"
    echo "   Iniciar: nohup bash /root/monitor_continuo.sh &"
fi

# Watchdog
if pgrep -f "watchdog_daemon.sh" > /dev/null; then
    echo "🐕 Watchdog: ✅ ATIVO"
else
    echo "🐕 Watchdog: ⚠️ NÃO ATIVO"
    echo "   Iniciar: nohup bash /root/watchdog_daemon.sh &"
fi

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📋 COMANDOS ÚTEIS:"
echo "════════════════════════════════════════════════════════════"
echo "   Monitor real-time:  tail -f /root/brain_v3_FIXED.log"
echo "   Ver alertas:        cat /root/ALERTAS_CRITICOS.txt"
echo "   Health check:       bash /root/health_check.sh"
echo "   Este status:        bash /root/status_rapido.sh"
echo "════════════════════════════════════════════════════════════"
