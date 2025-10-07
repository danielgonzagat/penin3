#!/bin/bash
# 🔍 INTELLIGENCE SYSTEM STATUS CHECKER
# Verifica status de todos os componentes do sistema

echo "═════════════════════════════════════════════════════════════"
echo "🧠 INTELLIGENCE SYSTEM STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
echo "═════════════════════════════════════════════════════════════"

echo ""
echo "📊 PROCESSOS ATIVOS:"
echo "-------------------"

# Brain Daemon
BRAIN_COUNT=$(pgrep -f "brain_daemon_real_env.py" | wc -l)
echo "🧠 Brain Daemon: $BRAIN_COUNT instância(s)"
if [ $BRAIN_COUNT -eq 0 ]; then
    echo "   ⚠️  WARNING: Brain Daemon não está rodando!"
fi

# Darwin Evolution
DARWIN_COUNT=$(pgrep -f "darwin_runner.py" | wc -l)
echo "🧬 Darwin Evolution: $DARWIN_COUNT instância(s)"
if [ $DARWIN_COUNT -eq 0 ]; then
    echo "   ⚠️  WARNING: Darwin não está rodando!"
fi

# Monitores
GUARDIAN_COUNT=$(pgrep -f "darwin_continuity_guardian" | wc -l)
echo "👁️  Darwin Guardian: $GUARDIAN_COUNT instância(s)"

HEALTH_COUNT=$(pgrep -f "health_monitor_phase1" | wc -l)
echo "🏥 Health Monitor: $HEALTH_COUNT instância(s)"

EMERGENCE_COUNT=$(pgrep -f "monitor_emergencia" | wc -l)
echo "🌟 Emergence Monitor: $EMERGENCE_COUNT instância(s)"

echo ""
echo "📈 MÉTRICAS (últimas 24h):"
echo "--------------------------"

# Brain Metrics
BRAIN_METRICS=$(sqlite3 /root/intelligence_system/data/intelligence.db \
    "SELECT COUNT(*) FROM brain_metrics WHERE timestamp > $(date -d '24 hours ago' +%s)" 2>/dev/null)
echo "🧠 Brain Metrics: $BRAIN_METRICS registros"

LATEST_EPISODE=$(sqlite3 /root/intelligence_system/data/intelligence.db \
    "SELECT MAX(episode) FROM brain_metrics" 2>/dev/null)
echo "📊 Latest Episode: $LATEST_EPISODE"

LATEST_ENERGY=$(sqlite3 /root/intelligence_system/data/intelligence.db \
    "SELECT energy FROM brain_metrics ORDER BY timestamp DESC LIMIT 1" 2>/dev/null)
echo "⚡ Latest Energy: $LATEST_ENERGY"

# Emergence Tracking
RAW_METRICS=$(sqlite3 /root/intelligence_system/data/emergence_surprises.db \
    "SELECT COUNT(*) FROM metrics_raw" 2>/dev/null)
echo "📊 Raw Metrics: $RAW_METRICS"

DIAGNOSTICS=$(sqlite3 /root/intelligence_system/data/emergence_surprises.db \
    "SELECT COUNT(*) FROM diagnostics WHERE sigma >= 2.0" 2>/dev/null)
echo "🔬 Diagnostics (≥2σ): $DIAGNOSTICS"

SURPRISES=$(sqlite3 /root/intelligence_system/data/emergence_surprises.db \
    "SELECT COUNT(*) FROM surprises WHERE sigma > 3.0" 2>/dev/null)
echo "🌟 Surprises (>3σ): $SURPRISES"

if [ $SURPRISES -gt 0 ]; then
    echo ""
    echo "🚨 HIGH-SIGMA SURPRISES DETECTED:"
    sqlite3 /root/intelligence_system/data/emergence_surprises.db \
        "SELECT metric_name, sigma, actual_value FROM surprises WHERE sigma > 3.0 ORDER BY sigma DESC LIMIT 5" 2>/dev/null
fi

echo ""
echo "💾 WORM LEDGER:"
echo "--------------"
if [ -f /root/darwinacci_omega/data/worm.csv ]; then
    WORM_LINES=$(wc -l < /root/darwinacci_omega/data/worm.csv)
    echo "📜 Darwinacci WORM: $WORM_LINES entradas"
fi

if [ -f /root/UNIFIED_BRAIN/worm.log ]; then
    BRAIN_WORM_LINES=$(wc -l < /root/UNIFIED_BRAIN/worm.log)
    echo "🧠 Brain WORM: $BRAIN_WORM_LINES entradas"
fi

echo ""
echo "🖥️  RECURSOS DO SISTEMA:"
echo "----------------------"
LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
echo "📊 Load Average: $LOAD"

MEM=$(free -h | grep Mem | awk '{print "Used: "$3" / "$2" ("$3/$2*100"%)"'}')
echo "💾 Memory: $MEM"

DISK=$(df -h / | tail -1 | awk '{print "Used: "$3" / "$2" ("$5")"}')
echo "💿 Disk: $DISK"

echo ""
echo "═════════════════════════════════════════════════════════════"
echo "✅ Status check complete"
echo "═════════════════════════════════════════════════════════════"