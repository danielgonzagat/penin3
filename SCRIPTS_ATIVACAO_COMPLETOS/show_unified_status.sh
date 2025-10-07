#!/bin/bash
# Dashboard unificado - mostra TODA inteligência ativa

clear
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🧠 STATUS UNIFICADO DE INTELIGÊNCIA - $(date)"
echo "════════════════════════════════════════════════════════════════════════════════"
echo

# UNIFIED_BRAIN
if [ -f /root/UNIFIED_BRAIN/dashboard.txt ]; then
    echo "✅ UNIFIED_BRAIN (SISTEMA PRINCIPAL):"
    cat /root/UNIFIED_BRAIN/dashboard.txt
    echo
else
    echo "⚠️  UNIFIED_BRAIN dashboard não encontrado"
    echo
fi

# Database stats
echo "📊 DATABASE METRICS:"
sqlite3 /root/intelligence_system/data/intelligence.db << SQL
.mode column
.headers on
SELECT 
    'Total Episodes' as metric,
    COUNT(*) as value 
FROM brain_metrics
UNION ALL
SELECT 
    'Last Update',
    datetime(MAX(timestamp), 'unixepoch', 'localtime')
FROM brain_metrics
UNION ALL
SELECT
    'Avg Coherence (last 10)',
    ROUND(AVG(coherence), 4)
FROM (SELECT coherence FROM brain_metrics ORDER BY timestamp DESC LIMIT 10)
UNION ALL
SELECT
    'Avg Novelty (last 10)',
    ROUND(AVG(novelty), 4)
FROM (SELECT novelty FROM brain_metrics ORDER BY timestamp DESC LIMIT 10);
SQL
echo

# Processos ativos
echo "🔄 PROCESSOS DE INTELIGÊNCIA ATIVOS:"
ps aux | grep -E "(brain_daemon|darwin_runner|unified_agi)" | grep -v grep | \
    awk '{printf "   PID %s: %s (CPU: %.1f%%, MEM: %.1f%%, TIME: %s)\n", $2, $11, $3, $4, $10}' || echo "   Nenhum"
echo

# Últimas conquistas
echo "🏆 ÚLTIMAS CONQUISTAS:"
tail -50 /root/UNIFIED_BRAIN/logs/unified_brain.log 2>/dev/null | \
    grep "NEW BEST" | tail -3 || echo "   Nenhuma recente"
echo

echo "════════════════════════════════════════════════════════════════════════════════"
echo "💡 Para monitorar em tempo real: tail -f /root/UNIFIED_BRAIN/logs/unified_brain.log | grep 'NEW BEST'"
echo "════════════════════════════════════════════════════════════════════════════════"
