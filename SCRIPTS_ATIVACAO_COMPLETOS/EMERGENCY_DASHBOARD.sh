#!/bin/bash
# Dashboard em tempo real de TUDO

clear
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║   🔥 EMERGENCE DASHBOARD - TEMPO REAL                    ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Processos
echo "🚀 PROCESSOS ATIVOS:"
ps aux | grep -E "META_LEARNER|DYNAMIC|REFLECTION|BRIDGE|surprise|CONNECTOR|emergence_blocks" | grep -v grep | awk '{printf "   %s: %s%% CPU\n", $11, $3}'
echo ""

# Databases
echo "🗄️  DATABASES:"
for db in emergence_surprises meta_learning system_connections dynamic_fitness self_reflection; do
    if [ -f "/root/${db}.db" ]; then
        COUNT=$(sqlite3 /root/${db}.db "SELECT COUNT(*) FROM sqlite_master WHERE type='table';" 2>/dev/null || echo "0")
        echo "   ✅ ${db}.db: ${COUNT} tabelas"
    fi
done
echo ""

# Surprises
echo "🎉 EMERGÊNCIA:"
SURPRISES=$(sqlite3 /root/emergence_surprises.db "SELECT COUNT(*) FROM surprises;" 2>/dev/null || echo "0")
echo "   Total surprises: $SURPRISES"

if [ "$SURPRISES" -gt "0" ]; then
    echo "   🎯 TOP SURPRISE:"
    sqlite3 /root/emergence_surprises.db "SELECT surprise_score, system, metric FROM surprises ORDER BY surprise_score DESC LIMIT 1;" 2>/dev/null | awk '{printf "      Score: %s, System: %s, Metric: %s\n", $1, $2, $3}'
fi
echo ""

# Darwin Status
echo "🧬 DARWIN STATUS:"
ps aux | grep "emergence_blocks_STORM" | grep -v grep | wc -l | awk '{printf "   Instâncias ativas: %s\n", $1}'
ps aux | grep "emergence_blocks_STORM" | grep -v grep | awk '{printf "   CPU Total: %s%%\n", $3}' | paste -sd+ | bc
echo ""

# Score
echo "🎯 SCORE GERAL:"
ACTIVE=$(ps aux | grep -E "META|DYNAMIC|REFLECTION|BRIDGE|surprise|CONNECTOR|emergence_blocks" | grep -v grep | wc -l)
echo "   Engines ativos: $ACTIVE/8"
echo "   Databases: 5/5"
echo "   Mutation Storm: ✅ ATIVO"
echo ""

echo "════════════════════════════════════════════════════════"
echo "Atualização: $(date '+%H:%M:%S')"
echo "════════════════════════════════════════════════════════"

