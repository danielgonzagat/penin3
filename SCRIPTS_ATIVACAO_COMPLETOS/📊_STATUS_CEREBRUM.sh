#!/bin/bash
# 📊 STATUS DO DARWINACCI CEREBRUM

PID_FILE="/root/darwinacci_omega/cerebrum.pid"
LOG_FILE="/root/darwinacci_omega/logs/cerebrum_24_7.log"

echo "═══════════════════════════════════════════════════════════════════════════"
echo "📊 DARWINACCI CEREBRUM - STATUS"
echo "═══════════════════════════════════════════════════════════════════════════"
echo ""

# Verificar se está rodando
if [ ! -f "$PID_FILE" ]; then
    echo "❌ Cerebrum NÃO ESTÁ RODANDO"
    echo "   Para iniciar: bash /root/⚡_ATIVAR_CEREBRUM_24_7.sh"
    exit 1
fi

PID=$(cat "$PID_FILE")
if ! ps -p "$PID" > /dev/null 2>&1; then
    echo "❌ Cerebrum MORTO (PID $PID não existe)"
    echo "   Para reiniciar: bash /root/⚡_ATIVAR_CEREBRUM_24_7.sh"
    exit 1
fi

echo "✅ Cerebrum ATIVO"
echo "   PID: $PID"
echo "   Uptime: $(ps -p $PID -o etime= | xargs)"
echo "   Memory: $(ps -p $PID -o rss= | awk '{printf "%.1f MB", $1/1024}')"
echo "   CPU: $(ps -p $PID -o %cpu= | xargs)%"
echo ""

# Status dos logs
if [ -f "$LOG_FILE" ]; then
    echo "📋 ÚLTIMOS LOGS (últimas 20 linhas):"
    echo "─────────────────────────────────────────────────────────────────────────"
    tail -20 "$LOG_FILE"
    echo "─────────────────────────────────────────────────────────────────────────"
    echo ""
    
    # Extrair métricas
    echo "📊 MÉTRICAS RECENTES:"
    # Contagem simples por padrão (não depende de PCRE)
    echo "   Cycles completos: $(grep -c 'CYCLE.*COMPLETE' "$LOG_FILE" 2>/dev/null || echo 0)"
    # Extrações compatíveis POSIX usando awk
    TRANSFERS=$(awk -F'Total Transfers: ' '/Total Transfers:/ {print $2}' "$LOG_FILE" | tail -1)
    EVALS=$(awk -F'Total Evaluations: ' '/Total Evaluations:/ {print $2}' "$LOG_FILE" | tail -1)
    SUCCESS=$(awk -F'Success Rate: ' '/Success Rate:/ {print $2}' "$LOG_FILE" | tail -1)
    echo "   Transfers totais: ${TRANSFERS:-0}"
    echo "   Evaluations totais: ${EVALS:-0}"
    echo "   Success rate: ${SUCCESS:-N/A}"
fi

# Status dos checkpoints
CKPT_DIR="/root/darwinacci_omega/checkpoints"
if [ -d "$CKPT_DIR" ]; then
    CKPT_COUNT=$(ls -1 "$CKPT_DIR"/*.pkl 2>/dev/null | wc -l)
    echo ""
    echo "💾 CHECKPOINTS:"
    echo "   Total: $CKPT_COUNT"
    if [ $CKPT_COUNT -gt 0 ]; then
        echo "   Mais recente: $(ls -t "$CKPT_DIR"/*.pkl 2>/dev/null | head -1 | xargs basename)"
    fi
fi

# Status do WORM
WORM_FILE="/root/darwinacci_omega/cerebrum_worm.csv"
if [ -f "$WORM_FILE" ]; then
    WORM_SIZE=$(du -h "$WORM_FILE" | cut -f1)
    WORM_LINES=$(wc -l < "$WORM_FILE")
    echo ""
    echo "📜 WORM LEDGER:"
    echo "   Size: $WORM_SIZE"
    echo "   Entries: $WORM_LINES"
fi

echo ""
echo "═══════════════════════════════════════════════════════════════════════════"