#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# CLEANUP CRÍTICO - Execute AGORA para liberar 70% CPU
# ═══════════════════════════════════════════════════════════════

set -e

echo ""
echo "🔴 AUDIT

ORIA BRUTAL: Sistema em sobrecarga massiva!"
echo "📊 Load atual: $(uptime | awk -F'load average:' '{print $2}')"
echo ""

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}🧹 FASE 1: Matando processos duplicados...${NC}"

# Contar processos antes
V7_BEFORE=$(pgrep -fc "V7_RUNNER_DAEMON" || echo "0")
EMER_BEFORE=$(pgrep -fc "EMERGENCE_CATALYST" || echo "0")

echo "   V7_RUNNER_DAEMON: $V7_BEFORE instâncias (deveria ser 1)"
echo "   EMERGENCE_CATALYST: $EMER_BEFORE instâncias (deveria ser 1)"

# Matar todos exceto o mais recente de cada
if [ "$V7_BEFORE" -gt 1 ]; then
    echo -e "${RED}   ⚠️  Matando $(($V7_BEFORE - 1)) V7_RUNNER duplicados...${NC}"
    pgrep -f "V7_RUNNER_DAEMON" | head -n -1 | xargs -r kill -9
fi

if [ "$EMER_BEFORE" -gt 1 ]; then
    echo -e "${RED}   ⚠️  Matando $(($EMER_BEFORE - 1)) EMERGENCE_CATALYST duplicados...${NC}"
    pgrep -f "EMERGENCE_CATALYST" | head -n -1 | xargs -r kill -9
fi

sleep 3

# Contar depois
V7_AFTER=$(pgrep -fc "V7_RUNNER_DAEMON" || echo "0")
EMER_AFTER=$(pgrep -fc "EMERGENCE_CATALYST" || echo "0")

echo -e "${GREEN}✓ V7_RUNNER: $V7_BEFORE → $V7_AFTER${NC}"
echo -e "${GREEN}✓ EMERGENCE_CATALYST: $EMER_BEFORE → $EMER_AFTER${NC}"

echo ""
echo -e "${YELLOW}🔥 FASE 2: Verificando llama-server...${NC}"

LLAMA_CPU=$(ps aux | grep llama-server | grep -v grep | awk '{print $3}' | head -1 || echo "0")
echo "   llama-server CPU: ${LLAMA_CPU}%"

if (( $(echo "$LLAMA_CPU > 1000" | bc -l) )); then
    echo -e "${RED}   ⚠️  CRÍTICO: llama-server consumindo ${LLAMA_CPU}% CPU${NC}"
    echo "   Reiniciando com limites..."
    systemctl restart llama-qwen.service || echo "   ⚠️  Service não encontrado (normal se não configurado)"
fi

echo ""
echo -e "${YELLOW}📊 FASE 3: Status final...${NC}"

sleep 2

LOAD_AFTER=$(uptime | awk -F'load average:' '{print $2}' | awk -F',' '{print $1}' | tr -d ' ')
PROCS=$(ps aux | grep python3 | grep -c -v grep)

echo ""
echo "════════════════════════════════════════════════"
echo -e "${GREEN}✅ CLEANUP CONCLUÍDO${NC}"
echo "════════════════════════════════════════════════"
echo "Load average: $LOAD_AFTER"
echo "Processos Python ativos: $PROCS"
echo ""
echo "💡 PRÓXIMO PASSO:"
echo "   cat /root/AUDITORIA_BRUTAL_COMPLETA.md | less"
echo ""
echo "🎯 Para ver TOP 10 candidatos a inteligência real:"
echo "   grep -A5 'TIER S' /root/AUDITORIA_BRUTAL_COMPLETA.md"
echo ""
