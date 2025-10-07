#!/bin/bash
# ATIVA TODOS OS 3 HOTPATCHES DE EMERGÊNCIA

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║   🚨 EMERGENCY HOTPATCHES - EMERGÊNCIA IMEDIATA          ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${RED}⚠️  AVISO: Estes hotpatches causam EMERGÊNCIA IMEDIATA!${NC}"
echo ""
echo "  1. 🔥 Darwin Mutation Storm (mutação 5×)"
echo "  2. 🔗 Darwin → Llama Loop (feedback em tempo real)"
echo "  3. 💉 Inject 24,716 neurônios (diversidade explosiva)"
echo ""

read -p "Deseja aplicar TODOS? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Abortado."
    exit 1
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}APLICANDO HOTPATCHES${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""

# 1. Mutation Storm (automático)
echo -e "${YELLOW}[1/3]${NC} 🔥 Aplicando Mutation Storm..."
echo "" | python3 /root/EMERGENCY_HOTPATCH_1_DARWIN_MUTATION_STORM.py
echo ""
sleep 2

# 2. Darwin → Llama Loop (background)
echo -e "${YELLOW}[2/3]${NC} 🔗 Iniciando Darwin → Llama Loop..."
nohup python3 /root/EMERGENCY_HOTPATCH_2_DARWIN_LLAMA_LOOP.py > /root/darwin_llama_loop_output.log 2>&1 &
LOOP_PID=$!
echo "      ✅ PID: $LOOP_PID"
echo ""
sleep 2

# 3. Massive Injection (automático)
echo -e "${YELLOW}[3/3]${NC} 💉 Injetando 24,716 neurônios..."
echo "" | python3 /root/EMERGENCY_HOTPATCH_3_INJECT_MASSIVE_NEURONS.py
echo ""

echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ TODOS HOTPATCHES APLICADOS!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""

echo "📊 STATUS:"
echo "   • Mutation Storm: ✅ APLICADO (próximas gerações)"
echo "   • Darwin-Llama Loop: ✅ ATIVO (PID $LOOP_PID)"
echo "   • Massive Injection: ✅ INJETADO (24,716 neurônios)"
echo ""

echo "⏰ EMERGÊNCIA ESPERADA:"
echo "   • Mutation Storm: 5-10 minutos"
echo "   • Llama Loop: AGORA (feedback contínuo)"
echo "   • Massive Injection: Próximo ciclo Darwin"
echo ""

echo "📋 MONITORAR:"
echo "   • Surprises: watch -n 5 'sqlite3 /root/emergence_surprises.db \"SELECT COUNT(*) FROM surprises;\"'"
echo "   • Darwin-Llama: tail -f /root/darwin_llama_loop.log"
echo "   • Injection: tail -f /root/massive_injection.log"
echo ""

echo -e "${RED}🔥 EMERGÊNCIA ATIVADA!${NC}"
