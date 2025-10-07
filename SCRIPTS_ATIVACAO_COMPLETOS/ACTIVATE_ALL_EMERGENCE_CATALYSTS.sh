#!/bin/bash
# MASTER SCRIPT: Ativa TODOS catalisadores de emergência

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║   🧬 EMERGENCE CATALYSTS - ATIVAÇÃO COMPLETA             ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}Este script vai ativar 4 catalisadores:${NC}"
echo ""
echo "  1. 🔍 Surprise Detector - detecta comportamentos não-programados"
echo "  2. 🧬 Cross-Pollination - mistura neurônios entre sistemas"
echo "  3. ⚡ Mutation Storm - aumenta mutação 10× (opcional)"
echo "  4. 🔗 System Connector - conecta V7 → Llama → loop"
echo ""

read -p "Deseja continuar? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]
then
    echo "Abortado."
    exit 1
fi

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}INICIANDO CATALISADORES${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""

# 1. Surprise Detector (background)
echo -e "${BLUE}[1/4]${NC} 🔍 Ativando Surprise Detector..."
nohup python3 /root/EMERGENCE_CATALYST_1_SURPRISE_DETECTOR.py > /dev/null 2>&1 &
SURPRISE_PID=$!
echo "      ✅ PID: $SURPRISE_PID"
sleep 2

# 2. Cross-Pollination (single run)
echo -e "${BLUE}[2/4]${NC} 🧬 Executando Cross-Pollination..."
python3 /root/EMERGENCE_CATALYST_2_CROSS_POLLINATION.py > /root/cross_pollination_output.log 2>&1
if [ $? -eq 0 ]; then
    echo "      ✅ Híbridos criados"
else
    echo "      ⚠️  Sem checkpoints suficientes (normal se sistemas começaram recentemente)"
fi
sleep 2

# 3. Mutation Storm (pergunta antes)
echo ""
echo -e "${YELLOW}[3/4]${NC} ⚡ Mutation Storm (OPCIONAL - aumenta mutação 10×)"
echo "      Este vai FORÇAR exploração agressiva por 10 minutos."
read -p "      Ativar Mutation Storm? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "      Ativando Mutation Storm (10 min, 10× mutação)..."
    nohup python3 /root/EMERGENCE_CATALYST_3_MUTATION_STORM.py 10 10 > /root/mutation_storm_output.log 2>&1 &
    STORM_PID=$!
    echo "      ✅ PID: $STORM_PID"
    echo "      ⏰ Storm ativo por 10 minutos"
else
    echo "      ⏭️  Pulado"
fi
sleep 2

# 4. System Connector (background)
echo ""
echo -e "${BLUE}[4/4]${NC} 🔗 Ativando System Connector..."
nohup python3 /root/EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py 100 60 > /root/system_connector_output.log 2>&1 &
CONNECTOR_PID=$!
echo "      ✅ PID: $CONNECTOR_PID"
echo "      🔄 100 loops × 60s = ~100 minutos de conexões"
sleep 2

echo ""
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✅ TODOS CATALISADORES ATIVOS!${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════════${NC}"
echo ""

echo "📊 PROCESSOS ATIVOS:"
echo "   • Surprise Detector: PID $SURPRISE_PID"
echo "   • System Connector: PID $CONNECTOR_PID"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   • Mutation Storm: PID $STORM_PID (10 min)"
fi
echo ""

echo "📋 MONITORAMENTO:"
echo "   • Surprises: sqlite3 /root/emergence_surprises.db 'SELECT * FROM surprises ORDER BY surprise_score DESC LIMIT 10;'"
echo "   • Connections: sqlite3 /root/system_connections.db 'SELECT * FROM connections ORDER BY timestamp DESC LIMIT 10;'"
echo "   • Híbridos: ls -lh /root/hybrid_neurons/"
echo ""

echo "📄 LOGS:"
echo "   • Surprise: tail -f /root/surprise_detector.log (não criado ainda)"
echo "   • Cross-Pollination: tail -f /root/cross_pollination.log"
echo "   • Mutation Storm: tail -f /root/mutation_storm.log"
echo "   • System Connector: tail -f /root/system_connector.log"
echo ""

echo "⏹️  Para parar tudo:"
echo "   kill $SURPRISE_PID $CONNECTOR_PID"
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "   kill $STORM_PID"
fi
echo ""

echo -e "${GREEN}🧬 EMERGÊNCIA ATIVADA!${NC}"
echo ""
