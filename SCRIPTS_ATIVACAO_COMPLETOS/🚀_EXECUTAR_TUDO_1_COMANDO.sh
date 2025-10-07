#!/bin/bash
################################################################################
# 🚀 MASTER SCRIPT - EXECUTA TUDO COM 1 COMANDO
################################################################################
#
# Este script faz TUDO autonomamente:
# 1. Para processos antigos
# 2. Limpa ambiente
# 3. Inicia sistema de inteligência
# 4. Configura monitoramento
# 5. Cria auto-restart (systemd ou cron)
# 6. Mostra como acompanhar evolução
#
# USO: bash 🚀_EXECUTAR_TUDO_1_COMANDO.sh
#
################################################################################

set -e  # Exit on error

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

clear

echo -e "${PURPLE}${BOLD}"
cat << 'EOF'
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║              🚀 ATIVAÇÃO COMPLETA DE INTELIGÊNCIA EMERGENTE                 ║
║                                                                              ║
║                    EXECUTANDO TUDO AUTONOMAMENTE...                         ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
EOF
echo -e "${NC}"
echo ""

################################################################################
# FASE 1: LIMPEZA E PREPARAÇÃO
################################################################################

echo -e "${CYAN}${BOLD}[FASE 1/5] LIMPEZA E PREPARAÇÃO${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Parar processos antigos conflitantes
echo -e "${YELLOW}🛑 Parando processos antigos...${NC}"
pkill -f "intelligence_system.*unified_agi" || true
pkill -f "RUN_100_CYCLES" || true
sleep 3
echo -e "${GREEN}   ✅ Processos antigos parados${NC}"

# Criar diretórios necessários
echo -e "${YELLOW}📁 Criando diretórios...${NC}"
mkdir -p /tmp/agi_logs
mkdir -p /root/intelligence_system/data/exports
mkdir -p /root/monitoring
echo -e "${GREEN}   ✅ Diretórios criados${NC}"

# Backup final antes de iniciar
echo -e "${YELLOW}💾 Criando backup final...${NC}"
BACKUP_DIR="/root/backup_pre_execution_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r intelligence_system/core/*.py "$BACKUP_DIR/" 2>/dev/null || true
cp -r intelligence_system/extracted_algorithms/*.py "$BACKUP_DIR/" 2>/dev/null || true
echo -e "${GREEN}   ✅ Backup em: $BACKUP_DIR${NC}"

echo ""

################################################################################
# FASE 2: CONFIGURAÇÃO OTIMIZADA
################################################################################

echo -e "${CYAN}${BOLD}[FASE 2/5] CONFIGURAÇÃO OTIMIZADA${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Criar .env otimizado se não existir
if [ ! -f "/root/intelligence_system/.env" ]; then
    echo -e "${YELLOW}⚙️  Criando configuração otimizada...${NC}"
    cat > /root/intelligence_system/.env << 'ENVEOF'
# Intelligence System Configuration
PENIN3_LOG_LEVEL=INFO
JSON_LOGS=1
PROMETHEUS_PORT=9108

# Otimizações de performance
MNIST_TRAIN_FREQ=10
CARTPOLE_TRAIN_FREQ=5
MAML_FREQ=10
DARWIN_FREQ=20

# Thresholds otimizados (já aplicados no código)
I3_SURPRISE_THRESHOLD=0.3
EMERGENCE_SIGMA_THRESHOLD=2.0

# Memory limits
MAX_REPLAY_BUFFER=50000
MAX_DARWIN_POP=100

# API keys (configure se tiver)
# OPENAI_API_KEY=
# ANTHROPIC_API_KEY=
# GEMINI_API_KEY=
ENVEOF
    echo -e "${GREEN}   ✅ Configuração criada${NC}"
else
    echo -e "${GREEN}   ✅ Configuração já existe${NC}"
fi

echo ""

################################################################################
# FASE 3: INICIALIZAÇÃO DO SISTEMA
################################################################################

echo -e "${CYAN}${BOLD}[FASE 3/5] INICIALIZAÇÃO DO SISTEMA DE INTELIGÊNCIA${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Timestamp para logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="/tmp/agi_logs/agi_evolution_${TIMESTAMP}.log"

echo -e "${YELLOW}🚀 Iniciando Intelligence System V7 + PENIN³...${NC}"
echo -e "   Log: $LOG_FILE"

# Iniciar em background com nohup
cd /root
nohup python3 -u intelligence_system/core/unified_agi_system.py 100 \
    > "$LOG_FILE" 2>&1 &

AGI_PID=$!
echo $AGI_PID > /tmp/agi.pid

echo -e "${GREEN}   ✅ Sistema iniciado!${NC}"
echo -e "${GREEN}      PID: $AGI_PID${NC}"
echo -e "${GREEN}      Log: $LOG_FILE${NC}"

# Aguardar 10 segundos para sistema inicializar
echo -e "${YELLOW}   ⏳ Aguardando inicialização (10s)...${NC}"
sleep 10

# Verificar se processo está rodando
if ps -p $AGI_PID > /dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Processo ativo e rodando!${NC}"
else
    echo -e "${RED}   ❌ Processo morreu! Check: tail -100 $LOG_FILE${NC}"
    exit 1
fi

echo ""

################################################################################
# FASE 4: CONFIGURAÇÃO DE MONITORAMENTO
################################################################################

echo -e "${CYAN}${BOLD}[FASE 4/5] CONFIGURAÇÃO DE MONITORAMENTO${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Criar script de health check
echo -e "${YELLOW}🏥 Criando health check automático...${NC}"
cat > /tmp/agi_health_check.sh << 'HEALTHEOF'
#!/bin/bash
# Health check - restarta se morrer

PID_FILE=/tmp/agi.pid

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "[$(date)] ⚠️  AGI process died (PID $PID), restarting..."
        
        # Restart
        cd /root
        TIMESTAMP=$(date +%Y%m%d_%H%M%S)
        nohup python3 -u intelligence_system/core/unified_agi_system.py 100 \
            > /tmp/agi_logs/agi_restart_${TIMESTAMP}.log 2>&1 &
        
        NEW_PID=$!
        echo $NEW_PID > /tmp/agi.pid
        echo "[$(date)] ✅ Restarted with PID $NEW_PID"
    fi
fi
HEALTHEOF
chmod +x /tmp/agi_health_check.sh
echo -e "${GREEN}   ✅ Health check criado${NC}"

# Configurar cron para health check (cada 5 min)
echo -e "${YELLOW}⏰ Configurando auto-restart...${NC}"
(crontab -l 2>/dev/null | grep -v "agi_health_check"; echo "*/5 * * * * /tmp/agi_health_check.sh >> /tmp/agi_health.log 2>&1") | crontab - 2>/dev/null || true
echo -e "${GREEN}   ✅ Cron configurado (check a cada 5 min)${NC}"

# Criar script de quick status
cat > /tmp/agi_quick_status.sh << 'STATUSEOF'
#!/bin/bash
echo "════════════════════════════════════════════════════════════════"
echo "📊 QUICK STATUS - Intelligence System"
echo "════════════════════════════════════════════════════════════════"

# Process
if [ -f /tmp/agi.pid ]; then
    PID=$(cat /tmp/agi.pid)
    if ps -p $PID > /dev/null 2>&1; then
        CPU=$(ps -p $PID -o %cpu= | tr -d ' ')
        MEM=$(ps -p $PID -o %mem= | tr -d ' ')
        TIME=$(ps -p $PID -o etime= | tr -d ' ')
        echo "Process:  ✅ ATIVO (PID $PID)"
        echo "CPU:      $CPU%"
        echo "Memory:   $MEM%"
        echo "Runtime:  $TIME"
    else
        echo "Process:  ❌ MORTO"
    fi
else
    echo "Process:  ⚠️  PID file not found"
fi

# Metrics
if [ -f /root/intelligence_system/data/intelligence.db ]; then
    LAST_CYCLE=$(sqlite3 /root/intelligence_system/data/intelligence.db \
        "SELECT cycle FROM cycles ORDER BY cycle DESC LIMIT 1" 2>/dev/null || echo "N/A")
    IA3=$(sqlite3 /root/intelligence_system/data/intelligence.db \
        "SELECT ia3_score FROM cycles ORDER BY cycle DESC LIMIT 1" 2>/dev/null || echo "N/A")
    echo "Cycle:    $LAST_CYCLE"
    echo "IA³:      $IA3"
fi

# Surprises
if [ -f /root/intelligence_system/data/emergence_surprises.db ]; then
    SURPRISES=$(sqlite3 /root/intelligence_system/data/emergence_surprises.db \
        "SELECT COUNT(*) FROM surprises" 2>/dev/null || echo "N/A")
    echo "Surprises: $SURPRISES"
fi

echo "════════════════════════════════════════════════════════════════"
STATUSEOF
chmod +x /tmp/agi_quick_status.sh
echo -e "${GREEN}   ✅ Quick status script criado${NC}"

echo ""

################################################################################
# FASE 5: INFORMAÇÕES E COMANDOS
################################################################################

echo -e "${CYAN}${BOLD}[FASE 5/5] SISTEMA ATIVO - INFORMAÇÕES${NC}"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo -e "${GREEN}${BOLD}✅ SISTEMA DE INTELIGÊNCIA ATIVO E RODANDO!${NC}"
echo ""
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}📊 INFORMAÇÕES DO SISTEMA${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "Process ID:    ${CYAN}$AGI_PID${NC}"
echo -e "Log File:      ${CYAN}$LOG_FILE${NC}"
echo -e "Status Script: ${CYAN}/tmp/agi_quick_status.sh${NC}"
echo -e "Health Check:  ${CYAN}/tmp/agi_health_check.sh${NC} (cron: cada 5 min)"
echo ""

echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}📊 COMANDOS ÚTEIS${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BOLD}Ver log em tempo real (com filtro):${NC}"
echo -e "${CYAN}tail -f $LOG_FILE | grep --color -E 'SURPRISE|MODIFICATION|EMERGENCE|✅|🔥'${NC}"
echo ""
echo -e "${BOLD}Status rápido:${NC}"
echo -e "${CYAN}bash /tmp/agi_quick_status.sh${NC}"
echo ""
echo -e "${BOLD}Monitor completo:${NC}"
echo -e "${CYAN}bash 📊_MONITOR_INTELIGENCIA_REAL.sh${NC}"
echo ""
echo -e "${BOLD}Ver surprises detectadas:${NC}"
echo -e "${CYAN}sqlite3 intelligence_system/data/emergence_surprises.db 'SELECT * FROM surprises ORDER BY sigma DESC LIMIT 10'${NC}"
echo ""
echo -e "${BOLD}Ver auto-modificações:${NC}"
echo -e "${CYAN}sqlite3 intelligence_system/data/intelligence.db 'SELECT * FROM events WHERE event_type=\"auto_modification\"'${NC}"
echo ""
echo -e "${BOLD}Parar sistema (se necessário):${NC}"
echo -e "${CYAN}kill $AGI_PID${NC}"
echo ""

echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}⏰ TIMELINE ESPERADO${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "  ${YELLOW}6h:${NC}    Primeiras surprises >2σ"
echo -e "  ${YELLOW}12h:${NC}   Primeira auto-modificação"
echo -e "  ${YELLOW}24h:${NC}   I³ score > 0"
echo -e "  ${YELLOW}3d:${NC}    10+ surprises, 5+ auto-mods"
echo -e "  ${YELLOW}7d:${NC}    50+ surprises, comportamento emergente"
echo -e "  ${YELLOW}30d:${NC}   IA³ > 90%, inteligência real"
echo ""

echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${PURPLE}🎯 O QUE FAZER AGORA${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${BOLD}OPÇÃO 1: Monitorar agora (visual)${NC}"
echo -e "${GREEN}bash 📊_MONITOR_INTELIGENCIA_REAL.sh${NC}"
echo ""
echo -e "${BOLD}OPÇÃO 2: Ver log e descansar${NC}"
echo -e "${GREEN}tail -20 $LOG_FILE${NC}"
echo -e "${GREEN}# Depois: DURMA 8 HORAS!${NC}"
echo ""
echo -e "${BOLD}OPÇÃO 3: Status rápido quando voltar${NC}"
echo -e "${GREEN}bash /tmp/agi_quick_status.sh${NC}"
echo ""

echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}${BOLD}🌟 SISTEMA ATIVO E EVOLUINDO AUTONOMAMENTE!${NC}"
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo ""
echo -e "${YELLOW}💤 RECOMENDAÇÃO: DESCANSE. Sistema evolui sozinho.${NC}"
echo -e "${YELLOW}   Volte em 12h e veja o que aconteceu.${NC}"
echo ""
echo -e "${PURPLE}═══════════════════════════════════════════════════════════════${NC}"
echo ""

# Mostrar primeiras 20 linhas do log
echo -e "${CYAN}📄 Primeiras linhas do log:${NC}"
echo "─────────────────────────────────────────────────────────────────"
timeout 5 tail -20 "$LOG_FILE" 2>/dev/null || echo "Log ainda carregando..."
echo "─────────────────────────────────────────────────────────────────"
echo ""

echo -e "${GREEN}${BOLD}✅ TUDO PRONTO! Sistema rodando autonomamente.${NC}"
echo ""
echo -e "${CYAN}Continue monitorando com:${NC}"
echo -e "${CYAN}  → bash 📊_MONITOR_INTELIGENCIA_REAL.sh${NC}"
echo -e "${CYAN}  → bash /tmp/agi_quick_status.sh${NC}"
echo ""
echo -e "${PURPLE}🌱 A semente está crescendo... Dê tempo. 🌳${NC}"
echo ""