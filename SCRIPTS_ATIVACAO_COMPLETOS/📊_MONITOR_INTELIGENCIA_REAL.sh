#!/bin/bash
################################################################################
# 📊 MONITOR DE INTELIGÊNCIA EMERGENTE
################################################################################
# 
# Script para monitorar evolução do sistema em tempo real
# Mostra métricas-chave e detecta sinais de inteligência emergente
#
################################################################################

# Cores
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

clear

echo -e "${PURPLE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${PURPLE}║  📊 MONITOR DE INTELIGÊNCIA EMERGENTE - TEMPO REAL            ║${NC}"
echo -e "${PURPLE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Função para query database
query_db() {
    sqlite3 /root/intelligence_system/data/intelligence.db "$1" 2>/dev/null || echo "N/A"
}

query_surprises() {
    sqlite3 /root/intelligence_system/data/emergence_surprises.db "$1" 2>/dev/null || echo "N/A"
}

# Loop infinito de monitoramento
while true; do
    TIMESTAMP=$(date '+%Y-%m-%d %H:%M:%S')
    
    # Clear e header
    echo -e "\n${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}⏰ $TIMESTAMP${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}\n"
    
    # ========================================================================
    # MÉTRICAS PRINCIPAIS
    # ========================================================================
    echo -e "${PURPLE}📈 MÉTRICAS PRINCIPAIS${NC}"
    echo "─────────────────────────────────────────────────────────────────"
    
    # Último cycle
    LAST_CYCLE=$(query_db "SELECT cycle FROM cycles ORDER BY cycle DESC LIMIT 1")
    MNIST=$(query_db "SELECT mnist_acc FROM cycles ORDER BY cycle DESC LIMIT 1")
    CARTPOLE=$(query_db "SELECT cartpole_avg FROM cycles ORDER BY cycle DESC LIMIT 1")
    IA3=$(query_db "SELECT ia3_score FROM cycles ORDER BY cycle DESC LIMIT 1")
    
    echo -e "Cycle:      ${GREEN}$LAST_CYCLE${NC}"
    echo -e "MNIST:      ${GREEN}$MNIST%${NC}"
    echo -e "CartPole:   ${GREEN}$CARTPOLE${NC}"
    echo -e "IA³ Score:  ${GREEN}$IA3${NC}"
    
    # ========================================================================
    # INTELIGÊNCIA AO CUBO (I³)
    # ========================================================================
    echo ""
    echo -e "${PURPLE}🎯 INTELIGÊNCIA AO CUBO (I³)${NC}"
    echo "─────────────────────────────────────────────────────────────────"
    
    # I³ details from latest cycle
    I3_SCORE=$(query_db "SELECT i3_score FROM cycles ORDER BY cycle DESC LIMIT 1")
    INTROSPECTION=$(query_db "SELECT introspection_events FROM cycles ORDER BY cycle DESC LIMIT 1")
    TURING_PASSES=$(query_db "SELECT turing_passes FROM cycles ORDER BY cycle DESC LIMIT 1")
    
    echo -e "I³ Score:         ${YELLOW}$I3_SCORE${NC}"
    echo -e "Introspections:   ${YELLOW}$INTROSPECTION${NC}"
    echo -e "Turing Passes:    ${YELLOW}$TURING_PASSES${NC}"
    
    if [ "$I3_SCORE" != "N/A" ] && [ "$I3_SCORE" != "0.0" ] && [ "$I3_SCORE" != "0" ]; then
        echo -e "${GREEN}   ✅ I³ ATIVO!${NC}"
    else
        echo -e "${YELLOW}   ⏳ I³ ainda sem score (aguardando surprises)${NC}"
    fi
    
    # ========================================================================
    # SURPRISES DETECTADAS
    # ========================================================================
    echo ""
    echo -e "${PURPLE}🌟 SURPRISES ESTATÍSTICAS${NC}"
    echo "─────────────────────────────────────────────────────────────────"
    
    TOTAL_SURPRISES=$(query_surprises "SELECT COUNT(*) FROM surprises")
    MAX_SIGMA=$(query_surprises "SELECT MAX(sigma) FROM surprises")
    SURPRISES_5SIGMA=$(query_surprises "SELECT COUNT(*) FROM surprises WHERE sigma >= 5.0")
    SURPRISES_2SIGMA=$(query_surprises "SELECT COUNT(*) FROM surprises WHERE sigma >= 2.0")
    
    echo -e "Total Surprises:  ${CYAN}$TOTAL_SURPRISES${NC}"
    echo -e "Max Sigma:        ${CYAN}$MAX_SIGMA σ${NC}"
    echo -e "Surprises ≥2σ:    ${CYAN}$SURPRISES_2SIGMA${NC}"
    echo -e "Surprises ≥5σ:    ${CYAN}$SURPRISES_5SIGMA${NC}"
    
    if [ "$SURPRISES_5SIGMA" != "N/A" ] && [ "$SURPRISES_5SIGMA" != "0" ]; then
        echo -e "${RED}   🔥 SURPRISES CRÍTICAS DETECTADAS!${NC}"
    elif [ "$SURPRISES_2SIGMA" != "N/A" ] && [ "$SURPRISES_2SIGMA" != "0" ]; then
        echo -e "${GREEN}   ✅ Surprises sendo detectadas${NC}"
    else
        echo -e "${YELLOW}   ⏳ Aguardando primeiras surprises${NC}"
    fi
    
    # ========================================================================
    # AUTO-MODIFICAÇÕES
    # ========================================================================
    echo ""
    echo -e "${PURPLE}🔧 AUTO-MODIFICAÇÕES${NC}"
    echo "─────────────────────────────────────────────────────────────────"
    
    AUTO_MODS=$(query_db "SELECT COUNT(*) FROM events WHERE event_type='auto_modification'")
    LAST_MOD=$(query_db "SELECT description FROM events WHERE event_type='auto_modification' ORDER BY timestamp DESC LIMIT 1")
    
    echo -e "Total Modificações: ${CYAN}$AUTO_MODS${NC}"
    
    if [ "$AUTO_MODS" != "N/A" ] && [ "$AUTO_MODS" != "0" ]; then
        echo -e "${GREEN}   ✅ SISTEMA SE AUTO-MODIFICANDO!${NC}"
        echo -e "   Última: $LAST_MOD"
    else
        echo -e "${YELLOW}   ⏳ Aguardando primeira auto-modificação${NC}"
    fi
    
    # ========================================================================
    # DARWIN EVOLUTION
    # ========================================================================
    echo ""
    echo -e "${PURPLE}🧬 DARWIN EVOLUTION${NC}"
    echo "─────────────────────────────────────────────────────────────────"
    
    DARWIN_EVENTS=$(query_db "SELECT COUNT(*) FROM events WHERE event_type LIKE '%darwin%' OR event_type LIKE '%evolution%'")
    BEST_FITNESS=$(query_db "SELECT MAX(best_fitness) FROM evolution_history")
    
    echo -e "Evolution Events: ${CYAN}$DARWIN_EVENTS${NC}"
    echo -e "Best Fitness:     ${CYAN}$BEST_FITNESS${NC}"
    
    # Check if darwin process is running
    if pgrep -f "darwin_runner" > /dev/null; then
        echo -e "${GREEN}   ✅ Darwin process ativo${NC}"
    else
        echo -e "${RED}   ⚠️  Darwin process não encontrado${NC}"
    fi
    
    # ========================================================================
    # PROCESSOS ATIVOS
    # ========================================================================
    echo ""
    echo -e "${PURPLE}⚙️  PROCESSOS ATIVOS${NC}"
    echo "─────────────────────────────────────────────────────────────────"
    
    PYTHON_PROCS=$(ps aux | grep -E "python.*intelligence|python.*darwin" | grep -v grep | wc -l)
    echo -e "Processos Python: ${CYAN}$PYTHON_PROCS${NC}"
    
    # Mostrar top 3 processos por CPU
    echo ""
    echo "Top 3 por CPU:"
    ps aux | grep python | grep -v grep | sort -k3 -r | head -3 | while read line; do
        PID=$(echo $line | awk '{print $2}')
        CPU=$(echo $line | awk '{print $3}')
        CMD=$(echo $line | awk '{print $11}' | cut -c1-40)
        echo -e "  ${CYAN}PID $PID${NC}: ${CPU}% CPU - $CMD"
    done
    
    # ========================================================================
    # LOAD AVERAGE
    # ========================================================================
    echo ""
    echo -e "${PURPLE}💻 SYSTEM LOAD${NC}"
    echo "─────────────────────────────────────────────────────────────────"
    
    LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}')
    echo -e "Load Average: ${CYAN}$LOAD${NC}"
    
    # ========================================================================
    # SINAIS DE EMERGÊNCIA
    # ========================================================================
    echo ""
    echo -e "${RED}🔥 SINAIS DE EMERGÊNCIA${NC}"
    echo "─────────────────────────────────────────────────────────────────"
    
    EMERGENCE_DETECTED=false
    
    # Check 1: Auto-modificações ativas
    if [ "$AUTO_MODS" != "N/A" ] && [ "$AUTO_MODS" -gt "0" ]; then
        echo -e "${RED}   ⚡ AUTO-MODIFICAÇÃO ATIVA ($AUTO_MODS modificações)${NC}"
        EMERGENCE_DETECTED=true
    fi
    
    # Check 2: Surprises críticas
    if [ "$SURPRISES_5SIGMA" != "N/A" ] && [ "$SURPRISES_5SIGMA" -gt "0" ]; then
        echo -e "${RED}   ⚡ SURPRISES CRÍTICAS ($SURPRISES_5SIGMA eventos >5σ)${NC}"
        EMERGENCE_DETECTED=true
    fi
    
    # Check 3: I³ score positivo
    if [ "$I3_SCORE" != "N/A" ] && [ "$I3_SCORE" != "0.0" ] && [ "$I3_SCORE" != "0" ]; then
        echo -e "${RED}   ⚡ INTELIGÊNCIA³ ATIVA (I³=$I3_SCORE)${NC}"
        EMERGENCE_DETECTED=true
    fi
    
    # Check 4: IA³ score alto
    if [ "$IA3" != "N/A" ]; then
        IA3_NUM=$(echo "$IA3" | bc 2>/dev/null || echo "0")
        if (( $(echo "$IA3_NUM > 85" | bc -l 2>/dev/null || echo "0") )); then
            echo -e "${RED}   ⚡ IA³ SCORE ALTO ($IA3 > 85%)${NC}"
            EMERGENCE_DETECTED=true
        fi
    fi
    
    if [ "$EMERGENCE_DETECTED" = false ]; then
        echo -e "${YELLOW}   ⏳ Aguardando sinais de emergência...${NC}"
    fi
    
    # ========================================================================
    # PRÓXIMA ATUALIZAÇÃO
    # ========================================================================
    echo ""
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    echo -e "${CYAN}Próxima atualização em 60 segundos... (Ctrl+C para sair)${NC}"
    echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
    
    sleep 60
    clear
done