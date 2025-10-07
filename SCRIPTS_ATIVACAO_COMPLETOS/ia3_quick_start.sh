#!/bin/bash
#==============================================================================
# IA³ QUICK START - Script de Inicialização Rápida
#==============================================================================

set -e  # Parar em caso de erro

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
cat << "EOF"
    ██╗ █████╗ ██████╗     
    ██║██╔══██╗╚════██╗    
    ██║███████║ █████╔╝    
    ██║██╔══██║ ╚═══██╗    
    ██║██║  ██║██████╔╝    
    ╚═╝╚═╝  ╚═╝╚═════╝     
    
    QUICK START SYSTEM
EOF
echo -e "${NC}"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   IA³ - Sistema de Inicialização${NC}"
echo -e "${GREEN}========================================${NC}\n"

# Função para verificar comando
check_command() {
    if command -v $1 &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 instalado"
        return 0
    else
        echo -e "${RED}✗${NC} $1 não encontrado"
        return 1
    fi
}

# Função para verificar módulo Python
check_python_module() {
    if python3 -c "import $1" 2>/dev/null; then
        echo -e "${GREEN}✓${NC} $1 instalado"
        return 0
    else
        echo -e "${YELLOW}⚠${NC} $1 não encontrado"
        return 1
    fi
}

# 1. Verificar dependências básicas
echo -e "${BLUE}[1/5] Verificando dependências básicas...${NC}"
check_command python3
check_command pip3
check_command git

# 2. Verificar módulos Python
echo -e "\n${BLUE}[2/5] Verificando módulos Python...${NC}"
MODULES_OK=true

for module in torch numpy scipy sklearn matplotlib; do
    if ! check_python_module $module; then
        MODULES_OK=false
    fi
done

if [ "$MODULES_OK" = false ]; then
    echo -e "\n${YELLOW}Alguns módulos estão faltando. Deseja instalar? (s/n)${NC}"
    read -p "> " install_choice
    
    if [ "$install_choice" = "s" ]; then
        echo -e "${GREEN}Instalando dependências...${NC}"
        pip3 install torch numpy scipy scikit-learn matplotlib numba asyncio
    fi
fi

# 3. Verificar arquivos IA³
echo -e "\n${BLUE}[3/5] Verificando arquivos IA³...${NC}"
FILES_OK=true

for file in ia3_true_core.py ia3_universal_connector.py ia3_auto_evolution.py ia3_activate.py; do
    if [ -f "/root/$file" ]; then
        echo -e "${GREEN}✓${NC} $file"
    else
        echo -e "${RED}✗${NC} $file não encontrado"
        FILES_OK=false
    fi
done

if [ "$FILES_OK" = false ]; then
    echo -e "${RED}Arquivos IA³ não encontrados!${NC}"
    echo -e "${YELLOW}Execute este script no diretório correto.${NC}"
    exit 1
fi

# 4. Criar diretórios necessários
echo -e "\n${BLUE}[4/5] Criando diretórios...${NC}"
mkdir -p /root/ia3_logs
mkdir -p /root/ia3_checkpoints
mkdir -p /root/ia3_data
mkdir -p /root/neural_farm_prod
mkdir -p /root/teis_v2_out_prod
echo -e "${GREEN}✓${NC} Diretórios criados"

# 5. Verificar processos em execução
echo -e "\n${BLUE}[5/5] Verificando processos...${NC}"

# Verificar se já está rodando
if pgrep -f "ia3_activate.py" > /dev/null; then
    echo -e "${YELLOW}⚠ Sistema IA³ já está em execução!${NC}"
    echo -e "PID: $(pgrep -f ia3_activate.py)"
    echo -e "\nDeseja parar o sistema atual? (s/n)"
    read -p "> " stop_choice
    
    if [ "$stop_choice" = "s" ]; then
        echo -e "${YELLOW}Parando sistema atual...${NC}"
        pkill -f ia3_activate.py
        sleep 2
    else
        echo -e "${YELLOW}Mantendo sistema atual em execução.${NC}"
        exit 0
    fi
fi

# Menu de opções
echo -e "\n${CYAN}========================================${NC}"
echo -e "${CYAN}         MENU DE INICIALIZAÇÃO${NC}"
echo -e "${CYAN}========================================${NC}\n"

echo -e "${GREEN}1)${NC} Iniciar sistema completo (recomendado)"
echo -e "${GREEN}2)${NC} Iniciar apenas o Core"
echo -e "${GREEN}3)${NC} Iniciar apenas Evolution Engine"
echo -e "${GREEN}4)${NC} Modo teste (5 minutos)"
echo -e "${GREEN}5)${NC} Ver status do sistema"
echo -e "${GREEN}6)${NC} Sair"

echo -e "\n${YELLOW}Escolha uma opção:${NC}"
read -p "> " choice

case $choice in
    1)
        echo -e "\n${GREEN}Iniciando sistema IA³ completo...${NC}"
        echo -e "${YELLOW}Pressione Ctrl+C para parar${NC}\n"
        sleep 2
        python3 /root/ia3_activate.py
        ;;
        
    2)
        echo -e "\n${GREEN}Iniciando apenas o Core...${NC}"
        python3 /root/ia3_true_core.py
        ;;
        
    3)
        echo -e "\n${GREEN}Iniciando Evolution Engine...${NC}"
        python3 /root/ia3_auto_evolution.py
        ;;
        
    4)
        echo -e "\n${GREEN}Modo teste - 5 minutos...${NC}"
        timeout 300 python3 /root/ia3_activate.py
        echo -e "\n${GREEN}Teste concluído!${NC}"
        ;;
        
    5)
        echo -e "\n${BLUE}Status do Sistema IA³:${NC}\n"
        
        # Verificar processos
        echo -e "${CYAN}Processos ativos:${NC}"
        ps aux | grep -E "ia3|neural_farm|teis" | grep -v grep || echo "Nenhum processo IA³ ativo"
        
        # Verificar logs
        echo -e "\n${CYAN}Últimas linhas do log:${NC}"
        if [ -f /root/ia3_activation.log ]; then
            tail -5 /root/ia3_activation.log
        else
            echo "Nenhum log encontrado"
        fi
        
        # Verificar checkpoints
        echo -e "\n${CYAN}Checkpoints salvos:${NC}"
        ls -la /root/ia3_evolution_checkpoint*.json 2>/dev/null | wc -l | xargs echo "Total:"
        
        # Uso de recursos
        echo -e "\n${CYAN}Uso de recursos:${NC}"
        echo "CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')%"
        echo "RAM: $(free -h | grep Mem | awk '{print $3 "/" $2}')"
        ;;
        
    6)
        echo -e "${GREEN}Saindo...${NC}"
        exit 0
        ;;
        
    *)
        echo -e "${RED}Opção inválida!${NC}"
        exit 1
        ;;
esac
