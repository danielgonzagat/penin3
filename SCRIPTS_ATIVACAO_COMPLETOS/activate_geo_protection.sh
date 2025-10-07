#!/bin/bash
# Script de Ativação Completa do Sistema de Proteção Geográfica
# Autor: Sistema de Segurança
# Data: 2025-01-17

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
echo -e "${CYAN}"
echo "╔══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    🛡️  SISTEMA DE PROTEÇÃO GEOGRÁFICA 🛡️                    ║"
echo "║                           BLOQUEIO MASSIVO ATIVADO                          ║"
echo "║                              APENAS BRASIL 🇧🇷                              ║"
echo "╚══════════════════════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Verificar se é root
if [ "$EUID" -ne 0 ]; then
    echo -e "${RED}❌ Este script deve ser executado como root${NC}"
    exit 1
fi

# Função de logging
log_message() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') - $1" | tee -a /var/log/geo_protection_activation.log
}

# Função para mostrar progresso
show_progress() {
    local step="$1"
    local total="$2"
    local description="$3"
    
    local percentage=$((step * 100 / total))
    local filled=$((percentage / 2))
    local empty=$((50 - filled))
    
    printf "\r${BLUE}["
    printf "%*s" $filled | tr ' ' '█'
    printf "%*s" $empty | tr ' ' '░'
    printf "] %d%% - %s${NC}" $percentage "$description"
}

echo -e "${YELLOW}🚀 Iniciando ativação do sistema de proteção geográfica...${NC}"
log_message "Iniciando ativação do sistema de proteção geográfica"

# Passo 1: Instalar dependências
show_progress 1 10 "Instalando dependências Python..."
pip3 install requests ipaddress > /dev/null 2>&1
log_message "Dependências Python instaladas"

# Passo 2: Criar diretórios necessários
show_progress 2 10 "Criando diretórios de log..."
mkdir -p /var/log/geo_monitoring
mkdir -p /var/www/html
mkdir -p /etc/nginx/conf.d
log_message "Diretórios criados"

# Passo 3: Configurar permissões
show_progress 3 10 "Configurando permissões..."
chmod +x /root/geo_protection_system.py
chmod +x /root/fastapi_geo_middleware.py
chmod +x /root/geo_monitoring_system.py
chmod +x /root/firewall_geo_protection.sh
log_message "Permissões configuradas"

# Passo 4: Testar configuração do Nginx
show_progress 4 10 "Testando configuração do Nginx..."
nginx -t > /dev/null 2>&1
if [ $? -eq 0 ]; then
    log_message "Configuração do Nginx válida"
else
    echo -e "${RED}❌ Erro na configuração do Nginx${NC}"
    log_message "ERRO: Configuração do Nginx inválida"
    exit 1
fi

# Passo 5: Recarregar Nginx
show_progress 5 10 "Recarregando Nginx..."
systemctl reload nginx > /dev/null 2>&1
log_message "Nginx recarregado"

# Passo 6: Iniciar firewall
show_progress 6 10 "Iniciando firewall de proteção..."
/root/firewall_geo_protection.sh start > /dev/null 2>&1 &
log_message "Firewall iniciado"

# Passo 7: Iniciar monitoramento
show_progress 7 10 "Iniciando sistema de monitoramento..."
python3 /root/geo_monitoring_system.py > /dev/null 2>&1 &
log_message "Sistema de monitoramento iniciado"

# Passo 8: Testar sistema de proteção
show_progress 8 10 "Testando sistema de proteção..."
python3 /root/geo_protection_system.py > /dev/null 2>&1
log_message "Sistema de proteção testado"

# Passo 9: Configurar serviços para inicialização automática
show_progress 9 10 "Configurando inicialização automática..."
cat > /etc/systemd/system/geo-protection.service << EOF
[Unit]
Description=Sistema de Proteção Geográfica
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart=/root/firewall_geo_protection.sh start
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable geo-protection.service > /dev/null 2>&1
log_message "Serviço configurado para inicialização automática"

# Passo 10: Finalização
show_progress 10 10 "Finalizando ativação..."
sleep 2
log_message "Sistema de proteção geográfica ativado com sucesso"

echo ""
echo -e "${GREEN}✅ SISTEMA DE PROTEÇÃO GEOGRÁFICA ATIVADO COM SUCESSO! ✅${NC}"
echo ""

# Mostrar estatísticas
echo -e "${BLUE}📊 === ESTATÍSTICAS DO SISTEMA ===${NC}"
echo -e "${YELLOW}🌍 País permitido:${NC} Brasil (BR)"
echo -e "${YELLOW}🚫 Países bloqueados:${NC} Todos os outros"
echo -e "${YELLOW}🛡️ Proteção ativa em:${NC}"
echo "   - Nginx (porta 8080)"
echo "   - FastAPI/Uvicorn (porta 8010)"
echo "   - LiteLLM (porta 8003)"
echo "   - Firewall (iptables)"

echo ""
echo -e "${BLUE}📁 === ARQUIVOS DE LOG ===${NC}"
echo -e "${YELLOW}📝 Logs principais:${NC}"
echo "   - /var/log/geo_protection.log"
echo "   - /var/log/nginx/geo_protection.log"
echo "   - /var/log/fastapi_geo_protection.log"
echo "   - /var/log/firewall_geo_protection.log"
echo "   - /var/log/geo_monitoring.log"

echo ""
echo -e "${BLUE}🔧 === COMANDOS DE GERENCIAMENTO ===${NC}"
echo -e "${YELLOW}📊 Ver estatísticas:${NC}"
echo "   python3 /root/geo_protection_system.py"
echo "   /root/firewall_geo_protection.sh stats"

echo -e "${YELLOW}🛑 Parar proteção:${NC}"
echo "   /root/firewall_geo_protection.sh stop"
echo "   systemctl stop geo-protection"

echo -e "${YELLOW}🔄 Reiniciar proteção:${NC}"
echo "   systemctl restart geo-protection"

echo ""
echo -e "${RED}⚠️  ATENÇÃO: Sistema configurado para bloquear TODOS os países exceto Brasil!${NC}"
echo -e "${RED}⚠️  Apenas usuários com IPs brasileiros terão acesso aos serviços.${NC}"
echo ""

# Teste final
echo -e "${CYAN}🧪 === TESTE FINAL ===${NC}"
echo "Testando bloqueio de IPs não brasileiros..."

# Testar com IPs conhecidos
test_ips=("8.8.8.8" "1.1.1.1" "208.67.222.222")
for ip in "${test_ips[@]}"; do
    country=$(curl -s --max-time 5 "http://ip-api.com/json/$ip" | grep -o '"country":"[^"]*"' | cut -d'"' -f4)
    if [ "$country" != "Brazil" ]; then
        echo -e "${GREEN}✅ IP $ip ($country) será bloqueado${NC}"
    else
        echo -e "${YELLOW}⚠️  IP $ip ($country) será permitido${NC}"
    fi
done

echo ""
echo -e "${GREEN}🎉 SISTEMA DE PROTEÇÃO GEOGRÁFICA TOTALMENTE OPERACIONAL! 🎉${NC}"
echo -e "${PURPLE}🇧🇷 Apenas usuários do Brasil terão acesso aos seus serviços! 🇧🇷${NC}"
echo ""

log_message "Ativação completa finalizada com sucesso"