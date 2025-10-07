#!/bin/bash
# Script de configuração do Grafana para Qwen2.5-Coder-7B

echo "🚀 Configurando Grafana para Qwen2.5-Coder-7B..."

# Instalar Docker se não estiver instalado
if ! command -v docker &> /dev/null; then
    echo "📦 Instalando Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    systemctl start docker
    systemctl enable docker
fi

# Instalar Grafana
echo "📊 Instalando Grafana..."
docker run -d \
    --name grafana \
    -p 3000:3000 \
    -e "GF_SECURITY_ADMIN_PASSWORD=admin" \
    -e "GF_INSTALL_PLUGINS=grafana-clock-panel,grafana-simple-json-datasource" \
    grafana/grafana:latest

# Aguardar Grafana iniciar
echo "⏳ Aguardando Grafana iniciar..."
sleep 30

# Configurar Prometheus como fonte de dados
echo "🔗 Configurando Prometheus como fonte de dados..."
curl -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "name": "Prometheus",
        "type": "prometheus",
        "url": "http://host.docker.internal:9090",
        "access": "proxy",
        "isDefault": true
    }' \
    http://admin:admin@localhost:3000/api/datasources

# Importar dashboard
echo "📋 Importando dashboard..."
curl -X POST \
    -H "Content-Type: application/json" \
    -d @/root/qwen_grafana_dashboard.json \
    http://admin:admin@localhost:3000/api/dashboards/db

echo "✅ Grafana configurado!"
echo "🌐 Acesse: http://localhost:3000"
echo "👤 Usuário: admin"
echo "🔑 Senha: admin"
