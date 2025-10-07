#!/bin/bash
"""
Script de inicialização do Sistema Autônomo Qwen2.5-Coder
"""

echo "🚀 Iniciando Sistema Autônomo Qwen2.5-Coder..."
echo "=================================================="

# Verifica se está rodando como root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Este script deve ser executado como root"
    exit 1
fi

# Função para verificar se um serviço está rodando
check_service() {
    local service_name=$1
    local port=$2
    
    if systemctl is-active --quiet "$service_name"; then
        echo "✅ $service_name está rodando"
        return 0
    else
        echo "❌ $service_name não está rodando"
        return 1
    fi
}

# Função para verificar se uma porta está em uso
check_port() {
    local port=$1
    if ss -ltnp | grep -q ":$port "; then
        echo "✅ Porta $port está em uso"
        return 0
    else
        echo "❌ Porta $port não está em uso"
        return 1
    fi
}

# Verifica dependências
echo "🔍 Verificando dependências..."

# Python 3
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 encontrado"
else
    echo "❌ Python 3 não encontrado"
    exit 1
fi

# Docker
if command -v docker &> /dev/null; then
    echo "✅ Docker encontrado"
else
    echo "❌ Docker não encontrado"
    exit 1
fi

# Git
if command -v git &> /dev/null; then
    echo "✅ Git encontrado"
else
    echo "❌ Git não encontrado"
    exit 1
fi

# Verifica serviços
echo "🔍 Verificando serviços..."

# Qwen server
if check_port 8013; then
    echo "✅ Servidor Qwen está rodando na porta 8013"
else
    echo "⚠️ Servidor Qwen não está rodando, tentando iniciar..."
    systemctl start llama-qwen.service
    sleep 10
    
    if check_port 8013; then
        echo "✅ Servidor Qwen iniciado com sucesso"
    else
        echo "❌ Falha ao iniciar servidor Qwen"
        exit 1
    fi
fi

# Verifica arquivos necessários
echo "🔍 Verificando arquivos do sistema..."

required_files=(
    "/root/qwen_interactive.py"
    "/root/safety_gates.py"
    "/root/darwinacci_omega.py"
    "/root/openhands_config.py"
    "/root/prometheus_telemetry.py"
    "/root/autonomous_system.py"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "✅ $file encontrado"
    else
        echo "❌ $file não encontrado"
        exit 1
    fi
done

# Instala dependências Python
echo "📦 Instalando dependências Python..."

pip3 install --upgrade pip
pip3 install requests psutil

# Cria diretórios necessários
echo "📁 Criando diretórios necessários..."

mkdir -p /root/logs
mkdir -p /root/backups
mkdir -p /root/evolution_logs
mkdir -p /root/telemetry_data

# Configura permissões
echo "🔐 Configurando permissões..."

chmod +x /root/qwen_interactive.py
chmod +x /root/safety_gates.py
chmod +x /root/darwinacci_omega.py
chmod +x /root/openhands_config.py
chmod +x /root/prometheus_telemetry.py
chmod +x /root/autonomous_system.py

# Cria arquivo de configuração
echo "⚙️ Criando arquivo de configuração..."

cat > /root/.autonomous_system_config.json << 'EOF'
{
  "system": {
    "name": "Qwen2.5-Coder Autonomous System",
    "version": "1.0.0",
    "qwen_url": "http://127.0.0.1:8013",
    "model_id": "/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf"
  },
  "safety_gates": {
    "enabled": true,
    "strict_mode": true,
    "max_file_size": 1048576,
    "max_execution_time": 30
  },
  "evolution": {
    "enabled": true,
    "interval": 300,
    "max_concurrent_tasks": 3
  },
  "telemetry": {
    "enabled": true,
    "prometheus_port": 9090,
    "ledger_file": "/root/worm_ledger.json"
  },
  "openhands": {
    "enabled": true,
    "tools": ["shell", "git", "tests"],
    "safety_level": "strict"
  }
}
EOF

# Cria service file para systemd
echo "🔧 Criando service file para systemd..."

cat > /etc/systemd/system/autonomous-system.service << 'EOF'
[Unit]
Description=Qwen2.5-Coder Autonomous System
After=network.target llama-qwen.service
Requires=llama-qwen.service

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart=/usr/bin/python3 /root/autonomous_system.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Recarrega systemd
systemctl daemon-reload

# Habilita serviço
systemctl enable autonomous-system.service

echo "✅ Sistema configurado com sucesso!"
echo "=================================================="
echo "Comandos disponíveis:"
echo "  systemctl start autonomous-system    - Inicia o sistema"
echo "  systemctl stop autonomous-system     - Para o sistema"
echo "  systemctl status autonomous-system   - Mostra status"
echo "  journalctl -u autonomous-system -f   - Mostra logs em tempo real"
echo "  python3 /root/qwen_interactive.py    - Interface interativa do Qwen"
echo "  python3 /root/autonomous_system.py   - Sistema principal"
echo "=================================================="

# Pergunta se deve iniciar o sistema
read -p "Deseja iniciar o sistema autônomo agora? (s/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[SsYy]$ ]]; then
    echo "🚀 Iniciando sistema autônomo..."
    systemctl start autonomous-system.service
    sleep 5
    
    if systemctl is-active --quiet autonomous-system.service; then
        echo "✅ Sistema autônomo iniciado com sucesso!"
        echo "📊 Para monitorar: journalctl -u autonomous-system -f"
    else
        echo "❌ Falha ao iniciar sistema autônomo"
        echo "📋 Logs: journalctl -u autonomous-system --no-pager"
    fi
else
    echo "⏸️ Sistema configurado mas não iniciado"
    echo "💡 Execute 'systemctl start autonomous-system' para iniciar"
fi

echo "🎉 Configuração concluída!"
