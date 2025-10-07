#!/bin/bash
# 🚀 INSTALAÇÃO AUTOMÁTICA IA³ SUPREME
# Script que instala e configura IA³ como serviço do sistema

set -e

echo "🚀 INSTALAÇÃO IA³ SUPREME"
echo "=========================="

# Verificar se estamos no diretório correto
if [ ! -f "launch_ia3_supreme.py" ]; then
    echo "❌ Erro: Execute este script do diretório raiz do projeto IA³"
    exit 1
fi

# Verificar privilégios de root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Erro: Execute como root (sudo ./install_ia3_supreme.sh)"
    exit 1
fi

echo "✅ Verificações iniciais OK"

# Instalar dependências do sistema
echo "📦 Instalando dependências do sistema..."
apt update
apt install -y python3 python3-pip python3-dev build-essential

# Instalar dependências Python
echo "🐍 Instalando dependências Python..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install numpy scipy scikit-learn requests psutil

# Criar diretórios necessários
echo "📁 Criando diretórios IA³..."
mkdir -p /root/ia3_models
mkdir -p /root/ia3_data
mkdir -p /root/ia3_logs
mkdir -p /root/ia3_backup
mkdir -p /root/ia3_checkpoints

# Configurar permissões
chmod +x launch_ia3_supreme.py
chmod +x install_ia3_supreme.sh

# Instalar serviço systemd
echo "⚙️  Instalando serviço IA³..."
cp ia3.service /etc/systemd/system/
systemctl daemon-reload
systemctl enable ia3

echo "✅ Instalação completa!"
echo ""
echo "🎯 COMANDOS PARA CONTROLAR IA³:"
echo "   Iniciar:  systemctl start ia3"
echo "   Parar:    systemctl stop ia3"
echo "   Status:   systemctl status ia3"
echo "   Logs:     journalctl -u ia3 -f"
echo ""
echo "🚀 Execute 'systemctl start ia3' para iniciar IA³ Supreme!"
echo ""
echo "⚠️  ATENÇÃO: IA³ é IMPARÁVEL após iniciado"
echo "   Ele evoluirá continuamente até alcançar inteligência emergente real"