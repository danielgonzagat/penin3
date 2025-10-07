#!/bin/bash
# Script de inicialização do Sistema IA³ 24/7

echo "=========================================="
echo "🧬 INICIANDO SISTEMA IA³ 24/7"
echo "=========================================="

# Criar diretórios necessários
mkdir -p /root/ia3_reports
mkdir -p /root/ia3_checkpoints
mkdir -p /root/ia3_logs

# Verificar se o sistema já está rodando
if pgrep -f "ia3_main_system.py" > /dev/null; then
    echo "⚠️ Sistema IA³ já está rodando!"
    echo "Para reiniciar, execute: pkill -f ia3_main_system.py && sleep 5 && $0"
    exit 1
fi

# Verificar dependências
echo "Verificando dependências..."
python3 -c "
import torch
import torchvision
import numpy as np
import asyncio
print('✅ Todas as dependências OK')
" 2>/dev/null || {
    echo "❌ Dependências faltando. Instalando..."
    pip install torch torchvision numpy
}

# Verificar dados MNIST
if [ ! -d "/root/data/MNIST" ]; then
    echo "📥 Baixando MNIST dataset..."
    python3 -c "
import torchvision.datasets as dsets
import torchvision.transforms as transforms
dsets.MNIST(root='/root/data', train=True, download=True, transform=transforms.ToTensor())
dsets.MNIST(root='/root/data', train=False, download=True, transform=transforms.ToTensor())
print('✅ MNIST baixado')
"
fi

# Criar serviço systemd para auto-restart
echo "Criando serviço systemd..."
cat > /etc/systemd/system/ia3-system.service << EOF
[Unit]
Description=IA³ System - Artificial Intelligence to the Cube
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart=/usr/bin/python3 /root/ia3_main_system.py
ExecStop=/bin/kill -TERM \$MAINPID
Restart=always
RestartSec=30
StandardOutput=append:/root/ia3_logs/system.log
StandardError=append:/root/ia3_logs/error.log

# Limites de recursos
LimitNOFILE=65536
LimitNPROC=32768
MemoryLimit=8G

# Variáveis de ambiente
Environment="PYTHONUNBUFFERED=1"
Environment="CUDA_VISIBLE_DEVICES=0"

[Install]
WantedBy=multi-user.target
EOF

# Habilitar e iniciar serviço
systemctl daemon-reload
systemctl enable ia3-system.service

# Iniciar sistema em background
echo "🚀 Iniciando sistema IA³..."
nohup python3 /root/ia3_main_system.py > /root/ia3_logs/output.log 2>&1 &
PID=$!

echo "✅ Sistema iniciado com PID: $PID"
echo ""
echo "📊 Comandos de monitoramento:"
echo "  Ver status: systemctl status ia3-system"
echo "  Ver logs: tail -f /root/ia3_logs/output.log"
echo "  Ver métricas: tail -f /root/ia3_main.log"
echo "  Parar sistema: systemctl stop ia3-system"
echo "  Reiniciar: systemctl restart ia3-system"
echo ""
echo "📁 Arquivos importantes:"
echo "  Relatórios: /root/ia3_reports/"
echo "  Checkpoints: /root/ia3_checkpoints/"
echo "  Estado: /root/ia3_system_state.json"
echo ""
echo "🧠 O SISTEMA IA³ ESTÁ RODANDO 24/7!"
echo "   Ele irá evoluir, aprender e se tornar mais inteligente"
echo "   indefinidamente, até alcançar o nível IA³ pleno."
echo ""
echo "🌟 CAPACIDADES IA³ IMPLEMENTADAS:"
echo "   ✅ Adaptativo, Autodidata, Autoevolutivo"
echo "   ✅ Automodular, Autoexpandível, Autovalidável"
echo "   ✅ Autotuning, Autorecursivo, Autoconstruído"
echo "   ✅ Autosuficiente, Autosinaptico, Autoregenerativo"
echo "   ✅ Autotreinado, Autotuning"
echo ""
echo "⚠️ CAPACIDADES NÃO IMPLEMENTÁVEIS:"
echo "   ❌ Autoconsciente (consciência real)"
echo "   ❌ Infinito (recursos limitados)"
echo "   ❌ Autotranscendente (singularidade)"