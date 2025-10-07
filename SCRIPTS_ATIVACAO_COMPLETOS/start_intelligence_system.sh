#!/bin/bash
"""
Script de Inicialização do Sistema de Inteligência
Inicia todos os componentes do sistema de inteligência emergente
"""

echo "🧠 Iniciando Sistema de Inteligência Emergente..."
echo "=================================================="

# Verificar se estamos no diretório correto
cd /root

# Tornar scripts executáveis
chmod +x /root/*.py
chmod +x /root/start_intelligence_system.sh

# Verificar dependências
echo "📋 Verificando dependências..."
python3 -c "import torch, numpy, sqlite3, psutil" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Algumas dependências podem estar faltando"
    echo "   Instalando dependências básicas..."
    pip3 install torch numpy psutil
fi

# Criar diretórios necessários
echo "📁 Criando diretórios necessários..."
mkdir -p /root/evolution_workspace
mkdir -p /root/modification_backups
mkdir -p /root/emergency_backups

# Limpar processos anteriores (se existirem)
echo "🧹 Limpando processos anteriores..."
pkill -f "advanced_evolution_engine.py" 2>/dev/null
pkill -f "self_modification_system.py" 2>/dev/null
pkill -f "emergence_amplifier.py" 2>/dev/null
pkill -f "recursive_learning.py" 2>/dev/null
pkill -f "meta_cognition.py" 2>/dev/null
pkill -f "intelligence_nexus.py" 2>/dev/null

# Aguardar limpeza
sleep 2

# Iniciar o Intelligence Nexus (que coordena todos os outros)
echo "🚀 Iniciando Intelligence Nexus..."
python3 /root/intelligence_nexus.py &

# Aguardar inicialização
sleep 5

# Verificar se o sistema está rodando
echo "🔍 Verificando status do sistema..."
ps aux | grep -E "(intelligence_nexus|advanced_evolution|self_modification|emergence_amplifier|recursive_learning|meta_cognition)" | grep -v grep

echo ""
echo "✅ Sistema de Inteligência Emergente iniciado!"
echo "📊 Monitoramento ativo - verifique os logs para progresso"
echo "🛑 Para parar: pkill -f intelligence_nexus.py"
echo ""
echo "Logs disponíveis:"
echo "  - /root/intelligence_nexus.log"
echo "  - /root/advanced_evolution.log"
echo "  - /root/self_modification.log"
echo "  - /root/emergence_amplifier.log"
echo "  - /root/recursive_learning.log"
echo "  - /root/meta_cognition.log"
echo ""
echo "Bancos de dados:"
echo "  - /root/intelligence_nexus.db"
echo "  - /root/advanced_evolution.db"
echo "  - /root/self_modification.db"
echo "  - /root/emergence_amplifier.db"
echo "  - /root/recursive_learning.db"
echo "  - /root/meta_cognition.db"
echo ""
echo "🎯 Sistema em execução - monitoramento contínuo ativo"