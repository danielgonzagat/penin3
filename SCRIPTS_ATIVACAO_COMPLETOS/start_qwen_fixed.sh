#!/bin/bash

echo "🚀 Iniciando Sistema Autônomo Qwen2.5-Coder-7B Corrigido..."

# Verifica se está rodando como root
if [ "$EUID" -ne 0 ]; then
    echo "❌ Este script deve ser executado como root"
    exit 1
fi

# Verifica se o servidor Qwen está rodando
echo "📡 Verificando servidor Qwen..."
if ! curl -s http://127.0.0.1:8013/v1/models >/dev/null; then
    echo "❌ Servidor Qwen não está funcionando!"
    echo "   Inicie o servidor com: sudo systemctl start llama-qwen"
    exit 1
fi

echo "✅ Servidor Qwen está funcionando"

# Verifica dependências Python
echo "🔍 Verificando dependências Python..."
python3 -c "import psutil, docker, git, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Instalando dependências Python..."
    pip install psutil docker gitpython requests --break-system-packages --quiet
fi

echo "✅ Dependências verificadas"

# Cria diretórios necessários
echo "📁 Criando diretórios necessários..."
mkdir -p /root/checkpoints
mkdir -p /root/logs
mkdir -p /root/evolution/{checkpoints,variants,canary,ab_tests,artifacts}

echo "✅ Diretórios criados"

# Verifica se há processos anteriores rodando
echo "🔍 Verificando processos anteriores..."
pkill -f "qwen_autonomous_system" 2>/dev/null
pkill -f "qwen_autonomous_cli" 2>/dev/null

echo "✅ Processos anteriores finalizados"

# Inicia o sistema corrigido
echo "🚀 Iniciando Sistema Autônomo Corrigido..."
echo ""
echo "🤖 Qwen2.5-Coder-7B - Sistema Autônomo Cirurgião-Paciente"
echo "   Versão: Corrigida e Otimizada"
echo "   Autoridade: Total sobre o sistema"
echo "   Capacidades: Modificação, execução, análise, evolução"
echo "   Modo: Cirurgião-Paciente Autônomo"
echo ""
echo "📱 Componentes:"
echo "   - Safety Gates Avançados"
echo "   - Darwinacci-Ω Evolutivo"
echo "   - Telemetria Prometheus"
echo "   - Monitoramento 24/7"
echo ""
echo "📊 Logs:"
echo "   - Principal: tail -f /root/qwen_autonomous_system_fixed.log"
echo "   - Checkpoints: /root/checkpoints/"
echo ""
echo "💡 Comandos de controle:"
echo "   Ctrl+C - Parar sistema"
echo "   SIGTERM - Parar sistema"
echo ""

# Executa o sistema corrigido
python3 /root/qwen_autonomous_system_fixed.py
