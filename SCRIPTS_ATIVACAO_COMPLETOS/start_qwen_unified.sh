#!/bin/bash

echo "🚀 Iniciando Sistema Unificado REAL Qwen2.5-Coder-7B..."

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
python3 -c "import psutil, requests" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Instalando dependências Python..."
    pip install psutil requests --break-system-packages --quiet
fi

echo "✅ Dependências verificadas"

# Cria diretórios necessários
echo "📁 Criando diretórios necessários..."
mkdir -p /root/qwen_checkpoints
mkdir -p /root/qwen_sandbox
mkdir -p /root/logs

echo "✅ Diretórios criados"

# Verifica se há processos anteriores rodando
echo "🔍 Verificando processos anteriores..."
pkill -f "qwen_unified_system.py" 2>/dev/null
pkill -f "qwen_real_autonomous.py" 2>/dev/null
pkill -f "qwen_autonomous_cli.py" 2>/dev/null

echo "✅ Processos anteriores finalizados"

# Inicia o sistema unificado REAL
echo "🚀 Iniciando Sistema Unificado REAL..."
echo ""
echo "🤖 Qwen2.5-Coder-7B - Sistema Unificado REAL"
echo "   Versão: 100% REAL - Zero Simulações"
echo "   Autoridade: Total sobre o sistema"
echo "   Capacidades: Modificação, execução, análise, evolução REAIS"
echo "   Modo: Cirurgião-Paciente Autônomo REAL"
echo ""
echo "📱 Comandos especiais:"
echo "   /status - Mostra status do sistema"
echo "   /metrics - Mostra métricas do sistema"
echo "   /checkpoint - Cria checkpoint manual"
echo "   /help - Mostra ajuda"
echo "   /exit - Sair"
echo ""
echo "💡 Exemplos de interação REAL:"
echo "   'Analise o sistema e otimize a performance'"
echo "   'Verifique os logs e identifique problemas'"
echo "   'Implemente uma melhoria de segurança'"
echo "   'Otimize o uso de memória'"
echo ""
echo "📊 Logs:"
echo "   - Principal: tail -f /root/qwen_unified.log"
echo "   - Checkpoints: /root/qwen_checkpoints/"
echo "   - Sandbox: /root/qwen_sandbox/"
echo ""

# Executa o sistema unificado REAL
python3 /root/qwen_unified_system.py
