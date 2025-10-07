#!/bin/bash

echo "🚀 Iniciando Sistema Ultimate REAL Qwen2.5-Coder-7B..."

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
mkdir -p /root/qwen_ultimate_checkpoints
mkdir -p /root/qwen_ultimate_sandbox
mkdir -p /root/logs

echo "✅ Diretórios criados"

# Verifica se há processos anteriores rodando
echo "🔍 Verificando processos anteriores..."
pkill -f "qwen_.*\.py" 2>/dev/null
pkill -f "INVESTIGACAO_AUTONOMA" 2>/dev/null
pkill -f "qwen_autonomous_system" 2>/dev/null

echo "✅ Processos anteriores finalizados"

# Inicia o sistema ultimate REAL
echo "🚀 Iniciando Sistema Ultimate REAL..."
echo ""
echo "🤖 Qwen2.5-Coder-7B - Sistema Ultimate REAL"
echo "   Versão: ULTIMATE - Arquitetura Unificada"
echo "   Autoridade: Total sobre o sistema"
echo "   Capacidades: Modificação, execução, análise, evolução REAIS"
echo "   Modo: Cirurgião-Paciente Autônomo REAL"
echo "   Sistema: Ultimate com evolução contínua"
echo ""
echo "📱 Comandos especiais:"
echo "   /status - Mostra status do sistema"
echo "   /metrics - Mostra métricas do sistema"
echo "   /checkpoint - Cria checkpoint manual"
echo "   /evolution - Mostra dados de evolução"
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
echo "   - Principal: tail -f /root/qwen_ultimate.log"
echo "   - Checkpoints: /root/qwen_ultimate_checkpoints/"
echo "   - Sandbox: /root/qwen_ultimate_sandbox/"
echo ""

# Executa o sistema ultimate REAL
python3 /root/qwen_ultimate_system.py
