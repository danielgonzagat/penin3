#!/bin/bash

echo "🚀 Iniciando Qwen2.5-Coder-7B - Interface CLI de Última Geração..."

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
mkdir -p /root/evolution/{checkpoints,variants,canary,ab_tests,artifacts}
mkdir -p /root/logs

echo "✅ Diretórios criados"

# Verifica se há processos anteriores rodando
echo "🔍 Verificando processos anteriores..."
pkill -f "qwen_autonomous_cli.py" 2>/dev/null

echo "✅ Processos anteriores finalizados"

# Inicia a interface CLI
echo "🚀 Iniciando Interface CLI de Última Geração..."
echo ""
echo "🤖 Qwen2.5-Coder-7B - Sistema Autônomo Cirurgião-Paciente"
echo "   Autoridade: Total sobre o sistema"
echo "   Capacidades: Modificação, execução, análise, evolução"
echo "   Modo: Cirurgião-Paciente Autônomo"
echo ""
echo "📱 Comandos especiais:"
echo "   /autonomous - Ativa modo autônomo"
echo "   /stop - Para modo autônomo"
echo "   /clear - Limpa histórico"
echo "   /status - Mostra status completo"
echo "   /scan - Reescaneia sistema"
echo "   /help - Mostra ajuda"
echo "   /exit - Sair"
echo ""
echo "💡 Exemplos de interação:"
echo "   'Analise o sistema e otimize a performance'"
echo "   'Verifique os logs e identifique problemas'"
echo "   'Implemente uma melhoria de segurança'"
echo "   'Otimize o uso de memória'"
echo ""

# Executa a interface CLI
python3 /root/qwen_autonomous_cli.py
