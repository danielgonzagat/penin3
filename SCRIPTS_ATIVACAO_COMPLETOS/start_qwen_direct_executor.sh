#!/bin/bash

echo "🚀 Iniciando Sistema Executor Direto REAL Qwen2.5-Coder-7B..."

# Verificar se o servidor Qwen está funcionando
echo "📡 Verificando servidor Qwen..."
if curl -s -f http://127.0.0.1:8013/v1/models > /dev/null; then
    echo "✅ Servidor Qwen está funcionando"
else
    echo "❌ Servidor Qwen não está funcionando!"
    echo "   Iniciando serviço..."
    sudo systemctl start llama-qwen
    sleep 10
    
    if curl -s -f http://127.0.0.1:8013/v1/models > /dev/null; then
        echo "✅ Servidor Qwen iniciado com sucesso"
    else
        echo "❌ Falha ao iniciar servidor Qwen"
        echo "   Verifique o status: sudo systemctl status llama-qwen"
        exit 1
    fi
fi

# Verificar dependências Python
echo "🔍 Verificando dependências Python..."
python3 -c "import requests, psutil, json, threading, subprocess, signal, logging, tempfile, shutil, ast, re, hashlib, socket, multiprocessing, pathlib, dataclasses, enum, queue, select, fcntl, termios, tty" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Dependências verificadas"
else
    echo "❌ Dependências Python faltando"
    echo "   Instalando dependências..."
    pip3 install requests psutil --break-system-packages
fi

# Criar diretórios necessários
echo "📁 Criando diretórios necessários..."
mkdir -p /root/qwen_direct_executor_checkpoints
mkdir -p /root/qwen_direct_executor_sandbox
echo "✅ Diretórios criados"

# Verificar processos anteriores
echo "🔍 Verificando processos anteriores..."
pkill -f "qwen_direct_executor.py" 2>/dev/null || true
sleep 2
echo "✅ Processos anteriores finalizados"

# Iniciar sistema executor direto
echo "🚀 Iniciando Sistema Executor Direto REAL..."
echo ""
echo "🤖 Qwen2.5-Coder-7B - Sistema Executor Direto REAL"
echo "   Versão: Executor Direto - Zero Simulações"
echo "   Autoridade: Total sobre o sistema"
echo "   Capacidades: Modificação, execução, análise, evolução REAIS"
echo "   Modo: Cirurgião-Paciente Autônomo REAL"
echo "📱 Comandos especiais:"
echo "   /status - Mostra status do sistema"
echo "   /metrics - Mostra métricas do sistema"
echo "   /checkpoint - Cria checkpoint manual"
echo "   /evolution - Mostra dados de evolução"
echo "   /exec <cmd> - Executa comando direto"
echo "   /help - Mostra ajuda"
echo "   /exit - Sair"
echo "💡 Exemplos de interação REAL:"
echo "   'Analise o sistema e otimize a performance'"
echo "   'Verifique os logs e identifique problemas'"
echo "   'Implemente uma melhoria de segurança'"
echo "   'Otimize o uso de memória'"
echo "📊 Logs:"
echo "   - Principal: tail -f /root/qwen_direct_executor.log"
echo "   - Checkpoints: /root/qwen_direct_executor_checkpoints/"
echo "   - Sandbox: /root/qwen_direct_executor_sandbox/"
echo ""

# Executar sistema
python3 /root/qwen_direct_executor.py
