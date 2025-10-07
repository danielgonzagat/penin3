#!/bin/bash

echo "🚀 Iniciando Sistema Autônomo Qwen2.5-Coder-7B Completo..."

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
python3 -c "import psutil, docker, git, requests, prometheus_client" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "📦 Instalando dependências Python..."
    pip install psutil docker gitpython requests prometheus_client --break-system-packages --quiet
fi

echo "✅ Dependências verificadas"

# Cria diretórios necessários
echo "📁 Criando diretórios necessários..."
mkdir -p /root/evolution/{checkpoints,variants,canary,ab_tests,artifacts}
mkdir -p /root/templates
mkdir -p /root/logs

echo "✅ Diretórios criados"

# Verifica se há processos anteriores rodando
echo "🔍 Verificando processos anteriores..."
pkill -f "qwen_autonomous_system_main.py" 2>/dev/null
pkill -f "qwen_autonomous_agent.py" 2>/dev/null
pkill -f "qwen_cli_interface.py" 2>/dev/null

echo "✅ Processos anteriores finalizados"

# Menu de opções
echo ""
echo "Escolha uma opção:"
echo "1) Sistema Autônomo Completo (Recomendado)"
echo "2) Agente Autônomo Individual"
echo "3) Interface CLI Robusta"
echo "4) Interface Flask Web"
echo "5) Darwinacci-Ω Individual"
echo "6) Telemetria Prometheus Individual"
echo "7) Safety Gates Individual"
echo "8) Todas as interfaces"
echo ""

read -p "Digite sua escolha (1-8): " choice

case $choice in
    1)
        echo "🚀 Iniciando Sistema Autônomo Completo..."
        echo "   - Agente Autônomo Qwen"
        echo "   - Safety Gates Avançados"
        echo "   - Darwinacci-Ω Evolutivo"
        echo "   - Telemetria Prometheus"
        echo "   - Monitoramento 24/7"
        echo ""
        echo "📱 Acesse:"
        echo "   - Telemetria: http://localhost:8000/metrics"
        echo "   - Logs: tail -f /root/qwen_autonomous_system.log"
        echo ""
        python3 /root/qwen_autonomous_system_main.py
        ;;
    2)
        echo "🚀 Iniciando Agente Autônomo Individual..."
        python3 /root/qwen_autonomous_agent.py
        ;;
    3)
        echo "🚀 Iniciando Interface CLI Robusta..."
        python3 /root/qwen_cli_interface.py
        ;;
    4)
        echo "🚀 Iniciando Interface Flask Web..."
        python3 /root/qwen_flask_interface_alt.py
        ;;
    5)
        echo "🚀 Iniciando Darwinacci-Ω Individual..."
        python3 /root/darwinacci_omega_core.py
        ;;
    6)
        echo "🚀 Iniciando Telemetria Prometheus Individual..."
        python3 /root/prometheus_telemetry_advanced.py
        ;;
    7)
        echo "🚀 Iniciando Safety Gates Individual..."
        python3 /root/safety_gates_advanced.py
        ;;
    8)
        echo "🚀 Iniciando todas as interfaces..."
        echo "   - Interface Flask: http://localhost:5001"
        echo "   - Telemetria: http://localhost:8000/metrics"
        echo ""
        
        # Inicia Flask em background
        python3 /root/qwen_flask_interface_alt.py &
        FLASK_PID=$!
        
        # Inicia Telemetria em background
        python3 /root/prometheus_telemetry_advanced.py &
        TELEMETRY_PID=$!
        
        # Inicia Agente Autônomo em background
        python3 /root/qwen_autonomous_agent.py &
        AGENT_PID=$!
        
        echo "✅ Todas as interfaces iniciadas!"
        echo "   PIDs: Flask=$FLASK_PID, Telemetria=$TELEMETRY_PID, Agente=$AGENT_PID"
        echo ""
        echo "Pressione Ctrl+C para parar todas as interfaces..."
        
        # Aguarda Ctrl+C
        trap "kill $FLASK_PID $TELEMETRY_PID $AGENT_PID 2>/dev/null; echo '👋 Interfaces encerradas'; exit 0" INT
        wait
        ;;
    *)
        echo "❌ Opção inválida!"
        exit 1
        ;;
esac
