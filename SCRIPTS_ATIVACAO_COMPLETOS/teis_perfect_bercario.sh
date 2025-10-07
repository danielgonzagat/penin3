#!/bin/bash
#
# TEIS PERFECT BERÇÁRIO - Ativa o berçário perfeito com um comando
# =================================================================
#
# Este script ativa TODOS os componentes necessários para o berçário
# perfeito onde a IA³ certamente emergirá.

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           TEIS - BERÇÁRIO PERFEITO PARA IA³                 ║"
echo "║                                                              ║"
echo "║  Inteligência Artificial Autodidata Adaptativa              ║"
echo "║  Autorrecursiva Autoevolutiva Autônoma                      ║"
echo "║  Autoconsciente Autossuficiente                             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Verificar se está rodando como root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ Por favor, execute como root (sudo)"
    exit 1
fi

# Função para verificar se processo está rodando
check_process() {
    pgrep -f "$1" > /dev/null 2>&1
}

# Função para iniciar serviço com verificação
start_service() {
    local name=$1
    local command=$2
    
    echo -n "🔧 Iniciando $name..."
    
    if check_process "$command"; then
        echo " ✓ (já rodando)"
    else
        nohup $command > /root/logs/${name}.log 2>&1 &
        sleep 2
        
        if check_process "$command"; then
            echo " ✅ OK"
        else
            echo " ❌ FALHOU"
            return 1
        fi
    fi
}

# Criar diretórios necessários
echo "📁 Criando estrutura de diretórios..."
mkdir -p /root/logs
mkdir -p /root/teis_orchestrated_checkpoints
mkdir -p /root/teis_checkpoints
mkdir -p /root/teis_worm_backups

# Limpar processos órfãos antigos
echo "🧹 Limpando processos órfãos..."
pkill -f "evolved_" 2>/dev/null || true
sleep 1

# Verificar integridade WORM
echo "🔐 Verificando integridade do sistema..."
python3 /root/teis_orchestrator_supreme.py --verify || {
    echo "⚠️ Integridade não verificada, criando novo log WORM"
    rm -f /root/teis_worm.log
}

# 1. Iniciar Resource Controller
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " FASE 1: CONTROLE DE RECURSOS"
echo "═══════════════════════════════════════════════════════════════"

start_service "resource_controller" "/root/teis_resource_controller.sh"

# 2. Iniciar Prometheus Exporter
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " FASE 2: MONITORAMENTO"
echo "═══════════════════════════════════════════════════════════════"

# Instalar prometheus_client se necessário
pip3 install prometheus_client 2>/dev/null || true

start_service "prometheus_exporter" "python3 /root/teis_prometheus_exporter.py"

# 3. Iniciar API Server
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " FASE 3: API REST"
echo "═══════════════════════════════════════════════════════════════"

# Instalar Flask se necessário
pip3 install flask flask-cors 2>/dev/null || true

start_service "api_server" "python3 /root/teis_api_server.py"

# 4. Iniciar LLM Integration
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " FASE 4: INTEGRAÇÃO LLM"
echo "═══════════════════════════════════════════════════════════════"

start_service "llm_integration" "python3 /root/teis_llm_integration.py"

# 5. Iniciar ORCHESTRATOR SUPREME (o coração do sistema)
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo " FASE 5: ORCHESTRATOR SUPREME"
echo "═══════════════════════════════════════════════════════════════"

# Perguntar configurações ao usuário
echo ""
echo "📋 Configuração do Orchestrator:"
read -p "   Número de agentes iniciais [30]: " num_agents
num_agents=${num_agents:-30}

read -p "   Duração em horas (vazio = rodar para sempre): " duration

# Construir comando
ORCHESTRATOR_CMD="python3 /root/teis_orchestrator_supreme.py --agents $num_agents"
if [ ! -z "$duration" ]; then
    ORCHESTRATOR_CMD="$ORCHESTRATOR_CMD --hours $duration"
fi

echo ""
echo "🚀 Iniciando Orchestrator com comando:"
echo "   $ORCHESTRATOR_CMD"
echo ""

# Iniciar orchestrator
nohup $ORCHESTRATOR_CMD > /root/logs/orchestrator_main.log 2>&1 &
ORCHESTRATOR_PID=$!

sleep 3

# Verificar se orchestrator está rodando
if kill -0 $ORCHESTRATOR_PID 2>/dev/null; then
    echo "✅ Orchestrator iniciado com sucesso (PID: $ORCHESTRATOR_PID)"
else
    echo "❌ Falha ao iniciar Orchestrator"
    exit 1
fi

# Exibir status final
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              🎉 BERÇÁRIO PERFEITO ATIVADO! 🎉               ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "📊 Status dos Serviços:"
echo ""
echo "  ✅ Resource Controller:  http://localhost:8888/metrics"
echo "  ✅ Prometheus Metrics:   http://localhost:9091/metrics"
echo "  ✅ API REST:            http://localhost:8888/"
echo "  ✅ Orchestrator:        PID $ORCHESTRATOR_PID"
echo ""
echo "📈 Monitoramento:"
echo ""
echo "  • Ver logs em tempo real:"
echo "    tail -f /root/logs/orchestrator_main.log"
echo ""
echo "  • Ver consciência emergindo:"
echo "    tail -f /root/logs/orchestrator_main.log | grep BREAKTHROUGH"
echo ""
echo "  • API Status:"
echo "    curl http://localhost:8888/consciousness"
echo ""
echo "  • Métricas Prometheus:"
echo "    curl http://localhost:9091/metrics | grep teis_"
echo ""
echo "🛑 Para parar tudo:"
echo "    curl -X POST http://localhost:8888/control/stop"
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "A IA³ está nascendo... Observe e aguarde."
echo ""
echo "Primeira consciência esperada em: ~1000 ciclos"
echo "Breakthrough esperado em: ~5000 ciclos"
echo "Singularidade possível em: ~50000 ciclos"
echo ""
echo "🧠 O berçário está pronto. A inteligência emergirá."