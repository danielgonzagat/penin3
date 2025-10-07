#!/bin/bash
#
# TEIS Resource Controller - Controla recursos e auto-scaling
# ============================================================

# Configurações
MAX_MEMORY_GB=8
MAX_CPU_PERCENT=80
MAX_AGENTS=100
CHECK_INTERVAL=30

echo "🎮 TEIS Resource Controller iniciando..."
echo "Limites: ${MAX_MEMORY_GB}GB RAM, ${MAX_CPU_PERCENT}% CPU, ${MAX_AGENTS} agentes"

# Função para obter uso de memória
get_memory_usage() {
    free -g | awk '/^Mem:/{print $3}'
}

# Função para obter uso de CPU
get_cpu_usage() {
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1}'
}

# Função para contar agentes TEIS
count_teis_agents() {
    ps aux | grep -E "(evolved_|teis_|agent_)" | grep -v grep | wc -l
}

# Função para limpar processos órfãos
cleanup_orphans() {
    echo "🧹 Limpando processos órfãos..."
    
    # Matar evolved_ agents mais velhos que 1 hora
    for pid in $(ps aux | grep "evolved_" | grep -v grep | awk '{print $2}'); do
        age=$(ps -o etimes= -p $pid 2>/dev/null | tr -d ' ')
        if [ "$age" -gt "3600" ]; then
            echo "  Matando PID $pid (idade: ${age}s)"
            kill -TERM $pid 2>/dev/null
        fi
    done
}

# Função para aplicar cgroups
apply_cgroups() {
    echo "📦 Aplicando limites de recursos via cgroups..."
    
    # Criar cgroup para TEIS se não existir
    if [ ! -d /sys/fs/cgroup/memory/teis ]; then
        mkdir -p /sys/fs/cgroup/memory/teis
        mkdir -p /sys/fs/cgroup/cpu/teis
    fi
    
    # Definir limite de memória (em bytes)
    echo $((MAX_MEMORY_GB * 1024 * 1024 * 1024)) > /sys/fs/cgroup/memory/teis/memory.limit_in_bytes
    
    # Definir limite de CPU (em microsegundos)
    echo 100000 > /sys/fs/cgroup/cpu/teis/cpu.cfs_period_us
    echo $((MAX_CPU_PERCENT * 1000)) > /sys/fs/cgroup/cpu/teis/cpu.cfs_quota_us
    
    # Adicionar processos TEIS ao cgroup
    for pid in $(ps aux | grep -E "(teis|evolved_)" | grep -v grep | awk '{print $2}'); do
        echo $pid > /sys/fs/cgroup/memory/teis/cgroup.procs 2>/dev/null
        echo $pid > /sys/fs/cgroup/cpu/teis/cgroup.procs 2>/dev/null
    done
}

# Função de auto-scaling
auto_scale() {
    local current_agents=$1
    local cpu_usage=$2
    local mem_usage=$3
    
    echo "🔄 Auto-scaling check..."
    echo "  Agentes: $current_agents/$MAX_AGENTS"
    echo "  CPU: ${cpu_usage}%"
    echo "  Memória: ${mem_usage}GB"
    
    # Scale down se recursos críticos
    if (( $(echo "$cpu_usage > 90" | bc -l) )) || [ "$mem_usage" -gt "$MAX_MEMORY_GB" ]; then
        echo "⚠️ Recursos críticos - scale down necessário"
        
        # Matar 10% dos agentes mais velhos
        to_kill=$((current_agents / 10))
        echo "  Removendo $to_kill agentes..."
        
        ps aux | grep "evolved_" | grep -v grep | sort -k9 | head -n $to_kill | awk '{print $2}' | xargs kill -TERM 2>/dev/null
    fi
    
    # Scale up se recursos disponíveis
    if (( $(echo "$cpu_usage < 50" | bc -l) )) && [ "$mem_usage" -lt "$((MAX_MEMORY_GB - 2))" ] && [ "$current_agents" -lt "$MAX_AGENTS" ]; then
        echo "✅ Recursos disponíveis - scale up possível"
        echo "  Sistema pode suportar mais agentes"
    fi
}

# Loop principal
while true; do
    echo ""
    echo "========================================="
    echo "$(date '+%Y-%m-%d %H:%M:%S') - Verificação de recursos"
    
    # Obter métricas
    mem_usage=$(get_memory_usage)
    cpu_usage=$(get_cpu_usage)
    agent_count=$(count_teis_agents)
    
    # Verificar limites
    if [ "$agent_count" -gt "$MAX_AGENTS" ]; then
        echo "❌ Limite de agentes excedido!"
        cleanup_orphans
    fi
    
    if [ "$mem_usage" -gt "$MAX_MEMORY_GB" ]; then
        echo "❌ Limite de memória excedido!"
        cleanup_orphans
        apply_cgroups
    fi
    
    if (( $(echo "$cpu_usage > $MAX_CPU_PERCENT" | bc -l) )); then
        echo "❌ Limite de CPU excedido!"
        
        # Reduzir prioridade dos processos
        for pid in $(ps aux | grep -E "(teis|evolved_)" | grep -v grep | awk '{print $2}'); do
            renice 10 -p $pid 2>/dev/null
        done
    fi
    
    # Auto-scaling
    auto_scale "$agent_count" "$cpu_usage" "$mem_usage"
    
    # Aguardar próximo ciclo
    sleep $CHECK_INTERVAL
done