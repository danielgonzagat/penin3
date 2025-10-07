#!/bin/bash
# 🔍 VALIDAÇÃO COMPLETA DE TODOS SISTEMAS

echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║           🔍 VALIDAÇÃO COMPLETA DE TODOS OS SISTEMAS                     ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Função para checar processo
check_process() {
    local name=$1
    local pid_file=$2
    
    if [ -f "$pid_file" ]; then
        pid=$(cat $pid_file)
        if ps -p $pid > /dev/null 2>&1; then
            cpu=$(ps -p $pid -o %cpu= | tr -d ' ')
            mem=$(ps -p $pid -o %mem= | tr -d ' ')
            echo "✅ $name: RODANDO (PID $pid, CPU: $cpu%, MEM: $mem%)"
            return 0
        else
            echo "❌ $name: MORTO (PID $pid não existe)"
            return 1
        fi
    else
        echo "⚠️  $name: SEM PID FILE"
        return 1
    fi
}

echo "📊 PROCESSOS:"
echo "─────────────────────────────────────────────────────────────────────────────"

running=0
total=0

# Brain Daemon
total=$((total + 1))
check_process "Brain Daemon V5" "/root/brain_daemon.pid" && running=$((running + 1))

# Darwin Bridge
total=$((total + 1))
check_process "Darwin Bridge" "/root/darwin_bridge.pid" && running=$((running + 1))

# Needle Bridge  
total=$((total + 1))
check_process "Needle Bridge" "/root/needle_bridge.pid" && running=$((running + 1))

# Auto-Evolution
total=$((total + 1))
check_process "Auto-Evolution" "/root/auto_evolution.pid" && running=$((running + 1))

# Multi-Env Test
total=$((total + 1))
check_process "Multi-Env Test" "/root/multi_env_test.pid" && running=$((running + 1))

# Continuous Monitor
total=$((total + 1))
check_process "Continuous Monitor" "/root/continuous_monitor.pid" && running=$((running + 1))

# Neurogenesis
total=$((total + 1))
check_process "Neurogenesis" "/root/neurogenesis.pid" && running=$((running + 1))

echo ""
echo "📈 TOTAL: $running/$total processos rodando"
echo ""

# Arquivos criados
echo "📁 ARQUIVOS CRIADOS:"
echo "─────────────────────────────────────────────────────────────────────────────"

files=(
    "/root/UNIFIED_BRAIN/system_bridge.py"
    "/root/UNIFIED_BRAIN/safe_collective_consciousness.py"
    "/root/UNIFIED_BRAIN/code_evolution_engine.py"
    "/root/UNIFIED_BRAIN/true_godelian_incompleteness.py"
    "/root/UNIFIED_BRAIN/meta_curiosity_module.py"
    "/root/darwin_bridge_connector.py"
    "/root/needle_bridge_connector.py"
    "/root/auto_evolution_controller.py"
    "/root/multi_environment_tester.py"
    "/root/continuous_monitor.py"
    "/root/ab_testing_runner.py"
    "/root/neurogenesis_controller.py"
)

files_exist=0
for file in "${files[@]}"; do
    if [ -f "$file" ]; then
        size=$(ls -lh "$file" | awk '{print $5}')
        echo "✅ $file ($size)"
        files_exist=$((files_exist + 1))
    else
        echo "❌ $file (FALTANDO)"
    fi
done

echo ""
echo "📈 TOTAL: $files_exist/${#files[@]} arquivos criados"
echo ""

# Logs recentes
echo "📝 LOGS RECENTES (Brain Daemon):"
echo "─────────────────────────────────────────────────────────────────────────────"
tail -5 /root/brain_daemon_v5_complete.log 2>/dev/null | grep "Ep " || echo "⚠️  Sem logs de episódios ainda"

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════╗"
echo "║                     ✅ VALIDAÇÃO COMPLETA                                ║"
echo "╚═══════════════════════════════════════════════════════════════════════════╝"
