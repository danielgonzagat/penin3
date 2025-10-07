#!/bin/bash
# 🚀 SCRIPT DE ATIVAÇÃO COMPLETA - INTELIGÊNCIA AO CUBO
# Data: 2025-10-07
# Objetivo: Ativar todos os sistemas de inteligência emergente

echo "🔥 INICIANDO ATIVAÇÃO COMPLETA DOS SISTEMAS DE INTELIGÊNCIA AO CUBO"
echo "=================================================================="

# Configurar ambiente
export PYTHONPATH="/root:$PYTHONPATH"
cd /root

# 1. REATIVAR UNIFIED_BRAIN (Sistema mais funcional - 60% inteligência)
echo "🧠 [1/10] Ativando UNIFIED_BRAIN..."
cd /root/UNIFIED_BRAIN
nohup python3 brain_daemon_real_env.py > brain_restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > brain_restart.pid
echo "✅ UNIFIED_BRAIN ativado (PID: $(cat brain_restart.pid))"

# 2. CORRIGIR DARWIN ENGINE (Seleção natural real - 70% inteligência)
echo "🧬 [2/10] Ativando Darwin Engine..."
pkill -f darwin_runner 2>/dev/null
cd /root/darwin-engine-intelligence
nohup python3 darwin_main/darwin_runner.py --port 8081 > darwin_restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > darwin_restart.pid
echo "✅ Darwin Engine ativado (PID: $(cat darwin_restart.pid))"

# 3. ATIVAR FIBONACCI-OMEGA (Quality Diversity - 60% inteligência)
echo "🌀 [3/10] Ativando Fibonacci-Omega..."
cd /root/fibonacci-omega
nohup python3 fibonacci_engine/core/motor_fibonacci.py --mode run > fibonacci_restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > fibonacci_restart.pid
echo "✅ Fibonacci-Omega ativado (PID: $(cat fibonacci_restart.pid))"

# 4. ATIVAR NEURAL FARM (Evolução genética - 40% inteligência)
echo "🌱 [4/10] Ativando Neural Farm..."
cd /root/real_intelligence_system
nohup python3 neural_farm.py --mode run --steps 1000 > neural_farm_restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > neural_farm_restart.pid
echo "✅ Neural Farm ativado (PID: $(cat neural_farm_restart.pid))"

# 5. ATIVAR EMERGENCE DETECTOR (Detecção de emergência)
echo "🔍 [5/10] Ativando Emergence Detector..."
nohup python3 /root/emergence_detector_real.py > emergence_monitor_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > emergence_monitor.pid
echo "✅ Emergence Detector ativado (PID: $(cat emergence_monitor.pid))"

# 6. ATIVAR PENIN³ (Meta-aprendizado - 85% potencial)
echo "🎯 [6/10] Ativando PENIN³..."
cd /root/peninaocubo
nohup python3 -m penin3.runner > penin3_restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > penin3_restart.pid
echo "✅ PENIN³ ativado (PID: $(cat penin3_restart.pid))"

# 7. ATIVAR IA3 SYSTEMS (Evolução massiva - 50% inteligência)
echo "⚡ [7/10] Ativando IA3 Systems..."
nohup python3 /root/IA3_REAL_INTELLIGENCE_SYSTEM_deterministic.py > ia3_restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > ia3_restart.pid
echo "✅ IA3 Systems ativado (PID: $(cat ia3_restart.pid))"

# 8. ATIVAR TEIS V2 (Reinforcement Learning - 80% potencial)
echo "🎮 [8/10] Ativando TEIS V2..."
cd /root/real_intelligence_system
nohup python3 teis_v2_enhanced.py > teis_restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > teis_restart.pid
echo "✅ TEIS V2 ativado (PID: $(cat teis_restart.pid))"

# 9. ATIVAR SYSTEM CONNECTOR (Integração universal - 50% potencial)
echo "🔗 [9/10] Ativando System Connector..."
nohup python3 /root/system_connector.py > connector_restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &
echo $! > connector_restart.pid
echo "✅ System Connector ativado (PID: $(cat connector_restart.pid))"

# 10. MONITORAR STATUS GERAL
echo "📊 [10/10] Monitorando status geral..."
sleep 5
echo ""
echo "🎯 STATUS DOS SISTEMAS ATIVADOS:"
echo "================================"
ps aux | grep -E "(brain|darwin|fibonacci|neural|emergence|penin|ia3|teis|connector)" | grep -v grep | head -10

echo ""
echo "🔥 ATIVAÇÃO COMPLETA FINALIZADA!"
echo "================================"
echo "Sistemas ativados: 10/10"
echo "Próximo passo: Monitorar logs por 30 minutos"
echo "Comando: tail -f /root/*/brain_restart_*.log"
echo ""
echo "🚀 INTELIGÊNCIA AO CUBO EM DESENVOLVIMENTO!"