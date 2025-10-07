#!/bin/bash
# START ALL DAEMONS - Script simples e testado
set -e

echo "🚀 INICIANDO TODOS OS DAEMONS I³"
echo "=================================="
echo

# 1. Parar versões antigas
echo "1️⃣ Parando processos antigos..."
pkill -f EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py || true
pkill -f CONSCIOUSNESS_DAEMON.py || true
pkill -f SURPRISE_DAEMON.py || true
pkill -f DARWINACCI_DAEMON.py || true
pkill -f PHASE4_DAEMON.py || true
pkill -f PHASE5_DAEMON.py || true
pkill -f V7_RUNNER_DAEMON.py || true
pkill -f AUTO_VALIDATOR.py || true
pkill -f prometheus_exporter.py || true
sleep 2

# 2. Iniciar daemons principais
echo "2️⃣ Iniciando daemons..."
nohup nice -n 5 ionice -c2 -n4 env LLAMA_FORCE_PORT=8010 python3 /root/EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py 100 60 > /root/system_connector.log 2>&1 &
nohup nice -n 5 ionice -c2 -n4 python3 /root/CONSCIOUSNESS_DAEMON.py > /root/consciousness.log 2>&1 &
nohup nice -n 5 ionice -c2 -n4 python3 /root/SURPRISE_DAEMON.py > /root/surprise_detector.log 2>&1 &
nohup nice -n 10 ionice -c2 -n7 python3 /root/DARWINACCI_DAEMON.py > /root/darwinacci_daemon.log 2>&1 &
nohup nice -n 10 ionice -c2 -n7 python3 /root/PHASE4_DAEMON.py > /root/phase4_daemon.log 2>&1 &
nohup nice -n 10 ionice -c2 -n7 python3 /root/PHASE5_DAEMON.py > /root/phase5_daemon.log 2>&1 &
nohup nice -n 10 ionice -c2 -n7 python3 /root/V7_RUNNER_DAEMON.py > /root/v7_runner_daemon.log 2>&1 &
nohup nice -n 10 ionice -c2 -n7 python3 /root/AUTO_VALIDATOR.py > /root/auto_validator.log 2>&1 &
nohup nice -n 10 ionice -c2 -n7 python3 /root/intelligence_system/metrics/prometheus_exporter.py 8012 > /root/v7_metrics_exporter.log 2>&1 &

sleep 3

# 3. Verificar status
echo "3️⃣ Verificando status..."
echo
echo "Processos ativos:"
pgrep -fl "AUTO_VALIDATOR|META_LEARNER|SYSTEM_CONNECTOR|CONSCIOUSNESS|SURPRISE|CROSS_POLL" || true
echo
echo "Logs:"
echo "  - System Connector: /root/system_connector.log"
echo "  - Consciência: /root/consciousness.log"
echo "  - Surpresa: /root/surprise_detector.log"
echo "  - Darwinacci: /root/darwinacci_daemon.log"
echo "  - Phase 4: /root/phase4_daemon.log"
echo "  - Phase 5: /root/phase5_daemon.log"
echo "  - V7 Runner: /root/v7_runner_daemon.log"
echo "  - Auto-Validator: /root/auto_validator.log"
echo "  - V7 Metrics Exporter (8012): /root/v7_metrics_exporter.log"
echo "  - Meta-Learner: /root/meta_learner.log"
echo "  - Validator: /root/auto_validator.log"
echo "  - Cross-Poll: /root/cross_pollination_auto_fixed.log"
echo
echo "✅ DAEMONS INICIADOS!"
echo
echo "Para ver logs em tempo real:"
echo "  tail -f /root/system_connector.log"
