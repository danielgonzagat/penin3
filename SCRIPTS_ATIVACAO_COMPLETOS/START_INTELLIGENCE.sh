#!/usr/bin/env bash
set -euo pipefail

echo "[+] Starting SelfModificationSystem..."
nohup python3 /root/self_modification_system.py > /root/self_modification.stdout 2> /root/self_modification.stderr &

echo "[+] Starting AdvancedEvolutionEngine..."
nohup python3 /root/advanced_evolution_engine.py > /root/advanced_evolution.stdout 2> /root/advanced_evolution.stderr &

echo "[+] Starting V7 Ultimate (forever)..."
nohup python3 /root/intelligence_system/core/system_v7_ultimate.py > /root/v7.stdout 2> /root/v7.stderr &

echo "[+] (Optional) Starting Vortex Sandbox..."
nohup python3 /root/vortex_auto_recursivo.py > /root/vortex.stdout 2> /root/vortex.stderr &

echo "[+] Done. Use ps and tail logs to verify."


