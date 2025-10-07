#!/bin/bash

LOG=/root/500_cycles_output.log
REPORT=/root/FINAL_500_CYCLES_REPORT.txt

echo "🔍 Monitorando até completar 500 ciclos..."
echo "PID: $(cat /root/500_cycles.pid)"
echo "Log: $LOG"
echo ""

while true; do
    sleep 30
    
    # Verifica se processo ainda vivo
    PID=$(cat /root/500_cycles.pid 2>/dev/null)
    if ! ps -p $PID > /dev/null 2>&1; then
        echo "❌ Processo morreu!"
        break
    fi
    
    # Ciclo atual
    CURRENT=$(tail -100 "$LOG" 2>/dev/null | grep -oP "CYCLE \K\d+" | tail -1)
    
    if [ -z "$CURRENT" ]; then
        continue
    fi
    
    echo "[$(date +%H:%M:%S)] Ciclo $CURRENT/1560"
    
    # Completou?
    if [ "$CURRENT" -ge 1560 ]; then
        echo ""
        echo "🎉 500 CICLOS COMPLETADOS!"
        break
    fi
done

echo ""
echo "Gerando relatório final..."

cat > $REPORT << 'REPORT'
╔════════════════════════════════════════════════════════════════╗
║          🎉 500 CICLOS COMPLETADOS - RELATÓRIO FINAL          ║
╚════════════════════════════════════════════════════════════════╝

REPORT

# Adiciona estatísticas
python3 - << 'PY' >> $REPORT
import re

with open('/root/500_cycles_output.log', 'r') as f:
    logs = f.readlines()

cycles = [int(m.group(1)) for line in logs for m in [re.search(r'CYCLE (\d+)', line)] if m]
print(f"Ciclos executados: {len(cycles)}")
print(f"Range: {min(cycles) if cycles else 0} - {max(cycles) if cycles else 0}")
print()

mnist_count = len([l for l in logs if 'Training MNIST' in l])
print(f"MNIST treinos: {mnist_count}")

darwin_count = len([l for l in logs if 'Darwin evolution' in l])
print(f"Darwin ativações: {darwin_count}")

api_success = len([l for l in logs if 'Consulted' in l and 'APIs' in l])
print(f"API sucessos: {api_success}")

warnings = len([l for l in logs if 'WARNING' in l])
errors = len([l for l in logs if 'ERROR' in l])
print()
print(f"Warnings: {warnings}")
print(f"Errors: {errors}")
PY

cat $REPORT
echo ""
echo "✅ Relatório salvo em: $REPORT"
