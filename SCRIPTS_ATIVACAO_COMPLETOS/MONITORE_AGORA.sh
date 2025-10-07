#!/bin/bash

echo "🔍 MONITORANDO SISTEMA V7.0 - AGUARDAR 10 MINUTOS"
echo ""
echo "Sistema: PID $(cat /root/500_cycles.pid)"
echo "Ciclo atual: $(strings /root/500_cycles_output.log | grep 'CYCLE' | tail -1)"
echo ""

for i in {1..10}; do
    sleep 60
    CURRENT=$(strings /root/500_cycles_output.log | grep "CYCLE.*ULTIMATE" | tail -1)
    AVG=$(strings /root/500_cycles_output.log | grep "Avg(100)" | tail -1 | cut -d'=' -f2 | cut -d',' -f1)
    
    echo "[$i/10] $CURRENT | Avg: $AVG"
    
    # Check for engine executions
    if [ $i -eq 5 ]; then
        echo ""
        echo "✅ Verificando execuções (ciclo 1500+):"
        strings /root/500_cycles_output.log | grep -E "Auto-coding|MAML.*shot|Darwin.*evolution|Multi-modal" | tail -8
        echo ""
    fi
done

echo ""
echo "📊 RELATÓRIO FINAL (10 min):"
echo "Ciclo final: $(strings /root/500_cycles_output.log | grep 'CYCLE' | tail -1)"
echo "Performance: $(strings /root/500_cycles_output.log | grep 'Avg(100)' | tail -1)"
echo ""
echo "Engines executados:"
strings /root/500_cycles_output.log | grep -E "Auto-coding|MAML.*shot|Darwin.*Gen|Multi-modal.*process" | wc -l
echo "linhas com execuções"
