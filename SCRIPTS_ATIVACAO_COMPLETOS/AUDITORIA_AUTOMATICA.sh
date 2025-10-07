#!/bin/bash

echo "🔍 AUDITORIA AUTOMÁTICA - Aguardando Ciclo 1600"
echo "PID: $(cat /root/500_cycles.pid)"
echo ""

while true; do
    CICLO=$(strings /root/system_live.log | grep "CYCLE.*ULTIMATE" | tail -1 | grep -oP 'CYCLE \K[0-9]+')
    
    echo "[$CICLO/1600] Aguardando..."
    
    if [ "$CICLO" -ge 1600 ]; then
        echo ""
        echo "🎉 CICLO 1600 ATINGIDO!"
        echo ""
        echo "════════════════════════════════════════════════════════════"
        echo "VALIDAÇÃO: O QUE EXECUTOU NO CICLO 1600?"
        echo "════════════════════════════════════════════════════════════"
        
        echo ""
        echo "1️⃣ DARWIN executou?"
        strings /root/system_live.log | grep -A5 "Darwin.*Novelty" | tail -10
        
        echo ""
        echo "2️⃣ DARWIN inicializou população?"
        strings /root/system_live.log | grep "Pop initialized" | tail -3
        
        echo ""
        echo "3️⃣ NOVELTY scores calculados?"
        strings /root/system_live.log | grep -E "Novel:|novelty" | tail -5
        
        echo ""
        echo "4️⃣ AUTO-CODING executou?"
        strings /root/system_live.log | grep -A3 "Auto-coding.*improvement" | tail -8
        
        echo ""
        echo "5️⃣ APIS consultadas?"
        strings /root/system_live.log | grep -E "Consulting|API.*OK" | tail -5
        
        echo ""
        echo "6️⃣ INCOMPLETUDE detectou estagnação?"
        strings /root/system_live.log | grep -i "stagnation\|intervention" | tail -5
        
        break
    fi
    
    sleep 30
done

echo ""
echo "════════════════════════════════════════════════════════════"
echo "📊 RESUMO FINAL"
echo "════════════════════════════════════════════════════════════"
echo "Ciclo final: $(strings /root/system_live.log | grep 'CYCLE' | tail -1)"
echo "Performance: $(strings /root/system_live.log | grep 'Avg(100)' | tail -1)"
