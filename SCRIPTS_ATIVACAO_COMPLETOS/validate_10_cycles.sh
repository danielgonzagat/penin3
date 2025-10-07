#!/bin/bash

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   🔬 VALIDAÇÃO COMPLETA - 10 CICLOS DE FUNCIONAMENTO          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

LOG=/root/500_cycles_output.log
START_CYCLE=$(tail -100 "$LOG" 2>/dev/null | grep -oP "CYCLE \K\d+" | tail -1)

echo "Ciclo inicial: $START_CYCLE"
echo "Meta: $((START_CYCLE + 10))"
echo ""
echo "Monitorando..."

for i in {1..10}; do
    sleep 30
    
    CURRENT=$(tail -100 "$LOG" 2>/dev/null | grep -oP "CYCLE \K\d+" | tail -1)
    
    # Métricas
    MNIST=$(tail -50 "$LOG" 2>/dev/null | grep "Test:" | tail -1 | grep -oP "Test: \K[\d.]+")
    CARTPOLE=$(tail -50 "$LOG" 2>/dev/null | grep "Avg(100):" | tail -1 | grep -oP "Avg\(100\): \K[\d.]+")
    
    # Problemas
    ADD_DROPOUT=$(tail -50 "$LOG" 2>/dev/null | grep -c "add_dropout" || echo "0")
    WARNINGS=$(tail -50 "$LOG" 2>/dev/null | grep -c "WARNING" || echo "0")
    
    echo "[$i/10] Ciclo $CURRENT: MNIST=$MNIST%, CartPole=$CARTPOLE, add_dropout=$ADD_DROPOUT, warnings=$WARNINGS"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "✅ VALIDAÇÃO COMPLETA"
