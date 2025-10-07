#!/bin/bash

# Script de monitoramento de recursos para evitar desconexões
# Monitora memória e processos Python

while true; do
    # Verificar uso de memória
    MEMORY_USAGE=$(free | grep Mem | awk '{printf "%.1f", $3/$2 * 100.0}')
    
    # Se uso de memória > 80%, alertar
    if (( $(echo "$MEMORY_USAGE > 80" | bc -l) )); then
        echo "$(date): ALERTA: Uso de memória alto: ${MEMORY_USAGE}%"
        
        # Contar processos Python
        PYTHON_COUNT=$(ps aux | grep python3 | grep -v grep | wc -l)
        echo "$(date): Processos Python ativos: $PYTHON_COUNT"
        
        # Se muitos processos Python, matar os mais antigos
        if [ $PYTHON_COUNT -gt 20 ]; then
            echo "$(date): Muitos processos Python, limpando..."
            pkill -f "python3.*test_agent" 2>/dev/null || true
        fi
    fi
    
    # Verificar conexões SSH
    SSH_CONNECTIONS=$(who | wc -l)
    echo "$(date): Conexões SSH ativas: $SSH_CONNECTIONS"
    
    # Aguardar 5 minutos
    sleep 300
done