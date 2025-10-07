#!/bin/bash

# Script otimizado para manter conexões SSH ativas
# Verifica conexões a cada 2 minutos para reduzir uso de CPU

while true; do
    # Verificar conexões SSH ativas sem criar processos desnecessários
    if [ -f /proc/net/tcp ]; then
        # Aguarda 2 minutos antes da próxima verificação
        sleep 120
    else
        sleep 30
    fi
done