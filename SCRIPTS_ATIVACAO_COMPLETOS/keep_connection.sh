#!/bin/bash
# Conexão otimizada com o iMac
while true; do
    # Verificar se há conexões SSH ativas
    if pgrep -f "sshd.*pts" > /dev/null; then
        # Conexão ativa, apenas aguardar
        sleep 60
    else
        echo "$(date): Nenhuma conexão SSH ativa detectada"
        sleep 30
    fi
done
