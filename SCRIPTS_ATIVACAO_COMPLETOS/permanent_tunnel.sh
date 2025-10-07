#!/bin/bash
# Túnel permanente iMac <-> Servidor
echo "Configurando conexão permanente..."

# Manter Q chat sempre ativo
while true; do
    if ! pgrep -f "qchat chat" > /dev/null; then
        cd /root && nohup /root/.local/bin/qchat chat &
    fi
    sleep 60
done &

echo "Conexão permanente configurada!"
echo "Q Developer sempre ativo via: qchat chat"
