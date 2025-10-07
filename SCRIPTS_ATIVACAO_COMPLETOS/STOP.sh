#!/bin/bash
if [ -f "/root/sistema_funcional.pid" ]; then
    PID=$(cat /root/sistema_funcional.pid)
    kill $PID 2>/dev/null
    sleep 2
    rm /root/sistema_funcional.pid 2>/dev/null
    echo "✅ Parado"
else
    echo "⚠️  Não está rodando"
fi
