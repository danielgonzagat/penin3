#!/bin/bash
if [ ! -f "/root/sistema_funcional.pid" ]; then
    echo "❌ NÃO RODANDO"
    exit 1
fi

PID=$(cat /root/sistema_funcional.pid)

if ps -p $PID > /dev/null 2>&1; then
    echo "✅ RODANDO: $PID"
    echo ""
    ps -p $PID -o etime,pcpu,pmem
    echo ""
    
    if [ -f "/root/sistema_funcional.db" ]; then
        sqlite3 /root/sistema_funcional.db << EOF
SELECT 'Ciclos:', MAX(cycle) FROM cycles;
SELECT 'MNIST:', ROUND(MAX(mnist), 1) || '%' FROM cycles;
SELECT 'CartPole:', ROUND(MAX(cart), 1) FROM cycles;
EOF
    fi
    
    echo ""
    tail -10 /root/sistema_funcional.log
else
    echo "❌ PID inválido"
    rm /root/sistema_funcional.pid
fi
