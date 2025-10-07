#!/bin/bash

echo "🧠 UNIFICAÇÃO DEFINITIVA - CÉREBRO ÚNICO"
echo "========================================"

# Parar TODOS os sistemas antigos
echo "⏹️ Parando todos os sistemas antigos..."

# Parar processos falcon_
pkill -9 -f falcon_
echo "✅ Processos falcon_ parados"

# Parar processos penin_
pkill -9 -f penin_
echo "✅ Processos penin_ parados"

# Parar processos iaaa_
pkill -9 -f iaaa_
echo "✅ Processos iaaa_ parados"

# Aguardar um pouco
sleep 3

# Verificar se ainda há processos
echo "🔍 Verificando processos restantes..."
RESTANTES=$(ps aux | grep -E "(falcon_|iaaa_|penin_)" | grep -v grep | wc -l)

if [ $RESTANTES -gt 0 ]; then
    echo "⚠️ Ainda há $RESTANTES processos antigos rodando"
    echo "📋 Lista de processos restantes:"
    ps aux | grep -E "(falcon_|iaaa_|penin_)" | grep -v grep
else
    echo "✅ Todos os sistemas antigos foram parados"
fi

# Iniciar cérebro único
echo "🚀 Iniciando cérebro único..."
nohup python3 /root/cerebro_unico_daemon.py > /tmp/cerebro_unico_final.log 2>&1 &

# Aguardar inicialização
sleep 5

# Verificar se cérebro único está rodando
CEREBRO_PID=$(pgrep -f cerebro_unico_daemon.py)

if [ ! -z "$CEREBRO_PID" ]; then
    echo "✅ Cérebro único ativo (PID: $CEREBRO_PID)"
    echo "📝 Logs em: /tmp/cerebro_unico_final.log"
    
    # Mostrar status final
    echo ""
    echo "📊 STATUS FINAL DO SISTEMA UNIFICADO:"
    echo "====================================="
    echo "🧠 Cérebro único: ATIVO (PID: $CEREBRO_PID)"
    echo "💾 Memória unificada: $(ls -lh /root/unified_memory.db | awk '{print $5}')"
    echo "📝 Logs: /tmp/cerebro_unico_final.log"
    
    # Verificar sistemas antigos
    ANTIGOS=$(ps aux | grep -E "(falcon_|iaaa_|penin_)" | grep -v grep | wc -l)
    echo "🔄 Sistemas antigos: $ANTIGOS processos"
    
    if [ $ANTIGOS -eq 0 ]; then
        echo ""
        echo "🎉 UNIFICAÇÃO 100% CONCLUÍDA!"
        echo "✅ Sistema funcionando como cérebro único"
    else
        echo ""
        echo "⚠️ Ainda há $ANTIGOS sistemas antigos rodando"
        echo "🔄 Execute novamente se necessário"
    fi
    
else
    echo "❌ Erro ao iniciar cérebro único"
    echo "📝 Verifique os logs: /tmp/cerebro_unico_final.log"
fi

echo ""
echo "🔧 COMANDOS ÚTEIS:"
echo "=================="
echo "Ver status: ps aux | grep cerebro_unico"
echo "Ver logs: tail -f /tmp/cerebro_unico_final.log"
echo "Parar: pkill -f cerebro_unico_daemon.py"
echo "Testar: python3 testar_cerebro_unico.py test"