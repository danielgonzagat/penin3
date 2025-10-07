#!/bin/bash

echo "=========================================="
echo "🔄 UPGRADE: Unified System → Enhanced Evolution"
echo "=========================================="

# 1. Encontra e para o processo atual
echo "📍 Localizando unified_system_production.py..."
PID=$(ps aux | grep "unified_system_production.py" | grep -v grep | awk '{print $2}')

if [ ! -z "$PID" ]; then
    echo "✅ Processo encontrado: PID $PID"
    echo "🛑 Parando sistema atual..."
    kill -TERM $PID
    sleep 3
    
    # Verifica se parou
    if ps -p $PID > /dev/null; then
        echo "⚠️ Processo ainda rodando, forçando parada..."
        kill -9 $PID
    fi
    echo "✅ Sistema antigo parado"
else
    echo "ℹ️ Sistema unified_system_production não está rodando"
fi

# 2. Backup dos dados atuais
echo "💾 Fazendo backup dos dados..."
if [ -d "/root/unified_data" ]; then
    cp -r /root/unified_data /root/unified_data_backup_$(date +%Y%m%d_%H%M%S)
    echo "✅ Backup criado"
fi

# 3. Cria diretório para novo sistema
mkdir -p /root/unified_enhanced_data
mkdir -p /root/unified_enhanced_logs

# 4. Inicia o sistema Enhanced
echo "🚀 Iniciando Sistema Enhanced com Anti-Estagnação..."
nohup python3 /root/unified_system_enhanced_evolution.py > /root/unified_enhanced_logs/output.log 2>&1 &
NEW_PID=$!

echo "✅ Sistema Enhanced iniciado: PID $NEW_PID"

# 5. Verifica se está rodando
sleep 3
if ps -p $NEW_PID > /dev/null; then
    echo "✅ Sistema Enhanced rodando com sucesso!"
    echo ""
    echo "📊 Monitorar progresso:"
    echo "  tail -f /root/unified_enhanced_logs/output.log"
    echo ""
    echo "📈 Verificar quebra de plateau:"
    echo "  grep 'Plateau quebrado' /root/unified_enhanced_logs/output.log"
    echo ""
    echo "🔍 Ver fitness evolution:"
    echo "  grep 'Fitness:' /root/unified_enhanced_logs/output.log | tail -10"
else
    echo "❌ Erro ao iniciar Sistema Enhanced!"
    echo "Verificando logs..."
    tail -20 /root/unified_enhanced_logs/output.log
fi

echo "=========================================="
echo "✅ Upgrade completo!"
echo "=========================================="