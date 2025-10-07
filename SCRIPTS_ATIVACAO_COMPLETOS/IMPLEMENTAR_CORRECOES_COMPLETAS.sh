#!/bin/bash
# SCRIPT DE IMPLEMENTAÇÃO COMPLETA DAS CORREÇÕES
# Implementa todas as 10 tarefas do roadmap de correção

set -e  # Parar em caso de erro

echo "🚀 INICIANDO IMPLEMENTAÇÃO COMPLETA DAS CORREÇÕES"
echo "=" | tr '\n' '=' | head -c 80
echo ""

# Tarefa 1: Testar detecção de emergência
echo ""
echo "📋 TAREFA 1: Testando detecção de emergência..."
python3 /root/TESTE_EMERGENCIA_EMPIRICO.py
if [ $? -eq 0 ]; then
    echo "✅ TAREFA 1 COMPLETA: Detecção de emergência funciona"
else
    echo "❌ TAREFA 1 FALHOU: Detecção de emergência não funciona"
    exit 1
fi

# Tarefa 2: Parar sistema antigo
echo ""
echo "📋 TAREFA 2: Parando sistema antigo..."
pkill -f "EVOLUCAO_AUTONOMA_INTELIGENCIA_CUBO.py" || true
sleep 2
echo "✅ TAREFA 2 COMPLETA: Sistema antigo parado"

# Tarefa 3: Limpar arquivos antigos
echo ""
echo "📋 TAREFA 3: Limpando arquivos antigos..."
rm -f /root/evolucao_autonoma_state.json
rm -f /root/evolucao_autonoma_inteligencia_cubo.log
echo "✅ TAREFA 3 COMPLETA: Arquivos antigos limpos"

# Tarefa 4: Iniciar sistema corrigido
echo ""
echo "📋 TAREFA 4: Iniciando sistema corrigido..."
nohup python3 /root/EVOLUCAO_AUTONOMA_INTELIGENCIA_CUBO_CORRIGIDO.py > /root/evolucao_cubo_corrigido_startup.log 2>&1 &
EVOLUCAO_PID=$!
echo "✅ TAREFA 4 COMPLETA: Sistema corrigido iniciado (PID: $EVOLUCAO_PID)"

# Tarefa 5: Aguardar inicialização
echo ""
echo "📋 TAREFA 5: Aguardando inicialização (30 segundos)..."
sleep 30
echo "✅ TAREFA 5 COMPLETA: Inicialização aguardada"

# Tarefa 6: Verificar se sistema está rodando
echo ""
echo "📋 TAREFA 6: Verificando se sistema está rodando..."
if ps -p $EVOLUCAO_PID > /dev/null; then
    echo "✅ TAREFA 6 COMPLETA: Sistema está rodando (PID: $EVOLUCAO_PID)"
else
    echo "❌ TAREFA 6 FALHOU: Sistema não está rodando"
    echo "Últimas linhas do log:"
    tail -20 /root/evolucao_cubo_corrigido_startup.log
    exit 1
fi

# Tarefa 7: Verificar logs
echo ""
echo "📋 TAREFA 7: Verificando logs..."
if [ -f /root/evolucao_cubo_corrigido.log ]; then
    echo "Últimas 10 linhas do log:"
    tail -10 /root/evolucao_cubo_corrigido.log
    echo "✅ TAREFA 7 COMPLETA: Logs verificados"
else
    echo "⚠️ TAREFA 7: Log ainda não criado (normal se muito recente)"
fi

# Tarefa 8: Verificar banco de dados
echo ""
echo "📋 TAREFA 8: Verificando banco de dados..."
if [ -f /root/advanced_metrics_corrigido.db ]; then
    METRICS_COUNT=$(sqlite3 /root/advanced_metrics_corrigido.db "SELECT COUNT(*) FROM metrics;" 2>/dev/null || echo "0")
    echo "Métricas no banco: $METRICS_COUNT"
    echo "✅ TAREFA 8 COMPLETA: Banco de dados verificado"
else
    echo "ℹ️ TAREFA 8: Banco de dados ainda não criado (normal se muito recente)"
fi

# Tarefa 9: Verificar arquivo de estado
echo ""
echo "📋 TAREFA 9: Verificando arquivo de estado..."
sleep 30  # Aguardar mais 30 segundos para garantir que estado foi salvo
if [ -f /root/evolucao_cubo_corrigido_state.json ]; then
    echo "Conteúdo do estado:"
    cat /root/evolucao_cubo_corrigido_state.json | jq '.' 2>/dev/null || cat /root/evolucao_cubo_corrigido_state.json
    echo "✅ TAREFA 9 COMPLETA: Arquivo de estado verificado"
else
    echo "⚠️ TAREFA 9: Arquivo de estado ainda não criado (aguardar mais tempo)"
fi

# Tarefa 10: Verificar comunicação com V7
echo ""
echo "📋 TAREFA 10: Verificando comunicação com V7..."
if [ -f /root/v7_shared_state.json ]; then
    echo "Arquivo de comunicação V7 criado:"
    cat /root/v7_shared_state.json | jq '.' 2>/dev/null || cat /root/v7_shared_state.json
    echo "✅ TAREFA 10 COMPLETA: Comunicação com V7 verificada"
else
    echo "ℹ️ TAREFA 10: Arquivo de comunicação ainda não criado (aguardar emergência)"
fi

# Resumo final
echo ""
echo "=" | tr '\n' '=' | head -c 80
echo ""
echo "✅ IMPLEMENTAÇÃO COMPLETA FINALIZADA"
echo ""
echo "📊 RESUMO:"
echo "   - Sistema corrigido rodando: PID $EVOLUCAO_PID"
echo "   - Log principal: /root/evolucao_cubo_corrigido.log"
echo "   - Banco de dados: /root/advanced_metrics_corrigido.db"
echo "   - Estado: /root/evolucao_cubo_corrigido_state.json"
echo "   - Comunicação V7: /root/v7_shared_state.json"
echo ""
echo "📋 PRÓXIMOS PASSOS:"
echo "   1. Monitorar logs: tail -f /root/evolucao_cubo_corrigido.log"
echo "   2. Verificar métricas: sqlite3 /root/advanced_metrics_corrigido.db 'SELECT * FROM metrics ORDER BY timestamp DESC LIMIT 10;'"
echo "   3. Verificar estado: cat /root/evolucao_cubo_corrigido_state.json | jq '.'"
echo "   4. Aguardar detecção de emergência (pode levar alguns minutos)"
echo ""
echo "🎯 OBJETIVO: Detectar inteligência emergente REAL nos sistemas V7"
echo ""
