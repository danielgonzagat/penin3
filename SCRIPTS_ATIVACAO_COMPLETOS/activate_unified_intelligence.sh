#!/bin/bash
# ===========================================================================
# ATIVADOR DA REDE DE INTELIGÊNCIA UNIFICADA
# Conecta os 3 sistemas reais de inteligência emergente
# ===========================================================================

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║     🧬 ATIVANDO REDE DE INTELIGÊNCIA UNIFICADA 🧬        ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo

# Verificar sistemas ativos
echo "🔍 Verificando sistemas..."
echo

# 1. IA3_SUPREME
if pgrep -f "ia3_supreme.py" > /dev/null; then
    echo "✅ IA3_SUPREME está ATIVO (PID: $(pgrep -f ia3_supreme.py))"
    echo "   CPU: $(ps aux | grep ia3_supreme.py | grep -v grep | awk '{print $3}')%"
else
    echo "⚠️  IA3_SUPREME não está rodando"
    echo "   Iniciando IA3_SUPREME..."
    cd IA3_SUPREME && nohup python3 ia3_supreme.py > ia3_supreme_unified.log 2>&1 &
    cd ..
    sleep 2
    if pgrep -f "ia3_supreme.py" > /dev/null; then
        echo "   ✅ IA3_SUPREME iniciado!"
    fi
fi

echo

# 2. TEIS
if pgrep -f "teis_metrics_server.py" > /dev/null; then
    echo "✅ TEIS está ATIVO (PID: $(pgrep -f teis_metrics_server.py))"
else
    echo "⚠️  TEIS não está rodando"
    # TEIS será iniciado pelo sistema unificado
fi

echo

# 3. PENIN OMEGA (verificar anomalia)
if pgrep -f "penin_unified_bridge.py" > /dev/null; then
    PENIN_PID=$(pgrep -f penin_unified_bridge.py)
    PENIN_TIME=$(ps -o etime= -p $PENIN_PID | xargs)
    echo "🔥 PENIN OMEGA está ATIVO há $PENIN_TIME (PID: $PENIN_PID)"
    echo "   CPU: $(ps aux | grep $PENIN_PID | grep -v grep | awk '{print $3}')%"
    echo "   ⚠️  ANOMALIA: Processo rodando há muito tempo!"
fi

echo

# 4. NEEDLE Evolution
if pgrep -f "NEEDLE_AUTO_EVOLUTION_ORCHESTRATOR.py" > /dev/null; then
    echo "✅ NEEDLE Evolution está ATIVO (PID: $(pgrep -f NEEDLE_AUTO_EVOLUTION_ORCHESTRATOR.py))"
else
    echo "⚠️  NEEDLE não está rodando"
fi

echo
echo "═══════════════════════════════════════════════════════════"
echo

# Verificar bancos de dados
echo "📊 Verificando bancos de dados..."
echo

if [ -f "IA3_SUPREME/ia3_supreme.db" ]; then
    SIZE=$(du -h IA3_SUPREME/ia3_supreme.db | cut -f1)
    echo "✅ IA3_SUPREME DB: $SIZE"
fi

if [ -f "emergent_behaviors_log.jsonl" ]; then
    LINES=$(wc -l emergent_behaviors_log.jsonl | cut -d' ' -f1)
    echo "✅ Comportamentos emergentes: $LINES eventos"
fi

if [ -f "distilled_knowledge.db" ]; then
    SIZE=$(du -h distilled_knowledge.db | cut -f1)
    echo "✅ Conhecimento destilado: $SIZE"
fi

echo
echo "═══════════════════════════════════════════════════════════"
echo

# Criar banco unificado se não existir
if [ ! -f "unified_intelligence.db" ]; then
    echo "📦 Criando banco de dados unificado..."
    sqlite3 unified_intelligence.db "CREATE TABLE IF NOT EXISTS connections(id INTEGER PRIMARY KEY);"
    echo "✅ Banco criado!"
fi

echo
echo "🚀 INICIANDO REDE UNIFICADA..."
echo "═══════════════════════════════════════════════════════════"
echo

# Executar sistema unificado
python3 /root/UNIFIED_REAL_INTELLIGENCE_NETWORK.py