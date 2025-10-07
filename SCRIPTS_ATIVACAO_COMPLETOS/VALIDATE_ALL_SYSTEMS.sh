#!/bin/bash
# VALIDAÇÃO COMPLETA DE TODOS SISTEMAS

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║                                                           ║"
echo "║   ✅ VALIDAÇÃO COMPLETA - SISTEMAS REAIS                 ║"
echo "║                                                           ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

echo "🔍 VERIFICANDO PROCESSOS ATIVOS..."
echo ""

# 1. Darwin
if ps -p 1457787 > /dev/null 2>&1; then
    CPU=$(ps -p 1457787 -o %cpu --no-headers)
    echo "✅ Darwin (PID 1457787): CPU ${CPU}%"
else
    echo "❌ Darwin NÃO está rodando!"
fi

# 2. Surprise Detector
if pgrep -f "EMERGENCE_CATALYST_1" > /dev/null; then
    PID=$(pgrep -f "EMERGENCE_CATALYST_1")
    echo "✅ Surprise Detector (PID $PID): ATIVO"
else
    echo "⚠️  Surprise Detector não detectado"
fi

# 3. Meta-Learner
if pgrep -f "META_LEARNER" > /dev/null; then
    PID=$(pgrep -f "META_LEARNER")
    echo "✅ Meta-Learner (PID $PID): ATIVO"
else
    echo "⚠️  Meta-Learner não detectado"
fi

# 4. Cross-Pollination Auto
if pgrep -f "CROSS_POLLINATION_AUTO" > /dev/null; then
    PID=$(pgrep -f "CROSS_POLLINATION_AUTO")
    echo "✅ Cross-Pollination Auto (PID $PID): ATIVO"
else
    echo "⚠️  Cross-Pollination Auto não detectado"
fi

# 5. System Connector
if ps -p 112404 > /dev/null 2>&1; then
    echo "✅ System Connector (PID 112404): ATIVO"
else
    echo "⚠️  System Connector não detectado"
fi

echo ""
echo "🗄️  VERIFICANDO DATABASES..."
echo ""

for db in /root/emergence_surprises.db /root/meta_learning.db /root/system_connections.db; do
    if [ -f "$db" ]; then
        SIZE=$(du -h "$db" | cut -f1)
        echo "✅ $(basename $db): $SIZE"
    else
        echo "❌ $(basename $db): NÃO EXISTE"
    fi
done

echo ""
echo "📝 VERIFICANDO LOGS..."
echo ""

for log in /root/meta_learner_output.log /root/cross_pollination_auto_output.log /root/surprise_detector_continuous.log; do
    if [ -f "$log" ]; then
        LINES=$(wc -l < "$log")
        echo "✅ $(basename $log): $LINES linhas"
    else
        echo "⚠️  $(basename $log): não criado ainda"
    fi
done

echo ""
echo "🔥 VERIFICANDO MUTATION STORM..."
if grep -q "mutation_rate.*1.0.*STORM" /root/darwin-engine-intelligence/core/darwin_engine_real.py; then
    echo "✅ Mutation Storm ATIVO (mutation_rate = 1.0)"
else
    echo "❌ Mutation Storm NÃO aplicado"
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "VALIDAÇÃO COMPLETA!"
echo "════════════════════════════════════════════════════════"
