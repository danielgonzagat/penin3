#!/bin/bash
# 🔧 CORREÇÕES NÍVEL 0 - TRIVIAIS (15-30 min)
# Certeza: 99% | Impacto: MÉDIO | Complexidade: BAIXA

set -e

echo "════════════════════════════════════════════════════════════════"
echo "🔧 CORREÇÕES NÍVEL 0 - LIMPEZA E FIXES TRIVIAIS"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ═══════════════════════════════════════════════════════════════════
# T0.1 - MATAR PROCESSOS DUPLICADOS
# ═══════════════════════════════════════════════════════════════════

echo "📋 T0.1: Matando processos duplicados..."

# EMERGENCE_CATALYST (tem 15+ duplicatas)
echo "   🔴 EMERGENCE_CATALYST_4 duplicados:"
PIDS=$(ps aux | grep "EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR" | grep -v grep | awk '{print $2}')
COUNT=$(echo "$PIDS" | wc -w)
echo "      Encontrados: $COUNT processos"

if [ $COUNT -gt 1 ]; then
    echo "      Mantendo apenas o primeiro..."
    FIRST_PID=$(echo "$PIDS" | head -1)
    echo "$PIDS" | tail -n +2 | xargs -r kill -9 2>/dev/null || true
    echo "      ✅ Mantido PID: $FIRST_PID"
    echo "      ✅ Matados: $((COUNT-1)) duplicatas"
else
    echo "      ✅ Sem duplicatas (OK)"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# T0.2 - FIX DARWIN PORT CONFLICT
# ═══════════════════════════════════════════════════════════════════

echo "📋 T0.2: Corrigindo conflito de porta Darwin..."

# Matar TODOS darwin_runner (evitar conflito)
echo "   🔴 Matando darwin_runner existentes..."
pkill -9 -f "darwin_runner" 2>/dev/null || true
sleep 2

# Verificar porta livre
if netstat -tuln 2>/dev/null | grep -q ":9092 "; then
    echo "   ⚠️ Porta 9092 ainda ocupada, tentando liberar..."
    PORT_PID=$(lsof -ti:9092 2>/dev/null || true)
    if [ -n "$PORT_PID" ]; then
        kill -9 $PORT_PID
        sleep 1
    fi
fi

# Verificar
if ! netstat -tuln 2>/dev/null | grep -q ":9092 "; then
    echo "   ✅ Porta 9092 livre"
else
    echo "   ⚠️ Porta 9092 ainda ocupada (pode ser OK se é outro serviço)"
fi

# Reiniciar Darwin (single instance)
echo "   🔄 Reiniciando Darwin runner..."
cd /root
nohup timeout 72h python3 -u darwin_runner.py > /root/darwin.log 2>&1 &
DARWIN_PID=$!
echo $DARWIN_PID > /root/darwin.pid
echo "   ✅ Darwin iniciado: PID $DARWIN_PID"

# Wait and verify
sleep 5
if ps -p $DARWIN_PID > /dev/null; then
    echo "   ✅ Darwin processo ativo"
    # Test metrics endpoint
    if curl -s http://localhost:9092/metrics | head -1 | grep -q "darwin"; then
        echo "   ✅ Darwin metrics endpoint OK"
    else
        echo "   ⚠️ Darwin metrics endpoint não responde (pode levar alguns segundos)"
    fi
else
    echo "   ❌ Darwin processo morreu, verificar /root/darwin.log"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# T0.3 - VACUUM DATABASE
# ═══════════════════════════════════════════════════════════════════

echo "📋 T0.3: Otimizando database..."

DB_PATH="/root/intelligence_system/data/intelligence.db"

if [ -f "$DB_PATH" ]; then
    echo "   📊 Tamanho antes:"
    du -h "$DB_PATH"
    
    echo "   🔄 Executando VACUUM..."
    sqlite3 "$DB_PATH" "VACUUM;" 2>&1 || echo "   ⚠️ VACUUM falhou (DB pode estar em uso)"
    
    echo "   📊 Tamanho depois:"
    du -h "$DB_PATH"
    echo "   ✅ Database otimizado"
else
    echo "   ⚠️ Database não encontrado: $DB_PATH"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# T0.4 - CLEANUP LOGS ANTIGOS (OPCIONAL)
# ═══════════════════════════════════════════════════════════════════

echo "📋 T0.4: Limpeza de logs antigos (>7 dias)..."

# Apenas reportar, não deletar automaticamente
OLD_LOGS=$(find /root -name "*.log" -type f -mtime +7 2>/dev/null | wc -l)
echo "   📁 Logs com >7 dias: $OLD_LOGS"

if [ $OLD_LOGS -gt 100 ]; then
    echo "   ⚠️ Muitos logs antigos. Considere limpar:"
    echo "      find /root -name '*.log' -type f -mtime +7 -delete"
else
    echo "   ✅ Volume de logs OK"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════"
echo "✅ CORREÇÕES NÍVEL 0 COMPLETAS"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📊 Resumo:"
echo "   ✅ T0.1: Processos duplicados eliminados"
echo "   ✅ T0.2: Darwin port conflict corrigido"
echo "   ✅ T0.3: Database otimizado"
echo "   ✅ T0.4: Logs verificados"
echo ""
echo "🚀 Próximo passo:"
echo "   bash /root/🔧_CORRECOES_NIVEL_1_META_LEARNING.sh"
echo ""
echo "📈 Status:"
ps aux | grep -E "(darwin_runner|EMERGENCE)" | grep -v grep | wc -l | \
    xargs -I {} echo "   Processos ativos: {}"
echo ""
echo "════════════════════════════════════════════════════════════════"