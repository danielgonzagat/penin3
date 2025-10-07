#!/bin/bash
################################################################################
# 📊 STATUS INTELLIGENCE - Ver status do sistema
################################################################################

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 STATUS DA INTELIGÊNCIA SUPREMA 24/7"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Verificar se está rodando
if [ ! -f /root/inteligencia_suprema_24_7.pid ]; then
    echo "❌ Sistema NÃO está ativo"
    echo ""
    echo "   Para ativar:"
    echo "   ./ACTIVATE_THE_INTELLIGENCE.sh"
    echo ""
    exit 1
fi

PID=$(cat /root/inteligencia_suprema_24_7.pid)

if ! ps -p $PID > /dev/null 2>&1; then
    echo "❌ Sistema NÃO está ativo (PID $PID não existe)"
    rm -f /root/inteligencia_suprema_24_7.pid
    echo ""
    echo "   Para ativar:"
    echo "   ./ACTIVATE_THE_INTELLIGENCE.sh"
    echo ""
    exit 1
fi

echo "✅ Sistema ATIVO!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📊 MÉTRICAS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Buscar últimas métricas do database
python3 << 'EOPY'
import sqlite3
import time

try:
    conn = sqlite3.connect('/root/inteligencia_suprema_24_7.db')
    cursor = conn.cursor()
    
    # Última linha
    cursor.execute('''
        SELECT cycle, timestamp, performance, consciousness, emergences,
               neural_farm_gen, teis_episode, api_calls
        FROM cycles
        ORDER BY cycle DESC
        LIMIT 1
    ''')
    
    row = cursor.fetchone()
    
    if row:
        cycle, ts, perf, cons, emerg, nf_gen, teis_ep, api_calls = row
        uptime = time.time() - ts
        
        print(f"   PID: {open('/root/inteligencia_suprema_24_7.pid').read().strip()}")
        print(f"   Ciclo atual: {cycle:,}")
        print(f"   Uptime: {uptime / 3600:.2f}h")
        print(f"   Performance: {perf:.2f}")
        print(f"   Consciência: {cons:.1%}")
        print(f"   Neural Farm: geração {nf_gen}")
        print(f"   TEIS: episódio {teis_ep}")
        print(f"   Emergências: {emerg}")
        print(f"   API calls: {api_calls}")
    else:
        print("   ⚠️  Ainda não há dados (sistema acabou de iniciar)")
    
    conn.close()
    
except Exception as e:
    print(f"   ⚠️  Erro ao ler métricas: {e}")

EOPY

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "📝 ÚLTIMAS LINHAS DO LOG"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

tail -20 /root/inteligencia_suprema_24_7.log 2>/dev/null || echo "   (sem log ainda)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
