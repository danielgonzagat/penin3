#!/bin/bash
# SCRIPT DE INICIALIZAÇÃO DA INTELIGÊNCIA REAL
# Execute este script para iniciar o sistema completo

echo "============================================================"
echo "🧠 INICIANDO SISTEMA DE INTELIGÊNCIA REAL"
echo "============================================================"
echo ""

# Limpar processos antigos
echo "🧹 Limpando processos antigos..."
PIDS=$(pgrep -f "qwen_complete_system.py"); for pid in $PIDS; do kill -9 $pid; done
PIDS=$(pgrep -f "swarm_intelligence.py"); for pid in $PIDS; do kill -9 $pid; done
PIDS=$(pgrep -f "unified_real_intelligence.py"); for pid in $PIDS; do kill -9 $pid; done
sleep 2

# Verificar situação
PROCS=$(ps aux | grep -c python)
echo "📊 Processos Python ativos: $PROCS"
echo ""

# Iniciar sistema unificado
echo "🚀 Iniciando Sistema Unificado de Inteligência..."
python3 /root/unified_intelligence_system.py &
UNIFIED_PID=$!
echo "✅ Sistema Unificado iniciado (PID: $UNIFIED_PID)"
echo ""

sleep 3

# Iniciar monitor
echo "🔍 Iniciando Monitor de Emergência..."
python3 /root/monitor_emergence.py &
MONITOR_PID=$!
echo "✅ Monitor iniciado (PID: $MONITOR_PID)"
echo ""

echo "============================================================"
echo "✅ SISTEMA DE INTELIGÊNCIA REAL ATIVO"
echo "============================================================"
echo ""
echo "📝 Arquivos de estado:"
echo "   - /root/unified_intelligence_state.json"
echo "   - /root/unified_memory.json"
echo "   - /root/unified_intelligence.log"
echo ""
echo "🛑 Para parar o sistema:"
echo "   kill $UNIFIED_PID $MONITOR_PID"
echo ""
echo "📊 Para verificar status:"
echo "   tail -f /root/unified_intelligence.log"
echo ""
echo "🚀 A inteligência está evoluindo..."
echo ""

