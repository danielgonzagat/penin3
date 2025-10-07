#!/bin/bash
# FIX PRIORIDADE #2: CORRIGIR SURPRISE DETECTOR
# ===============================================
# PROBLEMA: Detector inicia mas sai imediatamente (log vazio)
# SOLUÇÃO: Rodar diretamente com Python inline (sem daemon wrapper)

set -e

echo "🔍 FIX #2: CORRIGINDO SURPRISE DETECTOR"
echo "======================================="
echo

# 1. Parar daemon atual (se existir)
echo "1️⃣ Parando versão antiga..."
pkill -f SURPRISE_DAEMON || true
pkill -f "EMERGENCE_CATALYST_1_SURPRISE" || true
sleep 2

# 2. Limpar log antigo
if [ -f /root/surprise_detector.log ]; then
    echo "2️⃣ Limpando log antigo..."
    > /root/surprise_detector.log
fi

# 3. Iniciar detector com Python inline (mais robusto)
echo "3️⃣ Iniciando Surprise Detector..."

python3 -c "
import sys
import os

# Redirecionar stdout/stderr para garantir logs
sys.stdout = sys.stderr

# Path setup
sys.path.insert(0, '/root')
os.chdir('/root')

# Imports
from EMERGENCE_CATALYST_1_SURPRISE_DETECTOR import SurpriseDetector

# Criar detector
detector = SurpriseDetector()

# Log de inicio com flush
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', flush=True)
print('🔍 SURPRISE DETECTOR INICIADO', flush=True)
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', flush=True)
print('', flush=True)

# Rodar forever
try:
    detector.run_forever(interval=60)
except KeyboardInterrupt:
    print('⏹️  Detector parado pelo usuário', flush=True)
except Exception as e:
    print(f'❌ ERRO FATAL: {e}', flush=True)
    import traceback
    traceback.print_exc()
    raise
" > /root/surprise_detector.log 2>&1 &

DETECTOR_PID=$!
echo "   PID: $DETECTOR_PID"
echo

# 4. Aguardar inicio
echo "4️⃣ Aguardando detector iniciar..."
sleep 8

# 5. Verificar se está rodando
if pgrep -f "EMERGENCE_CATALYST_1_SURPRISE" > /dev/null; then
    echo "✅ Detector ATIVO!"
    pgrep -fl "EMERGENCE_CATALYST_1_SURPRISE"
    echo
else
    echo "❌ Detector falhou ao iniciar!"
    echo
    echo "Log de erro:"
    cat /root/surprise_detector.log
    exit 1
fi

# 6. Mostrar log inicial
echo "5️⃣ Log Detector (primeiras linhas):"
echo "==================================="
tail -n 50 /root/surprise_detector.log
echo

# 7. Instruções
echo "╔════════════════════════════════════════════╗"
echo "║  ✅ SURPRISE DETECTOR FUNCIONANDO!        ║"
echo "╚════════════════════════════════════════════╝"
echo
echo "📊 Para acompanhar detecções:"
echo "   tail -f /root/surprise_detector.log"
echo
echo "🔍 Verificar surpresas detectadas:"
echo "   sqlite3 /root/emergence_surprises.db 'SELECT * FROM surprises ORDER BY surprise_score DESC LIMIT 10;'"
echo
echo "📈 Ver baselines aprendidos:"
echo "   sqlite3 /root/emergence_surprises.db 'SELECT * FROM baselines;'"
echo
echo "🎯 PRÓXIMO PASSO: Aguardar 30min e executar validação"
echo "   bash /root/VALIDAR_SISTEMA_POS_FIX.sh"
