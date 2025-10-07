#!/bin/bash
# ⚡ ATIVAR DARWINACCI CEREBRUM 24/7
#
# Este script:
# 1. Ativa o Cerebrum permanentemente
# 2. Conecta TODOS os sistemas simbioticamente
# 3. Evolui continuamente genomas universais
# 4. Registra tudo no WORM universal
#
# OBJETIVO: Transformar fragmentos em ORGANISMO ÚNICO

set -e

echo "═══════════════════════════════════════════════════════════════════════════"
echo "⚡ ATIVANDO DARWINACCI CEREBRUM - CENTRO NEURAL UNIVERSAL"
echo "═══════════════════════════════════════════════════════════════════════════"

# Criar diretório de logs
mkdir -p /root/darwinacci_omega/logs
mkdir -p /root/darwinacci_omega/checkpoints

# Configurar ambiente
export DARWINACCI_CHECKPOINTS="/root/darwinacci_omega/checkpoints"
export DARWINACCI_WORM_PATH="/root/darwinacci_omega/cerebrum_worm.csv"
export DARWINACCI_WORM_HEAD="/root/darwinacci_omega/cerebrum_worm_head.txt"
export DARWINACCI_TRIALS="3"
export PYTHONUNBUFFERED=1

# PID file
PID_FILE="/root/darwinacci_omega/cerebrum.pid"
LOG_FILE="/root/darwinacci_omega/logs/cerebrum_24_7.log"

# Verificar se já está rodando
if [ -f "$PID_FILE" ]; then
    OLD_PID=$(cat "$PID_FILE")
    if ps -p "$OLD_PID" > /dev/null 2>&1; then
        echo "⚠️  Cerebrum já está rodando (PID: $OLD_PID)"
        echo "   Use 'kill $OLD_PID' para parar antes de reiniciar"
        exit 1
    fi
fi

echo "📝 Logs: $LOG_FILE"
echo "💾 Checkpoints: $DARWINACCI_CHECKPOINTS"
echo "📜 WORM Ledger: $DARWINACCI_WORM_PATH"
echo ""

# Criar script Python de execução contínua
cat > /tmp/cerebrum_continuous.py << 'PYEOF'
import sys
import logging
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, '/root')

from darwinacci_cerebrum_universal import DarwinacciCerebrum

def run_continuous():
    """Executar Cerebrum continuamente com restart automático"""
    logger.info("="*80)
    logger.info("🧠 DARWINACCI CEREBRUM - MODO CONTÍNUO 24/7")
    logger.info("="*80)
    
    cycle_count = 0
    total_evaluations = 0
    start_time = time.time()
    
    while True:
        cycle_count += 1
        try:
            logger.info(f"\n{'─'*80}")
            logger.info(f"🔄 CYCLE #{cycle_count} STARTING...")
            logger.info(f"{'─'*80}")
            
            # Criar nova instância do Cerebrum
            cerebrum = DarwinacciCerebrum(seed=42 + cycle_count)
            
            # Executar evolução (7 cycles Darwin internos)
            champion = cerebrum.evolve(cycles=7)
            
            # Status
            status = cerebrum.get_status()
            total_evaluations += status['total_evaluations']
            
            elapsed = time.time() - start_time
            evals_per_hour = (total_evaluations / elapsed) * 3600 if elapsed > 0 else 0
            
            logger.info(f"\n📊 CYCLE #{cycle_count} COMPLETE:")
            cs = 0.0
            try:
                cs = float(getattr(champion, 'score', 0.0)) if champion else 0.0
            except Exception:
                cs = 0.0
            logger.info(f"   Champion Score: {cs:.4f}")
            logger.info(f"   Total Transfers: {status['total_transfers']}")
            try:
                sr = float(status.get('success_rate', 0.0))
            except Exception:
                sr = 0.0
            logger.info(f"   Success Rate: {sr*100:.1f}%")
            logger.info(f"   Total Evaluations: {total_evaluations}")
            logger.info(f"   Eval Rate: {evals_per_hour:.1f} evals/hour")
            logger.info(f"   Elapsed: {elapsed/3600:.2f} hours")

            # Write consolidated stats JSON
            try:
                import json, pathlib
                stats = {
                    'round': cycle_count,
                    'transfers': int(status.get('total_transfers', 0)),
                    'evals': int(total_evaluations),
                    'success_rate': float(status.get('success_rate', 0.0)),
                    'ts': int(time.time()),
                }
                pathlib.Path('/root/darwinacci_omega/cerebrum_stats.json').write_text(json.dumps(stats))
            except Exception:
                pass
            
            # Pequena pausa entre ciclos
            time.sleep(2)
            
        except KeyboardInterrupt:
            logger.info("\n⚠️  User interrupt detected. Shutting down gracefully...")
            break
        except Exception as e:
            logger.error(f"\n❌ ERROR in cycle #{cycle_count}: {e}", exc_info=True)
            logger.info("   Restarting in 5 seconds...")
            time.sleep(5)
            continue
    
    logger.info(f"\n✅ CEREBRUM SHUTDOWN COMPLETE")
    logger.info(f"   Total Cycles: {cycle_count}")
    logger.info(f"   Total Evaluations: {total_evaluations}")
    logger.info(f"   Total Runtime: {(time.time() - start_time)/3600:.2f} hours")

if __name__ == '__main__':
    run_continuous()
PYEOF

# Executar em background
echo "🚀 Starting Cerebrum in background..."
nohup python /tmp/cerebrum_continuous.py > "$LOG_FILE" 2>&1 &
CEREBRUM_PID=$!

# Salvar PID
echo "$CEREBRUM_PID" > "$PID_FILE"

echo ""
echo "✅ Cerebrum ATIVADO!"
echo "   PID: $CEREBRUM_PID"
echo "   Logs: tail -f $LOG_FILE"
echo "   Stop: kill $CEREBRUM_PID"
echo ""
echo "═══════════════════════════════════════════════════════════════════════════"

# Mostrar primeiros logs
sleep 3
echo ""
echo "📋 PRIMEIROS LOGS:"
echo "─────────────────────────────────────────────────────────────────────────"
tail -20 "$LOG_FILE"
echo "─────────────────────────────────────────────────────────────────────────"