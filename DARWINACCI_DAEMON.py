#!/usr/bin/env python3
"""
Darwinacci Daemon - Symbiotic Connector
- Evolves Darwinacci in short bursts
- Reads feedback signals (V7, Brain)
- Publishes best genome + stats for V7 bridge

Usage:
  python3 DARWINACCI_DAEMON.py [interval_seconds]
  python3 DARWINACCI_DAEMON.py --once

Env (optional):
  DARWINACCI_POP=24
  DARWINACCI_CYCLES=3
  DARWINACCI_SEED=77
"""
import os
import sys
import time
import json
import argparse
import logging
from datetime import datetime

# Ensure package resolves
sys.path.insert(0, '/root')

from intelligence_system.core.darwinacci_hub import (
    get_orchestrator,
    evolve_once,
    write_transfer,
    read_v7_feedback,
    read_brain_sync_signal,
    apply_feedback_to_engine,
)

def validate_feedback(feedback: dict) -> bool:
    """Valida feedback antes de aplicar"""
    required_keys = ['directive', 'mutation_rate_delta', 'reason']
    
    # Verificar chaves obrigatórias
    if not all(key in feedback for key in required_keys):
        return False
        
    # Validar valores
    if not isinstance(feedback.get('mutation_rate_delta'), (int, float)):
        return False
        
    if feedback.get('mutation_rate_delta', 0) < -0.5 or feedback.get('mutation_rate_delta', 0) > 0.5:
        return False
        
    return True

LOG_PATH = '/root/darwinacci_daemon.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def _now_iso() -> str:
    try:
        return datetime.utcnow().isoformat()
    except Exception:
        return ''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('interval', nargs='?', type=int, default=120,
                        help='Seconds between evolution cycles (default: 120)')
    parser.add_argument('--once', action='store_true', help='Run one cycle and exit')
    args = parser.parse_args()

    pop = int(os.getenv('DARWINACCI_POP', '24'))
    cyc = int(os.getenv('DARWINACCI_CYCLES', '3'))
    seed = int(os.getenv('DARWINACCI_SEED', '77'))

    logger.info('🧠 Darwinacci Daemon INICIADO')
    logger.info(f'   interval={args.interval}s pop={pop} cycles={cyc} seed={seed}')

    # Initialize orchestrator singleton with tuned parameters
    # Enable Prometheus exporter in engine via env
    os.environ.setdefault('DARWINACCI_PROMETHEUS', '1')
    os.environ.setdefault('DARWINACCI_PROM_PORT', '8011')
    get_orchestrator(activate=True, population_size=pop, max_cycles=cyc, seed=seed)

    def run_cycle():
        # Merge feedbacks
        v7_fb = read_v7_feedback() or {}
        brain = read_brain_sync_signal() or {}
        feedback = dict(v7_fb)
        if brain:
            feedback['brain_hint'] = brain
        if feedback:
            # Validar feedback antes de aplicar
            if validate_feedback(feedback):
                applied = apply_feedback_to_engine(feedback)
                logger.info(f'🔁 Feedback aplicado: {json.dumps(applied, ensure_ascii=False)}')
            else:
                logger.warning(f'⚠️ Feedback inválido ignorado: {feedback}')

        # CPU-adaptive cycles: back off when system is hot
        try:
            import psutil  # type: ignore
            cpu = psutil.cpu_percent(interval=0.1)
            if cpu > 85:
                # reduce internal cycles to lighten pressure
                try:
                    orch = get_orchestrator(activate=False)
                    old = getattr(orch, 'max_cycles_per_call', 3)
                    orch.max_cycles_per_call = max(1, old - 1)
                    logger.info('🌡️ CPU high (%.1f%%): cycles %s → %s', cpu, old, orch.max_cycles_per_call)
                except Exception:
                    pass
        except Exception:
            pass

        # Evolve once
        stats = evolve_once()
        if isinstance(stats, dict):
            logger.info(
                '🧬 Darwinacci: gen=%s best=%.4f avg=%.4f cov=%.2f%% nnov=%s',
                stats.get('generation'),
                float(stats.get('best_fitness', 0.0)),
                float(stats.get('avg_fitness', 0.0)),
                float(stats.get('coverage', 0.0)) * 100.0,
                stats.get('novelty_archive_size'),
            )
            ok = write_transfer(stats=stats)
            if ok:
                logger.info('📤 Transfer publicado para V7 bridge')
            else:
                logger.warning('⚠️ Falha ao publicar transfer (sem best genome?)')
        else:
            logger.warning('⚠️ Evolução retornou stats inválidos: %r', stats)

    try:
        if args.once:
            run_cycle()
            return 0
        while True:
            run_cycle()
            time.sleep(max(5, int(args.interval)))
    except KeyboardInterrupt:
        logger.info('⏹️ Darwinacci Daemon parado')
        return 0
    except Exception as e:
        logger.exception('💥 Erro fatal no Daemon: %s', e)
        time.sleep(5)
        return 1


if __name__ == '__main__':
    sys.exit(main())
