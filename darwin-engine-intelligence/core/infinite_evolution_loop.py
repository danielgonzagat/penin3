"""
✅ FASE 3.3: Loop Infinito Auto-Sustentável - Evolução 24/7
===========================================================

Sistema de evolução contínua sem intervenção humana.

Features:
- Loop infinito com checkpoints automáticos
- Recuperação de falhas (fault tolerance)
- Auto-ajuste de parâmetros
- Telemetria contínua
- Paradas de emergência (kill-switch)

Referências:
- Continuous Evolution
- Self-sustaining systems
- Fault-tolerant computing
- Production ML systems
"""

import time
import json
import signal
import sys
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import traceback

logger = logging.getLogger(__name__)


@dataclass
class EvolutionState:
    """Estado atual da evolução infinita"""
    generation: int
    total_time_seconds: float
    best_fitness_ever: float
    best_genome_ever: Dict
    cycles_completed: int
    failures_count: int
    last_checkpoint_time: float
    is_running: bool = True


class InfiniteEvolutionLoop:
    """
    Loop Infinito de Evolução - Sistema Auto-Sustentável 24/7.
    
    Features:
    - Roda indefinidamente (ou até atingir condição de parada)
    - Checkpoints automáticos a cada N gerações
    - Recuperação automática de falhas
    - Telemetria contínua
    - Kill-switch para parada de emergência
    
    Uso:
        # Criar loop infinito
        infinite_loop = InfiniteEvolutionLoop(
            evolution_function=evolve_one_generation,
            checkpoint_interval=10,
            output_dir=Path("/root/infinite_evolution")
        )
        
        # Iniciar evolução infinita
        infinite_loop.start(
            target_fitness=0.99,  # Parar se atingir
            max_hours=48,         # Parar após 48h
            max_failures=10       # Parar se falhar 10x
        )
        
        # Parar externamente
        infinite_loop.stop()
        
        # Retomar de checkpoint
        infinite_loop.resume_from_checkpoint()
    """
    
    def __init__(self,
                 evolution_function: Callable,
                 checkpoint_interval: int = 10,
                 output_dir: Path = Path("/root/infinite_evolution")):
        """
        Args:
            evolution_function: Função que executa 1 geração
            checkpoint_interval: Intervalo de checkpoints (gerações)
            output_dir: Diretório de saída
        """
        self.evolution_function = evolution_function
        self.checkpoint_interval = checkpoint_interval
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True)
        
        # Estado
        self.state = EvolutionState(
            generation=0,
            total_time_seconds=0.0,
            best_fitness_ever=0.0,
            best_genome_ever={},
            cycles_completed=0,
            failures_count=0,
            last_checkpoint_time=time.time()
        )
        
        # Controle
        self.should_stop = False
        self.telemetry_log = []
        
        # Setup signal handlers para kill-switch
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handler para sinais (CTRL+C, kill)"""
        logger.info(f"\n🛑 Kill-switch activated (signal {signum})")
        self.should_stop = True
        self._save_checkpoint(reason="kill_switch")
        sys.exit(0)
    
    def start(self,
             target_fitness: Optional[float] = None,
             max_hours: Optional[float] = None,
             max_failures: int = 10):
        """
        Inicia loop infinito de evolução.
        
        Args:
            target_fitness: Parar se atingir este fitness
            max_hours: Parar após N horas
            max_failures: Parar após N falhas consecutivas
        """
        logger.info("="*80)
        logger.info("♾️  INFINITE EVOLUTION LOOP - STARTING")
        logger.info("="*80)
        logger.info(f"   Target fitness: {target_fitness or 'None (run forever)'}")
        logger.info(f"   Max hours: {max_hours or 'None (run forever)'}")
        logger.info(f"   Max failures: {max_failures}")
        logger.info(f"   Checkpoint interval: {self.checkpoint_interval} generations")
        logger.info(f"   Output dir: {self.output_dir}")
        logger.info("="*80)
        
        start_time = time.time()
        consecutive_failures = 0
        
        try:
            while not self.should_stop:
                cycle_start = time.time()
                
                try:
                    # === EXECUTAR 1 GERAÇÃO ===
                    result = self.evolution_function(generation=self.state.generation)
                    
                    # Atualizar estado
                    self.state.generation += 1
                    self.state.cycles_completed += 1
                    consecutive_failures = 0  # Reset após sucesso
                    
                    # Atualizar best
                    if result['fitness'] > self.state.best_fitness_ever:
                        self.state.best_fitness_ever = result['fitness']
                        self.state.best_genome_ever = result['genome']
                        logger.info(f"   🏆 New best fitness: {result['fitness']:.4f}")
                    
                    # Telemetria
                    self._log_telemetry({
                        'generation': self.state.generation,
                        'fitness': result['fitness'],
                        'time': time.time() - cycle_start,
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    # === CHECKPOINT AUTOMÁTICO ===
                    if self.state.generation % self.checkpoint_interval == 0:
                        self._save_checkpoint(reason="interval")
                    
                    # === CONDIÇÕES DE PARADA ===
                    
                    # Target fitness atingido?
                    if target_fitness and self.state.best_fitness_ever >= target_fitness:
                        logger.info(f"✅ Target fitness reached: {self.state.best_fitness_ever:.4f}")
                        break
                    
                    # Max hours excedido?
                    elapsed_hours = (time.time() - start_time) / 3600
                    if max_hours and elapsed_hours >= max_hours:
                        logger.info(f"⏱️  Max hours reached: {elapsed_hours:.1f}h")
                        break
                    
                    # Log progresso
                    if self.state.generation % 10 == 0:
                        logger.info(f"   Gen {self.state.generation}: fitness={result['fitness']:.4f}, "
                                  f"elapsed={elapsed_hours:.1f}h, "
                                  f"best_ever={self.state.best_fitness_ever:.4f}")
                
                except Exception as e:
                    # === RECUPERAÇÃO DE FALHA ===
                    consecutive_failures += 1
                    self.state.failures_count += 1
                    
                    logger.error(f"❌ Failure in generation {self.state.generation}: {e}")
                    logger.error(traceback.format_exc())
                    
                    if consecutive_failures >= max_failures:
                        logger.error(f"🛑 Max failures ({max_failures}) reached. Stopping.")
                        break
                    
                    # Tentar recuperar
                    logger.info(f"   🔄 Attempting recovery (attempt {consecutive_failures}/{max_failures})...")
                    time.sleep(5)  # Esperar antes de tentar novamente
                    
                    # Tentar carregar último checkpoint
                    if consecutive_failures > 3:
                        logger.info("   📂 Loading last checkpoint...")
                        self.resume_from_checkpoint()
                
                # Small delay para não sobrecarregar CPU
                time.sleep(0.1)
        
        except KeyboardInterrupt:
            logger.info("\n🛑 Interrupted by user")
        
        finally:
            # === FINALIZAÇÃO ===
            self._save_checkpoint(reason="final")
            self._save_final_report(start_time)
            
            logger.info("\n" + "="*80)
            logger.info("♾️  INFINITE EVOLUTION LOOP - STOPPED")
            logger.info("="*80)
            logger.info(f"   Total generations: {self.state.generation}")
            logger.info(f"   Best fitness ever: {self.state.best_fitness_ever:.4f}")
            logger.info(f"   Failures: {self.state.failures_count}")
            logger.info(f"   Total time: {(time.time() - start_time)/3600:.2f}h")
            logger.info("="*80)
    
    def stop(self):
        """Para o loop (gracefully)"""
        logger.info("🛑 Stop signal received")
        self.should_stop = True
    
    def _save_checkpoint(self, reason: str):
        """Salva checkpoint do estado atual"""
        checkpoint_path = self.output_dir / f"checkpoint_gen_{self.state.generation}.json"
        
        checkpoint_data = {
            'state': asdict(self.state),
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)
        
        self.state.last_checkpoint_time = time.time()
        
        logger.info(f"   💾 Checkpoint saved: {checkpoint_path.name} (reason: {reason})")
    
    def resume_from_checkpoint(self):
        """Retoma evolução do último checkpoint"""
        # Encontrar checkpoint mais recente
        checkpoints = sorted(self.output_dir.glob("checkpoint_gen_*.json"))
        
        if not checkpoints:
            logger.warning("   ⚠️  No checkpoints found to resume from")
            return
        
        latest_checkpoint = checkpoints[-1]
        
        with open(latest_checkpoint, 'r') as f:
            checkpoint_data = json.load(f)
        
        # Restaurar estado
        state_dict = checkpoint_data['state']
        self.state = EvolutionState(**state_dict)
        
        logger.info(f"   📂 Resumed from checkpoint: {latest_checkpoint.name}")
        logger.info(f"      Generation: {self.state.generation}")
        logger.info(f"      Best fitness: {self.state.best_fitness_ever:.4f}")
    
    def _log_telemetry(self, data: Dict):
        """Registra telemetria"""
        self.telemetry_log.append(data)
        
        # Manter apenas últimos 1000
        if len(self.telemetry_log) > 1000:
            self.telemetry_log.pop(0)
        
        # Salvar telemetria a cada 100 gerações
        if self.state.generation % 100 == 0:
            telemetry_path = self.output_dir / "telemetry.json"
            with open(telemetry_path, 'w') as f:
                json.dump(self.telemetry_log, f, indent=2)
    
    def _save_final_report(self, start_time: float):
        """Salva relatório final"""
        report_path = self.output_dir / "final_report.json"
        
        report = {
            'summary': {
                'total_generations': self.state.generation,
                'total_time_hours': (time.time() - start_time) / 3600,
                'best_fitness_ever': self.state.best_fitness_ever,
                'failures_count': self.state.failures_count,
                'cycles_completed': self.state.cycles_completed
            },
            'best_genome': self.state.best_genome_ever,
            'final_state': asdict(self.state),
            'timestamp': datetime.now().isoformat()
        }
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"   📄 Final report saved: {report_path}")


def example_evolution_function(generation: int) -> Dict:
    """
    Exemplo de função de evolução (1 geração).
    
    Na prática, substituir por evolução real.
    """
    import random
    
    # Simular evolução de 1 geração
    fitness = 0.5 + (generation / 1000) + random.random() * 0.1
    genome = {'param1': random.random(), 'param2': random.randint(1, 10)}
    
    return {
        'fitness': fitness,
        'genome': genome
    }


def run_infinite_evolution_example():
    """Exemplo de uso do loop infinito"""
    
    # Criar loop
    loop = InfiniteEvolutionLoop(
        evolution_function=example_evolution_function,
        checkpoint_interval=10,
        output_dir=Path("/root/infinite_evolution_example")
    )
    
    # Iniciar (vai rodar até atingir fitness 0.99 ou 1 hora)
    loop.start(
        target_fitness=0.99,
        max_hours=1.0,
        max_failures=5
    )


if __name__ == "__main__":
    # Exemplo de uso
    logging.basicConfig(level=logging.INFO)
    run_infinite_evolution_example()
