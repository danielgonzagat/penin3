#!/usr/bin/env python3
"""
✅ DARWIN CONNECTOR - ATIVAÇÃO AUTÔNOMA
Conecta Darwin Evolution ao UNIFIED_BRAIN
"""

import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')
sys.path.insert(0, '/root')

import time
import torch
from pathlib import Path
from datetime import datetime
from brain_logger import brain_logger

# Import Darwin components
try:
    from intelligence_system.extracted_algorithms.darwin_engine_real import DarwinEngine, Individual
    from brain_system_integration import BrainDarwinEvolver
    DARWIN_AVAILABLE = True
except Exception as e:
    brain_logger.error(f"Darwin import failed: {e}")
    DARWIN_AVAILABLE = False

class DarwinConnector:
    """Conecta Darwin ao brain daemon automaticamente"""
    
    def __init__(self, brain_daemon):
        self.brain = brain_daemon
        
        if not DARWIN_AVAILABLE:
            brain_logger.error("❌ Darwin não disponível")
            self.enabled = False
            return
        
        # Criar Darwin engine
        try:
            self.darwin_engine = DarwinEngine(
                survival_rate=0.4,
                elite_size=5,
                min_fitness_threshold=0.0
            )
            
            self.darwin_evolver = BrainDarwinEvolver(self.brain.hybrid.core)
            
            self.generation = 0
            self.last_evolution_episode = 0
            self.evolution_interval = 10  # Evoluir a cada 10 episódios
            
            self.enabled = True
            brain_logger.info("✅ Darwin Connector ACTIVE")
            
        except Exception as e:
            brain_logger.error(f"Darwin connector init failed: {e}")
            self.enabled = False
    
    def should_evolve(self, episode: int) -> bool:
        """Decide se deve evoluir agora"""
        if not self.enabled:
            return False
        
        # Evoluir a cada N episódios
        if episode - self.last_evolution_episode >= self.evolution_interval:
            return True
        
        return False
    
    def evolve(self, episode_reward: float, episode: int) -> dict:
        """Executa evolução Darwin"""
        if not self.enabled:
            return {'success': False, 'reason': 'disabled'}
        
        try:
            # 1. Evaluate fitness
            fitness_report = self.darwin_evolver.evaluate_fitness(episode_reward)
            
            brain_logger.info(
                f"🧬 Darwin eval ep={episode}: "
                f"fitness={fitness_report['fitness']:.4f}, "
                f"should_evolve={fitness_report['should_evolve']}"
            )
            
            # 2. Evolve se recomendado
            if fitness_report['should_evolve']:
                evo_result = self.darwin_evolver.evolve_topology(
                    mutation_rate=0.1,
                    crossover_rate=0.3,
                    selection_pressure=0.5
                )
                
                if evo_result['success']:
                    self.generation += 1
                    self.last_evolution_episode = episode
                    
                    brain_logger.warning(
                        f"🧬 EVOLVED! Gen={self.generation}, "
                        f"Changes={evo_result['changes']}"
                    )
                    
                    # Log to WORM
                    if hasattr(self.brain.hybrid.core, 'worm'):
                        self.brain.hybrid.core.worm.append('darwin_evolution', {
                            'episode': episode,
                            'generation': self.generation,
                            'fitness': fitness_report['fitness'],
                            'changes': evo_result['changes']
                        })
                    
                    return {
                        'success': True,
                        'generation': self.generation,
                        'changes': evo_result['changes']
                    }
            
            return {
                'success': False,
                'reason': 'evolution_not_needed',
                'fitness': fitness_report['fitness']
            }
            
        except Exception as e:
            brain_logger.error(f"Darwin evolution failed: {e}")
            return {'success': False, 'error': str(e)}

# Monkey-patch brain daemon para adicionar Darwin
def inject_darwin_into_daemon():
    """Injeta Darwin connector no daemon rodando"""
    try:
        # Não podemos modificar daemon em execução facilmente
        # Mas podemos criar arquivo que daemon vai carregar
        
        marker_file = Path('/root/UNIFIED_BRAIN/ENABLE_DARWIN.flag')
        marker_file.write_text(f"enabled_at={datetime.now().isoformat()}\n")
        
        brain_logger.info("✅ Darwin enable flag created")
        
        return True
        
    except Exception as e:
        brain_logger.error(f"Darwin injection failed: {e}")
        return False

if __name__ == "__main__":
    brain_logger.info("🧬 Darwin Connector - standalone mode")
    brain_logger.info("Creating enable flag...")
    
    if inject_darwin_into_daemon():
        print("✅ Darwin connector configured")
        print("⚠️  Restart brain daemon to apply:")
        print("   pkill -f brain_daemon_real_env.py")
        print("   cd /root/UNIFIED_BRAIN")
        print("   nohup python3 -u brain_daemon_real_env.py &")
    else:
        print("❌ Failed to configure Darwin connector")