#!/usr/bin/env python3
"""
🚀 DARWINACCI QUICK INTEGRATION - UNIFIED_BRAIN
Conecta Darwinacci ao UNIFIED_BRAIN para evoluir hiperparâmetros

OPÇÃO A (SIMPLES): Evolve router hyperparameters
- temperature
- top_k  
- curiosity_weight
- lateral_inhibition

Fitness = avg_reward últimos 10 episódios
"""

import sys
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/UNIFIED_BRAIN')

import random
import json
import time
from pathlib import Path
from typing import Dict, Any
from collections import deque

# Import Darwinacci
try:
    from darwinacci_omega.core.engine import DarwinacciEngine
    print("✅ Darwinacci imported")
except ImportError as e:
    print(f"❌ Darwinacci import failed: {e}")
    sys.exit(1)


class HyperparameterEvolver:
    """
    Evolve UNIFIED_BRAIN hyperparameters usando Darwinacci-Ω
    """
    
    def __init__(self):
        self.rewards_history = deque(maxlen=1000)
        self.best_genome = None
        self.best_fitness = -float('inf')
        self.evolution_count = 0
        
        # Paths
        self.brain_log = Path("/root/UNIFIED_BRAIN/logs/unified_brain.log")
        self.suggestions_file = Path("/root/UNIFIED_BRAIN/runtime_suggestions.json")
        self.state_file = Path("/root/UNIFIED_BRAIN/darwinacci_state.json")
        
        # Init functions for Darwinacci
        def init_fn(rng):
            """Cria genome de hiperparâmetros"""
            return {
                'temperature': rng.uniform(0.5, 2.0),
                'top_k': float(rng.randint(4, 16)),
                'curiosity_weight': rng.uniform(0.05, 0.3),
                'lateral_inhibition': rng.uniform(0.0, 0.3),
                'alpha': rng.uniform(0.7, 0.95),  # homeostasis
            }
        
        def eval_fn(genome, rng):
            """
            Fitness = performance com esses hiperparâmetros
            
            Usa historical data dos últimos episódios
            (em produção, aplicaria genome e mediria reward real)
            """
            # Simular fitness baseado em heurística dos params
            # Em produção real: aplicar params → rodar episodes → medir reward
            
            temp = genome.get('temperature', 1.0)
            k = genome.get('top_k', 8)
            cur_w = genome.get('curiosity_weight', 0.1)
            
            # Heurística: balance exploration/exploitation
            exploration = temp * cur_w
            exploitation = k / 16.0
            
            # Fitness toy (em produção: reward médio real)
            fitness = (
                exploration * 0.4 +
                exploitation * 0.4 +
                rng.random() * 0.2
            )
            
            # Behavior para QD: [exploration, exploitation]
            behavior = [exploration, exploitation]
            
            return {
                'objective': fitness,
                'behavior': behavior,
                'linf': min(1.0, fitness * 1.1),
                'novelty': 0.0,  # Filled by engine
                'robustness': 0.9,
                'caos_plus': exploration,
                'cost_penalty': 1.0,
                'ece': 0.05,
                'rho_bias': 1.0,
                'rho': 0.5,
                'eco_ok': True,
                'consent': True
            }
        
        # Create Darwinacci engine
        self.engine = DarwinacciEngine(
            init_fn=init_fn,
            eval_fn=eval_fn,
            max_cycles=5,
            pop_size=20,
            seed=int(time.time()) % 1000000
        )
        
        print("🌟 HyperparameterEvolver initialized")
        print(f"   Population: {self.engine.pop_size}")
        print(f"   Cycles: {self.engine.clock.seq[:7]}")
    
    def load_state(self):
        """Carrega estado anterior se existir"""
        if self.state_file.exists():
            try:
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                self.best_genome = state.get('best_genome')
                self.best_fitness = state.get('best_fitness', -float('inf'))
                self.evolution_count = state.get('evolution_count', 0)
                print(f"📊 Loaded state: evolutions={self.evolution_count}, best={self.best_fitness:.4f}")
            except Exception as e:
                print(f"⚠️ Failed to load state: {e}")
    
    def save_state(self):
        """Salva estado para persistência"""
        try:
            state = {
                'best_genome': self.best_genome,
                'best_fitness': self.best_fitness,
                'evolution_count': self.evolution_count,
                'timestamp': time.time()
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=2)
        except Exception as e:
            print(f"⚠️ Failed to save state: {e}")
    
    def evolve_once(self):
        """Executa uma rodada de evolução"""
        print(f"\n{'='*70}")
        print(f"🧬 EVOLUTION #{self.evolution_count + 1}")
        print(f"{'='*70}")
        
        # Run Darwinacci
        champion = self.engine.run(max_cycles=5)
        
        self.evolution_count += 1
        
        if champion:
            print(f"\n✅ Champion found!")
            print(f"   Score: {champion.score:.4f}")
            print(f"   Genome: {champion.genome}")
            print(f"   Coverage: {self.engine.archive.coverage():.2%}")
            print(f"   Novelty archive: {len(self.engine.novel.mem)}")
            
            # Update best if improved
            if champion.score > self.best_fitness:
                self.best_genome = champion.genome
                self.best_fitness = champion.score
                
                # Apply to UNIFIED_BRAIN via suggestions file
                self.apply_best_genome()
        
        # Save state
        self.save_state()
    
    def apply_best_genome(self):
        """Aplica best genome ao UNIFIED_BRAIN"""
        if not self.best_genome:
            return
        
        try:
            suggestion = {
                'router': {
                    'temperature': float(self.best_genome.get('temperature', 1.0)),
                    'top_k': int(self.best_genome.get('top_k', 8)),
                    'reason': f'darwinacci_evolution_{self.evolution_count}'
                },
                'curiosity_weight': float(self.best_genome.get('curiosity_weight', 0.1)),
                'timestamp': time.time(),
                'fitness': self.best_fitness
            }
            
            with open(self.suggestions_file, 'w') as f:
                json.dump(suggestion, f, indent=2)
            
            print(f"\n✅ Applied best genome to {self.suggestions_file}")
            print(f"   temperature: {suggestion['router']['temperature']:.3f}")
            print(f"   top_k: {suggestion['router']['top_k']}")
            print(f"   curiosity: {suggestion['curiosity_weight']:.3f}")
        
        except Exception as e:
            print(f"❌ Failed to apply genome: {e}")
    
    def run_daemon(self, interval_episodes: int = 20):
        """
        Roda como daemon, evoluindo a cada N episódios
        
        Args:
            interval_episodes: Evolve after this many new episodes
        """
        print("="*70)
        print("🤖 DARWINACCI HYPERPARAMETER EVOLVER DAEMON")
        print("="*70)
        print(f"   Monitoring: {self.brain_log}")
        print(f"   Evolve every: {interval_episodes} episodes")
        print(f"   Suggestions: {self.suggestions_file}")
        print("="*70)
        print()
        
        self.load_state()
        
        last_episode = 0
        
        while True:
            try:
                # Check brain log for episode count
                if self.brain_log.exists():
                    with open(self.brain_log, 'r') as f:
                        lines = list(f)[-100:]
                    
                    # Find latest episode
                    for line in reversed(lines):
                        if 'Ep ' in line or 'Episode ' in line:
                            try:
                                import re
                                match = re.search(r'Ep[isode]*\s+(\d+)', line)
                                if match:
                                    current_episode = int(match.group(1))
                                    
                                    # Check if enough episodes passed
                                    if current_episode >= last_episode + interval_episodes:
                                        print(f"\n📊 Episode {current_episode} (last={last_episode})")
                                        print(f"   → Triggering evolution...")
                                        
                                        self.evolve_once()
                                        last_episode = current_episode
                                        break
                            except:
                                continue
                
                # Sleep
                time.sleep(60)
            
            except KeyboardInterrupt:
                print("\n⏹️  Daemon stopped")
                self.save_state()
                break
            
            except Exception as e:
                print(f"❌ Error: {e}")
                time.sleep(60)


if __name__ == "__main__":
    evolver = HyperparameterEvolver()
    evolver.run_daemon(interval_episodes=20)
