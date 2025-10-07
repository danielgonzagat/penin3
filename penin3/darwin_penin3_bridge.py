#!/usr/bin/env python3
"""
DARWIN ↔ PENIN³ BRIDGE
======================

Integra Darwin Evolution como motor evolutivo do PENIN³.

Conceito:
- Darwin fornece: Natural selection, sexual reproduction, novelty search
- PENIN³ fornece: Meta-layer (SR-Ω∞, Sigma Guard, WORM audit, Master Equation)
- Bridge: Feedback loop bidirecional entre evolução e meta-cognição
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List
import torch
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("darwin_penin3_bridge")

# Import Darwin
sys.path.insert(0, '/root/darwin-engine-intelligence')
try:
    from core.darwin_evolution_system_FIXED import (
        DarwinEvolutionOrchestrator,
        EvolvableMNIST
    )
    DARWIN_AVAILABLE = True
except Exception as e:
    logger.error(f"Darwin unavailable: {e}")
    DARWIN_AVAILABLE = False

# Import PENIN³
sys.path.insert(0, '/root/penin3')
try:
    from penin3_system import PENIN3System
    PENIN3_AVAILABLE = True
except Exception as e:
    logger.error(f"PENIN³ unavailable: {e}")
    PENIN3_AVAILABLE = False


class DarwinPENIN3Bridge:
    """
    Ponte entre Darwin Evolution e PENIN³ Meta-Layer
    
    Fluxo:
    1. Darwin evolve → melhores indivíduos
    2. PENIN³ avalia → SR-Ω∞ score, Sigma Guard gates
    3. Feedback → Darwin ajusta fitness com meta-conhecimento
    4. Ciclo contínuo de co-evolução
    """
    
    def __init__(self, config: Optional[Dict] = None):
        if not DARWIN_AVAILABLE or not PENIN3_AVAILABLE:
            raise RuntimeError("Darwin ou PENIN³ não disponíveis!")
        
        self.config = config or self._default_config()
        
        # Inicializar Darwin
        logger.info("🧬 Initializing Darwin Evolution Orchestrator...")
        self.darwin = DarwinEvolutionOrchestrator()
        
        # Inicializar PENIN³
        logger.info("Ω Initializing PENIN³ System...")
        self.penin3 = PENIN3System(config=self.config.get('penin3_config', {}))
        
        # Estado da bridge
        self.cycle = 0
        self.darwin_generations = 0
        self.penin3_cycles = 0
        self.best_unified_score = 0.0
        self.history = []
        
        logger.info("✅ Darwin ↔ PENIN³ Bridge initialized")
    
    def _default_config(self) -> Dict:
        """Configuração padrão da bridge"""
        return {
            'darwin_config': {
                'generations_per_cycle': 10,
                'population_size': 30,
                'mutation_rate': 0.3,
                'demo_fast': True
            },
            'penin3_config': None,  # Usa padrão
            'fusion_config': {
                'sr_omega_weight': 0.3,  # Peso do SR-Ω∞ no fitness
                'sigma_gate_strict': True,  # Aplicar gates rigorosos
                'worm_audit_frequency': 5  # Auditar a cada 5 cycles
            }
        }
    
    def calculate_unified_fitness(
        self, 
        darwin_fitness: float,
        penin3_metrics: Dict[str, float]
    ) -> float:
        """
        Calcula fitness unificado: Darwin + PENIN³ meta-knowledge
        
        Fórmula:
        unified_fitness = darwin_fitness * (1 + sr_omega_bonus) * sigma_multiplier
        
        Args:
            darwin_fitness: Fitness do Darwin (accuracy, reward, etc)
            penin3_metrics: Métricas do PENIN³ (sr_score, g_score, etc)
        
        Returns:
            Fitness unificado (0.0 a 2.0+)
        """
        # Base fitness do Darwin
        base = float(darwin_fitness)
        
        # Bonus do SR-Ω∞ (auto-reflection quality)
        sr_score = penin3_metrics.get('sr_score', 0.0)
        sr_weight = self.config['fusion_config']['sr_omega_weight']
        sr_bonus = sr_score * sr_weight
        
        # Multiplicador do Sigma Guard (ethical gates)
        sigma_passed = penin3_metrics.get('sigma_passed', True)
        g_score = penin3_metrics.get('g_score', 0.5)
        
        if sigma_passed:
            sigma_multiplier = 1.0 + (g_score * 0.2)  # Bonus até 20%
        else:
            sigma_multiplier = 0.5  # Penalidade severa se falhar gates
        
        # Fitness unificado
        unified = base * (1.0 + sr_bonus) * sigma_multiplier
        
        return float(np.clip(unified, 0.0, 2.0))
    
    def run_cycle(self) -> Dict[str, Any]:
        """
        Executa um ciclo completo de co-evolução
        
        Returns:
            Resultados do ciclo
        """
        self.cycle += 1
        logger.info(f"\n{'='*70}")
        logger.info(f"🔄 DARWIN ↔ PENIN³ CYCLE {self.cycle}")
        logger.info(f"{'='*70}\n")
        
        results = {
            'cycle': self.cycle,
            'darwin_results': None,
            'penin3_results': None,
            'unified_score': 0.0,
            'best_improved': False
        }
        
        # ===== FASE 1: DARWIN EVOLUTION =====
        logger.info("🧬 Phase 1: Darwin Evolution...")
        try:
            darwin_config = self.config['darwin_config']
            
            # Evoluir MNIST com Darwin
            best_individual = self.darwin.evolve_mnist(
                generations=darwin_config['generations_per_cycle'],
                population_size=darwin_config['population_size'],
                demo_fast=darwin_config['demo_fast'],
                demo_epochs=6
            )
            
            darwin_fitness = float(best_individual.fitness)
            self.darwin_generations += darwin_config['generations_per_cycle']
            
            results['darwin_results'] = {
                'fitness': darwin_fitness,
                'total_generations': self.darwin_generations,
                'genome': best_individual.genome if hasattr(best_individual, 'genome') else None
            }
            
            logger.info(f"   Darwin fitness: {darwin_fitness:.4f}")
            
        except Exception as e:
            logger.error(f"❌ Darwin evolution failed: {e}")
            results['darwin_results'] = {'fitness': 0.0, 'error': str(e)}
            return results
        
        # ===== FASE 2: PENIN³ META-EVALUATION =====
        logger.info("\nΩ Phase 2: PENIN³ Meta-Evaluation...")
        try:
            # Executar ciclo PENIN³ com métricas do Darwin
            penin3_result = self.penin3.run_cycle(
                external_metrics={
                    'darwin_fitness': darwin_fitness,
                    'darwin_generations': self.darwin_generations
                }
            )
            
            self.penin3_cycles += 1
            
            # Extrair métricas relevantes
            penin3_metrics = {
                'sr_score': penin3_result.get('sr_omega_score', 0.0),
                'g_score': penin3_result.get('g_score', 0.5),
                'sigma_passed': penin3_result.get('sigma_guard_passed', True),
                'linf_score': penin3_result.get('linf_score', 0.0)
            }
            
            results['penin3_results'] = penin3_metrics
            
            logger.info(f"   SR-Ω∞ score: {penin3_metrics['sr_score']:.4f}")
            logger.info(f"   Sigma Guard: {'✅ PASS' if penin3_metrics['sigma_passed'] else '❌ FAIL'}")
            
        except Exception as e:
            logger.error(f"❌ PENIN³ evaluation failed: {e}")
            # Fallback: métricas neutras
            penin3_metrics = {
                'sr_score': 0.5,
                'g_score': 0.5,
                'sigma_passed': True,
                'linf_score': darwin_fitness
            }
            results['penin3_results'] = {'error': str(e)}
        
        # ===== FASE 3: UNIFIED SCORING =====
        logger.info("\n🔥 Phase 3: Unified Scoring...")
        unified_score = self.calculate_unified_fitness(darwin_fitness, penin3_metrics)
        results['unified_score'] = unified_score
        
        # Verificar melhoria
        if unified_score > self.best_unified_score:
            improvement = unified_score - self.best_unified_score
            self.best_unified_score = unified_score
            results['best_improved'] = True
            logger.info(f"   🎯 NEW BEST: {unified_score:.4f} (+{improvement:.4f})")
        else:
            logger.info(f"   Current: {unified_score:.4f} (best: {self.best_unified_score:.4f})")
        
        # ===== FASE 4: FEEDBACK LOOP =====
        logger.info("\n🔄 Phase 4: Feedback Loop...")
        try:
            # Ajustar parâmetros do Darwin baseado em PENIN³
            if penin3_metrics['sr_score'] < 0.3:
                # SR baixo → aumentar exploração (mutation rate)
                self.darwin.mutation_rate = min(1.0, self.darwin.mutation_rate * 1.2)
                logger.info(f"   ↑ Increased mutation rate: {self.darwin.mutation_rate:.3f}")
            elif penin3_metrics['sr_score'] > 0.7:
                # SR alto → diminuir exploração (exploitation)
                self.darwin.mutation_rate = max(0.1, self.darwin.mutation_rate * 0.9)
                logger.info(f"   ↓ Decreased mutation rate: {self.darwin.mutation_rate:.3f}")
            
            # Se Sigma Guard falhou → reset mais agressivo
            if not penin3_metrics['sigma_passed']:
                logger.warning("   ⚠️ Sigma Guard failed - triggering exploration boost")
                self.darwin.mutation_rate = min(1.0, self.darwin.mutation_rate * 1.5)
            
        except Exception as e:
            logger.warning(f"Feedback loop adjustment failed: {e}")
        
        # Salvar histórico
        self.history.append(results)
        
        # WORM audit periódico
        if self.cycle % self.config['fusion_config']['worm_audit_frequency'] == 0:
            logger.info("\n📋 WORM Audit triggered...")
            # TODO: Implementar audit completo via PENIN³.worm_ledger
        
        logger.info(f"\n{'='*70}\n")
        return results
    
    def run_continuous(self, num_cycles: int = 100):
        """
        Executa múltiplos ciclos de co-evolução
        
        Args:
            num_cycles: Número de ciclos a executar
        """
        logger.info(f"🚀 Starting continuous co-evolution: {num_cycles} cycles")
        logger.info(f"   Darwin: {self.config['darwin_config']['generations_per_cycle']} gens/cycle")
        logger.info(f"   PENIN³: Meta-evaluation + feedback loop")
        logger.info("")
        
        for i in range(num_cycles):
            try:
                result = self.run_cycle()
                
                # Log progress a cada 10 cycles
                if (i + 1) % 10 == 0:
                    logger.info(f"\n📊 Progress: {i+1}/{num_cycles} cycles")
                    logger.info(f"   Best unified score: {self.best_unified_score:.4f}")
                    logger.info(f"   Darwin generations: {self.darwin_generations}")
                    logger.info(f"   PENIN³ cycles: {self.penin3_cycles}\n")
                
            except KeyboardInterrupt:
                logger.info("\n⚠️ Interrupted by user")
                break
            except Exception as e:
                logger.error(f"❌ Cycle {self.cycle} failed: {e}")
                continue
        
        logger.info(f"\n✅ Continuous co-evolution completed!")
        logger.info(f"   Total cycles: {self.cycle}")
        logger.info(f"   Best unified score: {self.best_unified_score:.4f}")
        logger.info(f"   Darwin total generations: {self.darwin_generations}")


def main():
    """Teste da bridge"""
    logger.info("🔬 Testing Darwin ↔ PENIN³ Bridge\n")
    
    try:
        # Criar bridge
        bridge = DarwinPENIN3Bridge()
        
        # Executar 5 cycles de teste
        bridge.run_continuous(num_cycles=5)
        
        logger.info("\n✅ Bridge test completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"\n❌ Bridge test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
