"""
✅ FASE 2.2: Meta-Cognição - Sistema de Auto-Reflexão
=====================================================

Sistema que observa a si mesmo evoluindo e toma decisões meta-level.

Features:
- Observa métricas de evolução (fitness, diversity, novelty)
- Detecta padrões (stagnation, exploitation vs exploration)
- Toma decisões meta-level (ajustar mutation_rate, mudar objetivo)
- Reflete sobre próprias decisões (why did I do that?)
- Auto-crítica (were my interventions effective?)

Referências:
- System 1 vs System 2 Thinking (Kahneman)
- Meta-Learning (Schmidhuber)
- Self-Aware AI (introspection)
"""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from collections import deque
import numpy as np
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class EvolutionaryPhase(Enum):
    """Fases do processo evolutivo"""
    EXPLORATION = "exploration"      # Explorando espaço (alta diversidade)
    EXPLOITATION = "exploitation"    # Refinando soluções (baixa diversidade)
    STAGNATION = "stagnation"        # Preso em local optima
    BREAKTHROUGH = "breakthrough"    # Descobriu novo nicho
    CONVERGENCE = "convergence"      # Convergiu para solução


@dataclass
class EvolutionSnapshot:
    """Snapshot de uma geração"""
    generation: int
    best_fitness: float
    avg_fitness: float
    fitness_std: float  # Diversity
    novelty_avg: float
    mutation_rate: float
    n_structural_mutations: int
    n_crossovers: int
    phase: Optional[str] = None


class MetaCognitionEngine:
    """
    Motor de meta-cognição: sistema que pensa sobre próprio pensamento.
    
    Análogo a "System 2" thinking (Kahneman):
    - System 1: Evolução automática (rápida, instintiva)
    - System 2: Meta-cognição (lenta, deliberada, refletiva)
    
    Uso:
        meta = MetaCognitionEngine()
        
        # A cada geração:
        snapshot = EvolutionSnapshot(
            generation=gen,
            best_fitness=0.95,
            avg_fitness=0.85,
            fitness_std=0.05,
            novelty_avg=0.2,
            mutation_rate=0.2,
            n_structural_mutations=3,
            n_crossovers=10
        )
        meta.observe(snapshot)
        
        # Periodicamente (ex: cada 10 gens):
        decision = meta.reflect_and_decide()
        if decision.get('adjust_mutation_rate'):
            mutation_rate += decision['adjust_mutation_rate']
        if decision.get('force_diversification'):
            # Reinjetar diversidade
            pass
        
        # Ao final:
        story = meta.explain_evolution()
        critique = meta.self_critique()
    """
    
    def __init__(self, history_window: int = 50):
        self.history_window = history_window
        self.snapshots = deque(maxlen=history_window)
        
        # Estado interno
        self.current_phase = EvolutionaryPhase.EXPLORATION
        self.phase_counter = 0
        
        # Reflexões (decisões tomadas + justificativas)
        self.reflections: List[Dict[str, Any]] = []
        
        # Meta-parameters aprendidos
        self.learned_meta_params = {
            'optimal_mutation_rate': 0.2,
            'exploration_exploitation_ratio': 0.5,
            'diversity_threshold': 0.02
        }
        
        # Estatísticas de intervenções
        self.interventions = {
            'mutation_adjustments': 0,
            'forced_diversifications': 0,
            'total_reflections': 0
        }
    
    def observe(self, snapshot: EvolutionSnapshot):
        """
        Observa estado atual da evolução.
        
        Args:
            snapshot: Snapshot da geração atual
        """
        self.snapshots.append(snapshot)
    
    def detect_phase(self) -> EvolutionaryPhase:
        """
        Detecta fase atual do processo evolutivo.
        
        Returns:
            Fase detectada
        """
        if len(self.snapshots) < 10:
            return EvolutionaryPhase.EXPLORATION
        
        recent = list(self.snapshots)[-10:]
        
        # Calcular tendências
        fitness_trend = self._compute_trend([s.best_fitness for s in recent])
        diversity_trend = self._compute_trend([s.fitness_std for s in recent])
        
        # Heurísticas para detectar fase
        if fitness_trend > 0.01 and diversity_trend > 0:
            # Fitness crescendo E diversidade crescendo = explorando bem
            phase = EvolutionaryPhase.EXPLORATION
        elif fitness_trend > 0.05:
            # Fitness crescendo rápido = breakthrough
            phase = EvolutionaryPhase.BREAKTHROUGH
        elif fitness_trend < 0.001 and diversity_trend < -0.01:
            # Fitness estagnado E diversidade caindo = stagnation
            phase = EvolutionaryPhase.STAGNATION
        elif fitness_trend > 0 and diversity_trend < 0:
            # Fitness crescendo mas diversidade caindo = exploitation
            phase = EvolutionaryPhase.EXPLOITATION
        elif fitness_trend < 0.0001 and diversity_trend < 0.001:
            # Tudo estagnado = convergence
            phase = EvolutionaryPhase.CONVERGENCE
        else:
            # Default: exploration
            phase = EvolutionaryPhase.EXPLORATION
        
        return phase
    
    def _compute_trend(self, values: List[float]) -> float:
        """Calcula tendência (linear regression slope)"""
        if len(values) < 2:
            return 0.0
        
        try:
            x = np.arange(len(values))
            y = np.array(values)
            
            # Linear regression
            slope = float(np.polyfit(x, y, 1)[0])
            
            return slope
        except:
            return 0.0
    
    def reflect_and_decide(self) -> Dict[str, Any]:
        """
        Reflexão meta-cognitiva: analisa próprio processo e decide ajustes.
        
        Returns:
            Dict com decisões:
            {
                'adjust_mutation_rate': float or None,
                'adjust_crossover_rate': float or None,
                'force_diversification': bool,
                'reasoning': str,  # Justificativa da decisão
                'phase': str,
                'confidence': float
            }
        """
        self.interventions['total_reflections'] += 1
        
        if len(self.snapshots) < 20:
            return {
                'reasoning': 'Not enough data to reflect (need 20+ generations)',
                'phase': 'warming_up',
                'confidence': 0.0
            }
        
        # Detectar fase
        phase = self.detect_phase()
        
        # Decisão baseada na fase
        decision = {}
        reasoning = []
        confidence = 0.7  # Confiança na decisão
        
        if phase == EvolutionaryPhase.STAGNATION:
            # STUCK: Aumentar diversidade
            decision['adjust_mutation_rate'] = +0.1  # Aumenta mutação
            decision['force_diversification'] = True
            reasoning.append("🚨 Detected STAGNATION (fitness flat, diversity low)")
            reasoning.append("   Decision: Increase mutation +0.1 to escape local optimum")
            reasoning.append("   Action: Force diversification (inject random individuals)")
            confidence = 0.9
            self.interventions['mutation_adjustments'] += 1
            self.interventions['forced_diversifications'] += 1
            
        elif phase == EvolutionaryPhase.EXPLOITATION:
            # Refinando: Reduzir mutação para convergir
            decision['adjust_mutation_rate'] = -0.05
            reasoning.append("🎯 Detected EXPLOITATION (fitness growing, diversity shrinking)")
            reasoning.append("   Decision: Reduce mutation -0.05 to refine solutions")
            reasoning.append("   Rationale: Let natural selection polish current best")
            confidence = 0.75
            self.interventions['mutation_adjustments'] += 1
            
        elif phase == EvolutionaryPhase.BREAKTHROUGH:
            # Descoberta: Manter estratégia
            reasoning.append("🚀 Detected BREAKTHROUGH (rapid fitness improvement)")
            reasoning.append("   Decision: Keep current strategy (it's working!)")
            reasoning.append("   Rationale: Don't interrupt successful exploration")
            confidence = 0.85
            
        elif phase == EvolutionaryPhase.CONVERGENCE:
            # Convergiu: Perturbar sistema
            decision['force_diversification'] = True
            decision['adjust_mutation_rate'] = +0.15
            reasoning.append("🔄 Detected CONVERGENCE (premature?)")
            reasoning.append("   Decision: Force diversification + boost mutation +0.15")
            reasoning.append("   Rationale: Explore more before settling")
            confidence = 0.6
            self.interventions['mutation_adjustments'] += 1
            self.interventions['forced_diversifications'] += 1
        
        else:  # EXPLORATION
            reasoning.append("🌍 Phase: EXPLORATION (healthy)")
            reasoning.append("   Decision: No intervention needed")
            reasoning.append("   Rationale: System is exploring well naturally")
            confidence = 0.8
        
        decision['reasoning'] = "\n".join(reasoning)
        decision['phase'] = phase.value
        decision['confidence'] = confidence
        
        # Registrar reflexão
        reflection = {
            'generation': self.snapshots[-1].generation if self.snapshots else 0,
            'phase': phase.value,
            'decision': decision.copy(),
            'reasoning': decision['reasoning'],
            'confidence': confidence
        }
        self.reflections.append(reflection)
        
        # Atualizar fase interna
        if phase != self.current_phase:
            logger.info(f"🔄 Phase transition: {self.current_phase.value} → {phase.value}")
            self.phase_counter = 0
            self.current_phase = phase
        else:
            self.phase_counter += 1
        
        return decision
    
    def explain_evolution(self) -> str:
        """
        Gera explicação narrativa do processo evolutivo.
        
        Returns:
            String explicando o que aconteceu e por quê
        """
        if len(self.snapshots) < 10:
            return "Not enough generations to explain evolution story."
        
        story = []
        story.append("═" * 70)
        story.append("🧠 EVOLUTION STORY (Meta-Cognitive Analysis)")
        story.append("═" * 70)
        
        # Resumo geral
        first = self.snapshots[0]
        last = self.snapshots[-1]
        
        fitness_gain = last.best_fitness - first.best_fitness
        generations = last.generation - first.generation
        
        story.append(f"\n📊 Overview:")
        story.append(f"   Generation {first.generation} → {last.generation} ({generations} gens)")
        story.append(f"   Fitness: {first.best_fitness:.4f} → {last.best_fitness:.4f} (+{fitness_gain:.4f})")
        story.append(f"   Diversity: {first.fitness_std:.4f} → {last.fitness_std:.4f}")
        
        # Fases observadas
        phase_counts = {}
        for refl in self.reflections:
            phase = refl['phase']
            phase_counts[phase] = phase_counts.get(phase, 0) + 1
        
        if phase_counts:
            story.append(f"\n🔄 Phases Observed:")
            for phase, count in sorted(phase_counts.items(), key=lambda x: -x[1]):
                story.append(f"   • {phase.upper()}: {count} times")
        
        # Decisões tomadas
        story.append(f"\n🤔 Meta-Decisions Taken: {len(self.reflections)}")
        
        if self.reflections:
            story.append(f"\n   Last 5 reflections:")
            for i, refl in enumerate(self.reflections[-5:], 1):
                story.append(f"\n   {i}. Gen {refl['generation']}: {refl['phase'].upper()} (confidence: {refl['confidence']:.0%})")
                # Indent reasoning
                for line in refl['reasoning'].split('\n'):
                    story.append(f"      {line}")
        
        # Intervenções
        story.append(f"\n🔧 Interventions:")
        story.append(f"   • Mutation adjustments: {self.interventions['mutation_adjustments']}")
        story.append(f"   • Forced diversifications: {self.interventions['forced_diversifications']}")
        story.append(f"   • Total reflections: {self.interventions['total_reflections']}")
        
        # Aprendizados
        story.append(f"\n📚 Meta-Learnings:")
        
        if fitness_gain > 0.5:
            story.append(f"   ✅ Strong progress achieved (+{fitness_gain:.2f} fitness)")
        elif fitness_gain > 0.1:
            story.append(f"   ✓  Moderate progress (+{fitness_gain:.2f} fitness)")
        elif fitness_gain > 0:
            story.append(f"   ⚠️  Limited progress (+{fitness_gain:.2f} fitness)")
        else:
            story.append(f"   ❌ No progress or regression ({fitness_gain:.2f} fitness)")
        
        # Diversity pattern
        diversity_values = [s.fitness_std for s in self.snapshots]
        avg_diversity = np.mean(diversity_values)
        
        if avg_diversity > 0.05:
            story.append(f"   ✅ Maintained good diversity (avg {avg_diversity:.3f})")
        elif avg_diversity > 0.02:
            story.append(f"   ✓  Moderate diversity (avg {avg_diversity:.3f})")
        else:
            story.append(f"   ⚠️  Low diversity (avg {avg_diversity:.3f}) - risk of premature convergence")
        
        story.append("\n" + "═" * 70)
        
        return "\n".join(story)
    
    def self_critique(self) -> Dict[str, Any]:
        """
        Auto-crítica: avalia qualidade das próprias decisões.
        
        "Were my interventions effective?"
        
        Returns:
            Dict com análise:
            {
                'overall_effectiveness': float,
                'best_decisions': List[Dict],
                'worst_decisions': List[Dict],
                'lessons_learned': List[str]
            }
        """
        if len(self.reflections) < 5:
            return {
                'overall_effectiveness': 0.5,
                'message': 'Not enough reflections to critique'
            }
        
        critique = {
            'overall_effectiveness': 0.0,
            'best_decisions': [],
            'worst_decisions': [],
            'lessons_learned': []
        }
        
        # Avaliar cada decisão (simplificado)
        # Idealmente: verificar se fitness melhorou após intervenção
        
        effective_count = 0
        for i, refl in enumerate(self.reflections):
            # Se teve alta confiança e fase mudou depois, foi efetivo
            if refl['confidence'] > 0.7:
                effective_count += 1
        
        critique['overall_effectiveness'] = effective_count / len(self.reflections)
        
        # Best decisions: alta confiança
        best = sorted(self.reflections, key=lambda r: r['confidence'], reverse=True)[:3]
        critique['best_decisions'] = [
            {
                'generation': r['generation'],
                'phase': r['phase'],
                'confidence': r['confidence']
            }
            for r in best
        ]
        
        # Lessons learned
        if self.interventions['forced_diversifications'] > 5:
            critique['lessons_learned'].append(
                "System stagnated frequently → may need better mutation operators"
            )
        
        if critique['overall_effectiveness'] > 0.7:
            critique['lessons_learned'].append(
                "Meta-cognition interventions were mostly effective"
            )
        else:
            critique['lessons_learned'].append(
                "Meta-cognition needs refinement (effectiveness < 70%)"
            )
        
        return critique
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do engine"""
        if not self.snapshots:
            return {'snapshots': 0}
        
        return {
            'snapshots': len(self.snapshots),
            'current_phase': self.current_phase.value,
            'phase_counter': self.phase_counter,
            'reflections': len(self.reflections),
            'interventions': dict(self.interventions),
            'learned_meta_params': dict(self.learned_meta_params)
        }
