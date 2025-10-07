#!/usr/bin/env python3
"""
♾️ RECURSIVE IMPROVEMENT ENGINE - FASE 4
Meta-meta-learning e auto-melhoria recursiva
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from collections import deque
from brain_logger import brain_logger

class MetaLearningHistory:
    """Histórico de meta-aprendizado"""
    def __init__(self, max_size: int = 1000):
        self.history = deque(maxlen=max_size)
        self.meta_patterns = []
    
    def add_episode(self, episode_data: Dict[str, Any]):
        """Adiciona episódio ao histórico"""
        self.history.append(episode_data)
    
    def extract_patterns(self) -> List[Dict[str, Any]]:
        """
        Extrai padrões de meta-aprendizado
        
        Returns:
            Lista de padrões descobertos
        """
        if len(self.history) < 20:
            return []
        
        patterns = []
        
        # Padrão 1: Intervenções efetivas
        interventions = [h for h in self.history if h.get('meta_intervention')]
        if interventions:
            effective = [i for i in interventions if i.get('reward_after', 0) > i.get('reward_before', 0)]
            if len(effective) > len(interventions) * 0.6:  # 60% de sucesso
                patterns.append({
                    'type': 'effective_interventions',
                    'confidence': len(effective) / len(interventions),
                    'description': 'Meta-controller interventions are generally effective'
                })
        
        # Padrão 2: Curiosity patterns
        curiosity_data = [h.get('curiosity_surprise', 0) for h in list(self.history)[-50:]]
        if curiosity_data and len(curiosity_data) > 10:
            trend = (curiosity_data[-10:][0] - curiosity_data[:10][0]) / max(curiosity_data[:10][0], 0.001)
            if trend < -0.3:  # Surprise decrescente
                patterns.append({
                    'type': 'learning_progress',
                    'confidence': abs(trend),
                    'description': 'System is learning (curiosity decreasing)'
                })
        
        # Padrão 3: Arquiteturas bem-sucedidas
        arch_performance = {}
        for h in self.history:
            arch = h.get('architecture', 'unknown')
            reward = h.get('reward', 0)
            if arch not in arch_performance:
                arch_performance[arch] = []
            arch_performance[arch].append(reward)
        
        if arch_performance:
            best_arch = max(arch_performance.items(), key=lambda x: sum(x[1])/len(x[1]))
            if len(best_arch[1]) > 5:  # Pelo menos 5 samples
                patterns.append({
                    'type': 'best_architecture',
                    'confidence': sum(best_arch[1]) / (len(best_arch[1]) * 100),  # Normalized
                    'description': f'Architecture {best_arch[0]} performs best',
                    'architecture': best_arch[0]
                })
        
        self.meta_patterns = patterns
        return patterns


class RecursiveImprovementEngine:
    """
    Engine de melhoria recursiva
    Aprende sobre seu próprio aprendizado e se melhora
    """
    def __init__(self):
        self.meta_history = MetaLearningHistory()
        self.improvement_cycles = []
        self.current_strategy = 'exploration'  # exploration, exploitation, synthesis
        self.strategy_performance = {
            'exploration': [],
            'exploitation': [],
            'synthesis': []
        }
    
    def observe_cycle(self, cycle_data: Dict[str, Any]):
        """
        Observa um ciclo de aprendizado
        
        Args:
            cycle_data: Dados do ciclo (episódio, interventions, etc.)
        """
        self.meta_history.add_episode(cycle_data)
        
        # Track strategy performance
        strategy = cycle_data.get('strategy', 'exploration')
        reward = cycle_data.get('reward', 0)
        self.strategy_performance[strategy].append(reward)
    
    def meta_learn(self) -> Dict[str, Any]:
        """
        Meta-aprendizado: aprende sobre o próprio processo de aprendizado
        
        Returns:
            Insights e recomendações meta
        """
        patterns = self.meta_history.extract_patterns()
        
        meta_insights = {
            'timestamp': datetime.now().isoformat(),
            'patterns_found': len(patterns),
            'patterns': patterns,
            'strategy_analysis': self._analyze_strategies(),
            'recommendations': []
        }
        
        # Gera recomendações baseadas em patterns
        for pattern in patterns:
            if pattern['type'] == 'effective_interventions' and pattern['confidence'] > 0.7:
                meta_insights['recommendations'].append({
                    'action': 'increase_intervention_frequency',
                    'reason': 'Interventions are highly effective',
                    'priority': 'HIGH'
                })
            
            elif pattern['type'] == 'learning_progress' and pattern['confidence'] > 0.3:
                meta_insights['recommendations'].append({
                    'action': 'maintain_current_strategy',
                    'reason': 'System is learning well',
                    'priority': 'MEDIUM'
                })
            
            elif pattern['type'] == 'best_architecture':
                meta_insights['recommendations'].append({
                    'action': f"prioritize_architecture_{pattern['architecture']}",
                    'reason': f"Architecture shows superior performance",
                    'priority': 'HIGH'
                })
        
        return meta_insights
    
    def _analyze_strategies(self) -> Dict[str, Any]:
        """Analisa performance de estratégias"""
        analysis = {}
        
        for strategy, rewards in self.strategy_performance.items():
            if rewards:
                analysis[strategy] = {
                    'avg_reward': sum(rewards) / len(rewards),
                    'samples': len(rewards),
                    'trend': self._calculate_trend(rewards)
                }
        
        # Seleciona melhor estratégia
        if analysis:
            best_strategy = max(analysis.items(), key=lambda x: x[1]['avg_reward'])
            analysis['best_strategy'] = best_strategy[0]
        
        return analysis
    
    def _calculate_trend(self, values: List[float]) -> str:
        """Calcula tendência"""
        if len(values) < 10:
            return 'insufficient_data'
        
        recent = values[-5:]
        old = values[:5]
        
        trend_value = (sum(recent)/len(recent) - sum(old)/len(old)) / max(sum(old)/len(old), 0.1)
        
        if trend_value > 0.1:
            return 'improving'
        elif trend_value < -0.1:
            return 'degrading'
        else:
            return 'stable'
    
    def select_next_strategy(self) -> str:
        """
        Seleciona próxima estratégia baseado em meta-learning
        
        Returns:
            Nome da estratégia
        """
        analysis = self._analyze_strategies()
        
        if not analysis or 'best_strategy' not in analysis:
            return 'exploration'  # Default
        
        best = analysis['best_strategy']
        best_trend = analysis[best].get('trend', 'stable')
        
        # Se melhor estratégia está melhorando, continua
        if best_trend == 'improving':
            self.current_strategy = best
        # Se está degradando, testa outra
        elif best_trend == 'degrading':
            strategies = ['exploration', 'exploitation', 'synthesis']
            strategies.remove(best)
            self.current_strategy = strategies[0]  # Tenta a próxima
        # Se estável, alterna para explorar
        else:
            if self.current_strategy == 'exploitation':
                self.current_strategy = 'exploration'
            else:
                self.current_strategy = 'exploitation'
        
        brain_logger.info(f"♾️ Strategy selected: {self.current_strategy}")
        return self.current_strategy
    
    def recursive_improve(self, current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """
        Melhoria recursiva completa
        
        Args:
            current_performance: Performance atual do sistema
            
        Returns:
            Plano de melhoria
        """
        # Meta-learn
        meta_insights = self.meta_learn()
        
        # Seleciona estratégia
        next_strategy = self.select_next_strategy()
        
        # Gera plano de melhoria
        improvement_plan = {
            'timestamp': datetime.now().isoformat(),
            'cycle': len(self.improvement_cycles),
            'meta_insights': meta_insights,
            'current_strategy': self.current_strategy,
            'next_strategy': next_strategy,
            'actions': []
        }
        
        # Define ações baseadas na estratégia
        if next_strategy == 'exploration':
            improvement_plan['actions'].extend([
                {'type': 'increase_curiosity_weight', 'param': 0.15},
                {'type': 'increase_temperature', 'param': 1.2},
                {'type': 'increase_top_k', 'param': 2}
            ])
        elif next_strategy == 'exploitation':
            improvement_plan['actions'].extend([
                {'type': 'decrease_curiosity_weight', 'param': 0.05},
                {'type': 'decrease_temperature', 'param': 0.8},
                {'type': 'focus_best_neurons', 'param': 'top_5'}
            ])
        else:  # synthesis
            improvement_plan['actions'].extend([
                {'type': 'synthesize_neuron', 'architecture': 'auto'},
                {'type': 'prune_weak_neurons', 'threshold': 0.3}
            ])
        
        # Adiciona recomendações do meta-learning
        for rec in meta_insights['recommendations']:
            if rec['priority'] == 'HIGH':
                improvement_plan['actions'].append({
                    'type': 'meta_recommendation',
                    'action': rec['action'],
                    'reason': rec['reason']
                })
        
        self.improvement_cycles.append(improvement_plan)
        
        brain_logger.warning(
            f"♾️ RECURSIVE IMPROVEMENT cycle {len(self.improvement_cycles)}: "
            f"{len(improvement_plan['actions'])} actions planned"
        )
        
        return improvement_plan
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do engine"""
        return {
            'total_cycles': len(self.improvement_cycles),
            'patterns_discovered': len(self.meta_history.meta_patterns),
            'current_strategy': self.current_strategy,
            'strategy_performance': {
                k: {
                    'avg': sum(v)/len(v) if v else 0,
                    'count': len(v)
                }
                for k, v in self.strategy_performance.items()
            },
            'last_cycle': self.improvement_cycles[-1] if self.improvement_cycles else None
        }


class SelfImprovementLoop:
    """
    Loop completo de auto-melhoria
    Integra todos os componentes de FASE 3 e 4
    """
    def __init__(self, synthesizer, recursive_engine):
        self.synthesizer = synthesizer
        self.recursive_engine = recursive_engine
        self.active = True
        self.iterations = 0
    
    def iterate(self, system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uma iteração do loop de auto-melhoria
        
        Args:
            system_state: Estado completo do sistema
            
        Returns:
            Ações a serem tomadas
        """
        self.iterations += 1
        
        # Observa ciclo atual
        cycle_data = {
            'episode': system_state.get('episode', 0),
            'reward': system_state.get('reward', 0),
            'strategy': self.recursive_engine.current_strategy,
            'curiosity_surprise': system_state.get('curiosity_surprise', 0),
            'architecture': system_state.get('architecture', 'default')
        }
        
        self.recursive_engine.observe_cycle(cycle_data)
        
        # A cada N episódios, executa melhoria recursiva
        if system_state.get('episode', 0) % 20 == 0:
            improvement_plan = self.recursive_engine.recursive_improve(system_state)
            
            # Checa se deve sintetizar
            if any(a['type'] == 'synthesize_neuron' for a in improvement_plan['actions']):
                synthesis_result = self.synthesizer.auto_synthesize(
                    system_state.get('analysis_report', {}),
                    system_state.get('performance_stats', {})
                )
                
                if synthesis_result:
                    improvement_plan['synthesis_result'] = synthesis_result['info']
            
            return improvement_plan
        
        return {'actions': []}
    
    def get_status(self) -> Dict[str, Any]:
        """Status do loop"""
        return {
            'active': self.active,
            'iterations': self.iterations,
            'recursive_engine_stats': self.recursive_engine.get_stats(),
            'synthesizer_stats': self.synthesizer.get_stats()
        }


if __name__ == "__main__":
    print("♾️ Recursive Improvement Engine - FASE 4")
    
    # Test
    engine = RecursiveImprovementEngine()
    
    # Simula ciclos
    for i in range(30):
        strategy = ['exploration', 'exploitation', 'synthesis'][i % 3]
        cycle_data = {
            'episode': i,
            'reward': 20 + i * 2,
            'strategy': strategy,
            'curiosity_surprise': 0.1 - i * 0.002
        }
        engine.observe_cycle(cycle_data)
    
    # Meta-learn
    insights = engine.meta_learn()
    print(f"\n✅ Meta-learning:")
    print(f"   Patterns found: {len(insights['patterns'])}")
    for p in insights['patterns']:
        print(f"   - {p['type']}: {p['description']}")
    
    # Recursive improve
    improvement = engine.recursive_improve({'episode': 30, 'reward': 50})
    print(f"\n✅ Improvement plan:")
    print(f"   Strategy: {improvement['next_strategy']}")
    print(f"   Actions: {len(improvement['actions'])}")
    
    stats = engine.get_stats()
    print(f"\n✅ Stats:")
    print(f"   Cycles: {stats['total_cycles']}")
    print(f"   Patterns: {stats['patterns_discovered']}")
    
    print("\n✅ Recursive Improvement OK")
