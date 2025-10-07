"""
✅ FASE 3.2: Sistema de Curiosidade - Busca Ativa por Novidade
==============================================================

Sistema que recompensa exploração de regiões desconhecidas.

Features:
- Recompensa intrínseca por novidade
- Predição de features (curiosity-driven)
- Contagem de visitas (exploration bonus)
- Recompensa por surpresa (prediction error)
- Motivação intrínseca vs extrínseca

Referências:
- Curiosity-driven Exploration (Pathak et al., 2017)
- Intrinsic Motivation (Schmidhuber)
- Random Network Distillation (Burda et al., 2018)
- Exploration bonuses in RL
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CuriosityMetrics:
    """Métricas de curiosidade"""
    novelty_score: float  # Quão novo é o estado
    prediction_error: float  # Erro de predição (surpresa)
    visit_count: int  # Quantas vezes visitou região similar
    intrinsic_reward: float  # Recompensa intrínseca total
    

class CuriosityEngine:
    """
    Motor de Curiosidade - busca ativa por novidade.
    
    Implementa:
    - Modelo Forward (prediz próximo estado)
    - Modelo Inverse (prediz ação que levou ao estado)
    - Exploration bonuses baseado em visitas
    - Recompensa intrínseca por novidade
    
    Motivação: "Curiosity = Desire to reduce uncertainty"
    
    Uso:
        curiosity = CuriosityEngine(state_dim=784, action_dim=10)
        
        # Durante evolução:
        state = observation
        next_state = next_observation
        action = predicted_action
        
        # Calcular curiosidade
        metrics = curiosity.compute_curiosity(state, next_state, action)
        
        # Combinar com recompensa extrínseca
        total_reward = external_reward + metrics.intrinsic_reward
        
        # Treinar modelos de curiosidade
        curiosity.update_models(state, next_state, action)
    """
    
    def __init__(self, 
                 state_dim: int,
                 action_dim: int,
                 feature_dim: int = 128,
                 learning_rate: float = 0.001):
        """
        Args:
            state_dim: Dimensão do estado
            action_dim: Dimensão da ação
            feature_dim: Dimensão do espaço de features
            learning_rate: Taxa de aprendizado
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feature_dim = feature_dim
        self.learning_rate = learning_rate
        
        # Modelos de curiosidade
        self.feature_encoder = self._build_feature_encoder()
        self.forward_model = self._build_forward_model()
        self.inverse_model = self._build_inverse_model()
        
        # Otimizadores
        self.optimizer_forward = torch.optim.Adam(
            list(self.feature_encoder.parameters()) + list(self.forward_model.parameters()),
            lr=learning_rate
        )
        self.optimizer_inverse = torch.optim.Adam(
            self.inverse_model.parameters(),
            lr=learning_rate
        )
        
        # Rastreamento de visitas (exploration bonus)
        self.visit_counts = defaultdict(int)
        self.state_history = []
        
        # Estatísticas
        self.total_states_seen = 0
        self.avg_novelty = 0.0
        self.avg_prediction_error = 0.0
    
    def _build_feature_encoder(self) -> nn.Module:
        """Encoder de features (compartilhado)"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim),
            nn.ReLU()
        )
    
    def _build_forward_model(self) -> nn.Module:
        """
        Modelo Forward: prediz feature(next_state) dado feature(state) + action.
        
        Feature(s_t), a_t → Feature(s_{t+1})
        """
        return nn.Sequential(
            nn.Linear(self.feature_dim + self.action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, self.feature_dim)
        )
    
    def _build_inverse_model(self) -> nn.Module:
        """
        Modelo Inverse: prediz ação dado feature(state) e feature(next_state).
        
        Feature(s_t), Feature(s_{t+1}) → a_t
        """
        return nn.Sequential(
            nn.Linear(self.feature_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, self.action_dim)
        )
    
    def compute_curiosity(self, 
                         state: np.ndarray,
                         next_state: np.ndarray,
                         action: np.ndarray) -> CuriosityMetrics:
        """
        Calcula métricas de curiosidade para uma transição.
        
        Args:
            state: Estado atual
            next_state: Próximo estado
            action: Ação tomada
        
        Returns:
            CuriosityMetrics com scores de curiosidade
        """
        self.total_states_seen += 1
        
        # Converter para tensors
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        action_t = torch.FloatTensor(action).unsqueeze(0)
        
        with torch.no_grad():
            # Encode states
            phi_state = self.feature_encoder(state_t)
            phi_next = self.feature_encoder(next_state_t)
            
            # Predição forward
            phi_next_pred = self.forward_model(torch.cat([phi_state, action_t], dim=1))
            
            # Erro de predição (surpresa) = curiosidade!
            prediction_error = nn.functional.mse_loss(phi_next_pred, phi_next).item()
            
            # Novelty score baseado em distância no espaço de features
            # (mais distante dos estados já vistos = mais novo)
            novelty = self._compute_novelty(phi_state)
            
            # Visit count (exploration bonus)
            state_key = self._hash_state(state)
            visit_count = self.visit_counts[state_key]
            self.visit_counts[state_key] += 1
            
            # Bonus por baixa contagem de visitas
            visit_bonus = 1.0 / (1.0 + visit_count) ** 0.5
            
            # Recompensa intrínseca total
            intrinsic_reward = (
                0.5 * prediction_error +  # Surpresa
                0.3 * novelty +           # Novidade
                0.2 * visit_bonus         # Exploração
            )
        
        # Atualizar estatísticas
        self.avg_novelty = 0.9 * self.avg_novelty + 0.1 * novelty
        self.avg_prediction_error = 0.9 * self.avg_prediction_error + 0.1 * prediction_error
        
        return CuriosityMetrics(
            novelty_score=float(novelty),
            prediction_error=float(prediction_error),
            visit_count=visit_count,
            intrinsic_reward=float(intrinsic_reward)
        )
    
    def _compute_novelty(self, phi_state: torch.Tensor) -> float:
        """
        Calcula novelty como distância média aos estados já vistos.
        
        Args:
            phi_state: Feature do estado
        
        Returns:
            Novelty score (0-1)
        """
        if len(self.state_history) == 0:
            return 1.0  # Primeiro estado = muito novo
        
        # Distância aos k vizinhos mais próximos
        k = min(10, len(self.state_history))
        distances = []
        
        for hist_state in self.state_history[-100:]:  # Últimos 100
            dist = torch.norm(phi_state - hist_state).item()
            distances.append(dist)
        
        distances.sort()
        avg_dist = np.mean(distances[:k])
        
        # Normalizar para [0, 1]
        novelty = np.tanh(avg_dist)
        
        # Adicionar ao histórico
        self.state_history.append(phi_state.detach().clone())
        
        # Manter histórico limitado
        if len(self.state_history) > 1000:
            self.state_history.pop(0)
        
        return float(novelty)
    
    def _hash_state(self, state: np.ndarray) -> str:
        """Hash de estado para contagem de visitas"""
        # Discretizar estado para criar buckets
        discretized = (state * 10).astype(int)
        return str(discretized.tobytes())
    
    def update_models(self, 
                     state: np.ndarray,
                     next_state: np.ndarray,
                     action: np.ndarray):
        """
        Treina modelos de curiosidade (forward e inverse).
        
        Args:
            state: Estado atual
            next_state: Próximo estado
            action: Ação tomada
        """
        # Converter para tensors
        state_t = torch.FloatTensor(state).unsqueeze(0)
        next_state_t = torch.FloatTensor(next_state).unsqueeze(0)
        action_t = torch.FloatTensor(action).unsqueeze(0)
        
        # === Treinar Forward Model ===
        self.optimizer_forward.zero_grad()
        
        phi_state = self.feature_encoder(state_t)
        phi_next = self.feature_encoder(next_state_t).detach()  # Target
        
        phi_next_pred = self.forward_model(torch.cat([phi_state, action_t], dim=1))
        
        loss_forward = nn.functional.mse_loss(phi_next_pred, phi_next)
        
        loss_forward.backward()
        self.optimizer_forward.step()
        
        # === Treinar Inverse Model ===
        self.optimizer_inverse.zero_grad()
        
        phi_state = self.feature_encoder(state_t).detach()
        phi_next = self.feature_encoder(next_state_t).detach()
        
        action_pred = self.inverse_model(torch.cat([phi_state, phi_next], dim=1))
        
        loss_inverse = nn.functional.mse_loss(action_pred, action_t)
        
        loss_inverse.backward()
        self.optimizer_inverse.step()
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas de curiosidade"""
        return {
            'total_states_seen': self.total_states_seen,
            'unique_states': len(self.visit_counts),
            'avg_novelty': float(self.avg_novelty),
            'avg_prediction_error': float(self.avg_prediction_error),
            'state_history_size': len(self.state_history)
        }
    
    def visualize_curiosity(self) -> str:
        """Visualização ASCII da curiosidade"""
        stats = self.get_statistics()
        
        vis = []
        vis.append("═" * 60)
        vis.append("🔍 CURIOSITY ENGINE REPORT")
        vis.append("═" * 60)
        vis.append(f"\n📊 Exploration Statistics:")
        vis.append(f"   Total states seen: {stats['total_states_seen']}")
        vis.append(f"   Unique states: {stats['unique_states']}")
        vis.append(f"   Avg novelty: {stats['avg_novelty']:.4f}")
        vis.append(f"   Avg prediction error: {stats['avg_prediction_error']:.4f}")
        
        # Coverage
        if stats['total_states_seen'] > 0:
            coverage = stats['unique_states'] / stats['total_states_seen']
            vis.append(f"\n🗺️  State space coverage: {coverage:.1%}")
        
        # Curiosity level
        curiosity_level = (stats['avg_novelty'] + stats['avg_prediction_error']) / 2
        vis.append(f"\n🤔 Overall curiosity level: {curiosity_level:.1%}")
        
        if curiosity_level > 0.5:
            vis.append("   → High curiosity (actively exploring)")
        elif curiosity_level > 0.2:
            vis.append("   → Moderate curiosity (balanced)")
        else:
            vis.append("   → Low curiosity (exploiting known regions)")
        
        vis.append("\n" + "═" * 60)
        
        return "\n".join(vis)


def integrate_curiosity_into_evolution(population: List,
                                      curiosity_weight: float = 0.3):
    """
    Integra curiosidade na evolução.
    
    Args:
        population: População de indivíduos
        curiosity_weight: Peso da recompensa intrínseca
    
    Returns:
        População com fitness ajustado por curiosidade
    """
    # Placeholder: na prática, cada indivíduo teria seu curiosity engine
    # e durante avaliação de fitness, adiciona recompensa intrínseca
    
    for ind in population:
        if hasattr(ind, 'curiosity_metrics'):
            # Combinar fitness extrínseco + intrínseco
            external_fitness = ind.fitness
            intrinsic_reward = ind.curiosity_metrics.intrinsic_reward
            
            ind.fitness = (1 - curiosity_weight) * external_fitness + curiosity_weight * intrinsic_reward
    
    return population
