"""
NOVELTY SYSTEM - Emergência de comportamentos inéditos
Baseado em Novelty Search (Lehman & Stanley)
"""
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from collections import deque

logger = logging.getLogger(__name__)

class NoveltySystem:
    """
    Sistema de busca por novidade
    Recompensa comportamentos DIFERENTES, não apenas melhores
    """
    
    def __init__(self, 
                 k_nearest: int = 15,
                 archive_size: int = 500,
                 novelty_threshold: float = 0.5):
        """
        Args:
            k_nearest: Número de vizinhos para calcular novelty
            archive_size: Tamanho máximo do archive
            novelty_threshold: Threshold para adicionar ao archive
        """
        self.k_nearest = k_nearest
        self.k = self.k_nearest  # Alias para external access
        self.archive_size = archive_size
        self.novelty_threshold = novelty_threshold
        
        # Archive de comportamentos únicos
        self.behavior_archive: List[np.ndarray] = []
        self.archive = self.behavior_archive  # Alias (shared reference)
        self.archive_metadata: List[Dict] = []
        
        # Estatísticas
        self.behaviors_evaluated = 0
        self.novel_behaviors_found = 0
        self.average_novelty = 0.0
        
        logger.info("🎨 Novelty System initialized")
        logger.info(f"   k={k_nearest}, archive={archive_size}, threshold={novelty_threshold}")
    
    def characterize_behavior(self, trajectory: List[Any]) -> np.ndarray:
        """
        Converte trajetória em vetor de características
        
        Args:
            trajectory: Lista de estados/ações
        
        Returns:
            Vetor de comportamento
        """
        # Exemplo: últimas N posições
        if not trajectory:
            return np.zeros(10)
        
        # Pegar características relevantes
        features = []
        for state in trajectory[-10:]:  # Últimos 10 estados
            if isinstance(state, (list, tuple, np.ndarray)):
                features.extend(list(state)[:4])  # Primeiras 4 dimensões
            elif isinstance(state, (int, float)):
                features.append(float(state))
        
        # Pad or truncate to fixed size
        behavior_vector = np.array(features[:40] if len(features) >= 40 
                                   else features + [0]*(40-len(features)))
        return behavior_vector
    
    def calculate_novelty(self, behavior: np.ndarray) -> float:
        """
        Calcula novelty score como distância média aos K vizinhos mais próximos
        
        Args:
            behavior: Vetor de comportamento
        
        Returns:
            Novelty score (quanto maior, mais novo)
        """
        self.behaviors_evaluated += 1
        
        if len(self.behavior_archive) == 0:
            return 1.0  # Primeiro comportamento é sempre novel
        
        # Calcular distâncias a todos no archive
        distances = []
        for archived_behavior in self.behavior_archive:
            dist = np.linalg.norm(behavior - archived_behavior)
            distances.append(dist)
        
        # Média dos K mais próximos
        k = min(self.k_nearest, len(distances))
        closest_k = sorted(distances)[:k]
        novelty = np.mean(closest_k)
        
        return novelty
    
    def add_to_archive(self, behavior: np.ndarray, metadata: Dict = None) -> bool:
        """
        Adiciona comportamento ao archive se suficientemente novel
        
        Args:
            behavior: Vetor de comportamento
            metadata: Informações adicionais (fitness, etc)
        
        Returns:
            True se adicionado
        """
        novelty = self.calculate_novelty(behavior)
        
        if novelty >= self.novelty_threshold:
            self.behavior_archive.append(behavior.copy())
            self.archive_metadata.append(metadata or {})
            self.novel_behaviors_found += 1
            
            # Limitar tamanho do archive
            if len(self.behavior_archive) > self.archive_size:
                # Remove o mais antigo
                self.behavior_archive.pop(0)
                self.archive_metadata.pop(0)
            
            logger.debug(f"✨ Novel behavior added (novelty={novelty:.3f})")
            return True
        
        return False
    
    def reward_novelty(self, behavior: np.ndarray, base_fitness: float, 
                      novelty_weight: float = 0.5) -> float:
        """
        Combina fitness base com novelty bonus
        
        Args:
            behavior: Vetor de comportamento
            base_fitness: Fitness original
            novelty_weight: Peso do bonus de novelty (0-1)
        
        Returns:
            Fitness ajustado
        """
        novelty = self.calculate_novelty(behavior)
        
        # Atualizar média
        self.average_novelty = (self.average_novelty * 0.99 + novelty * 0.01)
        
        # Combinar fitness e novelty
        combined_fitness = (1 - novelty_weight) * base_fitness + novelty_weight * novelty
        
        # Adicionar ao archive se novel
        self.add_to_archive(behavior, {'fitness': base_fitness, 'novelty': novelty})
        
        return combined_fitness
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas do sistema"""
        return {
            'behaviors_evaluated': self.behaviors_evaluated,
            'novel_behaviors': self.novel_behaviors_found,
            'archive_size': len(self.behavior_archive),
            'average_novelty': self.average_novelty,
            'novelty_rate': self.novel_behaviors_found / max(self.behaviors_evaluated, 1)
        }
    
    def reset_archive(self):
        """Limpa o archive (útil para experimentos)"""
        self.behavior_archive.clear()
        self.archive_metadata.clear()
        self.novel_behaviors_found = 0
        logger.info("🔄 Novelty archive reset")


class CuriosityDrivenLearning:
    """
    Aprendizado guiado por curiosidade
    Recompensa estados SURPREENDENTES
    """
    
    def __init__(self, prediction_model: Any = None):
        """
        Args:
            prediction_model: Modelo para prever próximo estado (opcional)
        """
        self.prediction_model = prediction_model
        self.state_visit_counts: Dict[tuple, int] = {}
        self.total_visits = 0
        
        # Estatísticas
        self.curiosity_rewards = deque(maxlen=1000)
        
        logger.info("🔍 Curiosity-Driven Learning initialized")
    
    def count_based_curiosity(self, state: Any) -> float:
        """
        Curiosidade baseada em contagem: estados raros = mais curiosos
        
        Args:
            state: Estado atual
        
        Returns:
            Curiosity reward
        """
        # Discretizar estado para contagem
        if isinstance(state, np.ndarray):
            state_key = tuple(np.round(state, decimals=1))
        else:
            state_key = tuple(state) if isinstance(state, (list, tuple)) else (state,)
        
        # Contar visitas
        self.state_visit_counts[state_key] = self.state_visit_counts.get(state_key, 0) + 1
        self.total_visits += 1
        
        # Curiosidade = 1 / sqrt(count)
        count = self.state_visit_counts[state_key]
        curiosity = 1.0 / np.sqrt(count)
        
        self.curiosity_rewards.append(curiosity)
        
        return curiosity
    
    def prediction_error_curiosity(self, state: Any, next_state: Any) -> float:
        """
        Curiosidade baseada em erro de predição
        
        Args:
            state: Estado atual
            next_state: Próximo estado observado
        
        Returns:
            Curiosity reward (erro de predição)
        """
        if self.prediction_model is None:
            # Fallback: count-based
            return self.count_based_curiosity(next_state)
        
        # TODO: Implementar predição real
        # predicted = self.prediction_model.predict(state)
        # prediction_error = np.linalg.norm(next_state - predicted)
        # return prediction_error
        
        return 0.1  # Placeholder
    
    def get_statistics(self) -> Dict[str, Any]:
        """Retorna estatísticas de curiosidade"""
        return {
            'unique_states': len(self.state_visit_counts),
            'total_visits': self.total_visits,
            'avg_curiosity': np.mean(self.curiosity_rewards) if self.curiosity_rewards else 0.0,
            'exploration_rate': len(self.state_visit_counts) / max(self.total_visits, 1)
        }
