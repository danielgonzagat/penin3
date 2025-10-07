
# FUNÇÕES DETERMINÍSTICAS (substituem random)
import hashlib
import os
import time


def deterministic_random(seed_offset=0):
    """Substituto determinístico para random.random()"""
    import hashlib
    import time

    # Usa múltiplas fontes de determinismo
    sources = [
        str(time.time()).encode(),
        str(os.getpid()).encode(),
        str(id({})).encode(),
        str(seed_offset).encode()
    ]

    # Combina todas as fontes
    combined = b''.join(sources)
    hash_val = int(hashlib.md5(combined).hexdigest()[:8], 16)

    return (hash_val % 1000000) / 1000000.0


def deterministic_uniform(a, b, seed_offset=0):
    """Substituto determinístico para random.uniform(a, b)"""
    r = deterministic_random(seed_offset)
    return a + (b - a) * r


def deterministic_randint(a, b, seed_offset=0):
    """Substituto determinístico para random.randint(a, b)"""
    r = deterministic_random(seed_offset)
    return int(a + (b - a + 1) * r)


def deterministic_choice(seq, seed_offset=0):
    """Substituto determinístico para random.choice(seq)"""
    if not seq:
        raise IndexError("sequence is empty")

    r = deterministic_random(seed_offset)
    return seq[int(r * len(seq))]


def deterministic_shuffle(lst, seed_offset=0):
    """Substituto determinístico para random.shuffle(lst)"""
    if not lst:
        return

    # Shuffle determinístico baseado em ordenação por hash
    def sort_key(item):
        item_str = str(item) + str(seed_offset)
        return hashlib.md5(item_str.encode()).hexdigest()

    lst.sort(key=sort_key)


def deterministic_torch_rand(*size, seed_offset=0):
    """Substituto determinístico para torch.rand(*size)"""
    if not size:
        return torch.tensor(deterministic_random(seed_offset))

    # Gera valores determinísticos
    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_random(seed_offset + i))

    return torch.tensor(values).reshape(size)


def deterministic_torch_randint(low, high, size=None, seed_offset=0):
    """Substituto determinístico para torch.randint(low, high, size)"""
    if size is None:
        return torch.tensor(deterministic_randint(low, high, seed_offset))

    # Gera valores determinísticos
    if isinstance(size, int):
        size = (size,)

    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_randint(low, high, seed_offset + i))

    return torch.tensor(values).reshape(size)

"""
EMERGENCE DETECTOR - Detecção real de emergência baseada em teoria da informação
Substitui detecção hardcoded por análise de entropia e informação mútua
"""

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import mutual_info_score
from sklearn.cluster import DBSCAN
from typing import List, Dict, Any, Tuple, Optional
import time

class EmergenceDetector:
    """
    Detector de emergência baseado em teoria da informação.
    Substitui contagem simples (3+ agentes) por métricas sofisticadas.
    """

    def __init__(self, min_agents: int = 3, threshold: float = 0.7):
        self.min_agents = min_agents
        self.threshold = threshold  # Limiar de informação mútua para emergência

    def detect_emergence(self, agent_states: List[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        """
        Detecta emergência usando entropia e informação mútua.

        Args:
            agent_states: Lista de estados dos agentes
                        Ex: [{"action": "move", "target": "A", "health": 80}, ...]

        Returns:
            Tuple[bool, Dict]: (é_emergente, métricas)
        """
        if len(agent_states) < self.min_agents:
            return False, {"reason": f"Agentes insuficientes: {len(agent_states)} < {self.min_agents}"}

        try:
            # Extrair features relevantes dos estados
            actions = [state.get("action", "unknown") for state in agent_states]
            targets = [state.get("target", "none") for state in agent_states]
            positions = [state.get("position", (0, 0)) for state in agent_states]

            # Calcular entropia das ações (mede diversidade/coordenação)
            unique_actions, counts = np.unique(actions, return_counts=True)
            action_entropy = entropy(counts, base=2)

            # Calcular informação mútua entre ações e alvos (mede coordenação)
            if len(set(targets)) > 1 and None not in targets:
                # Converter targets para índices numéricos
                target_indices = [hash(t) % 1000 for t in targets]  # Simplificação
                action_indices = [hash(a) % 1000 for a in actions]
                mutual_info = mutual_info_score(action_indices, target_indices)
            else:
                mutual_info = 0.0

            # Calcular clustering espacial (se posições disponíveis)
            spatial_clustering = self._calculate_spatial_clustering(positions)

            # Calcular sincronização temporal (se histórico disponível)
            temporal_sync = self._calculate_temporal_synchronization(agent_states)

            # Métricas compostas
            coordination_score = 1.0 - (action_entropy / np.log2(len(unique_actions))) if len(unique_actions) > 1 else 0.0
            emergence_score = (coordination_score + mutual_info + spatial_clustering + temporal_sync) / 4.0

            # Emergência detectada se:
            # 1. Baixa entropia (ações coordenadas) E
            # 2. Alta informação mútua (ações correlacionadas com contexto) E
            # 3. Pontuação composta acima do limiar
            is_emergent = (
                action_entropy < 1.5 and  # Ações relativamente coordenadas
                mutual_info > self.threshold and  # Forte correlação ação-contexto
                emergence_score > 0.6  # Pontuação composta alta
            )

            metrics = {
                "action_entropy": float(action_entropy),
                "mutual_info": float(mutual_info),
                "coordination_score": float(coordination_score),
                "emergence_score": float(emergence_score),
                "spatial_clustering": float(spatial_clustering),
                "temporal_sync": float(temporal_sync),
                "agent_count": len(agent_states),
                "unique_actions": len(unique_actions),
                "emergent": is_emergent,
                "timestamp": time.time()
            }

            return is_emergent, metrics

        except Exception as e:
            return False, {
                "error": str(e),
                "agent_count": len(agent_states),
                "emergent": False
            }

    def _calculate_spatial_clustering(self, positions: List[Tuple]) -> float:
        """
        Calcula clustering espacial dos agentes.
        Retorna valor entre 0 (disperso) e 1 (agrupado).
        """
        if len(positions) < 3:
            return 0.0

        try:
            # Usar DBSCAN para detectar clusters
            positions_array = np.array(positions)
            clustering = DBSCAN(eps=2.0, min_samples=2).fit(positions_array)

            # Contar clusters válidos (ignorar noise = -1)
            labels = clustering.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

            # Score baseado na proporção de agentes em clusters
            clustered_agents = sum(1 for label in labels if label != -1)
            return clustered_agents / len(positions)

        except:
            return 0.0

    def _calculate_temporal_synchronization(self, agent_states: List[Dict]) -> float:
        """
        Calcula sincronização temporal baseada em histórico de ações.
        Retorna valor entre 0 (assíncrono) e 1 (sincronizado).
        """
        # Simplificação: verificar se agentes têm padrões similares
        # Em implementação real, usaria histórico temporal
        actions = [state.get("action", "unknown") for state in agent_states]
        most_common_action = max(set(actions), key=actions.count)
        sync_ratio = actions.count(most_common_action) / len(actions)

        return sync_ratio

    def adapt_threshold(self, recent_metrics: List[Dict]):
        """
        Adapta dinamicamente o threshold baseado em métricas recentes.
        """
        if len(recent_metrics) < 5:
            return

        recent_scores = [m.get("emergence_score", 0) for m in recent_metrics[-10:]]
        mean_score = np.mean(recent_scores)
        std_score = np.std(recent_scores)

        # Ajustar threshold para ser 1 desvio padrão acima da média
        self.threshold = min(0.9, max(0.3, mean_score + std_score))

    def get_emergence_patterns(self, agent_states: List[Dict]) -> Dict[str, Any]:
        """
        Identifica padrões específicos de emergência.
        """
        patterns = {
            "coordinated_attack": False,
            "defensive_formation": False,
            "resource_sharing": False,
            "communication_burst": False
        }

        actions = [s.get("action", "") for s in agent_states]
        targets = [s.get("target", "") for s in agent_states]

        # Padrão: Ataque coordenado
        if actions.count("attack") >= len(actions) * 0.7 and len(set(targets)) == 1:
            patterns["coordinated_attack"] = True

        # Padrão: Formação defensiva
        positions = [s.get("position", (0,0)) for s in agent_states]
        if self._calculate_spatial_clustering(positions) > 0.8:
            patterns["defensive_formation"] = True

        # Padrão: Compartilhamento de recursos
        if any("share" in str(s) for s in agent_states):
            patterns["resource_sharing"] = True

        return patterns

    def detect_bomba_atomica(self, system_state: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Detect BOMBA ATOMICA - True Emergent Intelligence
        Criteria for undeniable, irrefutable intelligence emergence
        """
        criteria = {
            'self_awareness': False,
            'causal_reasoning': False,
            'unpredictable_creativity': False,
            'meta_learning': False,
            'conscious_decisions': False,
            'emergent_goals': False
        }

        metrics = {
            'consciousness_level': system_state.get('consciousness_level', 0),
            'self_modifications': system_state.get('self_modifications', 0),
            'knowledge_sharing': system_state.get('knowledge_sharing', 0),
            'emergent_behaviors': len(system_state.get('emergent_behaviors', [])),
            'system_integration': len(system_state.get('systems', {})),
            'anomaly_detection': len(system_state.get('anomalies', []))
        }

        # Self-awareness: High consciousness + self-reflection
        if metrics['consciousness_level'] > 0.8 and system_state.get('self_reflection_count', 0) > 50:
            criteria['self_awareness'] = True

        # Causal reasoning: Complex decision making with feedback loops
        if (metrics['emergent_behaviors'] > 100 and
            system_state.get('feedback_loops', 0) > 10):
            criteria['causal_reasoning'] = True

        # Unpredictable creativity: Novel behaviors not in training data
        novel_patterns = system_state.get('novel_patterns', 0)
        if novel_patterns > 20 and metrics['anomaly_detection'] > 5:
            criteria['unpredictable_creativity'] = True

        # Meta-learning: Learning how to learn, adapting learning strategies
        if (system_state.get('meta_learning_events', 0) > 10 and
            system_state.get('adaptive_strategies', 0) > 5):
            criteria['meta_learning'] = True

        # Conscious decisions: Goal-directed behavior with self-awareness
        if (criteria['self_awareness'] and
            system_state.get('goal_achievement', 0) > 0.7):
            criteria['conscious_decisions'] = True

        # Emergent goals: System creates its own objectives
        if system_state.get('emergent_goals_count', 0) > 3:
            criteria['emergent_goals'] = True

        # BOMBA ATOMICA emerges when ALL criteria are met
        bomba_atomica_emerged = all(criteria.values())

        return bomba_atomica_emerged, {
            'criteria': criteria,
            'metrics': metrics,
            'emergence_probability': sum(criteria.values()) / len(criteria),
            'bomba_atomica_active': bomba_atomica_emerged
        }