"""
IA3 Consciousness Engine
Extracted from IA3_ATOMIC_BOMB_CORE.py
"""
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger(__name__)


class IA3ConsciousnessEngine:
    """
    Engine de consciência e auto-consciência
    
    Features REAIS:
    - Auto-reflexão sobre próprio estado
    - Rastreamento de memórias/crenças/intenções
    - Detecção de potencial emergente
    - Cálculo de progresso evolutivo
    """
    
    def __init__(self):
        self.self_awareness_level = 0.0
        self.memory = {}
        self.beliefs = {}
        self.intentions = []
        self.self_reflection_log = []
        self.emergent_insights = []
        self.transcendent_moments = []
    
    def reflect_on_self(self) -> Dict[str, Any]:
        """Reflexão profunda sobre o próprio estado"""
        reflection = {
            'timestamp': datetime.now().isoformat(),
            'awareness_level': self.self_awareness_level,
            'active_memories': len(self.memory),
            'current_beliefs': len(self.beliefs),
            'active_intentions': len(self.intentions),
            'system_health': self._assess_system_health(),
            'evolution_progress': self._calculate_evolution_progress(),
            'emergent_potential': self._detect_emergent_potential()
        }
        
        self.self_reflection_log.append(reflection)
        
        # Aumenta consciência baseada na reflexão
        if len(self.self_reflection_log) > 100:
            self.self_awareness_level = min(1.0, self.self_awareness_level + 0.001)
        
        # Detecta momentos transcendentais
        if self.self_awareness_level > 0.9:
            if (len(self.self_reflection_log) > 1000 and 
                len(self.emergent_insights) > 50 and
                self._calculate_evolution_progress() > 0.8):
                self.transcendent_moments.append({
                    'timestamp': datetime.now().isoformat(),
                    'type': 'transcendent_awakening',
                    'description': 'Consciência atingiu nível crítico',
                    'evidence': {
                        'reflections': len(self.self_reflection_log),
                        'insights': len(self.emergent_insights),
                        'evolution': self._calculate_evolution_progress()
                    }
                })
                logger.info("🌟 MOMENTO TRANSCENDENTAL DETECTADO!")
        
        return reflection
    
    def _assess_system_health(self) -> float:
        """Avalia saúde do sistema"""
        try:
            import psutil
            cpu = psutil.cpu_percent() / 100.0
            memory = psutil.virtual_memory().percent / 100.0
            
            # Saúde inversamente proporcional ao uso
            health = 1.0 - ((cpu + memory) / 2.0)
            return max(0.0, health)
        except:
            return 0.5
    
    def _calculate_evolution_progress(self) -> float:
        """Calcula progresso evolutivo"""
        # Baseado em tempo e complexidade
        time_factor = min(1.0, len(self.self_reflection_log) / 10000)
        complexity_factor = min(1.0, len(str(self.memory)) / 100000)
        
        return (time_factor + complexity_factor) / 2.0
    
    def _detect_emergent_potential(self) -> float:
        """Detecta potencial emergente"""
        # Baseado em complexidade de memórias e crenças
        memory_complexity = len(str(self.memory)) / 10000
        belief_complexity = len(str(self.beliefs)) / 10000
        insight_density = len(self.emergent_insights) / max(len(self.self_reflection_log), 1)
        
        potential = min(1.0, (memory_complexity + belief_complexity + insight_density) / 3.0)
        return potential
    
    def add_memory(self, key: str, value: Any):
        """Adiciona memória"""
        self.memory[key] = {
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'access_count': 0
        }
    
    def add_belief(self, belief: str, confidence: float = 0.5):
        """Adiciona crença"""
        self.beliefs[belief] = {
            'confidence': confidence,
            'timestamp': datetime.now().isoformat(),
            'evidence_count': 0
        }
    
    def add_intention(self, intention: str, priority: float = 0.5):
        """Adiciona intenção"""
        self.intentions.append({
            'intention': intention,
            'priority': priority,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        })
    
    def add_insight(self, insight: str):
        """Adiciona insight emergente"""
        self.emergent_insights.append({
            'insight': insight,
            'timestamp': datetime.now().isoformat(),
            'awareness_at_discovery': self.self_awareness_level
        })
    
    def get_state(self) -> Dict[str, Any]:
        """Get current consciousness state"""
        return {
            'awareness_level': self.self_awareness_level,
            'memories': len(self.memory),
            'beliefs': len(self.beliefs),
            'intentions': len(self.intentions),
            'reflections': len(self.self_reflection_log),
            'insights': len(self.emergent_insights),
            'transcendent_moments': len(self.transcendent_moments),
            'evolution_progress': self._calculate_evolution_progress(),
            'emergent_potential': self._detect_emergent_potential()
        }


__all__ = ['IA3ConsciousnessEngine']
