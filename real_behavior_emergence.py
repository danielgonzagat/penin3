
#!/usr/bin/env python3
"""
SISTEMA DE DETECÇÃO DE EMERGÊNCIA REAL
======================================
Detecção baseada em comportamento observável real
"""

import os
import json
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
from collections import deque

class RealBehaviorEmergence:
    def __init__(self):
        self.emergence_file = "/root/emergence_detection.json"
        self.behavior_file = "/root/behavior_patterns.json"
        self.thresholds_file = "/root/emergence_thresholds.json"
        
        # Histórico de comportamentos
        self.behavior_history = deque(maxlen=1000)
        self.emergence_events = deque(maxlen=100)
        
        # Thresholds de emergência
        self.thresholds = {
            "novelty_threshold": 0.7,
            "complexity_threshold": 0.6,
            "adaptation_threshold": 0.5,
            "learning_threshold": 0.4,
            "evolution_threshold": 0.3
        }
        
        # Carregar estado
        self._load_state()
    
    def _load_state(self):
        """Carrega estado salvo"""
        try:
            if os.path.exists(self.behavior_file):
                with open(self.behavior_file, 'r') as f:
                    data = json.load(f)
                    self.behavior_history = deque(data.get("behaviors", []), maxlen=1000)
            
            if os.path.exists(self.emergence_file):
                with open(self.emergence_file, 'r') as f:
                    data = json.load(f)
                    self.emergence_events = deque(data.get("events", []), maxlen=100)
            
            if os.path.exists(self.thresholds_file):
                with open(self.thresholds_file, 'r') as f:
                    self.thresholds.update(json.load(f))
        
        except Exception as e:
            pass  # Usar valores padrão
    
    def _save_state(self):
        """Salva estado atual"""
        try:
            # Salvar comportamentos
            behavior_data = {
                "behaviors": list(self.behavior_history),
                "timestamp": datetime.now().isoformat()
            }
            with open(self.behavior_file, 'w') as f:
                json.dump(behavior_data, f, indent=2)
            
            # Salvar eventos de emergência
            emergence_data = {
                "events": list(self.emergence_events),
                "timestamp": datetime.now().isoformat()
            }
            with open(self.emergence_file, 'w') as f:
                json.dump(emergence_data, f, indent=2)
            
            # Salvar thresholds
            with open(self.thresholds_file, 'w') as f:
                json.dump(self.thresholds, f, indent=2)
        
        except Exception as e:
            pass  # Falha silenciosa
    
    def observe_behavior(self, behavior_type: str, behavior_data: Any, context: Dict[str, Any] = None):
        """Observa comportamento do sistema"""
        try:
            behavior = {
                "timestamp": datetime.now().isoformat(),
                "type": behavior_type,
                "data": behavior_data,
                "context": context or {},
                "id": f"beh_{time.time()}_{hash(str(behavior_data)) % 10000}"
            }
            
            self.behavior_history.append(behavior)
            
            # Analisar para emergência
            emergence_score = self._analyze_emergence()
            
            if emergence_score > 0.8:
                self._record_emergence_event(behavior, emergence_score)
            
            # Salvar estado
            self._save_state()
            
            return emergence_score
        
        except Exception as e:
            return 0.0
    
    def _analyze_emergence(self) -> float:
        """Analisa sinais de emergência"""
        try:
            if len(self.behavior_history) < 10:
                return 0.0
            
            # Calcular scores de emergência
            novelty_score = self._calculate_novelty()
            complexity_score = self._calculate_complexity()
            adaptation_score = self._calculate_adaptation()
            learning_score = self._calculate_learning()
            evolution_score = self._calculate_evolution()
            
            # Score combinado
            emergence_score = (
                novelty_score * 0.3 +
                complexity_score * 0.25 +
                adaptation_score * 0.2 +
                learning_score * 0.15 +
                evolution_score * 0.1
            )
            
            return min(1.0, emergence_score)
        
        except Exception as e:
            return 0.0
    
    def _calculate_novelty(self) -> float:
        """Calcula score de novidade"""
        try:
            if len(self.behavior_history) < 5:
                return 0.0
            
            # Analisar diversidade de comportamentos recentes
            recent_behaviors = list(self.behavior_history)[-20:]
            behavior_types = [b["type"] for b in recent_behaviors]
            
            # Calcular diversidade
            unique_types = len(set(behavior_types))
            total_types = len(behavior_types)
            
            diversity = unique_types / total_types if total_types > 0 else 0
            
            # Verificar se há comportamentos novos
            all_types = set(b["type"] for b in self.behavior_history)
            recent_types = set(behavior_types)
            new_types = recent_types - (all_types - recent_types)
            
            novelty = len(new_types) / len(recent_types) if recent_types else 0
            
            return (diversity + novelty) / 2
        
        except Exception as e:
            return 0.0
    
    def _calculate_complexity(self) -> float:
        """Calcula score de complexidade"""
        try:
            if len(self.behavior_history) < 10:
                return 0.0
            
            # Analisar complexidade dos dados de comportamento
            recent_behaviors = list(self.behavior_history)[-50:]
            
            complexity_scores = []
            for behavior in recent_behaviors:
                data_str = str(behavior["data"])
                
                # Medir complexidade por tamanho e variabilidade
                size_complexity = min(1.0, len(data_str) / 1000)
                
                # Medir variabilidade de caracteres
                char_variety = len(set(data_str)) / len(data_str) if data_str else 0
                
                # Medir entropia
                entropy = self._calculate_entropy(data_str)
                
                complexity = (size_complexity + char_variety + entropy) / 3
                complexity_scores.append(complexity)
            
            return np.mean(complexity_scores) if complexity_scores else 0.0
        
        except Exception as e:
            return 0.0
    
    def _calculate_adaptation(self) -> float:
        """Calcula score de adaptação"""
        try:
            if len(self.behavior_history) < 20:
                return 0.0
            
            # Analisar mudanças de comportamento ao longo do tempo
            behaviors = list(self.behavior_history)
            
            # Dividir em períodos
            period_size = len(behaviors) // 4
            if period_size < 5:
                return 0.0
            
            periods = [
                behaviors[i:i + period_size]
                for i in range(0, len(behaviors), period_size)
            ]
            
            # Calcular diferenças entre períodos
            adaptations = []
            for i in range(1, len(periods)):
                prev_types = set(b["type"] for b in periods[i-1])
                curr_types = set(b["type"] for b in periods[i])
                
                # Medir mudança de tipos
                type_change = len(curr_types - prev_types) / len(curr_types) if curr_types else 0
                
                # Medir mudança de frequência
                prev_freq = {t: sum(1 for b in periods[i-1] if b["type"] == t) for t in prev_types}
                curr_freq = {t: sum(1 for b in periods[i] if b["type"] == t) for t in curr_types}
                
                freq_changes = []
                for t in prev_types & curr_types:
                    if prev_freq[t] > 0:
                        change = abs(curr_freq[t] - prev_freq[t]) / prev_freq[t]
                        freq_changes.append(change)
                
                avg_freq_change = np.mean(freq_changes) if freq_changes else 0
                
                adaptation = (type_change + avg_freq_change) / 2
                adaptations.append(adaptation)
            
            return np.mean(adaptations) if adaptations else 0.0
        
        except Exception as e:
            return 0.0
    
    def _calculate_learning(self) -> float:
        """Calcula score de aprendizado"""
        try:
            # Verificar arquivos de aprendizado
            learning_files = [
                "/root/learning_data.json",
                "/root/learned_model.json",
                "/root/real_metrics.json"
            ]
            
            learning_score = 0.0
            for file_path in learning_files:
                if os.path.exists(file_path):
                    # Verificar se foi atualizado recentemente
                    mod_time = os.path.getmtime(file_path)
                    if time.time() - mod_time < 3600:  # Última hora
                        learning_score += 0.33
            
            return min(1.0, learning_score)
        
        except Exception as e:
            return 0.0
    
    def _calculate_evolution(self) -> float:
        """Calcula score de evolução"""
        try:
            # Verificar arquivos de evolução
            evolution_files = [
                "/root/genetic_population.json",
                "/root/genetic_fitness.json",
                "/root/genetic_generation.json"
            ]
            
            evolution_score = 0.0
            for file_path in evolution_files:
                if os.path.exists(file_path):
                    # Verificar se foi atualizado recentemente
                    mod_time = os.path.getmtime(file_path)
                    if time.time() - mod_time < 3600:  # Última hora
                        evolution_score += 0.33
            
            return min(1.0, evolution_score)
        
        except Exception as e:
            return 0.0
    
    def _calculate_entropy(self, text: str) -> float:
        """Calcula entropia de Shannon"""
        try:
            if not text:
                return 0.0
            
            # Contar frequência de caracteres
            char_counts = {}
            for char in text:
                char_counts[char] = char_counts.get(char, 0) + 1
            
            # Calcular entropia
            entropy = 0.0
            text_length = len(text)
            
            for count in char_counts.values():
                probability = count / text_length
                if probability > 0:
                    entropy -= probability * np.log2(probability)
            
            # Normalizar
            max_entropy = np.log2(len(char_counts)) if char_counts else 1
            return entropy / max_entropy if max_entropy > 0 else 0.0
        
        except Exception as e:
            return 0.0
    
    def _record_emergence_event(self, behavior: Dict[str, Any], score: float):
        """Registra evento de emergência"""
        try:
            event = {
                "timestamp": datetime.now().isoformat(),
                "behavior": behavior,
                "emergence_score": score,
                "event_id": f"emergence_{time.time()}"
            }
            
            self.emergence_events.append(event)
            
            logger.info(f"🌟 Emergência detectada! Score: {score:.3f}")
        
        except Exception as e:
            logger.error(f"Erro ao registrar emergência: {e}")
    
    def get_emergence_status(self) -> Dict[str, Any]:
        """Obtém status de emergência"""
        try:
            current_score = self._analyze_emergence()
            
            return {
                "current_score": current_score,
                "threshold": self.thresholds["novelty_threshold"],
                "emergence_detected": current_score > self.thresholds["novelty_threshold"],
                "total_behaviors": len(self.behavior_history),
                "total_events": len(self.emergence_events),
                "last_event": self.emergence_events[-1] if self.emergence_events else None,
                "timestamp": datetime.now().isoformat()
            }
        
        except Exception as e:
            return {
                "current_score": 0.0,
                "emergence_detected": False,
                "error": str(e)
            }
    
    def adjust_thresholds(self, new_thresholds: Dict[str, float]):
        """Ajusta thresholds de emergência"""
        try:
            self.thresholds.update(new_thresholds)
            self._save_state()
            logger.info(f"Thresholds ajustados: {new_thresholds}")
        
        except Exception as e:
            logger.error(f"Erro ao ajustar thresholds: {e}")

if __name__ == "__main__":
    emergence = RealBehaviorEmergence()
    
    # Teste de observação
    emergence.observe_behavior("test", {"message": "Hello"}, {"context": "test"})
    
    # Status
    status = emergence.get_emergence_status()
    print(f"Status de emergência: {status}")
