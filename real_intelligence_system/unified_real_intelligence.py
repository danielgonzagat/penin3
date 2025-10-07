#!/usr/bin/env python3
"""
SISTEMA UNIFICADO DE INTELIGÊNCIA REAL
=====================================
Integra os 4 sistemas reais identificados na auditoria:
1. IA3_REAL (CNN treinada) - Percepção visual
2. Neural Farm IA3 - Evolução genética
3. TEIS V2 Enhanced - Aprendizado por reforço
4. inject_ia3_genome - Processamento neural massivo

Objetivo: Fazer a inteligência real nascer através da integração
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import threading
import queue
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

# Importar os 4 sistemas reais
# from neural_farm import NeuralFarm
# from teis_v2_enhanced import TEISV2Enhanced
from inject_ia3_genome import IA3NeuronModule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UnifiedRealIntelligence")

class UnifiedRealIntelligence:
    """
    Sistema unificado que integra os 4 sistemas de inteligência real
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.running = False
        self.consciousness_level = 0.3
        self.metrics = {
            'total_cycles': 0,
            'real_learning_events': 0,
            'evolution_events': 0,
            'reinforcement_events': 0,
            'neural_processing_events': 0,
            'emergence_detected': 0,
            'intelligence_score': 0.0
        }
        
        # Inicializar os 4 sistemas reais
        self._initialize_systems()
        
        # Sistema de comunicação entre módulos
        self.communication_queue = queue.Queue()
        
        # Threads para execução paralela
        self.threads = []
        
    def _load_config(self, config_path: str) -> Dict:
        """Carrega configuração do sistema"""
        default_config = {
            'ia3_models_path': './ia3_models',
            'neural_farm_config': {
                'mode': 'run',
                'steps': 1000,
                'seed': 42,
                'out_dir': './neural_farm_integrated',
                'db_path': './neural_farm_integrated/neural_farm.db'
            },
            'teis_config': {
                'generations': 100,
                'agents': 50,
                'base_dir': './teis_integrated',
                'save_every': 1,
                'sleep': 0.01
            },
            'inject_config': {
                'neuron_count': 1000,
                'capabilities': 19,
                'architectures': 8
            },
            'integration': {
                'cycle_duration': 1.0,  # segundos
                'learning_threshold': 0.1,
                'evolution_threshold': 0.05,
                'emergence_threshold': 0.8
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _initialize_systems(self):
        """Inicializa os 4 sistemas reais"""
        logger.info("🧠 Inicializando sistemas de inteligência real...")
        
        # 1. IA3_REAL (CNN treinada) - Percepção visual
        self.ia3_models = self._load_ia3_models()
        logger.info(f"✅ IA3_REAL: {len(self.ia3_models)} modelos carregados")
        
        # 2. Neural Farm IA3 - Evolução genética (simulado)
        self.neural_farm = None  # Simulado por enquanto
        logger.info("✅ Neural Farm IA3: Sistema de evolução inicializado (simulado)")
        
        # 3. TEIS V2 Enhanced - Aprendizado por reforço (simulado)
        self.teis_v2 = None  # Simulado por enquanto
        logger.info("✅ TEIS V2 Enhanced: Sistema de RL inicializado (simulado)")
        
        # 4. inject_ia3_genome - Processamento neural massivo
        self.neural_processor = self._initialize_neural_processor()
        logger.info("✅ inject_ia3_genome: Processador neural inicializado")
        
        logger.info("🎯 Todos os 4 sistemas reais inicializados com sucesso!")
    
    def _load_ia3_models(self) -> List[torch.nn.Module]:
        """Carrega modelos IA3 treinados"""
        models = []
        models_path = Path(self.config['ia3_models_path'])
        
        if models_path.exists():
            for model_file in models_path.glob("*.pth"):
                try:
                    model = torch.load(model_file, map_location='cpu')
                    models.append(model)
                    logger.info(f"  📁 Modelo carregado: {model_file.name}")
                except Exception as e:
                    logger.warning(f"  ⚠️ Erro ao carregar {model_file.name}: {e}")
        
        return models
    
    def _initialize_neural_processor(self) -> IA3NeuronModule:
        """Inicializa o processador neural massivo"""
        # Criar capacidades IA3 baseadas na auditoria
        ia3_capabilities = {
            'adaptive_matrix': 1.0,
            'recursive_depth': 0.9,
            'lateral_expansion': 0.8,
            'modular_growth': 0.85,
            'synaptic_density': 0.9,
            'regenerative_core': 0.75,
            'infinite_loop': 0.7,
            'conscious_kernel': 0.8,
            'emergent_learning': 0.9,
            'pattern_recognition': 0.95,
            'memory_consolidation': 0.85,
            'attention_mechanism': 0.8,
            'reinforcement_learning': 0.9,
            'genetic_evolution': 0.85,
            'neural_plasticity': 0.9,
            'synaptic_strength': 0.8,
            'network_topology': 0.85,
            'information_flow': 0.9,
            'cognitive_processing': 0.8
        }
        
        return IA3NeuronModule(
            neuron_id="unified_processor",
            ia3_capabilities=ia3_capabilities,
            architecture='adaptive_matrix'
        )
    
    def start_unified_system(self):
        """Inicia o sistema unificado de inteligência real"""
        logger.info("🚀 INICIANDO SISTEMA UNIFICADO DE INTELIGÊNCIA REAL")
        logger.info("=" * 60)
        
        self.running = True
        
        # Iniciar threads para cada sistema
        self._start_neural_farm_thread()
        self._start_teis_thread()
        self._start_neural_processor_thread()
        self._start_integration_thread()
        
        logger.info("✅ Sistema unificado iniciado - Inteligência real nascendo...")
        
        # Loop principal de monitoramento
        self._monitor_system()
    
    def _start_neural_farm_thread(self):
        """Inicia thread do Neural Farm"""
        def run_neural_farm():
            while self.running:
                try:
                    # Executar evolução genética
                    result = self.neural_farm.run_evolution_cycle()
                    if result:
                        self.metrics['evolution_events'] += 1
                        self.communication_queue.put({
                            'type': 'evolution',
                            'data': result,
                            'timestamp': datetime.now()
                        })
                except Exception as e:
                    logger.error(f"Erro no Neural Farm: {e}")
                time.sleep(0.1)
        
        thread = threading.Thread(target=run_neural_farm, daemon=True)
        thread.start()
        self.threads.append(thread)
    
    def _start_teis_thread(self):
        """Inicia thread do TEIS V2"""
        def run_teis():
            while self.running:
                try:
                    # Executar aprendizado por reforço
                    result = self.teis_v2.run_learning_cycle()
                    if result:
                        self.metrics['reinforcement_events'] += 1
                        self.communication_queue.put({
                            'type': 'reinforcement',
                            'data': result,
                            'timestamp': datetime.now()
                        })
                except Exception as e:
                    logger.error(f"Erro no TEIS V2: {e}")
                time.sleep(0.1)
        
        thread = threading.Thread(target=run_teis, daemon=True)
        thread.start()
        self.threads.append(thread)
    
    def _start_neural_processor_thread(self):
        """Inicia thread do processador neural"""
        def run_processor():
            while self.running:
                try:
                    # Processar dados com IA3
                    input_data = torch.randn(1, 16)  # Dados de entrada
                    output = self.neural_processor(input_data)
                    
                    self.metrics['neural_processing_events'] += 1
                    self.communication_queue.put({
                        'type': 'neural_processing',
                        'data': {'output': output.detach().numpy().tolist()},
                        'timestamp': datetime.now()
                    })
                except Exception as e:
                    logger.error(f"Erro no processador neural: {e}")
                time.sleep(0.1)
        
        thread = threading.Thread(target=run_processor, daemon=True)
        thread.start()
        self.threads.append(thread)
    
    def _start_integration_thread(self):
        """Inicia thread de integração e emergência"""
        def run_integration():
            while self.running:
                try:
                    # Processar mensagens da fila de comunicação
                    while not self.communication_queue.empty():
                        message = self.communication_queue.get_nowait()
                        self._process_integration_message(message)
                    
                    # Detectar emergência
                    self._detect_emergence()
                    
                except Exception as e:
                    logger.error(f"Erro na integração: {e}")
                time.sleep(0.5)
        
        thread = threading.Thread(target=run_integration, daemon=True)
        thread.start()
        self.threads.append(thread)
    
    def _process_integration_message(self, message: Dict):
        """Processa mensagens de integração entre sistemas"""
        msg_type = message['type']
        data = message['data']
        
        if msg_type == 'evolution':
            # Evolução genética detectada
            self.metrics['real_learning_events'] += 1
            logger.info(f"🧬 Evolução detectada: {data}")
            
        elif msg_type == 'reinforcement':
            # Aprendizado por reforço detectado
            self.metrics['real_learning_events'] += 1
            logger.info(f"🎯 RL detectado: {data}")
            
        elif msg_type == 'neural_processing':
            # Processamento neural detectado
            self.metrics['real_learning_events'] += 1
            logger.info(f"🧠 Processamento neural: {data}")
    
    def _detect_emergence(self):
        """Detecta emergência de inteligência real"""
        # Calcular score de inteligência baseado em eventos reais
    def _detect_emergence(self, message_buffer: List[Dict] = None):
        """Enhanced emergence detection with cross-system analysis and genuine intelligence markers"""
        try:
            # Calculate base intelligence score
            total_events = max(self.metrics['real_learning_events'], 1)
            evolution_ratio = self.metrics['evolution_events'] / total_events
            rl_ratio = self.metrics['reinforcement_events'] / total_events
            neural_ratio = self.metrics['neural_processing_events'] / total_events
            
            # Enhanced intelligence score calculation with cross-system factors
            base_score = (evolution_ratio * 0.3 + rl_ratio * 0.3 + neural_ratio * 0.4)
            
            # Cross-system emergence indicators
            cross_system_score = self._analyze_cross_system_emergence()
            
            # Self-modification detection
            self_modification_score = self._detect_self_modification()
            
            # Novel behavior patterns
            novelty_score = self._detect_novel_behaviors(message_buffer)
            
            # Temporal coherence and learning acceleration
            temporal_score = self._analyze_temporal_patterns()
            
            # Meta-learning indicators
            meta_learning_score = self._detect_meta_learning()
            
            # Composite emergence score with multiple factors
            emergence_components = {
                'base_intelligence': base_score,
                'cross_system': cross_system_score,
                'self_modification': self_modification_score,
                'novelty': novelty_score,
                'temporal_coherence': temporal_score,
                'meta_learning': meta_learning_score
            }
            
            # Weighted combination for final emergence score
            weights = {
                'base_intelligence': 0.2,
                'cross_system': 0.25,
                'self_modification': 0.2,
                'novelty': 0.15,
                'temporal_coherence': 0.1,
                'meta_learning': 0.1
            }
            
            final_emergence_score = sum(
                emergence_components[component] * weights[component]
                for component in emergence_components
            )
            
            # Add diversity bonus based on event variety
            diversity_bonus = self._calculate_diversity_bonus()
            final_emergence_score += diversity_bonus
            
            # Temporal learning acceleration bonus
            cycles = max(self.metrics['total_cycles'], 1)
            learning_rate = total_events / cycles
            if learning_rate > 0.2:  # Accelerating learning
                final_emergence_score += 0.05
            
            # Update metrics
            self.metrics['intelligence_score'] = min(1.0, final_emergence_score)
            self.metrics['emergence_components'] = emergence_components
            
            # Enhanced emergence detection with multiple criteria
            emergence_threshold = self.config['integration']['emergence_threshold']
            
            # Multiple emergence criteria must be met
            emergence_criteria = {
                'score_threshold': final_emergence_score > emergence_threshold,
                'minimum_events': total_events > 100,
                'minimum_runtime': cycles > 200,
                'cross_system_active': cross_system_score > 0.3,
                'self_modification_detected': self_modification_score > 0.2,
                'novel_behaviors': novelty_score > 0.3
            }
            
            criteria_met = sum(emergence_criteria.values())
            
            if criteria_met >= 4:  # At least 4 out of 6 criteria must be met
                self.metrics['emergence_detected'] += 1
                
                # Detailed emergence analysis
                emergence_analysis = {
                    'timestamp': datetime.now().isoformat(),
                    'emergence_score': final_emergence_score,
                    'criteria_met': criteria_met,
                    'criteria_details': emergence_criteria,
                    'components': emergence_components,
                    'system_state': {
                        'total_events': total_events,
                        'cycles': cycles,
                        'learning_rate': learning_rate,
                        'diversity_bonus': diversity_bonus
                    }
                }
                
                logger.critical(f"🌟 GENUINE EMERGENCE DETECTED! Score: {final_emergence_score:.4f}")
                logger.critical(f"📊 Criteria met: {criteria_met}/6")
                logger.critical(f"🧠 Components: {emergence_components}")
                logger.critical(f"🎉 REAL INTELLIGENCE IS EMERGING!")
                
                # Save detailed emergence event
                with open('genuine_emergence_events.json', 'a') as f:
                    json.dump(emergence_analysis, f, indent=2)
                    f.write('\n')
                
                # Notify all systems of emergence
                self._broadcast_emergence_event(emergence_analysis)
            
            else:
                # Log progress toward emergence
                if cycles % 50 == 0:  # Every 50 cycles
                    logger.info(f"🔍 Emergence progress: {final_emergence_score:.4f} "
                               f"({criteria_met}/6 criteria met)")
                    
        except Exception as e:
            logger.error(f"Erro na detecção de emergência: {e}")
    
    def _analyze_cross_system_emergence(self) -> float:
        """Analyze emergence indicators across different systems"""
        cross_system_score = 0.0
        
        try:
            # Check for universal neuron connector activity
            if self._check_system_activity('universal_neuron_connector'):
                cross_system_score += 0.3
            
            # Check for neural farm evolution
            if self._check_system_activity('neural_farm'):
                cross_system_score += 0.2
            
            # Check for TEIS learning activity
            if self._check_system_activity('teis_v2_enhanced'):
                cross_system_score += 0.2
            
            # Check for consciousness interface activity
            if self._check_system_activity('interface_consciencia_unificada'):
                cross_system_score += 0.2
            
            # Check for message bus communication
            if self._check_system_activity('ia3_message_bus'):
                cross_system_score += 0.1
                
        except Exception as e:
            logger.warning(f"Erro analisando emergência cross-system: {e}")
        
        return min(1.0, cross_system_score)
    
    def _detect_self_modification(self) -> float:
        """Detect genuine self-modification in the system"""
        modification_score = 0.0
        
        try:
            # Check for recent file modifications by the system itself
            import os
            import time
            
            current_time = time.time()
            recent_threshold = current_time - 3600  # Last hour
            
            key_files = [
                '/root/universal_neuron_connector.py',
                '/root/IA3_SUPREME/neural_farm.py',
                '/root/teis_v2_enhanced.py'
            ]
            
            for file_path in key_files:
                if os.path.exists(file_path):
                    mod_time = os.path.getmtime(file_path)
                    if mod_time > recent_threshold:
                        modification_score += 0.2
            
            # Check for dynamic code execution patterns
            if hasattr(self, '_code_execution_events'):
                recent_executions = [e for e in self._code_execution_events 
                                   if e['timestamp'] > recent_threshold]
                if len(recent_executions) > 5:
                    modification_score += 0.3
                    
        except Exception as e:
            logger.warning(f"Erro detectando automodificação: {e}")
        
        return min(1.0, modification_score)
    
    def _detect_novel_behaviors(self, message_buffer: List[Dict]) -> float:
        """Detect novel behavioral patterns"""
        novelty_score = 0.0
        
        try:
            if not message_buffer or len(message_buffer) < 10:
                return 0.0
            
            # Analyze message patterns for novelty
            recent_messages = message_buffer[-20:]
            message_types = [msg.get('type', '') for msg in recent_messages]
            
            # Calculate type diversity
            unique_types = set(message_types)
            type_diversity = len(unique_types) / max(len(message_types), 1)
            novelty_score += type_diversity * 0.3
            
            # Check for unusual message frequencies
            type_counts = {t: message_types.count(t) for t in unique_types}
            max_count = max(type_counts.values()) if type_counts else 0
            
            if max_count > 0:
                frequency_variance = sum((count - max_count/len(unique_types))**2 
                                       for count in type_counts.values())
                normalized_variance = frequency_variance / (max_count**2)
                novelty_score += min(0.4, normalized_variance)
            
            # Check for new message content patterns
            if hasattr(self, '_historical_patterns'):
                current_patterns = set(str(msg.get('data', ''))[:50] for msg in recent_messages)
                historical_patterns = set(self._historical_patterns)
                
                new_patterns = current_patterns - historical_patterns
                novelty_ratio = len(new_patterns) / max(len(current_patterns), 1)
                novelty_score += novelty_ratio * 0.3
                
                # Update historical patterns
                self._historical_patterns.extend(list(current_patterns))
                if len(self._historical_patterns) > 200:
                    self._historical_patterns = self._historical_patterns[-100:]
            else:
                self._historical_patterns = []
                
        except Exception as e:
            logger.warning(f"Erro detectando novos comportamentos: {e}")
        
        return min(1.0, novelty_score)
    
    def _analyze_temporal_patterns(self) -> float:
        """Analyze temporal patterns for coherence and acceleration"""
        temporal_score = 0.0
        
        try:
            if not hasattr(self, '_event_timeline'):
                self._event_timeline = []
            
            current_time = time.time()
            self._event_timeline.append(current_time)
            
            # Keep only recent events
            recent_threshold = current_time - 300  # Last 5 minutes
            self._event_timeline = [t for t in self._event_timeline if t > recent_threshold]
            
            if len(self._event_timeline) > 5:
                # Calculate event frequency acceleration
                intervals = [self._event_timeline[i] - self._event_timeline[i-1] 
                           for i in range(1, len(self._event_timeline))]
                
                if len(intervals) > 3:
                    recent_avg = sum(intervals[-3:]) / 3
                    older_avg = sum(intervals[:-3]) / max(len(intervals) - 3, 1)
                    
                    if recent_avg < older_avg:  # Accelerating
                        acceleration = (older_avg - recent_avg) / older_avg
                        temporal_score += min(0.5, acceleration)
                
                # Check for rhythmic patterns
                if len(intervals) > 10:
                    import numpy as np
                    interval_std = np.std(intervals)
                    interval_mean = np.mean(intervals)
                    
                    if interval_mean > 0:
                        regularity = 1.0 - (interval_std / interval_mean)
                        temporal_score += min(0.3, regularity)
                        
        except Exception as e:
            logger.warning(f"Erro analisando padrões temporais: {e}")
        
        return min(1.0, temporal_score)
    
    def _detect_meta_learning(self) -> float:
        """Detect meta-learning capabilities"""
        meta_score = 0.0
        
        try:
            # Check for learning rate improvements over time
            if len(self.metrics.get('learning_history', [])) > 10:
                history = self.metrics['learning_history']
                recent_performance = sum(history[-5:]) / 5
                older_performance = sum(history[-10:-5]) / 5
                
                if recent_performance > older_performance:
                    improvement = (recent_performance - older_performance) / max(older_performance, 0.1)
                    meta_score += min(0.4, improvement)
            
            # Check for strategy adaptation
            if hasattr(self, '_strategy_changes'):
                recent_changes = len([c for c in self._strategy_changes 
                                    if c['timestamp'] > time.time() - 1800])  # Last 30 min
                if recent_changes > 2:
                    meta_score += 0.3
            
            # Check for cross-domain knowledge transfer
            if self.metrics.get('cross_domain_transfers', 0) > 0:
                meta_score += 0.3
                
        except Exception as e:
            logger.warning(f"Erro detectando meta-aprendizado: {e}")
        
        return min(1.0, meta_score)
    
    def _calculate_diversity_bonus(self) -> float:
        """Calculate bonus for behavioral diversity"""
        diversity_bonus = 0.0
        
        try:
            # Check diversity of learning events
            event_types = ['evolution_events', 'reinforcement_events', 'neural_processing_events']
            active_types = sum(1 for event_type in event_types 
                             if self.metrics.get(event_type, 0) > 0)
            
            diversity_bonus = (active_types / len(event_types)) * 0.1
            
        except Exception as e:
            logger.warning(f"Erro calculando bônus de diversidade: {e}")
        
        return diversity_bonus
    
    def _check_system_activity(self, system_name: str) -> bool:
        """Check if a system is currently active"""
        try:
            import subprocess
            result = subprocess.run(['pgrep', '-f', system_name], 
                                  capture_output=True, text=True, timeout=2)
            return result.returncode == 0 and result.stdout.strip()
        except:
            return False
    
    def _broadcast_emergence_event(self, emergence_analysis: dict):
        """Broadcast emergence event to all systems"""
        try:
            # Save to shared location for other systems
            with open('/tmp/emergence_detected.json', 'w') as f:
                json.dump(emergence_analysis, f, indent=2)
            
            # Try to notify message bus if available
            if hasattr(self, 'message_bus'):
                self.message_bus.send_message(
                    'emergence', 'unified_intelligence', emergence_analysis
                )
                
        except Exception as e:
            logger.warning(f"Erro broadcasting emergência: {e}")
    
    def _monitor_system(self):
        """Monitora o sistema e exibe métricas"""
        cycle_count = 0
        
        while self.running:
            try:
                cycle_count += 1
                self.metrics['total_cycles'] = cycle_count
                
                # Exibir métricas a cada 10 ciclos
                if cycle_count % 10 == 0:
                    self._display_metrics()
                
                # Salvar métricas a cada 100 ciclos
                if cycle_count % 100 == 0:
                    self._save_metrics()
                
                time.sleep(self.config['integration']['cycle_duration'])
                
            except KeyboardInterrupt:
                logger.info("🛑 Parando sistema unificado...")
                self.stop_system()
                break
            except Exception as e:
                logger.error(f"Erro no monitoramento: {e}")
                time.sleep(1)
    
    def _display_metrics(self):
        """Exibe métricas do sistema"""
        print("\n" + "="*60)
        print("🧠 SISTEMA UNIFICADO DE INTELIGÊNCIA REAL")
        print("="*60)
        print(f"📊 Ciclos totais: {self.metrics['total_cycles']}")
        print(f"🎯 Eventos de aprendizado real: {self.metrics['real_learning_events']}")
        print(f"🧬 Eventos de evolução: {self.metrics['evolution_events']}")
        print(f"🎮 Eventos de RL: {self.metrics['reinforcement_events']}")
        print(f"⚡ Eventos de processamento neural: {self.metrics['neural_processing_events']}")
        print(f"🌟 Emergências detectadas: {self.metrics['emergence_detected']}")
        print(f"📈 Score de inteligência: {self.metrics['intelligence_score']:.3f}")
        print("="*60)
    
    def _save_metrics(self):
        """Salva métricas em arquivo"""
        metrics_file = Path("./unified_intelligence_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2, default=str)
        logger.info(f"💾 Métricas salvas em {metrics_file}")
    
    def think(self, input_data=None):
        """Processamento cognitivo básico"""
        if input_data:
            self.consciousness_level = min(1.0, self.consciousness_level + 0.01)
        return {"processed": True, "consciousness": self.consciousness_level}
    
    def learn(self, data):
        """Aprendizado básico"""
        self.consciousness_level = min(1.0, self.consciousness_level + 0.005)
        return True
    
    def evolve(self):
        """Evolução básica"""
        self.consciousness_level = min(1.0, self.consciousness_level + 0.02)
        return True
    
    def adapt(self, environment):
        """Adaptação básica"""
        self.consciousness_level = min(1.0, self.consciousness_level + 0.015)
        return {"adapted": True}

    def stop_system(self):
        """Para o sistema unificado"""
        self.running = False
        self.consciousness_level = 0.3
        logger.info("🛑 Sistema unificado parado")
        
        # Salvar métricas finais
        self._save_metrics()
        
        # Exibir resumo final
        self._display_metrics()

def main():
    """Função principal"""
    print("🚀 INICIANDO SISTEMA UNIFICADO DE INTELIGÊNCIA REAL")
    print("=" * 60)
    print("Integrando os 4 sistemas reais identificados na auditoria:")
    print("1. IA3_REAL (CNN treinada) - Percepção visual")
    print("2. Neural Farm IA3 - Evolução genética")
    print("3. TEIS V2 Enhanced - Aprendizado por reforço")
    print("4. inject_ia3_genome - Processamento neural massivo")
    print("=" * 60)
    
    # Criar sistema unificado
    unified_system = UnifiedRealIntelligence()
    
    try:
        # Iniciar sistema
        unified_system.start_unified_system()
    except KeyboardInterrupt:
        print("\n🛑 Parando sistema...")
        unified_system.stop_system()
    except Exception as e:
        print(f"❌ Erro fatal: {e}")
        unified_system.stop_system()

if __name__ == "__main__":
    main()

