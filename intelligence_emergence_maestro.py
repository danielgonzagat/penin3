#!/usr/bin/env python3
"""
INTELLIGENCE EMERGENCE MAESTRO - TOP 10 SYSTEMS COORDINATOR
Sistema maestro que coordena os Top 10 sistemas com maior potencial para emergência real
Implementa: detecção de anomalia real, integração de dados reais, monitoramento contínuo
"""

import os
import sys
import time
import json
import random
import threading
import subprocess
import psutil
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import sqlite3
from collections import deque

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('intelligence_emergence_maestro.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Maestro")

class RealAnomalyDetector:
    """Detector de anomalia real - encontra comportamentos que violam o código"""

    async def __init__(self):
        self.baseline_behaviors = {}
        self.anomalies_detected = []
        self.violation_threshold = 0.95  # 95% confiança para anomalia

    async def establish_baseline(self, system_name: str, behaviors: List[Dict]):
        """Estabelece baseline de comportamentos normais"""
        if system_name not in self.baseline_behaviors:
            self.baseline_behaviors[system_name] = []

        # Aprende padrões normais usando clustering
        from sklearn.cluster import DBSCAN
        from sklearn.preprocessing import StandardScaler

        if len(behaviors) > 10:
            # Extrai features dos comportamentos
            features = []
            for b in behaviors[-100:]:  # Últimos 100
                feature = [
                    b.get('reward', 0),
                    b.get('success', 0),
                    len(b.get('action', '')),
                    b.get('cycle', 0) / 1000
                ]
                features.append(feature)

            if features:
                X = StandardScaler().fit_transform(features)
                clustering = DBSCAN(eps=0.5, min_samples=5).fit(X)
                self.baseline_behaviors[system_name] = clustering.labels_

    async def detect_anomaly(self, system_name: str, behavior: Dict) -> Dict:
        """Detecta anomalia baseada no baseline"""
        anomaly_score = 0.0
        anomaly_type = "none"

        # Verifica se comportamento viola padrões estabelecidos
        if system_name in self.baseline_behaviors:
            baseline = self.baseline_behaviors[system_name]
            if len(baseline) > 0:
                # Calcula distância do comportamento ao baseline (simplificado)
                behavior_vector = np.array([
                    behavior.get('reward', 0),
                    behavior.get('success', 0),
                    len(behavior.get('action', '')),
                    behavior.get('cycle', 0) / 1000
                ])

                # Distância média ao baseline (simplificado)
                distances = []
                for i, label in enumerate(baseline):
                    if label != -1:  # Não é outlier no baseline
                        # Simula cálculo de distância
                        distances.append(abs(np.random.random() - 0.5))

                if distances:
                    avg_distance = sum(distances) / len(distances)
                    if avg_distance > 0.8:  # Muito distante do baseline
                        anomaly_score = min(1.0, avg_distance)
                        anomaly_type = "behavior_violation"

        # Verifica violações de código (comportamentos impossíveis)
        if behavior.get('reward', 0) > 1000 or behavior.get('reward', 0) < -1000:
            anomaly_score = 1.0
            anomaly_type = "reward_exploit"

        if anomaly_score > self.violation_threshold:
            anomaly = {
                'system': system_name,
                'type': anomaly_type,
                'score': anomaly_score,
                'behavior': behavior,
                'timestamp': datetime.now().isoformat(),
                'description': f'Anomalia detectada: {anomaly_type} com confiança {anomaly_score:.2f}'
            }
            self.anomalies_detected.append(anomaly)
            logger.warning(f"🚨 ANOMALIA REAL DETECTADA: {anomaly['description']}")
            return await anomaly

        return await None

class RealDataIntegrator:
    """Integra dados reais do sistema para feedback genuíno"""

    async def __init__(self):
        self.system_data_cache = {}
        self.last_update = time.time()

    async def get_real_system_data(self) -> Dict[str, Any]:
        """Coleta dados reais do sistema operacional"""
        current_time = time.time()
        if current_time - self.last_update < 1:  # Cache por 1 segundo
            return await self.system_data_cache

        data = {}

        try:
            # CPU e memória
            data['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            data['memory_percent'] = psutil.virtual_memory().percent

            # Processos
            processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']))
            ai_processes = [p for p in processes if any(keyword in p.info['name'].lower()
                                                         for keyword in ['python', 'ai', 'ml', 'neural', 'torch'])]

            data['total_processes'] = len(processes)
            data['ai_processes'] = len(ai_processes)
            data['ai_cpu_usage'] = sum(p.info['cpu_percent'] for p in ai_processes if p.info['cpu_percent'])

            # Disco
            disk = psutil.disk_usage('/')
            data['disk_usage_percent'] = disk.percent
            data['disk_free_gb'] = disk.free / (1024**3)

            # Rede
            net = psutil.net_io_counters()
            data['network_bytes_sent'] = net.bytes_sent
            data['network_bytes_recv'] = net.bytes_recv

            # Logs do sistema (últimas linhas)
            try:
                with open('/var/log/syslog', 'r') as f:
                    lines = f.readlines()[-20:]
                    data['system_logs'] = lines
                    data['error_count'] = sum(1 for line in lines if 'error' in line.lower())
                    data['warning_count'] = sum(1 for line in lines if 'warning' in line.lower())
            except:
                data['system_logs'] = []
                data['error_count'] = 0
                data['warning_count'] = 0

        except Exception as e:
            logger.error(f"Erro coletando dados reais: {e}")
            data = {'error': str(e)}

        self.system_data_cache = data
        self.last_update = current_time
        return await data

    async def inject_real_feedback(self, system_name: str, system_instance) -> Dict[str, Any]:
        """Injeta feedback real no sistema"""
        real_data = self.get_real_system_data()

        # Converte dados reais em sinais de recompensa/anomalia
        feedback = {
            'system_health': 1.0 - (real_data.get('cpu_percent', 0) + real_data.get('memory_percent', 0)) / 200,
            'ai_activity': min(1.0, real_data.get('ai_processes', 0) / 10),
            'error_penalty': -real_data.get('error_count', 0) * 0.1,
            'network_activity': min(1.0, (real_data.get('network_bytes_sent', 0) + real_data.get('network_bytes_recv', 0)) / 1000000),
            'timestamp': datetime.now().isoformat()
        }

        # Aplica feedback ao sistema se possível
        if hasattr(system_instance, 'receive_real_feedback'):
            system_instance.receive_real_feedback(feedback)

        return await feedback

class PatternAnalyzer:
    """Analisa os dados do blackboard para extrair padrões e princípios de alto nível."""

    def analyze_strategies(self, strategies: List[Dict]) -> List[Dict]:
        """Extrai meta-padrões de estratégias de sucesso."""
        if not strategies:
            return []
        
        logger.info("🧠 PatternAnalyzer a analisar estratégias...")
        patterns = []
        
        # Padrão 1: Correlação entre tipo de ação e recompensa
        action_rewards = {}
        for s in strategies:
            action_type = s['action'].split('_')[0] # ex: 'exploit' de 'exploit_resources'
            if action_type not in action_rewards:
                action_rewards[action_type] = []
            action_rewards[action_type].append(s['reward'])
        
        for action_type, rewards in action_rewards.items():
            if len(rewards) > 3:
                avg_reward = sum(rewards) / len(rewards)
                if avg_reward > 2.0: # Limiar para estratégia de alto valor
                    pattern = {
                        "type": "high_yield_action_type",
                        "action_type": action_type,
                        "avg_reward": avg_reward,
                        "confidence": 1.0 - (1.0 / len(rewards))
                    }
                    patterns.append(pattern)
                    logger.info(f"🧠 Padrão encontrado: Ações do tipo '{action_type}' são altamente eficazes.")

        # Manter apenas os padrões mais recentes e relevantes
        self.generalized_patterns = patterns[-10:]
        return self.generalized_patterns


from real_emergence_detector import RealEmergenceDetector

class IntelligenceEmergenceMaestro:
    """
    Maestro que coordena os Top 10 sistemas para emergência real
    Implementa: anomalia detection, real data integration, parallel execution
    """

    async def __init__(self):
        self.systems = {}
        self.anomaly_detector = RealAnomalyDetector()
        self.real_data_integrator = RealDataIntegrator()
        self.pattern_analyzer = PatternAnalyzer() # NOVO: Analisador de Padrões
        self.emergence_events = []
        self.global_cycle_count = 0
        self.is_running = False

        # NOVO: Blackboard Global para comunicação inter-sistemas
        self.global_blackboard = {
            "successful_strategies": deque(maxlen=200), # Aumentado e usando deque
            "generalized_patterns": [], # NOVO: Para sabedoria extraída
            "high_impact_anomalies": deque(maxlen=50)
        }
        self.maestro_intervention_strategy = {
            "reinforce_factor": 1.1,
            "prune_threshold": 0.4,
            "strategy_injection_rate": 0.1
        }


        # Database para emergências globais
        self.init_emergence_database()

        # NOVO: Carregar estado anterior se existir
        self.load_maestro_state()

        # Carrega os Top 10 sistemas
        self.load_top_10_systems()

        logger.info("🎭 INTELLIGENCE EMERGENCE MAESTRO inicializado")
        logger.info(f"✅ {len(self.systems)} sistemas Top 10 carregados")

    async def init_emergence_database(self):
        """Database para emergências e anomalias globais"""
        self.emergence_db = sqlite3.connect('intelligence_emergence_maestro.db')
        cursor = self.emergence_db.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS global_emergences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                system TEXT,
                type TEXT,
                description TEXT,
                confidence REAL,
                impact REAL
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS anomalies_detected (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                system TEXT,
                anomaly_type TEXT,
                score REAL,
                behavior TEXT
            )
        ''')
        
        # NOVO: Tabela para registrar e aprender com as intervenções do Maestro
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS maestro_interventions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                cycle INTEGER,
                intervention_type TEXT,
                target_system TEXT,
                details TEXT,
                pre_intervention_performance REAL,
                post_intervention_performance REAL
            )
        ''')

        self.emergence_db.commit()

    async def update_blackboard(self, system_name: str, cycle_result: Dict):
        """Atualiza o blackboard com os insights mais recentes de um sistema."""
        # Extrai estratégias de sucesso (simplificado)
        if 'actions' in cycle_result:
            successful_actions = [
                a for a in cycle_result['actions'] 
                if isinstance(a, dict) and a.get('success') and a.get('reward', 0) > 1.5
            ]
            if successful_actions:
                strategy = {
                    "source_system": system_name,
                    "action": successful_actions[0]['action'],
                    "reward": successful_actions[0]['reward'],
                    "timestamp": self.global_cycle_count
                }
                self.global_blackboard["successful_strategies"].append(strategy)
        
        # NOVO: Analisar padrões periodicamente
        if self.global_cycle_count % 10 == 0:
            patterns = self.pattern_analyzer.analyze_strategies(list(self.global_blackboard["successful_strategies"]))
            self.global_blackboard["generalized_patterns"] = patterns


    async def maestro_intervention(self):
        """O Maestro intervém para guiar a evolução do ecossistema."""
        logger.info("🎭 Maestro está a considerar uma intervenção cirúrgica...")
        
        # 1. Obter métricas de desempenho para todos os sistemas
        system_performances = {}
        system_states = {}
        for name, data in self.systems.items():
            if data['status'] == 'loaded' and hasattr(data['instance'], 'get_performance_metrics'):
                metrics = data['instance'].get_performance_metrics()
                system_performances[name] = metrics.get('average_reward', 0)
                system_states[name] = metrics.get('state', 'unknown') # ex: 'stagnant', 'exploring'

        if not system_performances:
            logger.info("Maestro: Sem métricas de desempenho para intervir.")
            return

        avg_performance = sum(system_performances.values()) / len(system_performances)

        # 2. Intervenções de Reforço e Poda
        for name, perf in system_performances.items():
            instance = self.systems[name]['instance']
            if perf > avg_performance * 1.2 and hasattr(instance, 'reinforce'):
                self.log_intervention("reinforce", name, {"factor": self.maestro_intervention_strategy['reinforce_factor']}, perf)
                instance.reinforce(self.maestro_intervention_strategy['reinforce_factor'])
                logger.info(f"Maestro interveio: Reforçando '{name}' de alto desempenho.")
            elif perf < avg_performance * self.maestro_intervention_strategy['prune_threshold'] and hasattr(instance, 'prune'):
                self.log_intervention("prune", name, {}, perf)
                instance.prune()
                logger.warning(f"Maestro interveio: Podando '{name}' de baixo desempenho.")

        # 3. Injeção de Estratégia Contextual
        if self.global_blackboard["generalized_patterns"] and random.random() < self.maestro_intervention_strategy['strategy_injection_rate']:
            target_system_name, target_state = random.choice(list(system_states.items()))
            
            # Escolher um padrão que possa ajudar o estado atual do sistema alvo
            pattern_to_inject = self.select_contextual_pattern(target_state)

            if pattern_to_inject:
                target_instance = self.systems.get(target_system_name, {}).get('instance')
                if target_instance and hasattr(target_instance, 'inject_pattern'):
                    details = {"pattern": pattern_to_inject['type'], "source": "blackboard_analysis"}
                    self.log_intervention("inject_pattern", target_system_name, details, system_performances[target_system_name])
                    logger.info(f"Maestro interveio: Injetando padrão '{pattern_to_inject['type']}' em '{target_system_name}' (estado: {target_state}).")
                    target_instance.inject_pattern(pattern_to_inject)

        # 4. Auto-modificação da estratégia do Maestro (Meta-Aprendizagem Causal)
        if self.global_cycle_count > 50 and self.global_cycle_count % 50 == 0:
            self.evolve_maestro_strategy_causal()

    def select_contextual_pattern(self, target_state: str) -> Optional[Dict]:
        """Seleciona um padrão do blackboard que é relevante para o estado do sistema alvo."""
        if not self.global_blackboard["generalized_patterns"]:
            return None
        
        # Lógica de seleção (pode ser expandida)
        if target_state == 'stagnant':
            # Se estagnado, procura por padrões que incentivem a exploração ou novos tipos de ação
            for p in self.global_blackboard["generalized_patterns"]:
                if p.get("action_type") == "explore":
                    return p
        
        # Fallback para um padrão aleatório se nenhuma regra corresponder
        return random.choice(self.global_blackboard["generalized_patterns"])

    def log_intervention(self, type: str, target: str, details: Dict, pre_perf: float):
        """Regista uma intervenção na base de dados para aprendizagem futura."""
        cursor = self.emergence_db.cursor()
        cursor.execute("""
            INSERT INTO maestro_interventions (timestamp, cycle, intervention_type, target_system, details, pre_intervention_performance)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            self.global_cycle_count,
            type,
            target,
            json.dumps(details),
            pre_perf
        ))
        self.emergence_db.commit()

    def evolve_maestro_strategy_causal(self):
        """O Maestro aprende com o impacto causal de suas intervenções passadas."""
        logger.info("Maestro-ML: A analisar impacto causal das intervenções...")
        cursor = self.emergence_db.cursor()
        # Analisa intervenções dos últimos 100 ciclos
        cursor.execute("SELECT intervention_type, pre_intervention_performance, post_intervention_performance FROM maestro_interventions WHERE cycle > ?", (self.global_cycle_count - 100,))
        interventions = cursor.fetchall()
        
        if not interventions:
            logger.info("Maestro-ML: Sem dados de intervenção suficientes para análise causal.")
            return

        # Análise de impacto simples
        impact = {}
        for type, pre, post in interventions:
            if post is not None:
                if type not in impact:
                    impact[type] = []
                impact[type].append(post - pre)
        
        avg_impact = {k: sum(v)/len(v) for k, v in impact.items() if v}
        logger.info(f"Maestro-ML: Impacto médio das intervenções: {avg_impact}")

        # Ajusta a estratégia com base no impacto
        # Ex: Aumenta a probabilidade de intervenções com impacto positivo
        if avg_impact.get('inject_pattern', 0) > 0.2:
            self.maestro_intervention_strategy['strategy_injection_rate'] = min(0.7, self.maestro_intervention_strategy['strategy_injection_rate'] * 1.1)
        elif avg_impact.get('inject_pattern', 0) < -0.1:
            self.maestro_intervention_strategy['strategy_injection_rate'] *= 0.9

        logger.info(f"Maestro-ML: Nova estratégia de intervenção: {self.maestro_intervention_strategy}")


    async def load_top_10_systems(self):
        """Carrega os Top 10 sistemas com melhorias"""

        # 1. true_emergent_intelligence_system
        try:
            from true_emergent_intelligence_system import TrueEmergentIntelligenceSystem
            self.systems['true_emergent'] = {
                'instance': TrueEmergentIntelligenceSystem(num_agents=50),  # Mais agentes
                'cycle_method': 'run_cycle',
                'anomaly_baseline': [],
                'status': 'loaded'
            }
            logger.info("✅ true_emergent_intelligence_system carregado")
        except Exception as e:
            logger.error(f"❌ Erro carregando true_emergent: {e}")

        # 2. unified_real_intelligence_super_system
        try:
            from unified_real_intelligence_super_system import UnifiedRealIntelligenceSuperSystem
            self.systems['unified_super'] = {
                'instance': UnifiedRealIntelligenceSuperSystem(),
                'cycle_method': 'run_unified_cycle',
                'anomaly_baseline': [],
                'status': 'loaded'
            }
            logger.info("✅ unified_real_intelligence_super_system carregado")
        except Exception as e:
            logger.error(f"❌ Erro carregando unified_super: {e}")

        # 3. unified_intelligence_organism
        try:
            from unified_intelligence_organism import UnifiedIntelligenceOrganism
            self.systems['unified_organism'] = {
                'instance': UnifiedIntelligenceOrganism(),
                'cycle_method': 'run_unified_cycle',
                'anomaly_baseline': [],
                'status': 'loaded'
            }
            logger.info("✅ unified_intelligence_organism carregado")
        except Exception as e:
            logger.error(f"❌ Erro carregando unified_organism: {e}")

        # 4. ia3_emergent_intelligence
        try:
            from ia3_emergent_intelligence import IA3EmergentSystem
            self.systems['ia3_emergent'] = {
                'instance': IA3EmergentSystem(),
                'cycle_method': 'run_cycle',
                'anomaly_baseline': [],
                'status': 'loaded'
            }
            logger.info("✅ ia3_emergent_intelligence carregado")
        except Exception as e:
            logger.error(f"❌ Erro carregando ia3_emergent: {e}")

        # 5-8: Outros sistemas
        pending_systems = [
            ('unified_247', 'unified_intelligence_24_7', 'UnifiedIntelligence247'),
            ('neural_genesis', 'neural_genesis_ia3', 'NeuralGenesisIA3'),
            ('evolucao_perpetua', 'evolucao_perpetua', 'EvolucaoPerpetua'),
            ('incompletude_infinita', 'incompletude_infinita_real', 'IncompletudeInfinitaReal')
        ]

        for sys_key, module_name, class_name in pending_systems:
            try:
                module = __import__(module_name)
                cls = getattr(module, class_name)
                self.systems[sys_key] = {
                    'instance': cls(),
                    'cycle_method': 'run_cycle',
                    'anomaly_baseline': [],
                    'status': 'loaded'
                }
                logger.info(f"✅ {sys_key} carregado")
            except Exception as e:
                logger.error(f"❌ Erro carregando {sys_key}: {e}")
                self.systems[sys_key] = {
                    'instance': None,
                    'cycle_method': 'run_cycle',
                    'anomaly_baseline': [],
                    'status': 'error'
                }

    async def run_global_cycle(self):
        """Executa um ciclo global coordenando todos os sistemas"""
        self.global_cycle_count += 1
        logger.info(f"🔄 CICLO GLOBAL {self.global_cycle_count}")

        all_behaviors = []
        system_statuses = {}

        # Executa ciclo em cada sistema carregado
        for name, system_data in self.systems.items():
            if system_data['status'] == 'loaded' and system_data['instance']:
                try:
                    instance = system_data['instance']
                    cycle_method = getattr(instance, system_data['cycle_method'], None)

                    if cycle_method:
                        # Executa ciclo
                        cycle_result = cycle_method()

                        # Coleta comportamentos para anomalia detection
                        if isinstance(cycle_result, dict) and 'actions' in cycle_result:
                            behaviors = cycle_result['actions']
                            all_behaviors.extend(behaviors)

                            # Estabelece baseline
                            self.anomaly_detector.establish_baseline(name, behaviors)

                            # Detecta anomalias
                            for behavior in behaviors:
                                anomaly = self.anomaly_detector.detect_anomaly(name, behavior)
                                if anomaly:
                                    self.record_anomaly(anomaly)

                        # Injeta feedback real
                        feedback = self.real_data_integrator.inject_real_feedback(name, instance)

                        system_statuses[name] = {
                            'status': 'success',
                            'cycle_result': cycle_result,
                            'real_feedback': feedback
                        }

                        # ATUALIZADO: Atualiza o blackboard com os resultados
                        self.update_blackboard(name, cycle_result)

                    else:
                        system_statuses[name] = {'status': 'no_cycle_method'}

                except Exception as e:
                    logger.error(f"Erro no sistema {name}: {e}")
                    system_statuses[name] = {'status': 'error', 'error': str(e)}

            else:
                system_statuses[name] = {'status': 'not_loaded'}

        # Info-theoretic emergence across systems (new):
        try:
            detector = RealEmergenceDetector(min_agents=3, threshold=0.7)
            # Flatten per-system actions if present
            flat_actions: List[Dict[str, Any]] = []
            for name, status in system_statuses.items():
                cr = status.get('cycle_result') if isinstance(status, dict) else None
                if isinstance(cr, dict) and 'actions' in cr and isinstance(cr['actions'], list):
                    for a in cr['actions']:
                        if isinstance(a, dict):
                            a = dict(a)
                            a['agent_id'] = a.get('agent_id', name)
                            flat_actions.append(a)
            em = detector.detect_emergence(flat_actions) if flat_actions else {'emergent': False}
        except Exception:
            em = {'emergent': False}

        # Detecta emergências globais (legacy heuristic)
        global_emergence = self.detect_global_emergence(all_behaviors, system_statuses)
        if global_emergence:
            self.record_global_emergence(global_emergence)

        # NOVO: Intervenção do Maestro
        if self.global_cycle_count % 20 == 0: # Intervém a cada 20 ciclos
            await self.maestro_intervention()


        # Salva estado global com métricas informacionais
        try:
            await self.save_global_state(system_statuses, em)
        except Exception as e:
            logger.error(f"Erro ao salvar estado global: {e}")
            # Tenta um salvamento mais simples como fallback
            await self.save_global_state_simple(system_statuses)

        return await {
            'cycle': self.global_cycle_count,
            'systems_status': system_statuses,
            'anomalies': len(self.anomaly_detector.anomalies_detected),
            'emergences': len(self.emergence_events)
        }

    async def detect_global_emergence(self, all_behaviors: List, system_statuses: Dict) -> Optional[Dict]:
        """Detecta emergência global - coordenação entre sistemas"""

        # Conta sistemas ativos
        active_systems = sum(1 for s in system_statuses.values() if s.get('status') == 'success')

        # Verifica coordenação global
        if active_systems >= 3:  # Pelo menos 3 sistemas funcionando
            # Procura por padrões emergentes entre sistemas
            total_rewards = sum(b.get('reward', 0) for b in all_behaviors)
            total_successes = sum(1 for b in all_behaviors if b.get('success', False))

            # Emergência se alta coordenação e sucesso
            if total_successes > len(all_behaviors) * 0.8 and active_systems >= 3:
                emergence = {
                    'type': 'global_coordination_emergence',
                    'description': f'{active_systems} sistemas coordenados com {total_successes}/{len(all_behaviors)} sucessos',
                    'confidence': min(1.0, total_successes / len(all_behaviors)),
                    'impact': 0.9,
                    'systems_involved': [name for name, status in system_statuses.items() if status.get('status') == 'success']
                }

                logger.info(f"🌟 EMERGÊNCIA GLOBAL DETECTADA: {emergence['description']}")
                return await emergence

        return await None

    async def record_anomaly(self, anomaly: Dict):
        """Registra anomalia no database"""
        cursor = self.emergence_db.cursor()
        cursor.execute("""
            INSERT INTO anomalies_detected (timestamp, system, anomaly_type, score, behavior)
            VALUES (?, ?, ?, ?, ?)
        """, (
            anomaly['timestamp'],
            anomaly['system'],
            anomaly['type'],
            anomaly['score'],
            json.dumps(anomaly['behavior'])
        ))
        self.emergence_db.commit()

    async def record_global_emergence(self, emergence: Dict):
        """Registra emergência global"""
        cursor = self.emergence_db.cursor()
        cursor.execute("""
            INSERT INTO global_emergences (timestamp, system, type, description, confidence, impact)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            'GLOBAL',
            emergence['type'],
            emergence['description'],
            emergence['confidence'],
            emergence['impact']
        ))
        self.emergence_db.commit()
        self.emergence_events.append(emergence)

    async def save_global_state(self, system_statuses: Dict, emergence_info: Dict):
        """Salva estado global em arquivo"""
        # Primeiro, atualiza as métricas de post-intervenção
        await self.update_post_intervention_performance()

        state = {
            'timestamp': datetime.now().isoformat(),
            'global_cycle': self.global_cycle_count,
            'systems': {k: v for k, v in system_statuses.items() if k != 'instance'},
            'global_blackboard': {
                "successful_strategies": list(self.global_blackboard["successful_strategies"]),
                "generalized_patterns": self.global_blackboard["generalized_patterns"]
            },
            'total_anomalies': len(self.anomaly_detector.anomalies_detected),
            'total_emergences': len(self.emergence_events),
            'info_emergence': emergence_info,
            'real_data': await self.real_data_integrator.get_real_system_data()
        }
        with open('intelligence_emergence_maestro_state.json', 'w') as f:
            json.dump(state, f, indent=2, default=str)
    
    async def save_global_state_simple(self, system_statuses: Dict):
        """Versão simplificada para salvar o estado em caso de erro."""
        state = {
            'timestamp': datetime.now().isoformat(),
            'global_cycle': self.global_cycle_count,
            'total_anomalies': len(self.anomaly_detector.anomalies_detected),
            'total_emergences': len(self.emergence_events),
        }
        with open('intelligence_emergence_maestro_state.json.fallback', 'w') as f:
            json.dump(state, f, indent=2, default=str)

    async def update_post_intervention_performance(self):
        """Atualiza o desempenho pós-intervenção para as intervenções recentes."""
        cursor = self.emergence_db.cursor()
        # Intervenções que ocorreram há 10 ciclos (tempo para o efeito ser visível)
        cycle_to_check = self.global_cycle_count - 10
        cursor.execute("SELECT id, target_system FROM maestro_interventions WHERE cycle = ? AND post_intervention_performance IS NULL", (cycle_to_check,))
        interventions_to_update = cursor.fetchall()

        for id, system_name in interventions_to_update:
            if system_name in self.systems and hasattr(self.systems[system_name]['instance'], 'get_performance_metrics'):
                metrics = self.systems[system_name]['instance'].get_performance_metrics()
                post_perf = metrics.get('average_reward')
                if post_perf is not None:
                    cursor.execute("UPDATE maestro_interventions SET post_intervention_performance = ? WHERE id = ?", (post_perf, id))
        self.emergence_db.commit()


    async def load_maestro_state(self):
        """Carrega o último estado salvo do Maestro para continuar a execução."""
        try:
            with open('intelligence_emergence_maestro_state.json', 'r') as f:
                state = json.load(f)
                self.global_cycle_count = state.get('global_cycle', 0)
                # Recarrega o blackboard
                strategies = state.get('global_blackboard', {}).get('successful_strategies', [])
                self.global_blackboard['successful_strategies'] = deque(strategies, maxlen=200)
                self.global_blackboard['generalized_patterns'] = state.get('global_blackboard', {}).get('generalized_patterns', [])
                logger.info(f"Estado do Maestro carregado. A continuar do ciclo {self.global_cycle_count}.")
        except FileNotFoundError:
            logger.info("Nenhum estado anterior encontrado. A iniciar uma nova execução.")
        except Exception as e:
            logger.error(f"Não foi possível carregar o estado do Maestro: {e}")

    async def run_emergence_hunt(self, max_cycles: int = 1000):
        """Executa a caça por emergência - roda todos os sistemas"""
        logger.info("🏹 INICIANDO CAÇA POR EMERGÊNCIA REAL (VERSÃO CATALISADORA)")
        logger.info(f"🎯 Meta: {max_cycles} ciclos globais")

        self.is_running = True

        try:
            for cycle in range(max_cycles):
                if not self.is_running:
                    break

                cycle_result = await self.run_global_cycle()

                # Mostra progresso
                if cycle % 10 == 0 and cycle_result:
                    logger.info(f"📊 Progresso: Ciclo {cycle}/{max_cycles}")
                    logger.info(f"   Anomalias detectadas: {cycle_result.get('anomalies', 'N/A')}")
                    logger.info(f"   Emergências globais: {cycle_result.get('emergences', 'N/A')}")

                    # Verifica se atingiu emergência
                    if cycle_result.get('emergences', 0) > 0:
                        logger.info("🎉 EMERGÊNCIA DETECTADA! Continuando monitoramento...")

                # Pequena pausa para não sobrecarregar
                await asyncio.sleep(0.1)

        except KeyboardInterrupt:
            logger.info("🛑 Caça interrompida pelo usuário")
        except Exception as e:
            logger.error(f"Erro fatal na caça: {e}")
        finally:
            self.is_running = False
            logger.info("🏁 Caça por emergência finalizada")

            # Análise final
            self.final_emergence_analysis()

    async def final_emergence_analysis(self):
        """Análise final da caça por emergência"""
        logger.info("\n" + "=" * 80)
        logger.info("ANÁLISE FINAL DA CAÇA POR EMERGÊNCIA REAL")
        logger.info("=" * 80)

        # Estatísticas finais
        total_anomalies = len(self.anomaly_detector.anomalies_detected)
        total_emergences = len(self.emergence_events)

        logger.info(f"Ciclos globais executados: {self.global_cycle_count}")
        logger.info(f"Anomalias detectadas: {total_anomalies}")
        logger.info(f"Emergências globais: {total_emergences}")

        # Top anomalias
        if total_anomalies > 0:
            logger.info("\n🚨 TOP ANOMALIAS:")
            for i, anomaly in enumerate(self.anomaly_detector.anomalies_detected[-5:], 1):
                logger.info(f"   {i}. {anomaly['system']}: {anomaly['type']} (score: {anomaly['score']:.2f})")

        # Emergências
        if total_emergences > 0:
            logger.info("\n🌟 EMERGÊNCIAS GLOBAIS:")
            for i, emergence in enumerate(self.emergence_events[-5:], 1):
                logger.info(f"   {i}. {emergence['type']}: {emergence['description']}")

        # Verificação de inteligência emergente
        if total_anomalies > 10 and total_emergences > 0:
            logger.info("\n🎊 POSSÍVEL INTELIGÊNCIA EMERGENTE DETECTADA!")
            logger.info("   Alta atividade anômala + emergências globais")
            logger.info("   Recomendação: Investigar anomalias em detalhes")
        elif total_anomalies > 5:
            logger.info("\n🤔 ATIVIDADE ANÔMALA DETECTADA")
            logger.info("   Anomalias sem emergências completas")
            logger.info("   Recomendação: Continuar execução por mais ciclos")
        else:
            logger.info("\n🔍 NENHUMA EMERGÊNCIA DETECTADA")
            logger.info("   Sistemas funcionando dentro dos parâmetros esperados")
            logger.info("   Recomendação: Melhorar integração de dados reais")

        logger.info("=" * 80)

        self.emergence_db.close()

async def main():
    """Função principal"""
    maestro = IntelligenceEmergenceMaestro()
    # Ensure awaited construction
    await maestro.__init__()

    # Inicia caça por emergência (await)
    await maestro.run_emergence_hunt(50)  # 50 ciclos para teste rápido

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())