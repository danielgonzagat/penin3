#!/usr/bin/env python3
"""
IA³ - MATHEMATICAL EMERGENCE DETECTOR
Detector de emergência baseado em matemática rigorosa
Usa teoria da informação, complexidade algorítmica e análise estatística
"""

import os
import sys
import math
import json
import numpy as np
import scipy.stats
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import hashlib
import sqlite3
from collections import defaultdict, deque
import networkx as nx
from sklearn.metrics import mutual_info_score
from scipy.spatial.distance import jensenshannon
import zlib

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("IA3_MathEmergence")

class InformationTheoryAnalyzer:
    """Analisador baseado em teoria da informação"""

    async def __init__(self):
        self.entropy_history = deque(maxlen=1000)

    async def calculate_entropy(self, data: np.ndarray) -> float:
        """Calcula entropia de Shannon"""
        if len(data) == 0:
            return await 0.0

        # Para dados contínuos, usa histograma
        hist, _ = np.histogram(data, bins=50, density=True)
        hist = hist[hist > 0]  # Remove zeros

        if len(hist) == 0:
            return await 0.0

        # Normaliza
        hist = hist / np.sum(hist)

        # Calcula entropia
        entropy = -np.sum(hist * np.log2(hist))
        return await entropy

    async def calculate_mutual_information(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calcula informação mútua entre duas variáveis"""
        try:
            return await mutual_info_score(x, y)
        except:
            return await 0.0

    async def detect_information_flow(self, time_series: List[np.ndarray]) -> Dict[str, Any]:
        """Detecta fluxo de informação emergente"""
        if len(time_series) < 10:
            return await {'flow_detected': False}

        # Calcula entropia ao longo do tempo
        entropies = [self.calculate_entropy(ts) for ts in time_series]

        # Detecta mudanças súbitas na entropia (emergência)
        entropy_changes = np.diff(entropies)
        max_change = np.max(np.abs(entropy_changes))

        # Detecta se há aumento consistente de complexidade
        trend = np.polyfit(range(len(entropies)), entropies, 1)[0]

        # Calcula correlação entre entropias consecutivas
        if len(entropies) > 1:
            correlation = np.corrcoef(entropies[:-1], entropies[1:])[0,1]
        else:
            correlation = 0.0

        # Emergência se há mudança significativa E tendência positiva
        emergence_detected = max_change > 0.5 and trend > 0.01

        return await {
            'flow_detected': emergence_detected,
            'max_entropy_change': max_change,
            'entropy_trend': trend,
            'temporal_correlation': correlation,
            'entropy_series': entropies
        }

class ComplexityAnalyzer:
    """Analisador de complexidade algorítmica"""

    async def __init__(self):
        self.kolmogorov_history = deque(maxlen=500)

    async def kolmogorov_complexity(self, data: str) -> float:
        """Estima complexidade de Kolmogorov (comprimento mínimo)"""
        # Usa compressão zlib como proxy
        compressed = zlib.compress(data.encode('utf-8'))
        return await len(compressed) / len(data.encode('utf-8'))

    async def fractal_dimension(self, data: np.ndarray) -> float:
        """Calcula dimensão fractal aproximada"""
        if len(data) < 10:
            return await 1.0

        # Método de sandbox counting simplificado
        scales = [2, 4, 8, 16]
        counts = []

        for scale in scales:
            # Conta "caixas" ocupadas
            hist, _ = np.histogram(data, bins=scale)
            occupied = np.sum(hist > 0)
            counts.append(occupied)

        if len(counts) > 1:
            # Regressão log-log para dimensão
            log_scales = np.log(scales)
            log_counts = np.log(counts)
            slope, _ = np.polyfit(log_scales, log_counts, 1)
            return await slope

        return await 1.0

    async def detect_emergent_complexity(self, system_states: List[Any]) -> Dict[str, Any]:
        """Detecta complexidade emergente"""
        if len(system_states) < 5:
            return await {'complexity_emergence': False}

        # Converte estados para strings
        state_strings = [json.dumps(state, default=str, sort_keys=True) for state in system_states]

        # Calcula complexidade ao longo do tempo
        complexities = [self.kolmogorov_complexity(s) for s in state_strings]

        # Detecta se complexidade está aumentando
        complexity_trend = np.polyfit(range(len(complexities)), complexities, 1)[0]

        # Calcula dimensão fractal dos dados
        if system_states and isinstance(system_states[0], dict):
            # Extrai valores numéricos
            numeric_data = []
            for state in system_states:
                if isinstance(state, dict):
                    for key, value in state.items():
                        if isinstance(value, (int, float)):
                            numeric_data.append(value)

            fractal_dim = self.fractal_dimension(np.array(numeric_data)) if numeric_data else 1.0
        else:
            fractal_dim = 1.0

        # Emergência se complexidade aumenta E dimensão > 1
        emergence = complexity_trend > 0.001 and fractal_dim > 1.1

        return await {
            'complexity_emergence': emergence,
            'complexity_trend': complexity_trend,
            'fractal_dimension': fractal_dim,
            'complexity_series': complexities
        }

class StatisticalEmergenceDetector:
    """Detector estatístico de emergência"""

    async def __init__(self):
        self.baseline_distribution = None
        self.emergence_threshold = 3.0  # 3 sigma

    async def establish_baseline(self, data_points: List[np.ndarray]):
        """Estabelece distribuição baseline"""
        if len(data_points) < 10:
            return

        # Combina todos os pontos
        all_data = np.concatenate(data_points) if data_points else np.array([])

        if len(all_data) > 0:
            self.baseline_distribution = {
                'mean': np.mean(all_data),
                'std': np.std(all_data),
                'min': np.min(all_data),
                'max': np.max(all_data)
            }

    async def detect_statistical_anomaly(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Detecta anomalia estatística"""
        if self.baseline_distribution is None or len(current_data) == 0:
            return await {'anomaly_detected': False}

        # Calcula estatísticas atuais
        current_mean = np.mean(current_data)
        current_std = np.std(current_data)

        # Calcula desvios do baseline
        mean_deviation = abs(current_mean - self.baseline_distribution['mean'])
        std_deviation = abs(current_std - self.baseline_distribution['std'])

        # Normaliza desvios
        mean_sigma = mean_deviation / (self.baseline_distribution['std'] + 1e-6)
        std_sigma = std_deviation / (self.baseline_distribution['std'] + 1e-6)

        # Detecta emergência se desvio significativo
        anomaly_detected = mean_sigma > self.emergence_threshold or std_sigma > self.emergence_threshold

        return await {
            'anomaly_detected': anomaly_detected,
            'mean_sigma': mean_sigma,
            'std_sigma': std_sigma,
            'current_mean': current_mean,
            'current_std': current_std
        }

class NetworkEmergenceAnalyzer:
    """Analisador de emergência baseada em redes"""

    async def __init__(self):
        self.interaction_graph = nx.Graph()

    async def update_interaction_graph(self, interactions: List[Tuple[str, str, str]]):
        """Atualiza grafo de interações"""
        for system_a, system_b, interaction_type in interactions:
            if not self.interaction_graph.has_edge(system_a, system_b):
                self.interaction_graph.add_edge(system_a, system_b, weight=1)
            else:
                self.interaction_graph[system_a][system_b]['weight'] += 1

    async def analyze_network_emergence(self) -> Dict[str, Any]:
        """Analisa propriedades emergentes da rede"""
        if len(self.interaction_graph.nodes) < 2:
            return await {'network_emergence': False}

        # Calcula métricas de rede
        try:
            # Centralidade
            betweenness = nx.betweenness_centrality(self.interaction_graph)

            # Clustering coefficient
            clustering = nx.average_clustering(self.interaction_graph)

            # Diâmetro
            if nx.is_connected(self.interaction_graph):
                diameter = nx.diameter(self.interaction_graph)
            else:
                diameter = float('inf')

            # Modularidade (comunidades)
            try:
                from networkx.algorithms.community import greedy_modularity_communities
                communities = list(greedy_modularity_communities(self.interaction_graph))
                modularity = len(communities) / len(self.interaction_graph.nodes)
            except:
                modularity = 0.0

            # Detecta emergência se rede mostra estrutura complexa
            emergence = (
                clustering > 0.3 and  # Alta conectividade local
                modularity > 0.5 and  # Estrutura modular
                max(betweenness.values()) > 0.1  # Nós centrais importantes
            )

            return await {
                'network_emergence': emergence,
                'clustering_coefficient': clustering,
                'modularity': modularity,
                'max_betweenness': max(betweenness.values()),
                'diameter': diameter,
                'num_nodes': len(self.interaction_graph.nodes),
                'num_edges': len(self.interaction_graph.edges)
            }

        except Exception as e:
            logger.error(f"Erro analisando rede: {e}")
            return await {'network_emergence': False, 'error': str(e)}

class IA3MathematicalEmergenceDetector:
    """
    Detector de emergência baseado em matemática rigorosa
    Combina teoria da informação, complexidade algorítmica e análise estatística
    """

    async def __init__(self):
        self.info_analyzer = InformationTheoryAnalyzer()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.statistical_detector = StatisticalEmergenceDetector()
        self.network_analyzer = NetworkEmergenceAnalyzer()

        # Histórico para análise temporal
        self.system_states = deque(maxlen=1000)
        self.interactions = deque(maxlen=1000)

        # Database para emergências detectadas
        self.init_emergence_database()

        logger.info("🔬 IA³ Mathematical Emergence Detector inicializado")

    async def init_emergence_database(self):
        """Inicializa database para emergências"""
        self.emergence_db = sqlite3.connect('ia3_mathematical_emergence.db')
        cursor = self.emergence_db.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergence_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                detection_method TEXT,
                confidence REAL,
                evidence TEXT,
                mathematical_proof TEXT,
                timestamp TEXT,
                system_state TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergence_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                metric_name TEXT,
                value REAL,
                timestamp TEXT
            )
        ''')

        self.emergence_db.commit()

    async def analyze_system_state(self, system_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analisa estado completo do sistema para emergência"""
        self.system_states.append(system_data)

        # Análise de teoria da informação
        info_flow = self._analyze_information_flow()

        # Análise de complexidade
        complexity_analysis = self.complexity_analyzer.detect_emergent_complexity(list(self.system_states))

        # Análise estatística
        if hasattr(system_data, 'values') and system_data:
            numeric_values = [v for v in system_data.values() if isinstance(v, (int, float))]
            stat_analysis = self.statistical_detector.detect_statistical_anomaly(np.array(numeric_values))
        else:
            stat_analysis = {'anomaly_detected': False}

        # Análise de rede
        network_analysis = self.network_analyzer.analyze_network_emergence()

        # Combina evidências
        emergence_confidence = self._calculate_overall_confidence(
            info_flow, complexity_analysis, stat_analysis, network_analysis
        )

        # Detecta emergência se confiança > threshold
        emergence_detected = emergence_confidence > 0.8

        result = {
            'emergence_detected': emergence_detected,
            'confidence': emergence_confidence,
            'evidence': {
                'information_flow': info_flow,
                'complexity': complexity_analysis,
                'statistical': stat_analysis,
                'network': network_analysis
            },
            'mathematical_proof': self._generate_mathematical_proof(
                info_flow, complexity_analysis, stat_analysis, network_analysis
            ),
            'timestamp': datetime.now().isoformat()
        }

        # Registra se emergência detectada
        if emergence_detected:
            self._record_emergence_event(result)

        # Registra métricas
        self._record_metrics(result)

        return await result

    async def _analyze_information_flow(self) -> Dict[str, Any]:
        """Analisa fluxo de informação"""
        if len(self.system_states) < 5:
            return await {'flow_detected': False}

        # Extrai séries temporais de diferentes aspectos
        time_series = []
        for state in self.system_states:
            if isinstance(state, dict):
                # Converte estado para vetor numérico
                numeric_state = []
                for key, value in state.items():
                    if isinstance(value, (int, float)):
                        numeric_state.append(value)
                    elif isinstance(value, str):
                        # Hash da string
                        numeric_state.append(hash(value) % 1000 / 1000.0)
                    elif isinstance(value, (list, tuple)):
                        numeric_state.append(len(value) / 100.0)

                if numeric_state:
                    time_series.append(np.array(numeric_state))

        if time_series:
            return await self.info_analyzer.detect_information_flow(time_series)
        else:
            return await {'flow_detected': False}

    async def _calculate_overall_confidence(self, info_flow: Dict, complexity: Dict,
                                    statistical: Dict, network: Dict) -> float:
        """Calcula confiança geral de emergência"""
        weights = {
            'information_flow': 0.3,
            'complexity': 0.3,
            'statistical': 0.2,
            'network': 0.2
        }

        confidence = 0.0

        # Informação
        if info_flow.get('flow_detected', False):
            confidence += weights['information_flow'] * min(1.0, info_flow.get('max_entropy_change', 0) / 2.0)

        # Complexidade
        if complexity.get('complexity_emergence', False):
            confidence += weights['complexity'] * min(1.0, abs(complexity.get('complexity_trend', 0)) * 1000)

        # Estatística
        if statistical.get('anomaly_detected', False):
            confidence += weights['statistical'] * min(1.0, max(statistical.get('mean_sigma', 0),
                                                               statistical.get('std_sigma', 0)) / 5.0)

        # Rede
        if network.get('network_emergence', False):
            confidence += weights['network'] * min(1.0, network.get('clustering_coefficient', 0) * 2)

        return await min(1.0, confidence)

    async def _generate_mathematical_proof(self, info_flow: Dict, complexity: Dict,
                                   statistical: Dict, network: Dict) -> str:
        """Gera prova matemática da emergência"""
        proof_parts = []

        if info_flow.get('flow_detected'):
            proof_parts.append(f"Teoria da Informação: Fluxo detectado com ΔH = {info_flow.get('max_entropy_change', 0):.3f}")

        if complexity.get('complexity_emergence'):
            proof_parts.append(f"Complexidade Algorítmica: Tendência C(t) = {complexity.get('complexity_trend', 0):.6f}")

        if statistical.get('anomaly_detected'):
            proof_parts.append(f"Análise Estatística: Anomalia {max(statistical.get('mean_sigma', 0), statistical.get('std_sigma', 0)):.1f}σ")

        if network.get('network_emergence'):
            proof_parts.append(f"Análise de Redes: Clustering = {network.get('clustering_coefficient', 0):.3f}")

        if proof_parts:
            return await " ∩ ".join(proof_parts)
        else:
            return await "Nenhuma prova matemática estabelecida"

    async def _record_emergence_event(self, emergence_result: Dict):
        """Registra evento de emergência"""
        cursor = self.emergence_db.cursor()
        cursor.execute("""
            INSERT INTO emergence_events
            (detection_method, confidence, evidence, mathematical_proof, timestamp, system_state)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'mathematical_combined',
            emergence_result['confidence'],
            json.dumps(emergence_result['evidence']),
            emergence_result['mathematical_proof'],
            emergence_result['timestamp'],
            json.dumps(self.system_states[-1] if self.system_states else {})
        ))
        self.emergence_db.commit()

        logger.warning(f"🚨 EMERGÊNCIA MATEMÁTICA DETECTADA: {emergence_result['mathematical_proof']}")

    async def _record_metrics(self, emergence_result: Dict):
        """Registra métricas de análise"""
        cursor = self.emergence_db.cursor()

        evidence = emergence_result['evidence']
        timestamp = emergence_result['timestamp']

        metrics = [
            ('emergence_confidence', emergence_result['confidence']),
            ('info_flow_detected', 1.0 if evidence['information_flow'].get('flow_detected') else 0.0),
            ('complexity_emergence', 1.0 if evidence['complexity'].get('complexity_emergence') else 0.0),
            ('statistical_anomaly', 1.0 if evidence['statistical'].get('anomaly_detected') else 0.0),
            ('network_emergence', 1.0 if evidence['network'].get('network_emergence') else 0.0),
        ]

        for metric_name, value in metrics:
            cursor.execute("""
                INSERT INTO emergence_metrics (metric_name, value, timestamp)
                VALUES (?, ?, ?)
            """, (metric_name, value, timestamp))

        self.emergence_db.commit()

    async def get_emergence_history(self) -> List[Dict]:
        """Retorna histórico de emergências detectadas"""
        cursor = self.emergence_db.cursor()
        cursor.execute("""
            SELECT detection_method, confidence, mathematical_proof, timestamp
            FROM emergence_events
            ORDER BY timestamp DESC
            LIMIT 20
        """)

        history = []
        for row in cursor.fetchall():
            history.append({
                'method': row[0],
                'confidence': row[1],
                'proof': row[2],
                'timestamp': row[3]
            })

        return await history

    async def run_emergence_analysis_cycle(self) -> Dict[str, Any]:
        """Executa ciclo completo de análise de emergência"""
        logger.info("🔬 Executando análise matemática de emergência...")

        # Coleta dados atuais do sistema
        system_data = self._collect_current_system_data()

        # Analisa emergência
        result = self.analyze_system_state(system_data)

        # Reporta resultado
        if result['emergence_detected']:
            logger.warning(f"🎯 EMERGÊNCIA CONFIRMADA: {result['mathematical_proof']}")
        else:
            logger.info(f"📊 Análise concluída: confiança de emergência = {result['confidence']:.3f}")

        return await result

    async def _collect_current_system_data(self) -> Dict[str, Any]:
        """Coleta dados atuais de todos os sistemas"""
        system_data = {}

        # Conta processos
        try:
            import psutil
            ai_processes = [p for p in psutil.process_iter(['name']) if 'ai' in p.info['name'].lower() or 'ia3' in p.info['name'].lower()]
            system_data['ai_processes'] = len(ai_processes)
        except:
            system_data['ai_processes'] = 0

        # Conta arquivos de database
        db_files = [f for f in os.listdir('.') if f.endswith('.db')]
        system_data['database_files'] = len(db_files)

        # Conta arquivos Python
        py_files = [f for f in os.listdir('.') if f.endswith('.py')]
        system_data['python_files'] = len(py_files)

        # Tamanho total de logs
        try:
            log_size = sum(os.path.getsize(f) for f in os.listdir('.') if f.endswith('.log'))
            system_data['total_log_size'] = log_size
        except:
            system_data['total_log_size'] = 0

        # Timestamp
        system_data['analysis_timestamp'] = datetime.now().isoformat()

        return await system_data

# Instância global
ia3_mathematical_detector = IA3MathematicalEmergenceDetector()

if __name__ == "__main__":
    # Executa análise
    result = ia3_mathematical_detector.run_emergence_analysis_cycle()
    print(f"Análise de emergência: {result['emergence_detected']}")
    print(f"Confiança: {result['confidence']:.3f}")
    print(f"Prova matemática: {result['mathematical_proof']}")

    # Mostra histórico
    history = ia3_mathematical_detector.get_emergence_history()
    print(f"Histórico de emergências: {len(history)} eventos")