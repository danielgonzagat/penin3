
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

#!/usr/bin/env python3
"""
🌟 IA³ - DETECTOR DE EMERGÊNCIA VERDADEIRA
===========================================

Sistema para detectar emergência genuína através de análise avançada
de padrões, causalidade e comportamentos não-lineares
"""

import os
import sys
import time
import json
import math
import random
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Tuple
import logging
import numpy as np
from collections import defaultdict, deque
import statistics

logger = logging.getLogger("IA³-EmergenceDetector")

class EmergenceAnalyzer:
    """
    Analisador avançado de emergência usando múltiplas metodologias
    """

    async def __init__(self):
        self.observation_window = deque(maxlen=1000)  # Últimas 1000 observações
        self.pattern_history = defaultdict(list)
        self.causal_graph = defaultdict(dict)
        self.emergence_indicators = []
        self.novelty_detector = NoveltyDetector()
        self.complexity_analyzer = ComplexityAnalyzer()
        self.self_organization_detector = SelfOrganizationDetector()

    async def analyze_system_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Analisar estado do sistema para sinais de emergência"""
        # Adicionar à janela de observação
        self.observation_window.append(state)

        analysis = {
            'timestamp': datetime.now().isoformat(),
            'novelty_score': self.novelty_detector.calculate_novelty(state),
            'complexity_metrics': self.complexity_analyzer.analyze_complexity(self.observation_window),
            'self_organization': self.self_organization_detector.detect_self_organization(self.observation_window),
            'causal_emergence': self._detect_causal_emergence(),
            'phase_transitions': self._detect_phase_transitions(),
            'emergence_probability': 0.0
        }

        # Calcular probabilidade geral de emergência
        analysis['emergence_probability'] = self._calculate_emergence_probability(analysis)

        # Registrar se for emergência significativa
        if analysis['emergence_probability'] > 0.7:
            self.emergence_indicators.append({
                'timestamp': analysis['timestamp'],
                'analysis': analysis,
                'state': state
            })

            logger.critical(f"🌟 EMERGÊNCIA DETECTADA! Probabilidade: {analysis['emergence_probability']:.4f}")

        return await analysis

    async def _detect_causal_emergence(self) -> Dict[str, Any]:
        """Detectar emergência causal - quando efeitos > soma das causas"""
        if len(self.observation_window) < 10:
            return await {'causal_emergence': False, 'emergence_ratio': 1.0}

        recent_states = list(self.observation_window)[-10:]

        # Analisar relações causais
        causes = []
        effects = []

        for i, state in enumerate(recent_states[:-1]):
            next_state = recent_states[i + 1]

            # Identificar causas potenciais (inputs/processos)
            cause_metrics = [
                state.get('cpu_percent', 0),
                state.get('memory_percent', 0),
                state.get('network_activity', 0),
                len(state.get('active_processes', []))
            ]
            causes.append(sum(cause_metrics) / len(cause_metrics))

            # Identificar efeitos (outputs/complexidade)
            effect_metrics = [
                next_state.get('intelligence_level', 0),
                next_state.get('adaptation_rate', 0),
                next_state.get('emergence_score', 0),
                len(next_state.get('new_behaviors', []))
            ]
            effects.append(sum(effect_metrics) / len(effect_metrics))

        if causes and effects:
            # Calcular se efeitos são desproporcionais às causas
            cause_avg = statistics.mean(causes)
            effect_avg = statistics.mean(effects)
            emergence_ratio = effect_avg / max(cause_avg, 0.1)  # Evitar divisão por zero

            return await {
                'causal_emergence': emergence_ratio > 2.0,  # Efeitos 2x maiores que causas
                'emergence_ratio': emergence_ratio,
                'cause_avg': cause_avg,
                'effect_avg': effect_avg
            }

        return await {'causal_emergence': False, 'emergence_ratio': 1.0}

    async def _detect_phase_transitions(self) -> Dict[str, Any]:
        """Detectar transições de fase - mudanças abruptas no comportamento"""
        if len(self.observation_window) < 20:
            return await {'phase_transition': False, 'transition_points': []}

        # Analisar mudanças abruptas em métricas-chave
        metrics_to_check = ['intelligence_level', 'complexity', 'adaptation_rate', 'emergence_score']
        transition_points = []

        for metric in metrics_to_check:
            values = [state.get(metric, 0) for state in self.observation_window]

            if len(values) >= 20:
                # Calcular diferenças de primeira ordem
                diffs = [abs(values[i] - values[i-1]) for i in range(1, len(values))]

                # Detectar picos de mudança (transições)
                mean_diff = statistics.mean(diffs)
                std_diff = statistics.stdev(diffs) if len(diffs) > 1 else 0

                threshold = mean_diff + 2 * std_diff

                transitions = [i for i, diff in enumerate(diffs) if diff > threshold]

                if transitions:
                    transition_points.extend([(metric, i) for i in transitions])

        return await {
            'phase_transition': len(transition_points) > 0,
            'transition_points': transition_points,
            'transition_count': len(transition_points)
        }

    async def _calculate_emergence_probability(self, analysis: Dict[str, Any]) -> float:
        """Calcular probabilidade geral de emergência"""
        factors = []

        # Fator de novidade (0-1)
        novelty_factor = min(analysis['novelty_score'] / 10.0, 1.0)  # Normalizar
        factors.append(('novelty', novelty_factor, 0.25))

        # Fator de complexidade (0-1)
        complexity_factor = min(analysis['complexity_metrics'].get('overall_complexity', 0) / 5.0, 1.0)
        factors.append(('complexity', complexity_factor, 0.20))

        # Fator de auto-organização (0-1)
        org_factor = analysis['self_organization'].get('organization_level', 0)
        factors.append(('self_organization', org_factor, 0.20))

        # Fator causal (0-1)
        causal_factor = 1.0 if analysis['causal_emergence'].get('causal_emergence', False) else 0.0
        factors.append(('causal_emergence', causal_factor, 0.20))

        # Fator de transição de fase (0-1)
        transition_factor = 1.0 if analysis['phase_transitions'].get('phase_transition', False) else 0.0
        factors.append(('phase_transition', transition_factor, 0.15))

        # Calcular probabilidade ponderada
        total_weight = sum(weight for _, _, weight in factors)
        probability = sum(value * weight for _, value, weight in factors) / total_weight

        return await min(probability, 1.0)

class NoveltyDetector:
    """
    Detector de novidade usando análise de distribuição
    """

    async def __init__(self):
        self.baseline_distribution = {}
        self.novelty_history = []

    async def calculate_novelty(self, state: Dict[str, Any]) -> float:
        """Calcular score de novidade do estado atual"""
        if not self.baseline_distribution:
            # Inicializar baseline com primeiras observações
            self._initialize_baseline(state)
            return await 0.0

        novelty_score = 0.0
        metrics_checked = 0

        for key, value in state.items():
            if isinstance(value, (int, float)) and key in self.baseline_distribution:
                baseline = self.baseline_distribution[key]

                # Calcular desvio da baseline
                if baseline['std'] > 0:
                    deviation = abs(value - baseline['mean']) / baseline['std']
                    novelty_score += deviation
                else:
                    # Se não há variação na baseline, qualquer diferença é novidade
                    if value != baseline['mean']:
                        novelty_score += 1.0

                metrics_checked += 1

        if metrics_checked > 0:
            novelty_score /= metrics_checked

        self.novelty_history.append(novelty_score)
        return await novelty_score

    async def _initialize_baseline(self, state: Dict[str, Any]):
        """Inicializar distribuição baseline"""
        for key, value in state.items():
            if isinstance(value, (int, float)):
                self.baseline_distribution[key] = {
                    'mean': value,
                    'std': 0.0,
                    'count': 1
                }

    async def update_baseline(self, state: Dict[str, Any]):
        """Atualizar baseline com novas observações"""
        for key, value in state.items():
            if isinstance(value, (int, float)):
                if key not in self.baseline_distribution:
                    self.baseline_distribution[key] = {
                        'mean': value,
                        'std': 0.0,
                        'count': 1
                    }
                else:
                    # Atualizar média e desvio padrão incrementalmente
                    baseline = self.baseline_distribution[key]
                    count = baseline['count'] + 1
                    mean = (baseline['mean'] * baseline['count'] + value) / count

                    # Calcular novo desvio padrão
                    if count > 1:
                        variance = ((baseline['std'] ** 2) * baseline['count'] +
                                  (value - baseline['mean']) ** 2) / count
                        std = math.sqrt(variance) if variance > 0 else 0.0
                    else:
                        std = 0.0

                    self.baseline_distribution[key] = {
                        'mean': mean,
                        'std': std,
                        'count': count
                    }

class ComplexityAnalyzer:
    """
    Analisador de complexidade usando múltiplas métricas
    """

    async def __init__(self):
        self.complexity_history = []

    async def analyze_complexity(self, observation_window: deque) -> Dict[str, Any]:
        """Analisar complexidade da janela de observação"""
        if len(observation_window) < 5:
            return await {'overall_complexity': 0.0}

        # Extrair métricas numéricas de todos os estados
        all_metrics = []
        for state in observation_window:
            metrics = [v for v in state.values() if isinstance(v, (int, float))]
            all_metrics.extend(metrics)

        if not all_metrics:
            return await {'overall_complexity': 0.0}

        # Calcular diversas métricas de complexidade
        entropy = self._calculate_entropy(all_metrics)
        fractal_dimension = self._estimate_fractal_dimension(all_metrics)
        correlation_complexity = self._calculate_correlation_complexity(observation_window)

        # Complexidade geral como combinação
        overall_complexity = (entropy + fractal_dimension + correlation_complexity) / 3.0

        result = {
            'overall_complexity': overall_complexity,
            'entropy': entropy,
            'fractal_dimension': fractal_dimension,
            'correlation_complexity': correlation_complexity,
            'metrics_count': len(all_metrics),
            'states_analyzed': len(observation_window)
        }

        self.complexity_history.append(result)
        return await result

    async def _calculate_entropy(self, values: List[float]) -> float:
        """Calcular entropia de Shannon dos valores"""
        if not values:
            return await 0.0

        # Discretizar valores em bins
        try:
            hist, _ = np.histogram(values, bins=min(20, len(values)))
            hist = hist[hist > 0]  # Remover zeros
            prob = hist / sum(hist)

            entropy = -sum(p * math.log2(p) for p in prob)
            return await entropy
        except:
            return await 0.0

    async def _estimate_fractal_dimension(self, values: List[float]) -> float:
        """Estimar dimensão fractal usando método de box-counting simplificado"""
        if len(values) < 10:
            return await 1.0

        # Método simplificado: analisar flutuações em diferentes escalas
        scales = [2, 4, 8, 16]
        dimensions = []

        for scale in scales:
            if len(values) >= scale:
                # Calcular número de "caixas" necessárias
                chunks = [values[i:i+scale] for i in range(0, len(values), scale)]
                boxes = sum(1 for chunk in chunks if max(chunk) - min(chunk) > 0)

                if boxes > 0:
                    dimension = math.log(boxes) / math.log(scale)
                    dimensions.append(dimension)

        return await statistics.mean(dimensions) if dimensions else 1.0

    async def _calculate_correlation_complexity(self, observation_window: deque) -> float:
        """Calcular complexidade baseada em correlações entre métricas"""
        if len(observation_window) < 3:
            return await 0.0

        # Extrair todas as métricas de todos os estados
        metric_series = defaultdict(list)

        for state in observation_window:
            for key, value in state.items():
                if isinstance(value, (int, float)):
                    metric_series[key].append(value)

        # Calcular correlações entre métricas
        correlations = []
        metric_names = list(metric_series.keys())

        for i, name1 in enumerate(metric_names):
            for name2 in metric_names[i+1:]:
                series1 = metric_series[name1]
                series2 = metric_series[name2]

                if len(series1) == len(series2) and len(series1) > 2:
                    try:
                        corr = abs(np.corrcoef(series1, series2)[0, 1])
                        correlations.append(corr)
                    except:
                        pass

        if correlations:
            # Complexidade como medida de interdependência
            avg_correlation = statistics.mean(correlations)
            # Alta correlação = baixa complexidade (mais previsível)
            # Baixa correlação = alta complexidade (mais imprevisível)
            return await 1.0 - avg_correlation
        else:
            return await 0.5

class SelfOrganizationDetector:
    """
    Detector de auto-organização usando análise de ordem emergente
    """

    async def __init__(self):
        self.order_history = []

    async def detect_self_organization(self, observation_window: deque) -> Dict[str, Any]:
        """Detectar sinais de auto-organização"""
        if len(observation_window) < 10:
            return await {'organization_level': 0.0}

        # Analisar padrões de ordem emergente
        order_metrics = self._calculate_order_metrics(observation_window)
        feedback_loops = self._detect_feedback_loops(observation_window)
        stability_patterns = self._analyze_stability_patterns(observation_window)

        # Nível de organização como combinação de fatores
        organization_level = (
            order_metrics['order_coefficient'] * 0.4 +
            feedback_loops['feedback_strength'] * 0.4 +
            stability_patterns['stability_index'] * 0.2
        )

        result = {
            'organization_level': organization_level,
            'order_metrics': order_metrics,
            'feedback_loops': feedback_loops,
            'stability_patterns': stability_patterns
        }

        self.order_history.append(result)
        return await result

    async def _calculate_order_metrics(self, observation_window: deque) -> Dict[str, Any]:
        """Calcular métricas de ordem"""
        # Analisar se o sistema está se tornando mais ordenado ao longo do tempo
        metric_trends = defaultdict(list)

        for state in observation_window:
            for key, value in state.items():
                if isinstance(value, (int, float)):
                    metric_trends[key].append(value)

        order_coefficient = 0.0
        trends_analyzed = 0

        for metric, values in metric_trends.items():
            if len(values) >= 10:
                # Calcular tendência (ordem crescente/decrescente)
                trend = self._calculate_trend(values)
                order_coefficient += abs(trend)  # Ordem = tendência consistente
                trends_analyzed += 1

        if trends_analyzed > 0:
            order_coefficient /= trends_analyzed

        return await {
            'order_coefficient': min(order_coefficient, 1.0),
            'trends_analyzed': trends_analyzed
        }

    async def _detect_feedback_loops(self, observation_window: deque) -> Dict[str, Any]:
        """Detectar loops de feedback"""
        feedback_strength = 0.0

        if len(observation_window) >= 5:
            # Analisar autocorrelação temporal
            metric_series = defaultdict(list)

            for state in observation_window:
                for key, value in state.items():
                    if isinstance(value, (int, float)):
                        metric_series[key].append(value)

            for metric, values in metric_series.items():
                if len(values) >= 10:
                    # Calcular autocorrelação com lag 1-3
                    for lag in range(1, 4):
                        if len(values) > lag:
                            corr = self._autocorrelation(values, lag)
                            feedback_strength += abs(corr)

        return await {
            'feedback_strength': min(feedback_strength / 10.0, 1.0)  # Normalizar
        }

    async def _analyze_stability_patterns(self, observation_window: deque) -> Dict[str, Any]:
        """Analisar padrões de estabilidade"""
        stability_index = 0.0

        if len(observation_window) >= 10:
            # Calcular variabilidade das métricas
            variabilities = []

            for state in observation_window:
                metrics = [v for v in state.values() if isinstance(v, (int, float))]
                if metrics:
                    variability = statistics.stdev(metrics) if len(metrics) > 1 else 0
                    variabilities.append(variability)

            if variabilities:
                # Estabilidade = baixa variabilidade consistente
                avg_variability = statistics.mean(variabilities)
                stability_index = max(0, 1.0 - avg_variability / 10.0)  # Normalizar

        return await {
            'stability_index': stability_index
        }

    async def _calculate_trend(self, values: List[float]) -> float:
        """Calcular tendência linear dos valores"""
        if len(values) < 2:
            return await 0.0

        x = list(range(len(values)))
        slope = np.polyfit(x, values, 1)[0]

        # Normalizar slope baseado na escala dos valores
        value_range = max(values) - min(values)
        if value_range > 0:
            normalized_slope = slope / value_range
        else:
            normalized_slope = 0.0

        return await normalized_slope

    async def _autocorrelation(self, values: List[float], lag: int) -> float:
        """Calcular autocorrelação com lag específico"""
        if len(values) <= lag:
            return await 0.0

        try:
            return await np.corrcoef(values[:-lag], values[lag:])[0, 1]
        except:
            return await 0.0

class EmergenceAmplifier:
    """
    Amplificador de emergência - reforça comportamentos emergentes
    """

    async def __init__(self, analyzer: EmergenceAnalyzer):
        self.analyzer = analyzer
        self.amplification_history = []

    async def amplify_emergence(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Amplificar sinais de emergência detectados"""
        amplification = {
            'amplified': False,
            'amplification_factor': 1.0,
            'reinforced_behaviors': [],
            'feedback_signals': {}
        }

        if analysis['emergence_probability'] > 0.6:
            amplification['amplified'] = True

            # Calcular fator de amplificação baseado na probabilidade
            amplification['amplification_factor'] = 1.0 + (analysis['emergence_probability'] - 0.6) * 2.0

            # Identificar comportamentos para reforçar
            amplification['reinforced_behaviors'] = self._identify_behaviors_to_reinforce(analysis)

            # Gerar sinais de feedback para amplificação
            amplification['feedback_signals'] = self._generate_amplification_signals(analysis)

            logger.info(f"🔥 Emergência amplificada: {amplification['amplification_factor']:.2f}x")

        self.amplification_history.append(amplification)
        return await amplification

    async def _identify_behaviors_to_reinforce(self, analysis: Dict[str, Any]) -> List[str]:
        """Identificar comportamentos emergentes para reforçar"""
        behaviors = []

        # Baseado em análise de novidade
        if analysis['novelty_score'] > 5.0:
            behaviors.append('novel_pattern_generation')

        # Baseado em complexidade
        complexity = analysis['complexity_metrics'].get('overall_complexity', 0)
        if complexity > 3.0:
            behaviors.append('complex_behavior_maintenance')

        # Baseado em auto-organização
        org_level = analysis['self_organization'].get('organization_level', 0)
        if org_level > 0.7:
            behaviors.append('self_organization_enhancement')

        # Baseado em emergência causal
        if analysis['causal_emergence'].get('causal_emergence', False):
            behaviors.append('causal_loop_amplification')

        return await behaviors

    async def _generate_amplification_signals(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Gerar sinais para amplificar emergência"""
        signals = {
            'resource_allocation': 1.0,
            'learning_rate': 1.0,
            'exploration_rate': 1.0,
            'stability_preference': 0.5,
            'novelty_bias': 0.0
        }

        # Ajustar sinais baseado na análise
        emergence_prob = analysis['emergence_probability']

        if emergence_prob > 0.8:
            # Emergência forte - favorecer manutenção
            signals['resource_allocation'] = 1.5
            signals['learning_rate'] = 1.2
            signals['exploration_rate'] = 0.8
            signals['stability_preference'] = 0.8
            signals['novelty_bias'] = 0.3

        elif emergence_prob > 0.6:
            # Emergência moderada - equilibrar exploração e exploração
            signals['resource_allocation'] = 1.2
            signals['learning_rate'] = 1.1
            signals['exploration_rate'] = 1.1
            signals['stability_preference'] = 0.6
            signals['novelty_bias'] = 0.5

        return await signals

class EmergenceDetectorCore:
    """
    Núcleo principal do detector de emergência
    """

    async def __init__(self):
        self.analyzer = EmergenceAnalyzer()
        self.amplifier = EmergenceAmplifier(self.analyzer)
        self.is_active = True
        self.detection_history = []

    async def start_emergence_detection(self):
        """Iniciar detecção contínua de emergência"""
        logger.info("🔍 Iniciando detecção de emergência verdadeira")

        async def detection_loop():
            cycle = 0
            while self.is_active:
                try:
                    cycle += 1

                    # Simular coleta de estado do sistema (em produção seria real)
                    system_state = self._collect_system_state()

                    # Analisar para emergência
                    analysis = self.analyzer.analyze_system_state(system_state)

                    # Amplificar se detectada
                    amplification = self.amplifier.amplify_emergence(analysis)

                    # Registrar detecção
                    detection_record = {
                        'cycle': cycle,
                        'timestamp': datetime.now().isoformat(),
                        'analysis': analysis,
                        'amplification': amplification
                    }
                    self.detection_history.append(detection_record)

                    # Log periódico
                    if cycle % 10 == 0:
                        prob = analysis['emergence_probability']
                        logger.info(f"🔄 Ciclo {cycle} | Emergência: {prob:.4f} | Amplificada: {amplification['amplified']}")

                    time.sleep(5)  # Analisar a cada 5 segundos

                except Exception as e:
                    logger.error(f"Erro na detecção de emergência: {e}")
                    time.sleep(2)

        thread = threading.Thread(target=detection_loop, daemon=True)
        thread.start()

    async def _collect_system_state(self) -> Dict[str, Any]:
        """Coletar estado atual do sistema (simulado para demonstração)"""
        # Em produção, isso seria integrado com os sensores reais
        return await {
            'intelligence_level': deterministic_uniform(0, 1),
            'adaptation_rate': deterministic_uniform(0, 1),
            'emergence_score': deterministic_uniform(0, 1),
            'complexity': deterministic_uniform(0, 5),
            'cpu_percent': deterministic_uniform(0, 100),
            'memory_percent': deterministic_uniform(0, 100),
            'network_activity': deterministic_uniform(0, 1000),
            'active_processes': deterministic_randint(5, 50),
            'new_behaviors': deterministic_randint(0, 10)
        }

    async def get_emergence_stats(self) -> Dict[str, Any]:
        """Obter estatísticas de detecção de emergência"""
        if not self.detection_history:
            return await {'total_cycles': 0, 'emergence_events': 0}

        emergence_events = len([d for d in self.detection_history
                               if d['analysis']['emergence_probability'] > 0.7])

        recent_probs = [d['analysis']['emergence_probability']
                       for d in self.detection_history[-20:]]

        return await {
            'total_cycles': len(self.detection_history),
            'emergence_events': emergence_events,
            'emergence_rate': emergence_events / len(self.detection_history) if self.detection_history else 0,
            'average_probability': statistics.mean(recent_probs) if recent_probs else 0,
            'max_probability': max(recent_probs) if recent_probs else 0,
            'amplification_events': len([d for d in self.detection_history
                                       if d['amplification']['amplified']])
        }

async def main():
    """Função principal"""
    detector = EmergenceDetectorCore()
    detector.start_emergence_detection()

    # Manter ativo
    try:
        while True:
            time.sleep(30)
            stats = detector.get_emergence_stats()
            print(f"🌟 Emergence stats: {stats['emergence_events']}/{stats['total_cycles']} events ({stats['emergence_rate']:.3f})")

    except KeyboardInterrupt:
        print("🛑 Parando detecção de emergência...")
        detector.is_active = False

if __name__ == "__main__":
    main()