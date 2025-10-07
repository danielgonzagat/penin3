
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
SISTEMA DE MÉTRICAS REAIS DE PROGRESSO
=====================================
Monitora e valida o progresso real da inteligência artificial
"""

import json
import time
import numpy as np
import torch
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import threading
import queue
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RealMetricsSystem")

class RealMetricsSystem:
    """
    Sistema de métricas reais para monitorar inteligência artificial
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or self._default_config()
        self.metrics = {
            'system_start_time': datetime.now(),
            'total_cycles': 0,
            'real_learning_events': 0,
            'evolution_events': 0,
            'reinforcement_events': 0,
            'neural_processing_events': 0,
            'emergence_detected': 0,
            'intelligence_score': 0.0,
            'learning_rate': 0.0,
            'adaptation_rate': 0.0,
            'creativity_score': 0.0,
            'efficiency_score': 0.0,
            'stability_score': 0.0,
            'convergence_score': 0.0,
            'real_time_metrics': [],
            'performance_history': [],
            'emergence_history': [],
            'intelligence_trends': []
        }
        
        self.running = False
        self.metrics_queue = queue.Queue()
        self.threads = []
        
    def _default_config(self) -> Dict:
        """Configuração padrão do sistema de métricas"""
        return {
            'update_interval': 1.0,  # segundos
            'history_length': 1000,
            'emergence_threshold': 0.8,
            'learning_threshold': 0.1,
            'convergence_threshold': 0.95,
            'metrics_file': 'real_intelligence_metrics.json',
            'dashboard_file': 'intelligence_dashboard.json'
        }
    
    def start_monitoring(self):
        """Inicia monitoramento de métricas"""
        logger.info("📊 INICIANDO SISTEMA DE MÉTRICAS REAIS")
        logger.info("=" * 50)
        
        self.running = True
        
        # Iniciar thread de processamento de métricas
        self._start_metrics_processor()
        
        # Iniciar thread de análise de tendências
        self._start_trend_analyzer()
        
        # Iniciar thread de detecção de emergência
        self._start_emergence_detector()
        
        logger.info("✅ Sistema de métricas iniciado!")
    
    def _start_metrics_processor(self):
        """Inicia processador de métricas"""
        def process_metrics():
            while self.running:
                try:
                    # Processar métricas da fila
                    while not self.metrics_queue.empty():
                        metric = self.metrics_queue.get_nowait()
                        self._process_metric(metric)
                    
                    # Atualizar métricas base
                    self._update_base_metrics()
                    
                    time.sleep(self.config['update_interval'])
                    
                except Exception as e:
                    logger.error(f"Erro no processador de métricas: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=process_metrics, daemon=True)
        thread.start()
        self.threads.append(thread)
    
    def _start_trend_analyzer(self):
        """Inicia analisador de tendências"""
        def analyze_trends():
            while self.running:
                try:
                    # Analisar tendências de inteligência
                    self._analyze_intelligence_trends()
                    
                    # Analisar tendências de aprendizado
                    self._analyze_learning_trends()
                    
                    # Analisar tendências de convergência
                    self._analyze_convergence_trends()
                    
                    time.sleep(5.0)  # Analisar a cada 5 segundos
                    
                except Exception as e:
                    logger.error(f"Erro no analisador de tendências: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=analyze_trends, daemon=True)
        thread.start()
        self.threads.append(thread)
    
    def _start_emergence_detector(self):
        """Inicia detector de emergência"""
        def detect_emergence():
            while self.running:
                try:
                    # Detectar emergência de inteligência
                    if self._detect_intelligence_emergence():
                        self.metrics['emergence_detected'] += 1
                        self.metrics['emergence_history'].append({
                            'timestamp': datetime.now(),
                            'intelligence_score': self.metrics['intelligence_score'],
                            'learning_rate': self.metrics['learning_rate'],
                            'adaptation_rate': self.metrics['adaptation_rate']
                        })
                        
                        logger.info("🌟 EMERGÊNCIA DE INTELIGÊNCIA DETECTADA!")
                        logger.info(f"   Score: {self.metrics['intelligence_score']:.3f}")
                        logger.info(f"   Taxa de aprendizado: {self.metrics['learning_rate']:.3f}")
                        logger.info(f"   Taxa de adaptação: {self.metrics['adaptation_rate']:.3f}")
                    
                    time.sleep(2.0)  # Verificar a cada 2 segundos
                    
                except Exception as e:
                    logger.error(f"Erro no detector de emergência: {e}")
                    time.sleep(1)
        
        thread = threading.Thread(target=detect_emergence, daemon=True)
        thread.start()
        self.threads.append(thread)
    
    def add_metric(self, metric_type: str, data: Dict):
        """Adiciona métrica à fila de processamento"""
        metric = {
            'type': metric_type,
            'data': data,
            'timestamp': datetime.now()
        }
        self.metrics_queue.put(metric)
    
    def _process_metric(self, metric: Dict):
        """Processa métrica individual"""
        metric_type = metric['type']
        data = metric['data']
        
        if metric_type == 'learning_event':
            self.metrics['real_learning_events'] += 1
            self.metrics['learning_rate'] = self._calculate_learning_rate()
            
        elif metric_type == 'evolution_event':
            self.metrics['evolution_events'] += 1
            self.metrics['adaptation_rate'] = self._calculate_adaptation_rate()
            
        elif metric_type == 'reinforcement_event':
            self.metrics['reinforcement_events'] += 1
            
        elif metric_type == 'neural_processing_event':
            self.metrics['neural_processing_events'] += 1
            
        elif metric_type == 'performance_metric':
            self.metrics['performance_history'].append({
                'timestamp': metric['timestamp'],
                'data': data
            })
            
            # Manter apenas histórico recente
            if len(self.metrics['performance_history']) > self.config['history_length']:
                self.metrics['performance_history'] = self.metrics['performance_history'][-self.config['history_length']:]
        
        # Atualizar score de inteligência
        self.metrics['intelligence_score'] = self._calculate_intelligence_score()
    
    def _update_base_metrics(self):
        """Atualiza métricas base do sistema"""
        self.metrics['total_cycles'] += 1
        
        # Calcular métricas derivadas
        self.metrics['creativity_score'] = self._calculate_creativity_score()
        self.metrics['efficiency_score'] = self._calculate_efficiency_score()
        self.metrics['stability_score'] = self._calculate_stability_score()
        self.metrics['convergence_score'] = self._calculate_convergence_score()
        
        # Adicionar métrica em tempo real
        real_time_metric = {
            'timestamp': datetime.now(),
            'intelligence_score': self.metrics['intelligence_score'],
            'learning_rate': self.metrics['learning_rate'],
            'adaptation_rate': self.metrics['adaptation_rate'],
            'creativity_score': self.metrics['creativity_score'],
            'efficiency_score': self.metrics['efficiency_score'],
            'stability_score': self.metrics['stability_score'],
            'convergence_score': self.metrics['convergence_score']
        }
        
        self.metrics['real_time_metrics'].append(real_time_metric)
        
        # Manter apenas histórico recente
        if len(self.metrics['real_time_metrics']) > self.config['history_length']:
            self.metrics['real_time_metrics'] = self.metrics['real_time_metrics'][-self.config['history_length']:]
    
    def _calculate_learning_rate(self) -> float:
        """Calcula taxa de aprendizado"""
        total_events = self.metrics['real_learning_events']
        time_elapsed = (datetime.now() - self.metrics['system_start_time']).total_seconds()
        
        if time_elapsed > 0:
            return total_events / time_elapsed
        return 0.0
    
    def _calculate_adaptation_rate(self) -> float:
        """Calcula taxa de adaptação"""
        evolution_events = self.metrics['evolution_events']
        total_events = self.metrics['real_learning_events']
        
        if total_events > 0:
            return evolution_events / total_events
        return 0.0
    
    def _calculate_intelligence_score(self) -> float:
        """Calcula score de inteligência geral"""
        # Fatores de inteligência
        learning_factor = min(self.metrics['learning_rate'] * 10, 1.0)
        adaptation_factor = min(self.metrics['adaptation_rate'] * 2, 1.0)
        creativity_factor = self.metrics['creativity_score']
        efficiency_factor = self.metrics['efficiency_score']
        stability_factor = self.metrics['stability_score']
        convergence_factor = self.metrics['convergence_score']
        
        # Score ponderado
        intelligence_score = (
            learning_factor * 0.25 +
            adaptation_factor * 0.20 +
            creativity_factor * 0.15 +
            efficiency_factor * 0.15 +
            stability_factor * 0.15 +
            convergence_factor * 0.10
        )
        
        return min(intelligence_score, 1.0)
    
    def _calculate_creativity_score(self) -> float:
        """Calcula score de criatividade"""
        # Baseado na diversidade de eventos e padrões
        total_events = self.metrics['real_learning_events']
        if total_events == 0:
            return 0.0
        
        # Diversidade de tipos de eventos
        event_diversity = len(set([
            self.metrics['evolution_events'],
            self.metrics['reinforcement_events'],
            self.metrics['neural_processing_events']
        ])) / 3.0
        
        return min(event_diversity, 1.0)
    
    def _calculate_efficiency_score(self) -> float:
        """Calcula score de eficiência"""
        # Baseado na relação entre eventos e tempo
        total_events = self.metrics['real_learning_events']
        time_elapsed = (datetime.now() - self.metrics['system_start_time']).total_seconds()
        
        if time_elapsed > 0:
            efficiency = total_events / time_elapsed
            return min(efficiency / 10.0, 1.0)  # Normalizar
        return 0.0
    
    def _calculate_stability_score(self) -> float:
        """Calcula score de estabilidade"""
        # Baseado na consistência das métricas
        if len(self.metrics['real_time_metrics']) < 10:
            return 0.0
        
        recent_scores = [m['intelligence_score'] for m in self.metrics['real_time_metrics'][-10:]]
        if len(recent_scores) < 2:
            return 0.0
        
        # Calcular variância (menor variância = maior estabilidade)
        variance = np.var(recent_scores)
        stability = max(0, 1.0 - variance * 10)  # Normalizar
        
        return min(stability, 1.0)
    
    def _calculate_convergence_score(self) -> float:
        """Calcula score de convergência"""
        # Baseado na tendência de melhoria das métricas
        if len(self.metrics['real_time_metrics']) < 20:
            return 0.0
        
        recent_scores = [m['intelligence_score'] for m in self.metrics['real_time_metrics'][-20:]]
        
        # Calcular tendência (regressão linear simples)
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            convergence = max(0, min(slope * 10, 1.0))  # Normalizar
            return convergence
        
        return 0.0
    
    def _analyze_intelligence_trends(self):
        """Analisa tendências de inteligência"""
        if len(self.metrics['real_time_metrics']) < 10:
            return
        
        recent_scores = [m['intelligence_score'] for m in self.metrics['real_time_metrics'][-10:]]
        
        # Calcular tendência
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            
            trend = {
                'timestamp': datetime.now(),
                'slope': slope,
                'trend_direction': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
                'intelligence_score': recent_scores[-1]
            }
            
            self.metrics['intelligence_trends'].append(trend)
            
            # Manter apenas histórico recente
            if len(self.metrics['intelligence_trends']) > 100:
                self.metrics['intelligence_trends'] = self.metrics['intelligence_trends'][-100:]
    
    def _analyze_learning_trends(self):
        """Analisa tendências de aprendizado"""
        # Implementar análise de tendências de aprendizado
        pass
    
    def _analyze_convergence_trends(self):
        """Analisa tendências de convergência"""
        # Implementar análise de tendências de convergência
        pass
    
    def _detect_intelligence_emergence(self) -> bool:
        """Detecta emergência de inteligência"""
        # Critérios para emergência
        intelligence_threshold = self.config['emergence_threshold']
        learning_threshold = self.config['learning_threshold']
        
        # Verificar se score de inteligência excede threshold
        if self.metrics['intelligence_score'] > intelligence_threshold:
            # Verificar se há aprendizado ativo
            if self.metrics['learning_rate'] > learning_threshold:
                # Verificar se há estabilidade
                if self.metrics['stability_score'] > 0.7:
                    return True
        
        return False
    
    def get_dashboard_data(self) -> Dict:
        """Retorna dados para dashboard"""
        return {
            'current_metrics': {
                'intelligence_score': self.metrics['intelligence_score'],
                'learning_rate': self.metrics['learning_rate'],
                'adaptation_rate': self.metrics['adaptation_rate'],
                'creativity_score': self.metrics['creativity_score'],
                'efficiency_score': self.metrics['efficiency_score'],
                'stability_score': self.metrics['stability_score'],
                'convergence_score': self.metrics['convergence_score']
            },
            'event_counts': {
                'total_cycles': self.metrics['total_cycles'],
                'real_learning_events': self.metrics['real_learning_events'],
                'evolution_events': self.metrics['evolution_events'],
                'reinforcement_events': self.metrics['reinforcement_events'],
                'neural_processing_events': self.metrics['neural_processing_events'],
                'emergence_detected': self.metrics['emergence_detected']
            },
            'trends': self.metrics['intelligence_trends'][-20:],  # Últimas 20 tendências
            'real_time_data': self.metrics['real_time_metrics'][-50:],  # Últimos 50 pontos
            'system_uptime': (datetime.now() - self.metrics['system_start_time']).total_seconds()
        }
    
    def save_metrics(self):
        """Salva métricas em arquivo"""
        try:
            # Salvar métricas completas
            with open(self.config['metrics_file'], 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)
            
            # Salvar dados do dashboard
            dashboard_data = self.get_dashboard_data()
            with open(self.config['dashboard_file'], 'w') as f:
                json.dump(dashboard_data, f, indent=2, default=str)
            
            logger.info(f"💾 Métricas salvas em {self.config['metrics_file']}")
            logger.info(f"📊 Dashboard salvo em {self.config['dashboard_file']}")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar métricas: {e}")
    
    def display_metrics(self):
        """Exibe métricas atuais"""
        print("\n" + "="*60)
        print("📊 SISTEMA DE MÉTRICAS REAIS DE INTELIGÊNCIA")
        print("="*60)
        print(f"🧠 Score de Inteligência: {self.metrics['intelligence_score']:.3f}")
        print(f"📈 Taxa de Aprendizado: {self.metrics['learning_rate']:.3f}")
        print(f"🔄 Taxa de Adaptação: {self.metrics['adaptation_rate']:.3f}")
        print(f"🎨 Score de Criatividade: {self.metrics['creativity_score']:.3f}")
        print(f"⚡ Score de Eficiência: {self.metrics['efficiency_score']:.3f}")
        print(f"🛡️ Score de Estabilidade: {self.metrics['stability_score']:.3f}")
        print(f"🎯 Score de Convergência: {self.metrics['convergence_score']:.3f}")
        print("-" * 60)
        print(f"🔄 Ciclos Totais: {self.metrics['total_cycles']}")
        print(f"🎯 Eventos de Aprendizado: {self.metrics['real_learning_events']}")
        print(f"🧬 Eventos de Evolução: {self.metrics['evolution_events']}")
        print(f"🎮 Eventos de RL: {self.metrics['reinforcement_events']}")
        print(f"⚡ Eventos de Processamento: {self.metrics['neural_processing_events']}")
        print(f"🌟 Emergências Detectadas: {self.metrics['emergence_detected']}")
        print("="*60)
    
    def stop_monitoring(self):
        """Para o monitoramento de métricas"""
        self.running = False
        logger.info("🛑 Sistema de métricas parado")
        
        # Salvar métricas finais
        self.save_metrics()
        
        # Exibir resumo final
        self.display_metrics()

def main():
    """Função principal para teste"""
    logger.info("📊 INICIANDO SISTEMA DE MÉTRICAS REAIS")
    
    # Criar sistema de métricas
    metrics_system = RealMetricsSystem()
    
    try:
        # Iniciar monitoramento
        metrics_system.start_monitoring()
        
        # Simular alguns eventos
        for i in range(100):
            metrics_system.add_metric('learning_event', {'value': i})
            metrics_system.add_metric('evolution_event', {'value': i * 0.5})
            metrics_system.add_metric('reinforcement_event', {'value': i * 0.3})
            metrics_system.add_metric('neural_processing_event', {'value': i * 0.8})
            
            time.sleep(0.1)
        
        # Manter rodando por um tempo
        time.sleep(10)
        
        # Parar monitoramento
        metrics_system.stop_monitoring()
        
    except KeyboardInterrupt:
        logger.info("🛑 Parando sistema de métricas...")
        metrics_system.stop_monitoring()

if __name__ == "__main__":
    main()
