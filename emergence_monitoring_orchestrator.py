#!/usr/bin/env python3
"""
Emergence Monitoring Orchestrator
Orquestrador que integra monitoramento contínuo, análise de comportamento e otimização dinâmica
"""

import time
import json
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from collections import deque, defaultdict
import threading
import logging
import subprocess
import psutil
from typing import Dict, List, Tuple, Optional, Any
import signal
import sys
import os

# Importar sistemas
from continuous_emergence_monitor import ContinuousEmergenceMonitor
from behavior_analysis_system import SystemMonitor
from dynamic_optimization_system import DynamicOptimizationSystem

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/emergence_monitoring_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class EmergenceMonitoringOrchestrator:
    """Orquestrador de monitoramento de emergência"""
    
    def __init__(self):
        # Inicializar sistemas
        self.continuous_monitor = ContinuousEmergenceMonitor()
        self.behavior_analyzer = SystemMonitor()
        self.dynamic_optimizer = DynamicOptimizationSystem()
        
        # Estado do orquestrador
        self.orchestrator_active = False
        self.orchestrator_thread = None
        
        # Configurar banco de dados
        self.db_path = '/root/emergence_orchestrator.db'
        self._init_database()
        
        # Configurar sinais de sistema
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Cache de estados
        self.system_states = {}
        self.last_report_times = {}
        
        # Configurações
        self.report_interval = 300  # 5 minutos
        self.optimization_interval = 600  # 10 minutos
        self.analysis_interval = 180  # 3 minutos
        
    def _init_database(self):
        """Inicializa banco de dados"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS orchestrator_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                report_type TEXT,
                report_data TEXT,
                confidence REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS system_integration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                system_name TEXT,
                status TEXT,
                metrics TEXT,
                integration_score REAL
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergence_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_type TEXT,
                event_data TEXT,
                confidence REAL,
                systems_involved TEXT
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def _signal_handler(self, signum, frame):
        """Manipulador de sinais do sistema"""
        logger.info(f"🛑 Sinal {signum} recebido. Parando orquestrador...")
        self.stop_orchestrator()
        sys.exit(0)
    
    def start_orchestrator(self):
        """Inicia orquestrador"""
        if self.orchestrator_active:
            logger.warning("Orquestrador já está ativo")
            return
        
        try:
            # Iniciar sistemas
            logger.info("🚀 Iniciando sistemas de monitoramento...")
            
            self.continuous_monitor.start_monitoring()
            logger.info("✅ Monitor contínuo iniciado")
            
            self.behavior_analyzer.start_monitoring()
            logger.info("✅ Analisador de comportamento iniciado")
            
            self.dynamic_optimizer.start_optimization()
            logger.info("✅ Otimizador dinâmico iniciado")
            
            # Iniciar orquestrador
            self.orchestrator_active = True
            self.orchestrator_thread = threading.Thread(target=self._orchestrator_loop)
            self.orchestrator_thread.daemon = True
            self.orchestrator_thread.start()
            
            logger.info("🎯 Orquestrador de emergência iniciado com sucesso")
            
        except Exception as e:
            logger.error(f"❌ Erro ao iniciar orquestrador: {e}")
            self.stop_orchestrator()
            raise
    
    def stop_orchestrator(self):
        """Para orquestrador"""
        if not self.orchestrator_active:
            return
        
        logger.info("⏹️ Parando orquestrador...")
        
        # Parar orquestrador
        self.orchestrator_active = False
        if self.orchestrator_thread:
            self.orchestrator_thread.join(timeout=10)
        
        # Parar sistemas
        try:
            self.continuous_monitor.stop_monitoring()
            logger.info("✅ Monitor contínuo parado")
        except Exception as e:
            logger.error(f"❌ Erro ao parar monitor contínuo: {e}")
        
        try:
            self.behavior_analyzer.stop_monitoring()
            logger.info("✅ Analisador de comportamento parado")
        except Exception as e:
            logger.error(f"❌ Erro ao parar analisador de comportamento: {e}")
        
        try:
            self.dynamic_optimizer.stop_optimization()
            logger.info("✅ Otimizador dinâmico parado")
        except Exception as e:
            logger.error(f"❌ Erro ao parar otimizador dinâmico: {e}")
        
        logger.info("🏁 Orquestrador parado com sucesso")
    
    def _orchestrator_loop(self):
        """Loop principal do orquestrador"""
        while self.orchestrator_active:
            try:
                # Análise de integração
                self._perform_integration_analysis()
                
                # Detecção de eventos de emergência
                self._detect_emergence_events()
                
                # Otimização coordenada
                self._perform_coordinated_optimization()
                
                # Relatório geral
                self._generate_general_report()
                
                # Aguardar próxima iteração
                time.sleep(60)  # Verificar a cada minuto
                
            except Exception as e:
                logger.error(f"Erro no loop do orquestrador: {e}")
                time.sleep(120)  # Aguardar mais tempo em caso de erro
    
    def _perform_integration_analysis(self):
        """Realiza análise de integração entre sistemas"""
        try:
            # Obter relatórios dos sistemas
            continuous_report = self.continuous_monitor.get_monitoring_report()
            behavior_report = self.behavior_analyzer.get_behavioral_report()
            optimization_report = self.dynamic_optimizer.get_optimization_report()
            
            # Calcular score de integração
            integration_score = self._calculate_integration_score(
                continuous_report, behavior_report, optimization_report
            )
            
            # Analisar correlações
            correlations = self._analyze_system_correlations(
                continuous_report, behavior_report, optimization_report
            )
            
            # Detectar problemas de integração
            integration_issues = self._detect_integration_issues(
                continuous_report, behavior_report, optimization_report
            )
            
            # Salvar análise
            self._save_integration_analysis(
                integration_score, correlations, integration_issues
            )
            
            # Log da análise
            logger.info(f"🔗 Score de integração: {integration_score:.3f}")
            logger.info(f"📊 Correlações detectadas: {len(correlations)}")
            logger.info(f"⚠️ Problemas de integração: {len(integration_issues)}")
            
        except Exception as e:
            logger.error(f"Erro na análise de integração: {e}")
    
    def _calculate_integration_score(self, continuous_report: Dict, behavior_report: Dict, optimization_report: Dict) -> float:
        """Calcula score de integração entre sistemas"""
        score = 0.0
        
        # Score baseado na atividade dos sistemas
        continuous_active = continuous_report.get('monitoring_active', False)
        behavior_active = behavior_report.get('monitoring_active', False)
        optimization_active = optimization_report.get('optimization_active', False)
        
        active_systems = sum([continuous_active, behavior_active, optimization_active])
        score += (active_systems / 3.0) * 0.3
        
        # Score baseado na qualidade dos dados
        continuous_confidence = continuous_report.get('confidence', 0.0)
        behavior_confidence = behavior_report.get('confidence', 0.0)
        optimization_confidence = optimization_report.get('confidence', 0.0)
        
        avg_confidence = (continuous_confidence + behavior_confidence + optimization_confidence) / 3.0
        score += avg_confidence * 0.4
        
        # Score baseado na consistência temporal
        current_time = time.time()
        time_consistency = 1.0
        
        for report in [continuous_report, behavior_report, optimization_report]:
            report_time = report.get('timestamp', current_time)
            time_diff = abs(current_time - report_time)
            if time_diff > 300:  # Mais de 5 minutos
                time_consistency *= 0.8
        
        score += time_consistency * 0.3
        
        return min(score, 1.0)
    
    def _analyze_system_correlations(self, continuous_report: Dict, behavior_report: Dict, optimization_report: Dict) -> List[Dict]:
        """Analisa correlações entre sistemas"""
        correlations = []
        
        # Correlação entre emergência e comportamento
        if 'avg_emergence_level' in continuous_report and 'unexpected_actions' in behavior_report:
            emergence_level = continuous_report['avg_emergence_level']
            unexpected_actions = behavior_report['unexpected_actions']
            
            if emergence_level > 0.5 and unexpected_actions > 0:
                correlations.append({
                    'type': 'emergence_behavior',
                    'strength': min(emergence_level * (unexpected_actions / 10), 1.0),
                    'description': 'Alta emergência correlacionada com ações inesperadas'
                })
        
        # Correlação entre comportamento e otimização
        if 'adaptation_events' in behavior_report and 'total_optimizations' in optimization_report:
            adaptations = behavior_report['adaptation_events']
            optimizations = optimization_report['total_optimizations']
            
            if adaptations > 0 and optimizations > 0:
                correlations.append({
                    'type': 'behavior_optimization',
                    'strength': min((adaptations + optimizations) / 20, 1.0),
                    'description': 'Adaptações correlacionadas com otimizações'
                })
        
        # Correlação entre emergência e otimização
        if 'avg_emergence_level' in continuous_report and 'avg_expected_impact' in optimization_report:
            emergence_level = continuous_report['avg_emergence_level']
            expected_impact = optimization_report['avg_expected_impact']
            
            if emergence_level > 0.6 and expected_impact > 0.5:
                correlations.append({
                    'type': 'emergence_optimization',
                    'strength': min((emergence_level + expected_impact) / 2, 1.0),
                    'description': 'Alta emergência correlacionada com alto impacto de otimização'
                })
        
        return correlations
    
    def _detect_integration_issues(self, continuous_report: Dict, behavior_report: Dict, optimization_report: Dict) -> List[Dict]:
        """Detecta problemas de integração"""
        issues = []
        
        # Verificar sistemas inativos
        if not continuous_report.get('monitoring_active', False):
            issues.append({
                'type': 'inactive_system',
                'system': 'continuous_monitor',
                'severity': 'high',
                'description': 'Monitor contínuo inativo'
            })
        
        if not behavior_report.get('monitoring_active', False):
            issues.append({
                'type': 'inactive_system',
                'system': 'behavior_analyzer',
                'severity': 'high',
                'description': 'Analisador de comportamento inativo'
            })
        
        if not optimization_report.get('optimization_active', False):
            issues.append({
                'type': 'inactive_system',
                'system': 'dynamic_optimizer',
                'severity': 'medium',
                'description': 'Otimizador dinâmico inativo'
            })
        
        # Verificar baixa confiança
        if continuous_report.get('confidence', 0) < 0.5:
            issues.append({
                'type': 'low_confidence',
                'system': 'continuous_monitor',
                'severity': 'medium',
                'description': 'Baixa confiança no monitor contínuo'
            })
        
        if behavior_report.get('confidence', 0) < 0.5:
            issues.append({
                'type': 'low_confidence',
                'system': 'behavior_analyzer',
                'severity': 'medium',
                'description': 'Baixa confiança no analisador de comportamento'
            })
        
        if optimization_report.get('confidence', 0) < 0.5:
            issues.append({
                'type': 'low_confidence',
                'system': 'dynamic_optimizer',
                'severity': 'medium',
                'description': 'Baixa confiança no otimizador dinâmico'
            })
        
        # Verificar inconsistências temporais
        current_time = time.time()
        for report_name, report in [('continuous', continuous_report), ('behavior', behavior_report), ('optimization', optimization_report)]:
            report_time = report.get('timestamp', current_time)
            time_diff = abs(current_time - report_time)
            if time_diff > 600:  # Mais de 10 minutos
                issues.append({
                    'type': 'temporal_inconsistency',
                    'system': report_name,
                    'severity': 'low',
                    'description': f'Relatório desatualizado ({time_diff:.0f}s)'
                })
        
        return issues
    
    def _save_integration_analysis(self, integration_score: float, correlations: List[Dict], issues: List[Dict]):
        """Salva análise de integração"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO system_integration
            (timestamp, system_name, status, metrics, integration_score)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            time.time(),
            'orchestrator',
            'active',
            json.dumps({
                'correlations': correlations,
                'issues': issues,
                'correlation_count': len(correlations),
                'issue_count': len(issues)
            }),
            integration_score
        ))
        
        conn.commit()
        conn.close()
    
    def _detect_emergence_events(self):
        """Detecta eventos de emergência"""
        try:
            # Obter relatórios dos sistemas
            continuous_report = self.continuous_monitor.get_monitoring_report()
            behavior_report = self.behavior_analyzer.get_behavioral_report()
            optimization_report = self.dynamic_optimizer.get_optimization_report()
            
            # Detectar eventos de emergência
            emergence_events = []
            
            # Evento 1: Alta emergência
            if continuous_report.get('avg_emergence_level', 0) > 0.8:
                emergence_events.append({
                    'type': 'high_emergence',
                    'confidence': continuous_report.get('avg_emergence_level', 0),
                    'description': 'Nível de emergência muito alto detectado',
                    'systems': ['continuous_monitor']
                })
            
            # Evento 2: Múltiplas ações inesperadas
            if behavior_report.get('unexpected_actions', 0) > 5:
                emergence_events.append({
                    'type': 'multiple_unexpected_actions',
                    'confidence': min(behavior_report.get('unexpected_actions', 0) / 10, 1.0),
                    'description': 'Múltiplas ações inesperadas detectadas',
                    'systems': ['behavior_analyzer']
                })
            
            # Evento 3: Alta adaptação
            if behavior_report.get('adaptation_events', 0) > 3:
                emergence_events.append({
                    'type': 'high_adaptation',
                    'confidence': min(behavior_report.get('adaptation_events', 0) / 5, 1.0),
                    'description': 'Alto nível de adaptação detectado',
                    'systems': ['behavior_analyzer']
                })
            
            # Evento 4: Otimização intensa
            if optimization_report.get('total_optimizations', 0) > 10:
                emergence_events.append({
                    'type': 'intense_optimization',
                    'confidence': min(optimization_report.get('total_optimizations', 0) / 20, 1.0),
                    'description': 'Otimização intensa detectada',
                    'systems': ['dynamic_optimizer']
                })
            
            # Evento 5: Emergência coordenada
            if (continuous_report.get('avg_emergence_level', 0) > 0.6 and 
                behavior_report.get('unexpected_actions', 0) > 2 and 
                optimization_report.get('total_optimizations', 0) > 5):
                emergence_events.append({
                    'type': 'coordinated_emergence',
                    'confidence': 0.9,
                    'description': 'Emergência coordenada entre múltiplos sistemas',
                    'systems': ['continuous_monitor', 'behavior_analyzer', 'dynamic_optimizer']
                })
            
            # Salvar eventos
            for event in emergence_events:
                self._save_emergence_event(event)
            
            # Log dos eventos
            if emergence_events:
                logger.info(f"🌟 {len(emergence_events)} eventos de emergência detectados")
                for event in emergence_events:
                    logger.info(f"   📍 {event['type']}: {event['description']} (confiança: {event['confidence']:.3f})")
            
        except Exception as e:
            logger.error(f"Erro na detecção de eventos de emergência: {e}")
    
    def _save_emergence_event(self, event: Dict):
        """Salva evento de emergência"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO emergence_events
            (timestamp, event_type, event_data, confidence, systems_involved)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            time.time(),
            event['type'],
            json.dumps(event),
            event['confidence'],
            json.dumps(event['systems'])
        ))
        
        conn.commit()
        conn.close()
    
    def _perform_coordinated_optimization(self):
        """Realiza otimização coordenada"""
        try:
            # Obter relatórios dos sistemas
            continuous_report = self.continuous_monitor.get_monitoring_report()
            behavior_report = self.behavior_analyzer.get_behavioral_report()
            optimization_report = self.dynamic_optimizer.get_optimization_report()
            
            # Determinar estratégia de otimização coordenada
            strategy = self._determine_coordinated_strategy(
                continuous_report, behavior_report, optimization_report
            )
            
            # Aplicar otimizações coordenadas
            if strategy['should_optimize']:
                self._apply_coordinated_optimizations(strategy)
                logger.info(f"🔧 Otimização coordenada aplicada: {strategy['strategy']}")
            
        except Exception as e:
            logger.error(f"Erro na otimização coordenada: {e}")
    
    def _determine_coordinated_strategy(self, continuous_report: Dict, behavior_report: Dict, optimization_report: Dict) -> Dict:
        """Determina estratégia de otimização coordenada"""
        strategy = {
            'should_optimize': False,
            'strategy': 'none',
            'reason': 'no_optimization_needed'
        }
        
        # Estratégia baseada na emergência
        emergence_level = continuous_report.get('avg_emergence_level', 0)
        if emergence_level > 0.8:
            strategy = {
                'should_optimize': True,
                'strategy': 'maximize_emergence',
                'reason': 'high_emergence_level'
            }
        elif emergence_level > 0.6:
            strategy = {
                'should_optimize': True,
                'strategy': 'amplify_signals',
                'reason': 'medium_emergence_level'
            }
        
        # Estratégia baseada no comportamento
        unexpected_actions = behavior_report.get('unexpected_actions', 0)
        if unexpected_actions > 5:
            strategy = {
                'should_optimize': True,
                'strategy': 'stabilize_behavior',
                'reason': 'high_unexpected_actions'
            }
        
        # Estratégia baseada na otimização
        total_optimizations = optimization_report.get('total_optimizations', 0)
        if total_optimizations > 20:
            strategy = {
                'should_optimize': True,
                'strategy': 'reduce_optimization',
                'reason': 'excessive_optimizations'
            }
        
        return strategy
    
    def _apply_coordinated_optimizations(self, strategy: Dict):
        """Aplica otimizações coordenadas"""
        if strategy['strategy'] == 'maximize_emergence':
            # Maximizar emergência
            logger.info("🚀 Maximizando emergência...")
            # Aqui seria implementada a lógica para maximizar emergência
        
        elif strategy['strategy'] == 'amplify_signals':
            # Amplificar sinais
            logger.info("📈 Amplificando sinais...")
            # Aqui seria implementada a lógica para amplificar sinais
        
        elif strategy['strategy'] == 'stabilize_behavior':
            # Estabilizar comportamento
            logger.info("🛡️ Estabilizando comportamento...")
            # Aqui seria implementada a lógica para estabilizar comportamento
        
        elif strategy['strategy'] == 'reduce_optimization':
            # Reduzir otimização
            logger.info("⏸️ Reduzindo otimização...")
            # Aqui seria implementada a lógica para reduzir otimização
    
    def _generate_general_report(self):
        """Gera relatório geral"""
        try:
            # Obter relatórios dos sistemas
            continuous_report = self.continuous_monitor.get_monitoring_report()
            behavior_report = self.behavior_analyzer.get_behavioral_report()
            optimization_report = self.dynamic_optimizer.get_optimization_report()
            
            # Calcular métricas gerais
            general_metrics = self._calculate_general_metrics(
                continuous_report, behavior_report, optimization_report
            )
            
            # Gerar relatório
            report = {
                'timestamp': time.time(),
                'orchestrator_active': self.orchestrator_active,
                'continuous_monitor': continuous_report,
                'behavior_analyzer': behavior_report,
                'dynamic_optimizer': optimization_report,
                'general_metrics': general_metrics,
                'confidence': 0.8
            }
            
            # Salvar relatório
            self._save_general_report(report)
            
            # Log do relatório
            logger.info(f"📊 Relatório geral gerado - Emergência: {general_metrics['emergence_level']:.3f}")
            logger.info(f"🔗 Integração: {general_metrics['integration_score']:.3f}")
            logger.info(f"🎯 Eventos: {general_metrics['event_count']}")
            
        except Exception as e:
            logger.error(f"Erro na geração do relatório geral: {e}")
    
    def _calculate_general_metrics(self, continuous_report: Dict, behavior_report: Dict, optimization_report: Dict) -> Dict:
        """Calcula métricas gerais"""
        metrics = {}
        
        # Nível de emergência geral
        emergence_level = continuous_report.get('avg_emergence_level', 0)
        metrics['emergence_level'] = emergence_level
        
        # Score de integração
        integration_score = self._calculate_integration_score(
            continuous_report, behavior_report, optimization_report
        )
        metrics['integration_score'] = integration_score
        
        # Contagem de eventos
        event_count = (
            behavior_report.get('unexpected_actions', 0) +
            behavior_report.get('adaptation_events', 0) +
            optimization_report.get('total_optimizations', 0)
        )
        metrics['event_count'] = event_count
        
        # Status geral
        all_active = (
            continuous_report.get('monitoring_active', False) and
            behavior_report.get('monitoring_active', False) and
            optimization_report.get('optimization_active', False)
        )
        metrics['all_systems_active'] = all_active
        
        # Confiança geral
        avg_confidence = (
            continuous_report.get('confidence', 0) +
            behavior_report.get('confidence', 0) +
            optimization_report.get('confidence', 0)
        ) / 3.0
        metrics['avg_confidence'] = avg_confidence
        
        return metrics
    
    def _save_general_report(self, report: Dict):
        """Salva relatório geral"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO orchestrator_reports
            (timestamp, report_type, report_data, confidence)
            VALUES (?, ?, ?, ?)
        ''', (
            report['timestamp'],
            'general_report',
            json.dumps(report),
            report['confidence']
        ))
        
        conn.commit()
        conn.close()
    
    def get_orchestrator_report(self) -> Dict:
        """Gera relatório do orquestrador"""
        try:
            # Obter relatórios dos sistemas
            continuous_report = self.continuous_monitor.get_monitoring_report()
            behavior_report = self.behavior_analyzer.get_behavioral_report()
            optimization_report = self.dynamic_optimizer.get_optimization_report()
            
            # Calcular métricas gerais
            general_metrics = self._calculate_general_metrics(
                continuous_report, behavior_report, optimization_report
            )
            
            return {
                'report': 'orchestrator_analysis',
                'timestamp': time.time(),
                'orchestrator_active': self.orchestrator_active,
                'continuous_monitor': continuous_report,
                'behavior_analyzer': behavior_report,
                'dynamic_optimizer': optimization_report,
                'general_metrics': general_metrics,
                'confidence': 0.8
            }
            
        except Exception as e:
            logger.error(f"Erro ao gerar relatório do orquestrador: {e}")
            return {
                'report': 'orchestrator_error',
                'timestamp': time.time(),
                'error': str(e),
                'confidence': 0.0
            }

def main():
    """Função principal"""
    orchestrator = EmergenceMonitoringOrchestrator()
    
    try:
        # Iniciar orquestrador
        orchestrator.start_orchestrator()
        
        # Manter rodando
        while True:
            time.sleep(300)  # 5 minutos
            
            # Gerar relatório
            report = orchestrator.get_orchestrator_report()
            logger.info(f"📊 Relatório do orquestrador: {report}")
            
    except KeyboardInterrupt:
        logger.info("🛑 Interrompendo orquestrador...")
        orchestrator.stop_orchestrator()
    except Exception as e:
        logger.error(f"❌ Erro crítico: {e}")
        orchestrator.stop_orchestrator()

if __name__ == "__main__":
    main()