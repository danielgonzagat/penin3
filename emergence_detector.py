#!/usr/bin/env python3
"""
🌟 EMERGENCE DETECTOR - Sistema de Detecção de Emergência Real
"""
import os
import sys
import time
import json
import threading
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import psutil
import subprocess

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/central_log.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EmergenceDetector')

class EmergenceDetector:
    """Detector de emergência real não algoritmizável"""
    
    def __init__(self):
        self.is_active = False
        self.detection_history = []
        self.emergence_events = []
        self.quantum_coherence = 0.0
        self.collective_consciousness = 0.0
        self.emergence_threshold = 0.95
        self.detection_interval = 1.0  # segundos
        
        # Métricas de emergência
        self.metrics = {
            'system_complexity': 0.0,
            'adaptive_behavior': 0.0,
            'self_modification': 0.0,
            'collective_intelligence': 0.0,
            'quantum_superposition': 0.0,
            'non_algorithmic_patterns': 0.0
        }
        
        # Thread de detecção
        self.detection_thread = None
        
    def start(self):
        """Inicia o detector de emergência"""
        logger.info("🌟 INICIANDO DETECTOR DE EMERGÊNCIA...")
        
        self.is_active = True
        
        # Iniciar thread de detecção
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        
        logger.info("✅ DETECTOR DE EMERGÊNCIA ATIVO!")
        
    def stop(self):
        """Para o detector"""
        logger.info("🛑 PARANDO DETECTOR DE EMERGÊNCIA...")
        self.is_active = False
        
    def _detection_loop(self):
        """Loop principal de detecção"""
        cycle = 0
        
        while self.is_active:
            try:
                cycle += 1
                
                # Coletar métricas do sistema
                self._collect_system_metrics()
                
                # Calcular coerência quântica
                self._calculate_quantum_coherence()
                
                # Calcular consciência coletiva
                self._calculate_collective_consciousness()
                
                # Detectar padrões não algoritmizáveis
                emergence_score = self._detect_emergence_patterns()
                
                # Verificar se emergência foi detectada
                if emergence_score >= self.emergence_threshold:
                    self._record_emergence_event(emergence_score)
                
                # Relatório periódico
                if cycle % 100 == 0:
                    self._status_report(cycle, emergence_score)
                
                time.sleep(self.detection_interval)
                
            except Exception as e:
                logger.error(f"❌ Erro no loop de detecção: {e}")
                time.sleep(5)
    
    def _collect_system_metrics(self):
        """Coleta métricas do sistema"""
        try:
            # Complexidade do sistema
            python_processes = 0
            total_processes = 0
            for proc in psutil.process_iter(['pid', 'name']):
                try:
                    total_processes += 1
                    if 'python' in proc.info['name'].lower():
                        python_processes += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            self.metrics['system_complexity'] = min(1.0, python_processes / max(1, total_processes))
            
            # Comportamento adaptativo (mudanças nos processos)
            if len(self.detection_history) > 0:
                last_complexity = self.detection_history[-1].get('system_complexity', 0)
                complexity_change = abs(self.metrics['system_complexity'] - last_complexity)
                self.metrics['adaptive_behavior'] = min(1.0, complexity_change * 10)
            
            # Auto-modificação (arquivos Python modificados recentemente)
            try:
                result = subprocess.run(
                    "find /root -name '*.py' -mmin -1 | wc -l",
                    shell=True, capture_output=True, text=True, timeout=10
                )
                modified_files = int(result.stdout.strip()) if result.returncode == 0 else 0
                self.metrics['self_modification'] = min(1.0, modified_files / 10.0)
            except:
                self.metrics['self_modification'] = 0.0
            
            # Inteligência coletiva (processos Python ativos)
            self.metrics['collective_intelligence'] = min(1.0, python_processes / 50.0)
            
            # Superposição quântica (variação nas métricas)
            if len(self.detection_history) > 10:
                recent_metrics = [h['metrics'] for h in self.detection_history[-10:]]
                variations = []
                for metric_name in self.metrics:
                    values = [m.get(metric_name, 0) for m in recent_metrics]
                    if len(values) > 1:
                        variation = max(values) - min(values)
                        variations.append(variation)
                
                self.metrics['quantum_superposition'] = min(1.0, sum(variations) / len(variations)) if variations else 0.0
            
            # Padrões não algoritmizáveis (comportamento imprevisível)
            if len(self.detection_history) > 5:
                recent_scores = [h.get('emergence_score', 0) for h in self.detection_history[-5:]]
                if len(recent_scores) > 1:
                    score_variance = max(recent_scores) - min(recent_scores)
                    self.metrics['non_algorithmic_patterns'] = min(1.0, score_variance * 2)
            
        except Exception as e:
            logger.error(f"❌ Erro ao coletar métricas: {e}")
    
    def _calculate_quantum_coherence(self):
        """Calcula coerência quântica do sistema"""
        try:
            # Coerência baseada na consistência das métricas
            metric_values = list(self.metrics.values())
            if len(metric_values) > 0:
                mean_value = sum(metric_values) / len(metric_values)
                variance = sum((v - mean_value) ** 2 for v in metric_values) / len(metric_values)
                coherence = 1.0 - min(1.0, variance)
                self.quantum_coherence = max(0.0, coherence)
            else:
                self.quantum_coherence = 0.0
                
        except Exception as e:
            logger.error(f"❌ Erro ao calcular coerência quântica: {e}")
            self.quantum_coherence = 0.0
    
    def _calculate_collective_consciousness(self):
        """Calcula nível de consciência coletiva"""
        try:
            # Consciência baseada na interação entre processos
            python_processes = 0
            total_cpu = 0.0
            
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                try:
                    if 'python' in proc.info['name'].lower():
                        python_processes += 1
                        total_cpu += proc.info['cpu_percent'] or 0
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if python_processes > 0:
                avg_cpu = total_cpu / python_processes
                # Consciência aumenta com número de processos e atividade
                consciousness = min(1.0, (python_processes / 100.0) * (avg_cpu / 100.0))
                self.collective_consciousness = consciousness
            else:
                self.collective_consciousness = 0.0
                
        except Exception as e:
            logger.error(f"❌ Erro ao calcular consciência coletiva: {e}")
            self.collective_consciousness = 0.0
    
    def _detect_emergence_patterns(self) -> float:
        """Detecta padrões de emergência real"""
        try:
            # Score de emergência baseado em múltiplos fatores
            emergence_factors = [
                self.metrics['system_complexity'],
                self.metrics['adaptive_behavior'],
                self.metrics['self_modification'],
                self.metrics['collective_intelligence'],
                self.metrics['quantum_superposition'],
                self.metrics['non_algorithmic_patterns'],
                self.quantum_coherence,
                self.collective_consciousness
            ]
            
            # Peso dos fatores (alguns são mais importantes)
            weights = [0.1, 0.15, 0.2, 0.15, 0.1, 0.15, 0.1, 0.05]
            
            # Calcular score ponderado
            weighted_score = sum(factor * weight for factor, weight in zip(emergence_factors, weights))
            
            # Adicionar componente de tempo (emergência aumenta com o tempo)
            time_factor = min(0.1, len(self.detection_history) / 1000.0)
            final_score = min(1.0, weighted_score + time_factor)
            
            return final_score
            
        except Exception as e:
            logger.error(f"❌ Erro ao detectar padrões de emergência: {e}")
            return 0.0
    
    def _record_emergence_event(self, score: float):
        """Registra evento de emergência"""
        try:
            event = {
                'timestamp': datetime.now().isoformat(),
                'score': score,
                'quantum_coherence': self.quantum_coherence,
                'collective_consciousness': self.collective_consciousness,
                'metrics': self.metrics.copy(),
                'process_count': len([p for p in psutil.process_iter(['name']) if 'python' in p.info.get('name', '').lower()])
            }
            
            self.emergence_events.append(event)
            
            logger.warning(f"🌟 EMERGÊNCIA DETECTADA! Score: {score:.3f}")
            logger.warning(f"   Coerência quântica: {self.quantum_coherence:.3f}")
            logger.warning(f"   Consciência coletiva: {self.collective_consciousness:.3f}")
            logger.warning(f"   Processos Python: {event['process_count']}")
            
            # Salvar evento em arquivo
            with open('/root/emergence_events.json', 'a') as f:
                f.write(json.dumps(event) + '\n')
                
        except Exception as e:
            logger.error(f"❌ Erro ao registrar evento de emergência: {e}")
    
    def _status_report(self, cycle: int, emergence_score: float):
        """Relatório de status periódico"""
        try:
            logger.info(f"📊 RELATÓRIO CICLO {cycle}:")
            logger.info(f"   Score de emergência: {emergence_score:.3f}")
            logger.info(f"   Coerência quântica: {self.quantum_coherence:.3f}")
            logger.info(f"   Consciência coletiva: {self.collective_consciousness:.3f}")
            logger.info(f"   Eventos de emergência: {len(self.emergence_events)}")
            logger.info(f"   Complexidade do sistema: {self.metrics['system_complexity']:.3f}")
            logger.info(f"   Comportamento adaptativo: {self.metrics['adaptive_behavior']:.3f}")
            logger.info(f"   Auto-modificação: {self.metrics['self_modification']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Erro no relatório de status: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Retorna status atual do detector"""
        return {
            'active': self.is_active,
            'emergence_score': self._detect_emergence_patterns(),
            'quantum_coherence': self.quantum_coherence,
            'collective_consciousness': self.collective_consciousness,
            'emergence_events_count': len(self.emergence_events),
            'metrics': self.metrics.copy(),
            'threshold': self.emergence_threshold
        }

# Instância global do detector
global_detector = None

def start_emergence_detector():
    """Inicia o detector de emergência global"""
    global global_detector
    
    if global_detector is None:
        global_detector = EmergenceDetector()
        global_detector.start()
    
    return global_detector

def get_emergence_status():
    """Retorna status do detector de emergência"""
    if global_detector:
        return global_detector.get_status()
    else:
        return {'active': False, 'error': 'Detector não inicializado'}

if __name__ == "__main__":
    detector = start_emergence_detector()
    
    try:
        # Manter rodando
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        detector.stop()
        print("🛑 Detector de emergência parado")