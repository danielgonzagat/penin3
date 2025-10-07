#!/usr/bin/env python3
"""
🔍 SELF-ANALYSIS MODULE - FASE 2
Auto-diagnóstico e análise de bottlenecks
"""

import time
import torch
from collections import deque, defaultdict
from datetime import datetime
from typing import Dict, List, Any, Optional
from brain_logger import brain_logger

class PerformanceProfiler:
    """
    Profila performance de cada componente do sistema
    """
    def __init__(self, history_size: int = 100):
        self.timings = defaultdict(lambda: deque(maxlen=history_size))
        self.counts = defaultdict(int)
        self.active_timers = {}
    
    def start(self, component: str):
        """Inicia timer para componente"""
        self.active_timers[component] = time.time()
    
    def stop(self, component: str):
        """Para timer e registra"""
        if component in self.active_timers:
            elapsed = time.time() - self.active_timers[component]
            self.timings[component].append(elapsed)
            self.counts[component] += 1
            del self.active_timers[component]
            return elapsed
        return None
    
    def get_stats(self, component: str) -> Dict[str, float]:
        """Estatísticas de um componente"""
        if component not in self.timings or len(self.timings[component]) == 0:
            return {}
        
        times = list(self.timings[component])
        return {
            'avg': sum(times) / len(times),
            'min': min(times),
            'max': max(times),
            'last': times[-1],
            'count': self.counts[component],
            'p50': sorted(times)[len(times)//2],
            'p95': sorted(times)[int(len(times)*0.95)],
            'p99': sorted(times)[int(len(times)*0.99)]
        }
    
    def get_all_stats(self) -> Dict[str, Dict[str, float]]:
        """Estatísticas de todos componentes"""
        return {comp: self.get_stats(comp) for comp in self.timings.keys()}


class BottleneckDetector:
    """
    Detecta bottlenecks no sistema
    """
    def __init__(self, profiler: PerformanceProfiler):
        self.profiler = profiler
        self.bottlenecks = []
    
    def analyze(self) -> List[Dict[str, Any]]:
        """
        Analisa e identifica bottlenecks
        
        Returns:
            Lista de bottlenecks detectados
        """
        all_stats = self.profiler.get_all_stats()
        bottlenecks = []
        
        # Calcula tempo total
        total_time = sum(s['avg'] for s in all_stats.values())
        
        for component, stats in all_stats.items():
            if len(stats) == 0:
                continue
            
            # Bottleneck se componente consome > 30% do tempo
            percentage = (stats['avg'] / total_time * 100) if total_time > 0 else 0
            
            if percentage > 30:
                bottlenecks.append({
                    'component': component,
                    'avg_time': stats['avg'],
                    'percentage': percentage,
                    'severity': 'HIGH' if percentage > 50 else 'MEDIUM',
                    'p99': stats['p99'],
                    'recommendation': self._get_recommendation(component, stats)
                })
            
            # Bottleneck se variância muito alta (p99 > 3x avg)
            elif stats['p99'] > 3 * stats['avg']:
                bottlenecks.append({
                    'component': component,
                    'avg_time': stats['avg'],
                    'p99': stats['p99'],
                    'variance_ratio': stats['p99'] / stats['avg'],
                    'severity': 'MEDIUM',
                    'recommendation': f"Alta variância em {component}. Investigar casos extremos."
                })
        
        self.bottlenecks = sorted(bottlenecks, key=lambda x: x.get('percentage', 0), reverse=True)
        return self.bottlenecks
    
    def _get_recommendation(self, component: str, stats: Dict[str, float]) -> str:
        """Gera recomendação baseada no componente"""
        recommendations = {
            'forward': "Otimizar forward pass: batch operations, reduce model size",
            'backward': "Otimizar backward: gradient accumulation, mixed precision",
            'router': "Otimizar router: cache routing decisions, reduce top_k",
            'neuron_forward': "Otimizar neurons: simplify adapters, parallelize",
            'adapter_forward': "Simplificar adapters: reduce layers, use Linear",
            'training': "Otimizar training: increase batch size, optimize loss computation",
            'checkpoint': "Otimizar checkpoint: reduce frequency, async saving",
            'curiosity': "Otimizar curiosity: reduce model complexity, batch predictions"
        }
        
        return recommendations.get(component, f"Otimizar {component}: profiling detalhado necessário")


class SelfDiagnostics:
    """
    Sistema de auto-diagnóstico
    """
    def __init__(self, profiler: PerformanceProfiler, bottleneck_detector: BottleneckDetector):
        self.profiler = profiler
        self.bottleneck_detector = bottleneck_detector
        self.diagnostics_history = deque(maxlen=50)
    
    def run_diagnostics(self, brain_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executa diagnóstico completo
        
        Args:
            brain_state: Estado atual do brain
            
        Returns:
            Relatório de diagnóstico
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'performance': self._diagnose_performance(),
            'health': self._diagnose_health(brain_state),
            'bottlenecks': self.bottleneck_detector.analyze(),
            'recommendations': []
        }
        
        # Gera recomendações
        report['recommendations'] = self._generate_recommendations(report)
        
        # Salva histórico
        self.diagnostics_history.append(report)
        
        return report
    
    def _diagnose_performance(self) -> Dict[str, Any]:
        """
        Diagnóstico de performance com target configurável
        ✅ CORREÇÃO #6: Target via env var
        """
        all_stats = self.profiler.get_all_stats()
        
        # step_time total
        step_time = all_stats.get('step_total', {}).get('avg', 0)
        
        # ✅ Target configurável via env var
        import os
        target_step_time = float(os.getenv('TARGET_STEP_TIME_S', '1.0'))
        
        return {
            'step_time': {
                'current': step_time,
                'target': target_step_time,  # ✅ Usa variável
                'status': 'OK' if step_time < target_step_time else 'SLOW',
                'deviation': ((step_time - target_step_time) / target_step_time * 100) 
                             if step_time > 0 else 0
            },
            'component_breakdown': {
                comp: stats['avg'] for comp, stats in all_stats.items()
            }
        }
    
    def _diagnose_health(self, brain_state: Dict[str, Any]) -> Dict[str, Any]:
        """Diagnóstico de saúde do brain"""
        health = {
            'status': 'HEALTHY',
            'issues': []
        }
        
        # Check 1: Neurônios ativos
        active_neurons = brain_state.get('active_neurons', 0)
        if active_neurons < 8:
            health['issues'].append({
                'type': 'LOW_NEURONS',
                'severity': 'MEDIUM',
                'message': f"Apenas {active_neurons} neurônios ativos (ideal: 16+)"
            })
        
        # Check 2: Gradientes
        gradients_applied = brain_state.get('gradients_applied', 0)
        if gradients_applied == 0:
            health['issues'].append({
                'type': 'NO_GRADIENTS',
                'severity': 'HIGH',
                'message': "Nenhum gradiente aplicado - aprendizado parado"
            })
        
        # Check 3: Reward trend
        rewards = brain_state.get('recent_rewards', [])
        if len(rewards) >= 10:
            recent = rewards[-5:]
            old = rewards[-10:-5]
            if sum(recent) < sum(old) * 0.8:
                health['issues'].append({
                    'type': 'REWARD_DEGRADATION',
                    'severity': 'HIGH',
                    'message': "Reward degradou > 20% - possível catastrophic forgetting"
                })
        
        # Check 4: Loss
        avg_loss = brain_state.get('avg_loss', 0)
        if avg_loss > 100:
            health['issues'].append({
                'type': 'HIGH_LOSS',
                'severity': 'MEDIUM',
                'message': f"Loss muito alta ({avg_loss:.1f}) - possível instabilidade"
            })
        
        # Atualiza status
        if any(i['severity'] == 'HIGH' for i in health['issues']):
            health['status'] = 'CRITICAL'
        elif any(i['severity'] == 'MEDIUM' for i in health['issues']):
            health['status'] = 'WARNING'
        
        return health
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gera recomendações baseadas no diagnóstico"""
        recommendations = []
        
        # Performance recommendations
        perf = report['performance']
        if perf['step_time']['status'] == 'SLOW':
            recommendations.append({
                'category': 'PERFORMANCE',
                'priority': 'HIGH',
                'action': 'Reduzir step_time',
                'details': f"Atual: {perf['step_time']['current']:.3f}s, Target: 1.0s",
                'suggestions': [
                    "Reduzir top_k no router",
                    "Simplificar adapters",
                    "Usar mixed precision (AMP)",
                    "Cachear routing decisions"
                ]
            })
        
        # Bottleneck recommendations
        for bottleneck in report['bottlenecks']:
            if bottleneck['severity'] == 'HIGH':
                recommendations.append({
                    'category': 'BOTTLENECK',
                    'priority': 'HIGH',
                    'action': f"Otimizar {bottleneck['component']}",
                    'details': bottleneck['recommendation']
                })
        
        # Health recommendations
        for issue in report['health']['issues']:
            if issue['severity'] == 'HIGH':
                recommendations.append({
                    'category': 'HEALTH',
                    'priority': 'HIGH',
                    'action': f"Resolver {issue['type']}",
                    'details': issue['message']
                })
        
        return recommendations


class SelfAnalysisModule:
    """
    Módulo completo de auto-análise
    Integra profiling, bottleneck detection, e diagnostics
    """
    def __init__(self):
        self.profiler = PerformanceProfiler(history_size=100)
        self.bottleneck_detector = BottleneckDetector(self.profiler)
        self.diagnostics = SelfDiagnostics(self.profiler, self.bottleneck_detector)
        
        self.last_analysis = None
        self.analysis_interval = 10  # A cada 10 episódios
    
    def profile_component(self, component: str):
        """Context manager para profiling"""
        class ProfileContext:
            def __init__(self, profiler, comp):
                self.profiler = profiler
                self.component = comp
            
            def __enter__(self):
                self.profiler.start(self.component)
                return self
            
            def __exit__(self, *args):
                self.profiler.stop(self.component)
        
        return ProfileContext(self.profiler, component)
    
    def analyze(self, brain_state: Dict[str, Any], force: bool = False) -> Optional[Dict[str, Any]]:
        """
        Executa análise completa
        
        Args:
            brain_state: Estado atual do brain
            force: Força análise mesmo fora do intervalo
            
        Returns:
            Relatório ou None se não for momento de analisar
        """
        episode = brain_state.get('episode', 0)
        
        if not force and episode % self.analysis_interval != 0:
            return None
        
        brain_logger.info("🔍 Executando self-analysis...")
        
        report = self.diagnostics.run_diagnostics(brain_state)
        self.last_analysis = report
        
        # Log principais descobertas
        if report['bottlenecks']:
            brain_logger.warning(f"🔍 Bottlenecks detectados: {len(report['bottlenecks'])}")
            for b in report['bottlenecks'][:3]:  # Top 3
                brain_logger.warning(
                    f"  - {b['component']}: {b['percentage']:.1f}% "
                    f"({b['avg_time']*1000:.1f}ms avg) [{b['severity']}]"
                )
        
        if report['health']['status'] != 'HEALTHY':
            brain_logger.error(f"🔍 Health: {report['health']['status']}")
            for issue in report['health']['issues']:
                brain_logger.error(f"  - [{issue['severity']}] {issue['message']}")
        
        if report['recommendations']:
            brain_logger.info(f"🔍 Recomendações: {len(report['recommendations'])}")
            for rec in report['recommendations'][:3]:  # Top 3
                brain_logger.info(f"  - [{rec['priority']}] {rec['action']}")
        
        return report
    
    def get_summary(self) -> Dict[str, Any]:
        """Retorna sumário do self-analysis"""
        if not self.last_analysis:
            return {'status': 'no_analysis_yet'}
        
        return {
            'timestamp': self.last_analysis['timestamp'],
            'step_time': self.last_analysis['performance']['step_time'],
            'health': self.last_analysis['health']['status'],
            'bottlenecks_count': len(self.last_analysis['bottlenecks']),
            'recommendations_count': len(self.last_analysis['recommendations']),
            'top_bottleneck': self.last_analysis['bottlenecks'][0]['component'] if self.last_analysis['bottlenecks'] else None
        }


if __name__ == "__main__":
    print("🔍 Self-Analysis Module - FASE 2")
    
    # Test
    analysis = SelfAnalysisModule()
    
    # Simula alguns timings
    for i in range(20):
        with analysis.profile_component('forward'):
            time.sleep(0.05)
        
        with analysis.profile_component('backward'):
            time.sleep(0.02)
        
        with analysis.profile_component('router'):
            time.sleep(0.03)
    
    # Analisa
    brain_state = {
        'episode': 10,
        'active_neurons': 16,
        'gradients_applied': 10,
        'recent_rewards': [10, 20, 15, 25, 30, 28, 32, 35, 33, 38],
        'avg_loss': 15.5
    }
    
    report = analysis.analyze(brain_state, force=True)
    
    print("\n📊 Relatório:")
    print(f"  Status: {report['health']['status']}")
    print(f"  Bottlenecks: {len(report['bottlenecks'])}")
    print(f"  Recommendations: {len(report['recommendations'])}")
    
    if report['bottlenecks']:
        print("\n🔴 Top Bottleneck:")
        b = report['bottlenecks'][0]
        print(f"  {b['component']}: {b['percentage']:.1f}% [{b['severity']}]")
        print(f"  {b['recommendation']}")
    
    print("\n✅ Self-Analysis OK")
