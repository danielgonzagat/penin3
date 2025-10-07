#!/usr/bin/env python3
"""
📊 METRICS DASHBOARD - FASE 2
Real-time metrics e visualização
"""

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from collections import deque

class MetricsDashboard:
    """
    Dashboard de métricas em tempo real
    Gera relatórios legíveis para monitoramento
    """
    def __init__(self, output_path: str = "/root/UNIFIED_BRAIN/dashboard.txt"):
        self.output_path = Path(output_path)
        self.metrics_history = deque(maxlen=1000)
        self.last_update = None
    
    def update(self, metrics: Dict[str, Any]):
        """Atualiza métricas"""
        metrics['timestamp'] = datetime.now().isoformat()
        self.metrics_history.append(metrics)
        self.last_update = datetime.now()
    
    def render(self) -> str:
        """Renderiza dashboard como string"""
        if not self.metrics_history:
            return "No metrics yet"
        
        latest = self.metrics_history[-1]
        
        lines = []
        lines.append("═" * 80)
        lines.append("📊 UNIFIED BRAIN - METRICS DASHBOARD")
        lines.append("═" * 80)
        lines.append("")
        
        # System Status
        lines.append("🎯 SYSTEM STATUS")
        lines.append("─" * 80)
        status = latest.get('system_status', {})
        lines.append(f"  Status: {status.get('health', 'UNKNOWN')}")
        lines.append(f"  Uptime: {status.get('uptime', 'N/A')}")
        lines.append(f"  Episode: {latest.get('episode', 0)}")
        lines.append(f"  Total Steps: {latest.get('total_steps', 0)}")
        lines.append("")
        
        # Performance
        lines.append("⚡ PERFORMANCE")
        lines.append("─" * 80)
        perf = latest.get('performance', {})
        step_time = perf.get('step_time', 0)
        target = 1.0
        status_icon = "✅" if step_time < target else "⚠️"
        lines.append(f"  {status_icon} Step Time: {step_time:.3f}s (target: <{target}s)")
        lines.append(f"  Episodes/hour: {(3600 / (step_time * 30)):.1f}" if step_time > 0 else "  Episodes/hour: N/A")
        lines.append(f"  Throughput: {(1/step_time):.2f} steps/s" if step_time > 0 else "  Throughput: N/A")
        lines.append("")
        
        # Learning
        lines.append("🧠 LEARNING")
        lines.append("─" * 80)
        learning = latest.get('learning', {})
        lines.append(f"  Current Reward: {learning.get('current_reward', 0):.1f}")
        lines.append(f"  Best Reward: {learning.get('best_reward', 0):.1f}")
        lines.append(f"  Avg (last 100): {learning.get('avg_reward_100', 0):.1f}")
        lines.append(f"  Loss: {learning.get('loss', 0):.4f}")
        lines.append(f"  Gradients Applied: {learning.get('gradients_applied', 0)}")
        lines.append("")
        
        # Auto-Evolution (FASE 1)
        lines.append("🔄 AUTO-EVOLUTION (FASE 1)")
        lines.append("─" * 80)
        auto_evo = latest.get('auto_evolution', {})
        meta = auto_evo.get('meta_controller', {})
        curiosity = auto_evo.get('curiosity', {})
        darwin = auto_evo.get('darwin', {})
        
        lines.append(f"  Meta-Controller:")
        lines.append(f"    Interventions: {meta.get('total_interventions', 0)}")
        if meta.get('last_intervention'):
            lines.append(f"    Last: {meta['last_intervention']['type']} @ {meta['last_intervention']['episode']}")
        
        lines.append(f"  Curiosity:")
        lines.append(f"    Predictions: {curiosity.get('total_predictions', 0)}")
        lines.append(f"    Avg Surprise: {curiosity.get('avg_surprise', 0):.4f}")
        
        lines.append(f"  Darwin:")
        lines.append(f"    Frozen Neurons: {darwin.get('frozen_count', 0)}")
        lines.append(f"    Generations: {darwin.get('max_generation', 0)}")
        lines.append("")
        
        # Self-Analysis (FASE 2)
        lines.append("🔍 SELF-ANALYSIS (FASE 2)")
        lines.append("─" * 80)
        analysis = latest.get('self_analysis', {})
        if analysis:
            lines.append(f"  Health: {analysis.get('health', 'N/A')}")
            lines.append(f"  Bottlenecks: {analysis.get('bottlenecks_count', 0)}")
            if analysis.get('top_bottleneck'):
                lines.append(f"  Top Bottleneck: {analysis['top_bottleneck']}")
            lines.append(f"  Recommendations: {analysis.get('recommendations_count', 0)}")
        else:
            lines.append("  (Not yet analyzed)")
        lines.append("")
        
        # Resources
        lines.append("💾 RESOURCES")
        lines.append("─" * 80)
        resources = latest.get('resources', {})
        lines.append(f"  Memory (RSS): {resources.get('memory_rss_mb', 0):.1f} MB")
        lines.append(f"  GPU Memory: {resources.get('gpu_memory_mb', 0):.1f} MB")
        lines.append(f"  Active Neurons: {resources.get('active_neurons', 0)}")
        lines.append(f"  Soup Size: {resources.get('soup_size', 0)}")
        lines.append("")
        
        # Bottlenecks (if any)
        bottlenecks = latest.get('bottlenecks', [])
        if bottlenecks:
            lines.append("🔴 ACTIVE BOTTLENECKS")
            lines.append("─" * 80)
            for b in bottlenecks[:5]:  # Top 5
                lines.append(f"  [{b.get('severity', 'N/A')}] {b.get('component', 'N/A')}: {b.get('percentage', 0):.1f}%")
                lines.append(f"     → {b.get('recommendation', 'N/A')[:60]}")
            lines.append("")
        
        # Recommendations
        recommendations = latest.get('recommendations', [])
        if recommendations:
            lines.append("💡 RECOMMENDATIONS")
            lines.append("─" * 80)
            for rec in recommendations[:5]:  # Top 5
                lines.append(f"  [{rec.get('priority', 'N/A')}] {rec.get('action', 'N/A')}")
                if rec.get('details'):
                    lines.append(f"     {rec['details'][:70]}")
            lines.append("")
        
        lines.append("═" * 80)
        lines.append(f"Last Update: {self.last_update.strftime('%Y-%m-%d %H:%M:%S') if self.last_update else 'N/A'}")
        lines.append("═" * 80)
        
        return "\n".join(lines)
    
    def save(self):
        """
        Salva dashboard em arquivo com validação completa
        ✅ CORREÇÃO #2: Cria diretório pai + valida escrita
        """
        content = self.render()
        
        # ✅ Criar diretório pai se não existir
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Escrever arquivo
            self.output_path.write_text(content)
            
            # ✅ Validar que arquivo existe
            if not self.output_path.exists():
                raise IOError(f"Failed to create dashboard at {self.output_path}")
            
            # ✅ Validar tamanho mínimo (dashboard vazio = problema)
            file_size = self.output_path.stat().st_size
            if file_size < 100:
                raise IOError(
                    f"Dashboard file too small ({file_size} bytes), "
                    f"expected at least 100 bytes"
                )
            
            # ✅ Log sucesso com detalhes (fallback if logger missing)
            try:
                from brain_logger import brain_logger
                brain_logger.info(
                    f"📊 Dashboard saved successfully: "
                    f"{self.output_path} ({file_size} bytes)"
                )
            except Exception:
                print(f"[dashboard] saved: {self.output_path} ({file_size} bytes)")
            
        except Exception as e:
            # ✅ Log erro real (fallback if logger missing)
            try:
                from brain_logger import brain_logger
                brain_logger.error(f"❌ Dashboard save failed: {e}")
            except Exception:
                print(f"[dashboard] save failed: {e}")
            raise
    
    def get_trends(self, metric_path: str, window: int = 10) -> Dict[str, Any]:
        """
        Calcula tendências de uma métrica
        
        Args:
            metric_path: Caminho da métrica (ex: 'performance.step_time')
            window: Janela de análise
            
        Returns:
            Tendências (avg, min, max, trend)
        """
        if len(self.metrics_history) < window:
            return {}
        
        recent = list(self.metrics_history)[-window:]
        
        # Extrai valores
        values = []
        for m in recent:
            parts = metric_path.split('.')
            val = m
            for part in parts:
                val = val.get(part, {})
                if not isinstance(val, dict):
                    break
            
            if isinstance(val, (int, float)):
                values.append(val)
        
        if not values:
            return {}
        
        # Calcula tendência (linear)
        if len(values) >= 2:
            first_half = sum(values[:len(values)//2]) / (len(values)//2)
            second_half = sum(values[len(values)//2:]) / (len(values) - len(values)//2)
            trend = (second_half - first_half) / first_half if first_half != 0 else 0
        else:
            trend = 0
        
        return {
            'avg': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'last': values[-1],
            'trend': trend,
            'trend_direction': 'UP' if trend > 0.05 else ('DOWN' if trend < -0.05 else 'STABLE')
        }


if __name__ == "__main__":
    print("📊 Testing Metrics Dashboard...")
    
    dashboard = MetricsDashboard("/tmp/test_dashboard.txt")
    
    # Simula métricas
    for i in range(5):
        metrics = {
            'episode': i + 1,
            'total_steps': (i + 1) * 30,
            'system_status': {
                'health': 'HEALTHY',
                'uptime': f'{i+1}h'
            },
            'performance': {
                'step_time': 0.72 + i * 0.01
            },
            'learning': {
                'current_reward': 10 + i * 5,
                'best_reward': 50,
                'avg_reward_100': 25,
                'loss': 15.5 - i * 0.5,
                'gradients_applied': i + 1
            },
            'auto_evolution': {
                'meta_controller': {
                    'total_interventions': 0
                },
                'curiosity': {
                    'total_predictions': (i + 1) * 30,
                    'avg_surprise': 0.05 - i * 0.005
                },
                'darwin': {
                    'frozen_count': 0,
                    'max_generation': 40
                }
            },
            'self_analysis': {
                'health': 'HEALTHY',
                'bottlenecks_count': 1,
                'top_bottleneck': 'forward',
                'recommendations_count': 2
            },
            'resources': {
                'memory_rss_mb': 150 + i * 10,
                'gpu_memory_mb': 0,
                'active_neurons': 16,
                'soup_size': 238
            }
        }
        
        dashboard.update(metrics)
    
    # Renderiza
    print(dashboard.render())
    
    # Testa trends
    print("\n📈 Tendências:")
    trends = dashboard.get_trends('performance.step_time', window=5)
    print(f"  step_time: {trends}")
    
    print("\n✅ Dashboard OK")
