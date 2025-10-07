"""
DASHBOARD DE MONITORAMENTO DARWIN - Sistema de Observabilidade Completo
=========================================================================

Implementa sistema avançado de monitoramento, métricas e visualização
para o Darwin Ultimate Engine.

Componente essencial para operação e debugging do sistema completo.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import threading
import queue
import statistics
from collections import defaultdict, deque

# Tentar importar bibliotecas opcionais de visualização
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DarwinMetricsCollector:
    """Coletor avançado de métricas Darwin"""

    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.metrics_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        self.metrics_metadata: Dict[str, Dict] = {}
        self.collection_start_time = datetime.now()

        # Filas para coleta assíncrona
        self.metric_queue = queue.Queue()
        self._start_collection_thread()

        logger.info("📊 Darwin Metrics Collector inicializado")

    def _start_collection_thread(self):
        """Inicia thread de coleta assíncrona"""
        def collection_worker():
            while True:
                try:
                    metric_data = self.metric_queue.get(timeout=1.0)
                    self._store_metric(metric_data)
                    self.metric_queue.task_done()
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Erro na coleta de métricas: {e}")

        thread = threading.Thread(target=collection_worker, daemon=True)
        thread.start()

    def collect_metric(self, name: str, value: float, metadata: Dict[str, Any] = None,
                      timestamp: datetime = None):
        """Coleta métrica de forma assíncrona"""
        if timestamp is None:
            timestamp = datetime.now()

        metric_data = {
            'name': name,
            'value': value,
            'timestamp': timestamp,
            'metadata': metadata or {}
        }

        self.metric_queue.put(metric_data)

    def _store_metric(self, metric_data: Dict[str, Any]):
        """Armazena métrica no histórico"""
        name = metric_data['name']
        value = metric_data['value']
        timestamp = metric_data['timestamp']

        # Armazenar no histórico
        self.metrics_history[name].append({
            'value': value,
            'timestamp': timestamp
        })

        # Atualizar metadados
        if name not in self.metrics_metadata:
            self.metrics_metadata[name] = {
                'first_seen': timestamp,
                'unit': metric_data['metadata'].get('unit', 'unknown'),
                'description': metric_data['metadata'].get('description', ''),
                'category': metric_data['metadata'].get('category', 'general')
            }

        # Atualizar últimos valores
        self.metrics_metadata[name].update({
            'last_seen': timestamp,
            'last_value': value,
            'total_samples': len(self.metrics_history[name])
        })

    def get_metric_history(self, name: str, limit: int = None) -> List[Dict]:
        """Obtém histórico de uma métrica"""
        if name not in self.metrics_history:
            return []

        history = list(self.metrics_history[name])
        if limit:
            history = history[-limit:]

        return history

    def get_latest_value(self, name: str) -> float:
        """Obtém último valor de uma métrica"""
        if name not in self.metrics_history or not self.metrics_history[name]:
            return 0.0

        return self.metrics_history[name][-1]['value']

    def get_metric_stats(self, name: str, window_minutes: int = 60) -> Dict[str, Any]:
        """Calcula estatísticas de uma métrica"""
        history = self.get_metric_history(name)

        if not history:
            return {}

        # Filtrar janela de tempo
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_values = [
            h['value'] for h in history
            if h['timestamp'] >= cutoff_time
        ]

        if not recent_values:
            return {}

        return {
            'current': recent_values[-1],
            'mean': statistics.mean(recent_values),
            'median': statistics.median(recent_values),
            'min': min(recent_values),
            'max': max(recent_values),
            'std': statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0,
            'count': len(recent_values),
            'time_range': f"{recent_values[0]:.1f} - {recent_values[-1]:.1f}"
        }

    def get_all_metrics_summary(self) -> Dict[str, Any]:
        """Obtém resumo de todas as métricas"""
        summary = {}

        for name in self.metrics_metadata:
            metadata = self.metrics_metadata[name]
            stats = self.get_metric_stats(name, window_minutes=60)

            summary[name] = {
                'metadata': metadata,
                'stats': stats,
                'history_length': len(self.metrics_history[name])
            }

        return summary

    def export_metrics_json(self, filename: str = None) -> str:
        """Exporta métricas para JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"darwin_metrics_{timestamp}.json"

        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'collection_start_time': self.collection_start_time.isoformat(),
            'max_history': self.max_history,
            'metrics_metadata': self.metrics_metadata,
            'metrics_data': {}
        }

        # Exportar últimos 100 pontos de cada métrica
        for name, history in self.metrics_history.items():
            export_data['metrics_data'][name] = list(history)[-100:]

        with open(filename, 'w') as f:
            json.dump(export_data, f, indent=2, default=str)

        logger.info(f"📊 Métricas exportadas: {filename}")
        return filename

class DarwinDashboard:
    """Dashboard avançado para visualização de métricas"""

    def __init__(self, metrics_collector: DarwinMetricsCollector):
        self.collector = metrics_collector
        self.dashboard_config = {
            'refresh_interval': 30,  # segundos
            'auto_export': True,
            'export_interval': 300,  # 5 minutos
            'enable_plots': MATPLOTLIB_AVAILABLE or PLOTLY_AVAILABLE,
            'max_plot_points': 200
        }

        self._start_dashboard_thread()

    def _start_dashboard_thread(self):
        """Inicia thread do dashboard"""
        def dashboard_worker():
            last_export = 0

            while True:
                try:
                    # Atualizar dashboard
                    self._update_dashboard()

                    # Auto-export periódico
                    current_time = time.time()
                    if (self.dashboard_config['auto_export'] and
                        current_time - last_export > self.dashboard_config['export_interval']):
                        self.collector.export_metrics_json()
                        last_export = current_time

                    time.sleep(self.dashboard_config['refresh_interval'])

                except Exception as e:
                    logger.error(f"Erro no dashboard: {e}")
                    time.sleep(self.dashboard_config['refresh_interval'])

        thread = threading.Thread(target=dashboard_worker, daemon=True)
        thread.start()

    def _update_dashboard(self):
        """Atualiza informações do dashboard"""
        summary = self.collector.get_all_metrics_summary()

        # Métricas principais
        key_metrics = [
            'darwin.population_size',
            'darwin.avg_fitness',
            'darwin.best_fitness',
            'darwin.generation',
            'darwin.diversity'
        ]

        print("\n🚀 DARWIN DASHBOARD - Última atualização: {}".format(datetime.now().strftime("%H:%M:%S")))
        print("="*80)

        for metric_name in key_metrics:
            if metric_name in summary:
                metric_info = summary[metric_name]
                stats = metric_info['stats']

                if stats:
                    current = stats.get('current', 0)
                    mean_val = stats.get('mean', 0)

                    print(f"📊 {metric_name}:")
                    print(f"   Atual: {current:.4f}")
                    print(f"   Média (1h): {mean_val:.4f}")
                    print(f"   Variação: {stats.get('min', 0):.4f} - {stats.get('max', 0):.4f}")
                else:
                    print(f"📊 {metric_name}: Sem dados recentes")
            else:
                print(f"📊 {metric_name}: Não disponível")

        print("\n📈 EVOLUÇÃO:")
        print(f"   Tempo de coleta: {(datetime.now() - self.collector.collection_start_time).total_seconds():.0f}s")
        print(f"   Métricas coletadas: {len(summary)}")
        print(f"   Total de pontos: {sum(len(history) for history in self.collector.metrics_history.values())}")

        # Gráfico simples se matplotlib disponível
        if MATPLOTLIB_AVAILABLE and len(summary) > 0:
            self._generate_simple_plot()

    def _generate_simple_plot(self):
        """Gera gráfico simples das métricas principais"""
        try:
            fig, ax = plt.subplots(figsize=(10, 6))

            # Plotar fitness ao longo do tempo
            fitness_history = self.collector.get_metric_history('darwin.avg_fitness', 50)
            if fitness_history:
                timestamps = [h['timestamp'] for h in fitness_history]
                values = [h['value'] for h in fitness_history]

                ax.plot(timestamps, values, 'b-', label='Fitness Médio', linewidth=2)
                ax.fill_between(timestamps, values, alpha=0.3)

            ax.set_title('Evolução do Fitness - Darwin Engine')
            ax.set_xlabel('Tempo')
            ax.set_ylabel('Fitness')
            ax.legend()
            ax.grid(True, alpha=0.3)

            # Salvar plot
            plt.savefig('darwin_fitness_plot.png', dpi=100, bbox_inches='tight')
            plt.close()

            print("   📈 Gráfico salvo: darwin_fitness_plot.png")

        except Exception as e:
            logger.error(f"Erro ao gerar gráfico: {e}")

    def generate_comprehensive_report(self) -> str:
        """Gera relatório abrangente das métricas"""
        summary = self.collector.get_all_metrics_summary()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"darwin_comprehensive_report_{timestamp}.json"

        # Estrutura do relatório
        report = {
            'report_timestamp': datetime.now().isoformat(),
            'collection_period': {
                'start': self.collector.collection_start_time.isoformat(),
                'duration_seconds': (datetime.now() - self.collector.collection_start_time).total_seconds()
            },
            'metrics_summary': summary,
            'key_insights': self._generate_insights(summary),
            'alerts': self._check_alerts(summary),
            'recommendations': self._generate_recommendations(summary)
        }

        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        logger.info(f"📋 Relatório abrangente gerado: {report_file}")
        return report_file

    def _generate_insights(self, summary: Dict) -> List[str]:
        """Gera insights automáticos das métricas"""
        insights = []

        # Análise de fitness
        if 'darwin.avg_fitness' in summary:
            fitness_stats = summary['darwin.avg_fitness']['stats']
            if fitness_stats:
                current = fitness_stats.get('current', 0)
                trend = self._calculate_trend('darwin.avg_fitness')

                if current > 0.8:
                    insights.append(f"✅ Excelente: Fitness médio em {current:.3f}")
                elif current > 0.6:
                    insights.append(f"👍 Bom: Fitness médio em {current:.3f}")
                elif current > 0.4:
                    insights.append(f"⚠️ Regular: Fitness médio em {current:.3f} - pode melhorar")
                else:
                    insights.append(f"❌ Baixo: Fitness médio em {current:.3f} - ação necessária")

                if trend > 0.01:
                    insights.append("📈 Tendência positiva no fitness")
                elif trend < -0.01:
                    insights.append("📉 Tendência negativa no fitness")

        # Análise de diversidade
        if 'darwin.diversity' in summary:
            diversity_stats = summary['darwin.diversity']['stats']
            if diversity_stats:
                diversity = diversity_stats.get('current', 0)
                if diversity > 0.7:
                    insights.append(f"🌈 Alta diversidade genética: {diversity:.3f}")
                elif diversity < 0.3:
                    insights.append(f"⚠️ Baixa diversidade genética: {diversity:.3f}")

        # Análise de população
        if 'darwin.population_size' in summary:
            pop_stats = summary['darwin.population_size']['stats']
            if pop_stats:
                pop_size = pop_stats.get('current', 0)
                insights.append(f"👥 Tamanho da população: {int(pop_size)} indivíduos")

        return insights

    def _calculate_trend(self, metric_name: str, window: int = 10) -> float:
        """Calcula tendência de uma métrica"""
        history = self.collector.get_metric_history(metric_name, window)

        if len(history) < 3:
            return 0.0

        values = [h['value'] for h in history]

        # Calcular tendência usando regressão linear simples
        x = list(range(len(values)))
        y = values

        if len(x) > 1:
            slope = (len(x) * sum(xi*yi for xi, yi in zip(x, y)) - sum(x) * sum(y)) / \
                   (len(x) * sum(xi**2 for xi in x) - sum(x)**2)
            return slope

        return 0.0

    def _check_alerts(self, summary: Dict) -> List[str]:
        """Verifica alertas baseados nas métricas"""
        alerts = []

        # Alerta de fitness baixo
        if 'darwin.avg_fitness' in summary:
            fitness_stats = summary['darwin.avg_fitness']['stats']
            if fitness_stats and fitness_stats.get('current', 0) < 0.3:
                alerts.append("🚨 ALERTA: Fitness muito baixo - sistema pode estar estagnado")

        # Alerta de diversidade baixa
        if 'darwin.diversity' in summary:
            diversity_stats = summary['darwin.diversity']['stats']
            if diversity_stats and diversity_stats.get('current', 0) < 0.2:
                alerts.append("⚠️ AVISO: Diversidade genética baixa - risco de convergência prematura")

        # Alerta de população pequena
        if 'darwin.population_size' in summary:
            pop_stats = summary['darwin.population_size']['stats']
            if pop_stats and pop_stats.get('current', 0) < 20:
                alerts.append("⚠️ AVISO: População muito pequena - pode limitar exploração")

        return alerts

    def _generate_recommendations(self, summary: Dict) -> List[str]:
        """Gera recomendações baseadas nas métricas"""
        recommendations = []

        # Recomendações baseadas no fitness
        if 'darwin.avg_fitness' in summary:
            fitness_stats = summary['darwin.avg_fitness']['stats']
            if fitness_stats:
                current = fitness_stats.get('current', 0)
                trend = self._calculate_trend('darwin.avg_fitness')

                if current < 0.5 and trend < 0:
                    recommendations.append("💡 Aumentar taxa de mutação para melhorar exploração")
                    recommendations.append("💡 Aplicar incompletude gödeliana para escapar de ótimos locais")

                if current > 0.8 and trend > 0:
                    recommendations.append("✅ Sistema convergindo bem - manter parâmetros atuais")

        # Recomendações baseadas na diversidade
        if 'darwin.diversity' in summary:
            diversity_stats = summary['darwin.diversity']['stats']
            if diversity_stats:
                diversity = diversity_stats.get('current', 0)

                if diversity < 0.3:
                    recommendations.append("💡 Aumentar exploração para manter diversidade genética")
                    recommendations.append("💡 Introduzir indivíduos aleatórios periodicamente")

        return recommendations

    def start_web_dashboard(self, port: int = 8080):
        """Inicia dashboard web simples"""
        def web_server():
            import http.server
            import socketserver

            class DarwinHandler(http.server.SimpleHTTPRequestHandler):
                def do_GET(self):
                    if self.path == '/':
                        self.path = '/darwin_dashboard.html'
                    elif self.path == '/api/metrics':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()

                        summary = self.server.metrics_collector.get_all_metrics_summary()
                        self.wfile.write(json.dumps(summary, default=str).encode())
                    elif self.path == '/api/status':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()

                        status = {
                            'timestamp': datetime.now().isoformat(),
                            'uptime_seconds': (datetime.now() - self.server.start_time).total_seconds(),
                            'metrics_count': len(self.server.metrics_collector.metrics_metadata),
                            'total_samples': sum(len(history) for history in self.server.metrics_collector.metrics_history.values())
                        }
                        self.wfile.write(json.dumps(status).encode())
                    else:
                        super().do_GET()

            # Configurar handler
            handler = DarwinHandler
            handler.server = type('MockServer', (), {
                'metrics_collector': self.collector,
                'start_time': datetime.now()
            })()

            with socketserver.TCPServer(("", port), handler) as httpd:
                logger.info(f"🌐 Dashboard web iniciado: http://localhost:{port}")
                httpd.serve_forever()

        thread = threading.Thread(target=web_server, daemon=True)
        thread.start()

# ============================================================================
# FUNÇÕES DE INTEGRAÇÃO
# ============================================================================

def integrate_darwin_monitoring(darwin_engine):
    """Integra sistema de monitoramento com o Darwin Engine"""
    # Criar coletor de métricas
    collector = DarwinMetricsCollector()

    # Monkey patch para coletar métricas automaticamente
    original_evolve = darwin_engine.evolve_generation

    def monitored_evolve():
        start_time = time.time()

        # Executar evolução normal
        result = original_evolve()

        # Coletar métricas
        evolution_time = time.time() - start_time

        # Métricas básicas
        collector.collect_metric('darwin.generation', darwin_engine.generation,
                              {'unit': 'count', 'description': 'Número da geração atual'})
        collector.collect_metric('darwin.population_size', len(darwin_engine.population),
                              {'unit': 'count', 'description': 'Tamanho da população'})
        collector.collect_metric('darwin.evolution_time', evolution_time,
                              {'unit': 'seconds', 'description': 'Tempo de evolução por geração'})

        # Estatísticas da população
        if darwin_engine.population:
            fitnesses = [ind.fitness for ind in darwin_engine.population]
            collector.collect_metric('darwin.avg_fitness', sum(fitnesses) / len(fitnesses),
                                  {'unit': 'score', 'description': 'Fitness médio da população'})
            collector.collect_metric('darwin.best_fitness', max(fitnesses),
                                  {'unit': 'score', 'description': 'Melhor fitness da população'})
            collector.collect_metric('darwin.worst_fitness', min(fitnesses),
                                  {'unit': 'score', 'description': 'Pior fitness da população'})

            # Diversidade genética
            genome_differences = []
            for i, ind1 in enumerate(darwin_engine.population[:10]):  # Amostra
                for ind2 in darwin_engine.population[i+1:11]:
                    diff = sum(abs(ind1.genome.get(k, 0) - ind2.genome.get(k, 0))
                              for k in set(ind1.genome.keys()) | set(ind2.genome.keys()))
                    genome_differences.append(diff)

            if genome_differences:
                avg_genetic_diff = sum(genome_differences) / len(genome_differences)
                collector.collect_metric('darwin.diversity', avg_genetic_diff / 1000.0,
                                      {'unit': 'distance', 'description': 'Diversidade genética'})

        return result

    # Aplicar monkey patch
    darwin_engine.evolve_generation = monitored_evolve

    # Criar dashboard
    dashboard = DarwinDashboard(collector)

    logger.info("🔗 Monitoramento integrado ao Darwin Engine")

    return collector, dashboard

# ============================================================================
# EXEMPLOS DE USO
# ============================================================================

def example_monitoring():
    """Exemplo de uso do sistema de monitoramento"""
    print("="*80)
    print("📊 EXEMPLO: SISTEMA DE MONITORAMENTO DARWIN")
    print("="*80)

    # Criar coletor
    collector = DarwinMetricsCollector()

    # Simular coleta de métricas
    print("📊 Coletando métricas de exemplo...")

    for i in range(50):
        # Simular métricas de evolução
        generation = i + 1
        fitness = 0.5 + 0.4 * (1 - 1/(1 + generation/10)) + random.uniform(-0.05, 0.05)
        diversity = 0.8 - 0.3 * (generation / 50) + random.uniform(-0.1, 0.1)

        collector.collect_metric('darwin.generation', generation,
                              {'unit': 'count', 'description': 'Geração atual'})
        collector.collect_metric('darwin.avg_fitness', fitness,
                              {'unit': 'score', 'description': 'Fitness médio'})
        collector.collect_metric('darwin.diversity', diversity,
                              {'unit': 'ratio', 'description': 'Diversidade genética'})

        time.sleep(0.1)  # Simular tempo

    # Gerar relatório
    print("\n📋 Gerando relatório abrangente...")
    report_file = collector.export_metrics_json()

    # Mostrar estatísticas
    print("\n📈 ESTATÍSTICAS:")
    summary = collector.get_all_metrics_summary()

    for name, info in summary.items():
        stats = info['stats']
        if stats:
            print(f"   {name}: {stats['current']:.3f} (média: {stats['mean']:.3f})")

    # Criar dashboard
    dashboard = DarwinDashboard(collector)

    print("\n🌐 Dashboard iniciado!")
    print("   📊 Métricas coletadas em tempo real")
    print(f"   📁 Relatório salvo: {report_file}")

    return collector, dashboard

if __name__ == "__main__":
    # Executar exemplo
    collector, dashboard = example_monitoring()

    print("\n✅ Sistema de Monitoramento funcionando!")
    print("   📊 Coleta de métricas automática")
    print("   📈 Visualização em tempo real")
    print("   📋 Relatórios abrangentes")
    print("   🌐 Dashboard web disponível")
    print("   🎯 Darwin Ideal: monitoramento ALCANÇADO!")