#!/usr/bin/env python3
"""
📊 IA³ - MONITORAMENTO 24/7
===========================

Sistema de monitoramento contínuo 24 horas por dia
"""

import os
import sys
import time
import json
import subprocess
import psutil
import threading
import smtplib
import socket
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging
import requests
import sqlite3

logger = logging.getLogger("IA³-24/7-Monitoring")

class SystemMonitor:
    """
    Monitor principal do sistema
    """

    async def __init__(self):
        self.monitors = {
            'process_monitor': ProcessMonitor(),
            'resource_monitor': ResourceMonitor(),
            'emergence_monitor': EmergenceMonitor(),
            'performance_monitor': PerformanceMonitor(),
            'security_monitor': SecurityMonitor()
        }
        self.alerts = []
        self.metrics_history = []
        self.is_active = True

    async def start_24_7_monitoring(self):
        """Iniciar monitoramento 24/7"""
        logger.info("📊 Iniciando monitoramento 24/7 IA³")

        # Iniciar todos os monitores
        for monitor_name, monitor in self.monitors.items():
            monitor.start_monitoring()

        # Loop principal de monitoramento
        async def main_monitoring_loop():
            cycle = 0
            while self.is_active:
                try:
                    cycle += 1

                    # Coletar métricas de todos os monitores
                    all_metrics = {}
                    for monitor_name, monitor in self.monitors.items():
                        metrics = monitor.get_current_metrics()
                        all_metrics[monitor_name] = metrics

                    # Verificar alertas
                    alerts = self._check_for_alerts(all_metrics)

                    # Registrar métricas
                    metrics_record = {
                        'timestamp': datetime.now().isoformat(),
                        'cycle': cycle,
                        'metrics': all_metrics,
                        'alerts': alerts
                    }
                    self.metrics_history.append(metrics_record)

                    # Log periódico
                    if cycle % 60 == 0:  # A cada hora (60 ciclos de 1 min)
                        self._hourly_report(metrics_record)

                    # Limpar histórico antigo
                    if len(self.metrics_history) > 1440:  # Manter 24 horas (1440 minutos)
                        self.metrics_history = self.metrics_history[-720:]  # Manter 12 horas

                    time.sleep(60)  # Verificar a cada minuto

                except Exception as e:
                    logger.error(f"Erro no ciclo de monitoramento: {e}")
                    time.sleep(30)

        # Iniciar loop principal
        monitoring_thread = threading.Thread(target=main_monitoring_loop, daemon=True)
        monitoring_thread.start()

        # Iniciar limpeza periódica
        cleanup_thread = threading.Thread(target=self._periodic_cleanup, daemon=True)
        cleanup_thread.start()

    async def _check_for_alerts(self, all_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verificar alertas em todas as métricas"""
        alerts = []

        # Alertas críticos
        critical_alerts = self._check_critical_alerts(all_metrics)
        alerts.extend(critical_alerts)

        # Alertas de performance
        performance_alerts = self._check_performance_alerts(all_metrics)
        alerts.extend(performance_alerts)

        # Alertas de emergência
        emergence_alerts = self._check_emergence_alerts(all_metrics)
        alerts.extend(emergence_alerts)

        # Registrar alertas
        for alert in alerts:
            alert_record = {
                'timestamp': datetime.now().isoformat(),
                'level': alert['level'],
                'message': alert['message'],
                'metrics': alert.get('metrics', {})
            }
            self.alerts.append(alert_record)

            # Log do alerta
            if alert['level'] == 'critical':
                logger.critical(f"🚨 {alert['message']}")
            elif alert['level'] == 'warning':
                logger.warning(f"⚠️ {alert['message']}")
            else:
                logger.info(f"ℹ️ {alert['message']}")

        # Manter apenas alertas recentes
        if len(self.alerts) > 1000:
            self.alerts = self.alerts[-500:]

        return await alerts

    async def _check_critical_alerts(self, all_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verificar alertas críticos"""
        alerts = []

        # Sistema parado
        process_metrics = all_metrics.get('process_monitor', {})
        if process_metrics.get('ia3_processes_running', 0) == 0:
            alerts.append({
                'level': 'critical',
                'message': 'Nenhum processo IA³ em execução',
                'metrics': process_metrics
            })

        # Recursos críticos
        resource_metrics = all_metrics.get('resource_monitor', {})
        if resource_metrics.get('cpu_percent', 0) > 95:
            alerts.append({
                'level': 'critical',
                'message': f'CPU crítica: {resource_metrics["cpu_percent"]:.1f}%',
                'metrics': resource_metrics
            })

        if resource_metrics.get('memory_percent', 0) > 95:
            alerts.append({
                'level': 'critical',
                'message': f'Memória crítica: {resource_metrics["memory_percent"]:.1f}%',
                'metrics': resource_metrics
            })

        return await alerts

    async def _check_performance_alerts(self, all_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verificar alertas de performance"""
        alerts = []

        performance_metrics = all_metrics.get('performance_monitor', {})

        # Performance degradada
        if performance_metrics.get('response_time_avg', 0) > 10.0:  # 10 segundos
            alerts.append({
                'level': 'warning',
                'message': f'Response time alto: {performance_metrics["response_time_avg"]:.2f}s',
                'metrics': performance_metrics
            })

        # Taxa de erro alta
        if performance_metrics.get('error_rate', 0) > 0.1:  # 10%
            alerts.append({
                'level': 'warning',
                'message': f'Taxa de erro alta: {performance_metrics["error_rate"]:.2%}',
                'metrics': performance_metrics
            })

        return await alerts

    async def _check_emergence_alerts(self, all_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verificar alertas de emergência"""
        alerts = []

        emergence_metrics = all_metrics.get('emergence_monitor', {})

        # Emergência detectada
        if emergence_metrics.get('emergence_probability', 0) > 0.8:
            alerts.append({
                'level': 'info',
                'message': f'Emergência detectada: {emergence_metrics["emergence_probability"]:.3f}',
                'metrics': emergence_metrics
            })

        # Falta de progresso
        if emergence_metrics.get('evolution_stagnation', False):
            alerts.append({
                'level': 'warning',
                'message': 'Evolução estagnada - possível problema',
                'metrics': emergence_metrics
            })

        return await alerts

    async def _hourly_report(self, metrics_record: Dict[str, Any]):
        """Relatório horário"""
        metrics = metrics_record['metrics']
        alerts_count = len(metrics_record['alerts'])

        # Calcular estatísticas da hora
        hour_stats = {
            'avg_cpu': sum(m.get('cpu_percent', 0) for m in [metrics.get('resource_monitor', {})]),
            'avg_memory': sum(m.get('memory_percent', 0) for m in [metrics.get('resource_monitor', {})]),
            'emergence_probability': metrics.get('emergence_monitor', {}).get('emergence_probability', 0),
            'alerts_this_hour': alerts_count
        }

        logger.info(f"📊 Relatório horário: CPU {hour_stats['avg_cpu']:.1f}% | Memória {hour_stats['avg_memory']:.1f}% | Emergência {hour_stats['emergence_probability']:.3f} | Alertas: {alerts_count}")

    async def _periodic_cleanup(self):
        """Limpeza periódica de logs e dados antigos"""
        while self.is_active:
            try:
                # Limpar logs antigos (> 7 dias)
                for file in os.listdir('.'):
                    if file.endswith('.log'):
                        try:
                            if time.time() - os.path.getmtime(file) > 604800:  # 7 dias
                                os.remove(file)
                                logger.info(f"🗑️ Log antigo removido: {file}")
                        except:
                            pass

                # Limpar métricas antigas (> 48 horas)
                cutoff_time = datetime.now() - timedelta(hours=48)
                self.metrics_history = [
                    m for m in self.metrics_history
                    if datetime.fromisoformat(m['timestamp']) > cutoff_time
                ]

                time.sleep(3600)  # Limpar a cada hora

            except Exception as e:
                logger.error(f"Erro na limpeza periódica: {e}")
                time.sleep(1800)  # Tentar novamente em 30 min

    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Obter status completo do monitoramento"""
        latest_metrics = self.metrics_history[-1] if self.metrics_history else None
        recent_alerts = self.alerts[-10:] if self.alerts else []

        return await {
            'is_active': self.is_active,
            'latest_metrics': latest_metrics,
            'recent_alerts': recent_alerts,
            'total_alerts': len(self.alerts),
            'monitors_status': {name: monitor.is_monitoring for name, monitor in self.monitors.items()},
            'uptime': self._calculate_uptime()
        }

    async def _calculate_uptime(self) -> float:
        """Calcular uptime do monitoramento"""
        if not self.metrics_history:
            return await 0.0

        start_time = datetime.fromisoformat(self.metrics_history[0]['timestamp'])
        current_time = datetime.now()

        return await (current_time - start_time).total_seconds() / 3600  # Horas

class ProcessMonitor:
    """
    Monitor de processos
    """

    async def __init__(self):
        self.is_monitoring = False
        self.process_history = []

    async def start_monitoring(self):
        """Iniciar monitoramento de processos"""
        self.is_monitoring = True

        async def monitor_loop():
            while self.is_monitoring:
                try:
                    # Obter processos IA³
                    ia3_processes = []
                    for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
                        if 'ia3' in proc.info['name'].lower() or 'python' in proc.info['name'].lower():
                            # Verificar se é processo IA³
                            try:
                                cmdline = proc.cmdline()
                                if any('IA3' in cmd or 'ia3' in cmd.lower() for cmd in cmdline):
                                    ia3_processes.append(proc.info)
                            except:
                                pass

                    # Registrar métricas
                    metrics = {
                        'timestamp': datetime.now().isoformat(),
                        'ia3_processes_running': len(ia3_processes),
                        'total_processes': len(list(psutil.process_iter())),
                        'ia3_processes': ia3_processes
                    }

                    self.process_history.append(metrics)

                    # Manter histórico
                    if len(self.process_history) > 60:  # Última hora
                        self.process_history = self.process_history[-30:]

                    time.sleep(10)  # Verificar a cada 10 segundos

                except Exception as e:
                    logger.error(f"Erro no monitoramento de processos: {e}")
                    time.sleep(30)

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Obter métricas atuais de processos"""
        if self.process_history:
            return await self.process_history[-1]
        return await {'ia3_processes_running': 0, 'total_processes': 0}

class ResourceMonitor:
    """
    Monitor de recursos do sistema
    """

    async def __init__(self):
        self.is_monitoring = False
        self.resource_history = []

    async def start_monitoring(self):
        """Iniciar monitoramento de recursos"""
        self.is_monitoring = True

        async def monitor_loop():
            while self.is_monitoring:
                try:
                    metrics = {
                        'timestamp': datetime.now().isoformat(),
                        'cpu_percent': psutil.cpu_percent(interval=1),
                        'memory_percent': psutil.virtual_memory().percent,
                        'disk_percent': psutil.disk_usage('/').percent,
                        'network_bytes_sent': psutil.net_io_counters().bytes_sent,
                        'network_bytes_recv': psutil.net_io_counters().bytes_recv,
                        'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                    }

                    self.resource_history.append(metrics)

                    if len(self.resource_history) > 60:
                        self.resource_history = self.resource_history[-30:]

                    time.sleep(30)  # Verificar a cada 30 segundos

                except Exception as e:
                    logger.error(f"Erro no monitoramento de recursos: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Obter métricas atuais de recursos"""
        if self.resource_history:
            return await self.resource_history[-1]
        return await {}

class EmergenceMonitor:
    """
    Monitor de emergência
    """

    async def __init__(self):
        self.is_monitoring = False
        self.emergence_history = []

    async def start_monitoring(self):
        """Iniciar monitoramento de emergência"""
        self.is_monitoring = True

        async def monitor_loop():
            while self.is_monitoring:
                try:
                    # Verificar indicadores de emergência
                    emergence_files = len([f for f in os.listdir('.') if 'emergence' in f.lower()])
                    evolution_logs = 0

                    # Verificar logs recentes
                    log_files = [f for f in os.listdir('.') if f.endswith('.log')][:3]
                    for log_file in log_files:
                        try:
                            with open(log_file, 'r') as f:
                                content = f.read()
                                evolution_logs += content.lower().count('evolution')
                                evolution_logs += content.lower().count('emergence')
                        except:
                            pass

                    # Calcular probabilidade de emergência
                    emergence_probability = min(1.0, (emergence_files / 10.0 + evolution_logs / 100.0) / 2)

                    # Verificar estagnação
                    stagnation = False
                    if len(self.emergence_history) > 10:
                        recent_probs = [h['emergence_probability'] for h in self.emergence_history[-10:]]
                        if max(recent_probs) - min(recent_probs) < 0.01:  # Pouca variação
                            stagnation = True

                    metrics = {
                        'timestamp': datetime.now().isoformat(),
                        'emergence_probability': emergence_probability,
                        'emergence_files': emergence_files,
                        'evolution_logs': evolution_logs,
                        'evolution_stagnation': stagnation
                    }

                    self.emergence_history.append(metrics)

                    if len(self.emergence_history) > 60:
                        self.emergence_history = self.emergence_history[-30:]

                    time.sleep(60)  # Verificar a cada minuto

                except Exception as e:
                    logger.error(f"Erro no monitoramento de emergência: {e}")
                    time.sleep(120)

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Obter métricas atuais de emergência"""
        if self.emergence_history:
            return await self.emergence_history[-1]
        return await {'emergence_probability': 0.0}

class PerformanceMonitor:
    """
    Monitor de performance
    """

    async def __init__(self):
        self.is_monitoring = False
        self.performance_history = []
        self.response_times = []

    async def start_monitoring(self):
        """Iniciar monitoramento de performance"""
        self.is_monitoring = True

        async def monitor_loop():
            while self.is_monitoring:
                try:
                    # Simular medição de performance (em produção seria real)
                    start_time = time.time()
                    # Simular operação
                    time.sleep(0.01)
                    response_time = time.time() - start_time

                    self.response_times.append(response_time)

                    # Manter últimas 100 medições
                    if len(self.response_times) > 100:
                        self.response_times = self.response_times[-100:]

                    # Calcular métricas
                    avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
                    error_rate = 0.02  # Simulado - em produção seria real

                    metrics = {
                        'timestamp': datetime.now().isoformat(),
                        'response_time_avg': avg_response_time,
                        'response_time_min': min(self.response_times) if self.response_times else 0,
                        'response_time_max': max(self.response_times) if self.response_times else 0,
                        'error_rate': error_rate,
                        'throughput': len(self.response_times) / 60  # Por minuto
                    }

                    self.performance_history.append(metrics)

                    if len(self.performance_history) > 60:
                        self.performance_history = self.performance_history[-30:]

                    time.sleep(60)

                except Exception as e:
                    logger.error(f"Erro no monitoramento de performance: {e}")
                    time.sleep(120)

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Obter métricas atuais de performance"""
        if self.performance_history:
            return await self.performance_history[-1]
        return await {'response_time_avg': 0.0, 'error_rate': 0.0}

class SecurityMonitor:
    """
    Monitor de segurança
    """

    async def __init__(self):
        self.is_monitoring = False
        self.security_history = []

    async def start_monitoring(self):
        """Iniciar monitoramento de segurança"""
        self.is_monitoring = True

        async def monitor_loop():
            while self.is_monitoring:
                try:
                    # Verificar conexões suspeitas
                    suspicious_connections = 0
                    for conn in psutil.net_connections():
                        if conn.status == 'ESTABLISHED':
                            # Verificar portas suspeitas
                            if conn.laddr.port in [22, 23, 3389]:  # SSH, Telnet, RDP
                                suspicious_connections += 1

                    # Verificar arquivos suspeitos
                    suspicious_files = 0
                    dangerous_extensions = ['.exe', '.bat', '.sh', '.py']
                    for file in os.listdir('.'):
                        if any(file.endswith(ext) for ext in dangerous_extensions):
                            # Verificar se foi modificado recentemente
                            try:
                                if time.time() - os.path.getmtime(file) < 300:  # Últimos 5 min
                                    suspicious_files += 1
                            except:
                                pass

                    # Verificar tentativas de acesso
                    failed_logins = 0  # Em produção seria monitorado do sistema

                    metrics = {
                        'timestamp': datetime.now().isoformat(),
                        'suspicious_connections': suspicious_connections,
                        'suspicious_files': suspicious_files,
                        'failed_logins': failed_logins,
                        'security_status': 'good' if suspicious_connections == 0 and suspicious_files < 3 else 'warning'
                    }

                    self.security_history.append(metrics)

                    if len(self.security_history) > 60:
                        self.security_history = self.security_history[-30:]

                    time.sleep(300)  # Verificar a cada 5 minutos

                except Exception as e:
                    logger.error(f"Erro no monitoramento de segurança: {e}")
                    time.sleep(600)

        thread = threading.Thread(target=monitor_loop, daemon=True)
        thread.start()

    async def get_current_metrics(self) -> Dict[str, Any]:
        """Obter métricas atuais de segurança"""
        if self.security_history:
            return await self.security_history[-1]
        return await {'security_status': 'unknown'}

async def main():
    """Função principal"""
    print("📊 IA³ - MONITORAMENTO 24/7")
    print("=" * 30)

    # Inicializar monitoramento
    monitor = SystemMonitor()
    monitor.start_24_7_monitoring()

    # Manter ativo
    try:
        while True:
            time.sleep(300)  # Report a cada 5 minutos
            status = monitor.get_monitoring_status()
            alerts_count = len(status['recent_alerts'])
            print(f"📊 Monitoramento ativo: {status['uptime']:.1f}h | Alertas recentes: {alerts_count}")

    except KeyboardInterrupt:
        print("🛑 Parando monitoramento...")
        monitor.is_active = False

if __name__ == "__main__":
    main()