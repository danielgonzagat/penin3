#!/usr/bin/env python3
"""
🌍 IA³ - MOTOR DE FEEDBACK REAL
===============================

Sistema de feedback loops reais conectando IA³ ao mundo físico
"""

import os
import sys
import time
import psutil
import threading
import subprocess
import socket
import requests
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging
import platform

logger = logging.getLogger("IA³-RealFeedback")

class RealWorldSensor:
    """
    Sensor que coleta dados reais do mundo físico
    """

    async def __init__(self):
        self.sensors = {
            'system_resources': self._get_system_resources,
            'network_activity': self._get_network_activity,
            'disk_usage': self._get_disk_usage,
            'process_info': self._get_process_info,
            'external_connectivity': self._get_external_connectivity,
            'time_patterns': self._get_time_patterns,
            'environmental_data': self._get_environmental_data
        }
        self.history = {}
        self.baselines = {}

    async def collect_all_data(self) -> Dict[str, Any]:
        """Coletar dados de todos os sensores"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'sensors': {}
        }

        for sensor_name, sensor_func in self.sensors.items():
            try:
                sensor_data = sensor_func()
                data['sensors'][sensor_name] = sensor_data

                # Manter histórico
                if sensor_name not in self.history:
                    self.history[sensor_name] = []
                self.history[sensor_name].append(sensor_data)

                # Limitar histórico a últimas 100 entradas
                if len(self.history[sensor_name]) > 100:
                    self.history[sensor_name] = self.history[sensor_name][-100:]

            except Exception as e:
                logger.warning(f"Erro no sensor {sensor_name}: {e}")
                data['sensors'][sensor_name] = {'error': str(e)}

        return await data

    async def _get_system_resources(self) -> Dict[str, Any]:
        """Obter recursos reais do sistema"""
        return await {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'cpu_count': psutil.cpu_count(),
            'cpu_freq': psutil.cpu_freq().current if psutil.cpu_freq() else None,
            'memory_total': psutil.virtual_memory().total,
            'memory_used': psutil.virtual_memory().used,
            'memory_percent': psutil.virtual_memory().percent,
            'swap_total': psutil.swap_memory().total,
            'swap_used': psutil.swap_memory().used,
            'swap_percent': psutil.swap_memory().percent,
            'load_average': psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
        }

    async def _get_network_activity(self) -> Dict[str, Any]:
        """Obter atividade de rede real"""
        net_io = psutil.net_io_counters()
        net_connections = psutil.net_connections()

        return await {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errin': net_io.errin,
            'errout': net_io.errout,
            'dropin': net_io.dropin,
            'dropout': net_io.dropout,
            'active_connections': len([c for c in net_connections if c.status == 'ESTABLISHED']),
            'listening_ports': len([c for c in net_connections if c.status == 'LISTEN'])
        }

    async def _get_disk_usage(self) -> Dict[str, Any]:
        """Obter uso de disco real"""
        disk_usage = psutil.disk_usage('/')
        disk_io = psutil.disk_io_counters()

        return await {
            'total': disk_usage.total,
            'used': disk_usage.used,
            'free': disk_usage.free,
            'percent': disk_usage.percent,
            'read_count': disk_io.read_count if disk_io else 0,
            'write_count': disk_io.write_count if disk_io else 0,
            'read_bytes': disk_io.read_bytes if disk_io else 0,
            'write_bytes': disk_io.write_bytes if disk_io else 0
        }

    async def _get_process_info(self) -> Dict[str, Any]:
        """Obter informação de processos reais"""
        processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']))

        # Filtrar processos Python/IA³
        ai_processes = [p for p in processes if 'python' in p.info['name'].lower() or 'ia3' in p.info['name'].lower()]

        return await {
            'total_processes': len(processes),
            'ai_processes': len(ai_processes),
            'top_cpu_processes': sorted(processes, key=lambda x: x.info['cpu_percent'] or 0, reverse=True)[:5],
            'top_memory_processes': sorted(processes, key=lambda x: x.info['memory_percent'] or 0, reverse=True)[:5],
            'running_processes': len([p for p in processes if p.info['status'] == 'running'])
        }

    async def _get_external_connectivity(self) -> Dict[str, Any]:
        """Verificar conectividade externa real"""
        connectivity = {
            'internet_access': False,
            'dns_resolution': False,
            'latency': None,
            'public_ip': None
        }

        try:
            # Testar acesso à internet
            response = requests.get('https://www.google.com', timeout=5)
            connectivity['internet_access'] = response.status_code == 200

            # Resolver DNS
            socket.gethostbyname('google.com')
            connectivity['dns_resolution'] = True

            # Medir latência
            start_time = time.time()
            socket.create_connection(('google.com', 80), timeout=5)
            connectivity['latency'] = time.time() - start_time

            # Obter IP público
            try:
                response = requests.get('https://api.ipify.org', timeout=5)
                connectivity['public_ip'] = response.text.strip()
            except:
                pass

        except Exception as e:
            logger.debug(f"Erro na conectividade externa: {e}")

        return await connectivity

    async def _get_time_patterns(self) -> Dict[str, Any]:
        """Obter padrões temporais reais"""
        now = datetime.now()

        return await {
            'hour': now.hour,
            'minute': now.minute,
            'day_of_week': now.weekday(),
            'day_of_month': now.day,
            'month': now.month,
            'year': now.year,
            'is_weekend': now.weekday() >= 5,
            'is_business_hours': 9 <= now.hour <= 17,
            'season': self._get_season(now.month),
            'uptime_seconds': time.time() - psutil.boot_time()
        }

    async def _get_environmental_data(self) -> Dict[str, Any]:
        """Obter dados ambientais (simulados baseados em contexto)"""
        # Como não temos sensores reais, usar contexto do sistema
        cpu_temp = None
        try:
            # Tentar obter temperatura se disponível
            temps = psutil.sensors_temperatures()
            if temps and 'coretemp' in temps:
                cpu_temp = temps['coretemp'][0].current
        except:
            pass

        return await {
            'cpu_temperature': cpu_temp,
            'system_load': psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else None,
            'battery_percent': psutil.sensors_battery().percent if psutil.sensors_battery() else None,
            'fans_speed': None,  # Não disponível na maioria dos sistemas
            'ambient_noise': None,  # Simulado
            'light_level': None  # Simulado
        }

    async def _get_season(self, month: int) -> str:
        """Determinar estação do ano"""
        if month in [12, 1, 2]:
            return await 'winter'
        elif month in [3, 4, 5]:
            return await 'spring'
        elif month in [6, 7, 8]:
            return await 'summer'
        else:
            return await 'fall'

    async def get_anomalies(self) -> List[Dict[str, Any]]:
        """Detectar anomalias nos dados dos sensores"""
        anomalies = []
        current_data = self.collect_all_data()

        for sensor_name, sensor_data in current_data['sensors'].items():
            if sensor_name in self.history and len(self.history[sensor_name]) > 10:
                # Calcular baseline (média das últimas 10 leituras)
                recent_data = self.history[sensor_name][-10:]

                for key, value in sensor_data.items():
                    if isinstance(value, (int, float)) and value is not None:
                        # Calcular média e desvio padrão
                        values = [d.get(key) for d in recent_data if isinstance(d.get(key), (int, float))]
                        if values:
                            mean = sum(values) / len(values)
                            variance = sum((v - mean) ** 2 for v in values) / len(values)
                            std_dev = variance ** 0.5 if variance > 0 else 1

                            # Detectar anomalia (3 desvios padrão)
                            if abs(value - mean) > 3 * std_dev:
                                anomalies.append({
                                    'sensor': sensor_name,
                                    'metric': key,
                                    'value': value,
                                    'mean': mean,
                                    'std_dev': std_dev,
                                    'severity': 'high' if abs(value - mean) > 5 * std_dev else 'medium',
                                    'timestamp': current_data['timestamp']
                                })

        return await anomalies

    async def get_trends(self) -> Dict[str, Any]:
        """Analisar tendências nos dados"""
        trends = {}

        for sensor_name, history in self.history.items():
            if len(history) >= 20:
                trends[sensor_name] = {}

                # Analisar cada métrica
                for key in history[0].keys():
                    values = [h.get(key) for h in history[-20:] if isinstance(h.get(key), (int, float))]

                    if len(values) >= 10:
                        # Calcular tendência linear simples
                        x = list(range(len(values)))
                        slope = self._calculate_slope(x, values)

                        trend_direction = 'stable'
                        if slope > 0.1:
                            trend_direction = 'increasing'
                        elif slope < -0.1:
                            trend_direction = 'decreasing'

                        trends[sensor_name][key] = {
                            'slope': slope,
                            'direction': trend_direction,
                            'recent_avg': sum(values[-5:]) / 5,
                            'overall_avg': sum(values) / len(values)
                        }

        return await trends

    async def _calculate_slope(self, x: List[float], y: List[float]) -> float:
        """Calcular inclinação da linha de tendência"""
        if len(x) != len(y) or len(x) < 2:
            return await 0

        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(xi * yi for xi, yi in zip(x, y))
        sum_x2 = sum(xi ** 2 for xi in x)

        denominator = n * sum_x2 - sum_x ** 2
        if denominator == 0:
            return await 0

        slope = (n * sum_xy - sum_x * sum_y) / denominator
        return await slope

class FeedbackProcessor:
    """
    Processador que converte dados reais em feedback para evolução
    """

    async def __init__(self, sensor: RealWorldSensor):
        self.sensor = sensor
        self.feedback_rules = self._load_feedback_rules()
        self.feedback_history = []

    async def _load_feedback_rules(self) -> Dict[str, Any]:
        """Carregar regras de feedback"""
        return await {
            'resource_pressure': {
                'cpu_percent': {'threshold': 80, 'action': 'reduce_complexity', 'weight': 0.8},
                'memory_percent': {'threshold': 85, 'action': 'optimize_memory', 'weight': 0.9},
                'disk_percent': {'threshold': 90, 'action': 'cleanup_files', 'weight': 0.7}
            },
            'network_opportunities': {
                'internet_access': {'value': True, 'action': 'expand_connectivity', 'weight': 0.6},
                'active_connections': {'threshold': 10, 'action': 'increase_interaction', 'weight': 0.5}
            },
            'temporal_patterns': {
                'is_business_hours': {'value': True, 'action': 'increase_activity', 'weight': 0.4},
                'is_weekend': {'value': True, 'action': 'enable_maintenance', 'weight': 0.3}
            },
            'anomaly_response': {
                'high_severity': {'action': 'emergency_adaptation', 'weight': 1.0},
                'medium_severity': {'action': 'gradual_adaptation', 'weight': 0.7}
            }
        }

    async def process_feedback(self) -> Dict[str, Any]:
        """Processar feedback completo"""
        sensor_data = self.sensor.collect_all_data()
        anomalies = self.sensor.get_anomalies()
        trends = self.sensor.get_trends()

        feedback = {
            'timestamp': datetime.now().isoformat(),
            'sensor_data': sensor_data,
            'anomalies': anomalies,
            'trends': trends,
            'actions': self._generate_actions(sensor_data, anomalies, trends),
            'adaptation_signals': self._generate_adaptation_signals(sensor_data, trends)
        }

        self.feedback_history.append(feedback)

        # Limitar histórico
        if len(self.feedback_history) > 50:
            self.feedback_history = self.feedback_history[-50:]

        return await feedback

    async def _generate_actions(self, sensor_data: Dict[str, Any], anomalies: List[Dict[str, Any]], trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Gerar ações baseadas nos dados"""
        actions = []

        # Processar regras de feedback
        for category, rules in self.feedback_rules.items():
            if category == 'resource_pressure':
                actions.extend(self._process_resource_rules(sensor_data, rules))
            elif category == 'network_opportunities':
                actions.extend(self._process_network_rules(sensor_data, rules))
            elif category == 'temporal_patterns':
                actions.extend(self._process_temporal_rules(sensor_data, rules))
            elif category == 'anomaly_response':
                actions.extend(self._process_anomaly_rules(anomalies, rules))

        # Remover duplicatas e ordenar por prioridade
        unique_actions = {}
        for action in actions:
            key = action['action']
            if key not in unique_actions or action['priority'] > unique_actions[key]['priority']:
                unique_actions[key] = action

        return await sorted(unique_actions.values(), key=lambda x: x['priority'], reverse=True)

    async def _process_resource_rules(self, sensor_data: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Processar regras de pressão de recursos"""
        actions = []

        system_resources = sensor_data['sensors'].get('system_resources', {})

        for metric, rule in rules.items():
            value = system_resources.get(metric)
            if value is not None and value > rule['threshold']:
                actions.append({
                    'action': rule['action'],
                    'reason': f'{metric} at {value:.1f}% (threshold: {rule["threshold"]}%)',
                    'priority': rule['weight'] * (value - rule['threshold']) / 100,
                    'category': 'resource_management'
                })

        return await actions

    async def _process_network_rules(self, sensor_data: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Processar regras de oportunidades de rede"""
        actions = []

        network_data = sensor_data['sensors'].get('network_activity', {})

        for metric, rule in rules.items():
            value = network_data.get(metric)
            if value is not None:
                if metric == 'internet_access' and value == rule['value']:
                    actions.append({
                        'action': rule['action'],
                        'reason': 'Internet access available',
                        'priority': rule['weight'],
                        'category': 'connectivity'
                    })
                elif isinstance(rule.get('threshold'), (int, float)) and value > rule['threshold']:
                    actions.append({
                        'action': rule['action'],
                        'reason': f'{metric} at {value} (threshold: {rule["threshold"]})',
                        'priority': rule['weight'],
                        'category': 'connectivity'
                    })

        return await actions

    async def _process_temporal_rules(self, sensor_data: Dict[str, Any], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Processar regras temporais"""
        actions = []

        time_data = sensor_data['sensors'].get('time_patterns', {})

        for metric, rule in rules.items():
            value = time_data.get(metric)
            if value is not None and value == rule['value']:
                actions.append({
                    'action': rule['action'],
                    'reason': f'Temporal condition: {metric}',
                    'priority': rule['weight'],
                    'category': 'temporal_adaptation'
                })

        return await actions

    async def _process_anomaly_rules(self, anomalies: List[Dict[str, Any]], rules: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Processar regras de resposta a anomalias"""
        actions = []

        severity_counts = {'high': 0, 'medium': 0, 'low': 0}
        for anomaly in anomalies:
            severity_counts[anomaly['severity']] += 1

        for severity, count in severity_counts.items():
            if count > 0 and severity in rules:
                rule = rules[severity]
                actions.append({
                    'action': rule['action'],
                    'reason': f'{count} {severity} severity anomalies detected',
                    'priority': rule['weight'] * min(count, 5) / 5,  # Escalar com número de anomalias
                    'category': 'anomaly_response'
                })

        return await actions

    async def _generate_adaptation_signals(self, sensor_data: Dict[str, Any], trends: Dict[str, Any]) -> Dict[str, Any]:
        """Gerar sinais de adaptação baseados em tendências"""
        signals = {
            'complexity_adjustment': 0.0,
            'resource_allocation': 0.0,
            'learning_rate': 0.0,
            'exploration_rate': 0.0
        }

        # Ajustar complexidade baseada em recursos
        system_resources = sensor_data['sensors'].get('system_resources', {})
        cpu_pressure = system_resources.get('cpu_percent', 0) / 100
        memory_pressure = system_resources.get('memory_percent', 0) / 100

        resource_pressure = (cpu_pressure + memory_pressure) / 2
        signals['complexity_adjustment'] = -resource_pressure  # Reduzir complexidade se recursos sob pressão

        # Ajustar alocação de recursos baseada em tendências
        if 'system_resources' in trends:
            cpu_trend = trends['system_resources'].get('cpu_percent', {}).get('direction', 'stable')
            if cpu_trend == 'increasing':
                signals['resource_allocation'] = 0.2  # Aumentar alocação
            elif cpu_trend == 'decreasing':
                signals['resource_allocation'] = -0.1  # Reduzir alocação

        # Ajustar taxas de aprendizado baseadas em conectividade
        network_data = sensor_data['sensors'].get('network_activity', {})
        if network_data.get('internet_access'):
            signals['learning_rate'] = 0.1  # Aumentar aprendizado com internet
            signals['exploration_rate'] = 0.1  # Aumentar exploração

        # Ajustar baseado em padrões temporais
        time_data = sensor_data['sensors'].get('time_patterns', {})
        if time_data.get('is_business_hours'):
            signals['exploration_rate'] = 0.2  # Mais exploração durante horário comercial

        return await signals

class RealFeedbackEngine:
    """
    Motor principal de feedback real
    """

    async def __init__(self):
        self.sensor = RealWorldSensor()
        self.processor = FeedbackProcessor(self.sensor)
        self.is_active = True
        self.feedback_subscribers = []

    async def start_real_feedback_loop(self):
        """Iniciar loop de feedback real"""
        logger.info("🌍 Iniciando loop de feedback real")

        async def feedback_loop():
            cycle_count = 0
            while self.is_active:
                try:
                    cycle_count += 1

                    # Coletar e processar feedback
                    feedback = self.processor.process_feedback()

                    # Notificar subscribers
                    self._notify_subscribers(feedback)

                    # Executar ações recomendadas
                    self._execute_feedback_actions(feedback)

                    # Log periódico
                    if cycle_count % 10 == 0:
                        logger.info(f"🔄 Ciclo {cycle_count} | Anomalias: {len(feedback['anomalies'])} | Ações: {len(feedback['actions'])}")

                    time.sleep(30)  # Coletar a cada 30 segundos

                except Exception as e:
                    logger.error(f"Erro no loop de feedback: {e}")
                    time.sleep(10)

        thread = threading.Thread(target=feedback_loop, daemon=True)
        thread.start()

    async def subscribe_to_feedback(self, callback: Callable):
        """Inscrever callback para receber feedback"""
        self.feedback_subscribers.append(callback)
        logger.info(f"📡 Subscriber adicionado: {len(self.feedback_subscribers)} total")

    async def _notify_subscribers(self, feedback: Dict[str, Any]):
        """Notificar todos os subscribers"""
        for subscriber in self.feedback_subscribers:
            try:
                subscriber(feedback)
            except Exception as e:
                logger.warning(f"Erro ao notificar subscriber: {e}")

    async def _execute_feedback_actions(self, feedback: Dict[str, Any]):
        """Executar ações recomendadas pelo feedback"""
        actions = feedback.get('actions', [])

        for action in actions[:5]:  # Executar top 5 ações
            try:
                self._execute_action(action)
                logger.debug(f"✅ Ação executada: {action['action']}")
            except Exception as e:
                logger.warning(f"Erro ao executar ação {action['action']}: {e}")

    async def _execute_action(self, action: Dict[str, Any]):
        """Executar ação específica"""
        action_type = action['action']

        if action_type == 'reduce_complexity':
            self._reduce_system_complexity()
        elif action_type == 'optimize_memory':
            self._optimize_memory_usage()
        elif action_type == 'cleanup_files':
            self._cleanup_old_files()
        elif action_type == 'expand_connectivity':
            self._expand_network_connectivity()
        elif action_type == 'increase_activity':
            self._increase_system_activity()
        elif action_type == 'enable_maintenance':
            self._enable_maintenance_mode()
        elif action_type == 'emergency_adaptation':
            self._emergency_adaptation()
        elif action_type == 'gradual_adaptation':
            self._gradual_adaptation()

    async def _reduce_system_complexity(self):
        """Reduzir complexidade do sistema"""
        # Matar processos não essenciais
        try:
            processes = list(psutil.process_iter(['pid', 'name', 'cpu_percent']))
            python_processes = [p for p in processes if 'python' in p.info['name'].lower()]

            # Manter apenas os processos mais importantes
            if len(python_processes) > 5:
                to_kill = python_processes[5:]  # Manter top 5
                for proc in to_kill:
                    try:
                        proc.kill()
                        logger.info(f"🗑️ Processo terminado por redução de complexidade: {proc.info['name']}")
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Erro ao reduzir complexidade: {e}")

    async def _optimize_memory_usage(self):
        """Otimizar uso de memória"""
        try:
            # Limpar cache se possível
            import gc
            gc.collect()

            # Log de otimização
            memory_before = psutil.virtual_memory().percent
            logger.info(f"🧹 Otimização de memória executada (antes: {memory_before:.1f}%)")
        except Exception as e:
            logger.warning(f"Erro na otimização de memória: {e}")

    async def _cleanup_old_files(self):
        """Limpar arquivos antigos"""
        try:
            cleanup_count = 0
            for file in os.listdir('.'):
                if file.endswith(('.log', '.tmp', '.cache')):
                    try:
                        # Verificar se arquivo é antigo (> 1 dia)
                        if time.time() - os.path.getmtime(file) > 86400:
                            os.remove(file)
                            cleanup_count += 1
                    except:
                        pass

            if cleanup_count > 0:
                logger.info(f"🗑️ {cleanup_count} arquivos antigos removidos")
        except Exception as e:
            logger.warning(f"Erro na limpeza de arquivos: {e}")

    async def _expand_network_connectivity(self):
        """Expandir conectividade de rede"""
        # Implementação básica - apenas log
        logger.info("🌐 Expansão de conectividade solicitada")

    async def _increase_system_activity(self):
        """Aumentar atividade do sistema"""
        # Implementação básica - apenas log
        logger.info("⚡ Aumento de atividade solicitado")

    async def _enable_maintenance_mode(self):
        """Habilitar modo de manutenção"""
        # Implementação básica - apenas log
        logger.info("🔧 Modo de manutenção habilitado")

    async def _emergency_adaptation(self):
        """Adaptação de emergência"""
        logger.warning("🚨 Adaptação de emergência executada")

    async def _gradual_adaptation(self):
        """Adaptação gradual"""
        logger.info("🔄 Adaptação gradual executada")

    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Obter estatísticas de feedback"""
        return await {
            'total_feedback_cycles': len(self.processor.feedback_history),
            'anomalies_detected': sum(len(f['anomalies']) for f in self.processor.feedback_history),
            'actions_executed': sum(len(f['actions']) for f in self.processor.feedback_history),
            'subscribers': len(self.feedback_subscribers),
            'sensor_history_size': {k: len(v) for k, v in self.sensor.history.items()}
        }

async def main():
    """Função principal"""
    engine = RealFeedbackEngine()
    engine.start_real_feedback_loop()

    # Exemplo de subscriber
    async def feedback_handler(feedback):
        anomalies = feedback.get('anomalies', [])
        if anomalies:
            print(f"⚠️ {len(anomalies)} anomalias detectadas")

    engine.subscribe_to_feedback(feedback_handler)

    # Manter ativo
    try:
        while True:
            time.sleep(60)
            stats = engine.get_feedback_stats()
            print(f"🌍 Feedback stats: {stats['total_feedback_cycles']} cycles, {stats['anomalies_detected']} anomalies")

    except KeyboardInterrupt:
        print("🛑 Parando feedback real...")
        engine.is_active = False

if __name__ == "__main__":
    main()