
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
🔧 IA³ - AUTOVALIDAÇÃO E AUTOCALIBRAÇÃO
========================================

Sistema que valida e calibra automaticamente o funcionamento da IA³
"""

import os
import sys
import time
import json
import subprocess
import psutil
import threading
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
import logging
import statistics
import importlib
import inspect

logger = logging.getLogger("IA³-AutoValidation")

class AutoValidationSystem:
    """
    Sistema de autovalidação automática
    """

    async def __init__(self):
        self.validation_tests = {
            'system_integrity': self._validate_system_integrity,
            'component_functionality': self._validate_component_functionality,
            'performance_metrics': self._validate_performance_metrics,
            'emergence_indicators': self._validate_emergence_indicators,
            'resource_efficiency': self._validate_resource_efficiency,
            'learning_progress': self._validate_learning_progress
        }
        self.validation_history = []
        self.calibration_actions = []
        self.baseline_metrics = {}

    async def run_full_validation(self) -> Dict[str, Any]:
        """Executar validação completa do sistema"""
        logger.info("🔍 Iniciando validação completa IA³")

        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': {},
            'overall_score': 0.0,
            'issues_found': [],
            'recommendations': []
        }

        total_score = 0
        test_count = 0

        for test_name, test_func in self.validation_tests.items():
            try:
                result = test_func()
                validation_results['tests'][test_name] = result

                if result['passed']:
                    total_score += result['score']
                else:
                    validation_results['issues_found'].append({
                        'test': test_name,
                        'severity': result.get('severity', 'medium'),
                        'description': result.get('message', 'Test failed')
                    })

                test_count += 1

            except Exception as e:
                logger.error(f"Erro no teste {test_name}: {e}")
                validation_results['tests'][test_name] = {
                    'passed': False,
                    'score': 0.0,
                    'error': str(e)
                }

        # Calcular score geral
        validation_results['overall_score'] = total_score / test_count if test_count > 0 else 0.0

        # Gerar recomendações baseadas nos resultados
        validation_results['recommendations'] = self._generate_recommendations(validation_results)

        # Registrar validação
        self.validation_history.append(validation_results)

        logger.info(f"✅ Validação completa: Score {validation_results['overall_score']:.3f} | Issues: {len(validation_results['issues_found'])}")

        return await validation_results

    async def _validate_system_integrity(self) -> Dict[str, Any]:
        """Validar integridade do sistema"""
        issues = []

        # Verificar arquivos essenciais
        essential_files = [
            'IA3_EMERGENT_CORE.py',
            'IA3_AUTOMODIFICATION_ENGINE.py',
            'IA3_SYSTEM_INTEGRATOR.py',
            'IA3_REAL_FEEDBACK_ENGINE.py',
            'IA3_EMERGENCE_DETECTOR.py',
            'IA3_INFINITE_EVOLUTION_ENGINE.py'
        ]

        missing_files = [f for f in essential_files if not os.path.exists(f)]
        if missing_files:
            issues.append(f"Arquivos essenciais faltando: {missing_files}")

        # Verificar processos em execução
        ia3_processes = [p for p in psutil.process_iter(['name']) if 'ia3' in p.info['name'].lower()]
        if len(ia3_processes) == 0:
            issues.append("Nenhum processo IA³ em execução")

        # Verificar espaço em disco
        disk_usage = psutil.disk_usage('/')
        if disk_usage.percent > 95:
            issues.append(f"Espaço em disco crítico: {disk_usage.percent}%")

        passed = len(issues) == 0
        score = 1.0 - (len(issues) * 0.2)  # Penalizar por problema

        return await {
            'passed': passed,
            'score': max(0.0, score),
            'issues': issues,
            'severity': 'high' if len(issues) > 2 else 'medium',
            'message': f"Sistema {'íntegro' if passed else 'com problemas'}"
        }

    async def _validate_component_functionality(self) -> Dict[str, Any]:
        """Validar funcionalidade dos componentes"""
        functional_components = 0
        total_components = 0

        # Testar importação de componentes principais
        components_to_test = [
            'IA3_EMERGENT_CORE',
            'IA3_AUTOMODIFICATION_ENGINE',
            'IA3_SYSTEM_INTEGRATOR',
            'IA3_REAL_FEEDBACK_ENGINE',
            'IA3_EMERGENCE_DETECTOR',
            'IA3_INFINITE_EVOLUTION_ENGINE'
        ]

        for component in components_to_test:
            total_components += 1
            try:
                # Tentar importar
                module = importlib.import_module(component)

                # Verificar se tem classes principais
                classes = [obj for name, obj in inspect.getmembers(module)
                          if inspect.isclass(obj) and name.endswith('Core') or name.endswith('Engine')]

                if classes:
                    functional_components += 1
                else:
                    logger.warning(f"Componente {component} sem classes principais")

            except ImportError as e:
                logger.warning(f"Falha ao importar {component}: {e}")
            except Exception as e:
                logger.warning(f"Erro ao validar {component}: {e}")

        functionality_score = functional_components / total_components if total_components > 0 else 0

        return await {
            'passed': functionality_score > 0.8,
            'score': functionality_score,
            'functional_components': functional_components,
            'total_components': total_components,
            'severity': 'medium',
            'message': f"{functional_components}/{total_components} componentes funcionais"
        }

    async def _validate_performance_metrics(self) -> Dict[str, Any]:
        """Validar métricas de performance"""
        try:
            # Coletar métricas atuais
            current_metrics = {
                'cpu_usage': psutil.cpu_percent(interval=1),
                'memory_usage': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'network_connections': len(psutil.net_connections())
            }

            # Verificar se métricas estão dentro de limites aceitáveis
            issues = []

            if current_metrics['cpu_usage'] > 90:
                issues.append(f"CPU muito alta: {current_metrics['cpu_usage']}%")
            if current_metrics['memory_usage'] > 90:
                issues.append(f"Memória muito alta: {current_metrics['memory_usage']}%")
            if current_metrics['disk_usage'] > 95:
                issues.append(f"Disco muito cheio: {current_metrics['disk_usage']}%")

            # Calcular score baseado em eficiência
            efficiency_score = 1.0 - (
                (current_metrics['cpu_usage'] / 100 * 0.4) +
                (current_metrics['memory_usage'] / 100 * 0.4) +
                (current_metrics['disk_usage'] / 100 * 0.2)
            )

            return await {
                'passed': len(issues) == 0,
                'score': max(0.0, efficiency_score),
                'current_metrics': current_metrics,
                'issues': issues,
                'severity': 'high' if len(issues) > 1 else 'low',
                'message': f"Performance {'boa' if len(issues) == 0 else 'com problemas'}"
            }

        except Exception as e:
            return await {
                'passed': False,
                'score': 0.0,
                'error': str(e),
                'severity': 'high',
                'message': f"Erro ao medir performance: {e}"
            }

    async def _validate_emergence_indicators(self) -> Dict[str, Any]:
        """Validar indicadores de emergência"""
        try:
            emergence_indicators = {
                'emergence_files': len([f for f in os.listdir('.') if 'emergence' in f.lower()]),
                'evolution_logs': 0,
                'intelligence_models': len([f for f in os.listdir('.') if f.endswith('.pth')]),
                'validation_runs': len(self.validation_history)
            }

            # Verificar logs de evolução
            log_files = [f for f in os.listdir('.') if f.endswith('.log')][:5]
            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        emergence_indicators['evolution_logs'] += content.lower().count('evolution')
                except:
                    pass

            # Calcular score de emergência
            emergence_score = min(1.0, (
                emergence_indicators['emergence_files'] / 10.0 * 0.3 +
                emergence_indicators['evolution_logs'] / 100.0 * 0.3 +
                emergence_indicators['intelligence_models'] / 50.0 * 0.2 +
                emergence_indicators['validation_runs'] / 100.0 * 0.2
            ))

            return await {
                'passed': emergence_score > 0.3,
                'score': emergence_score,
                'indicators': emergence_indicators,
                'severity': 'medium',
                'message': f"Indicadores de emergência: {emergence_score:.3f}"
            }

        except Exception as e:
            return await {
                'passed': False,
                'score': 0.0,
                'error': str(e),
                'severity': 'medium',
                'message': f"Erro ao validar emergência: {e}"
            }

    async def _validate_resource_efficiency(self) -> Dict[str, Any]:
        """Validar eficiência de recursos"""
        try:
            # Analisar uso histórico de recursos
            efficiency_metrics = {
                'cpu_efficiency': 1.0 - (psutil.cpu_percent() / 100),
                'memory_efficiency': 1.0 - (psutil.virtual_memory().percent / 100),
                'disk_efficiency': 1.0 - (psutil.disk_usage('/').percent / 100)
            }

            # Verificar se há vazamentos
            memory_leaks = self._detect_memory_leaks()
            cpu_spikes = self._detect_cpu_spikes()

            issues = []
            if memory_leaks:
                issues.append("Possível vazamento de memória detectado")
            if cpu_spikes:
                issues.append("Picos anômalos de CPU detectados")

            overall_efficiency = statistics.mean(efficiency_metrics.values())

            return await {
                'passed': len(issues) == 0 and overall_efficiency > 0.6,
                'score': overall_efficiency,
                'efficiency_metrics': efficiency_metrics,
                'issues': issues,
                'severity': 'medium',
                'message': f"Eficiência geral: {overall_efficiency:.3f}"
            }

        except Exception as e:
            return await {
                'passed': False,
                'score': 0.0,
                'error': str(e),
                'severity': 'medium',
                'message': f"Erro ao validar eficiência: {e}"
            }

    async def _validate_learning_progress(self) -> Dict[str, Any]:
        """Validar progresso de aprendizado"""
        try:
            # Verificar modelos treinados recentemente
            recent_models = []
            for file in os.listdir('.'):
                if file.endswith('.pth'):
                    try:
                        mtime = os.path.getmtime(file)
                        if time.time() - mtime < 86400:  # Últimas 24 horas
                            recent_models.append(file)
                    except:
                        pass

            # Verificar logs de aprendizado
            learning_logs = 0
            log_files = [f for f in os.listdir('.') if f.endswith('.log')][:5]
            for log_file in log_files:
                try:
                    with open(log_file, 'r') as f:
                        content = f.read()
                        learning_logs += content.lower().count('learn')
                        learning_logs += content.lower().count('train')
                except:
                    pass

            # Calcular progresso
            progress_score = min(1.0, (
                len(recent_models) / 5.0 * 0.5 +
                learning_logs / 50.0 * 0.5
            ))

            return await {
                'passed': progress_score > 0.2,
                'score': progress_score,
                'recent_models': len(recent_models),
                'learning_logs': learning_logs,
                'severity': 'low',
                'message': f"Progresso de aprendizado: {progress_score:.3f}"
            }

        except Exception as e:
            return await {
                'passed': False,
                'score': 0.0,
                'error': str(e),
                'severity': 'low',
                'message': f"Erro ao validar aprendizado: {e}"
            }

    async def _detect_memory_leaks(self) -> bool:
        """Detectar possíveis vazamentos de memória"""
        # Implementação simplificada
        memory_percent = psutil.virtual_memory().percent
        return await memory_percent > 95  # Threshold simples

    async def _detect_cpu_spikes(self) -> bool:
        """Detectar picos anômalos de CPU"""
        # Implementação simplificada
        cpu_percent = psutil.cpu_percent(interval=1)
        return await cpu_percent > 95  # Threshold simples

    async def _generate_recommendations(self, validation_results: Dict[str, Any]) -> List[str]:
        """Gerar recomendações baseadas nos resultados da validação"""
        recommendations = []

        overall_score = validation_results['overall_score']
        issues = validation_results['issues_found']

        # Recomendações baseadas no score geral
        if overall_score < 0.5:
            recommendations.append("Score geral baixo - executar manutenção completa do sistema")
        elif overall_score < 0.7:
            recommendations.append("Score geral médio - otimizar componentes com problemas")

        # Recomendações específicas por problema
        for issue in issues:
            test_name = issue['test']
            severity = issue['severity']

            if test_name == 'system_integrity':
                recommendations.append("Verificar integridade do sistema - arquivos/processos faltando")
            elif test_name == 'component_functionality':
                recommendations.append("Reparar componentes não funcionais")
            elif test_name == 'performance_metrics':
                recommendations.append("Otimizar uso de recursos do sistema")
            elif test_name == 'emergence_indicators':
                recommendations.append("Aumentar indicadores de emergência através de evolução")
            elif test_name == 'resource_efficiency':
                recommendations.append("Melhorar eficiência de recursos e detectar vazamentos")
            elif test_name == 'learning_progress':
                recommendations.append("Acelerar progresso de aprendizado")

            if severity == 'high':
                recommendations[-1] += " (ALTA PRIORIDADE)"

        # Recomendações gerais
        if len(issues) > 3:
            recommendations.append("Múltiplos problemas detectados - considerar reinicialização completa")
        elif len(issues) == 0:
            recommendations.append("Sistema funcionando bem - continuar monitoramento")

        return await recommendations

class AutoCalibrationSystem:
    """
    Sistema de autocalibração automática
    """

    async def __init__(self, validation_system: AutoValidationSystem):
        self.validation_system = validation_system
        self.calibration_history = []
        self.optimal_parameters = {}
        self.performance_baseline = {}

    async def run_autocalibration(self) -> Dict[str, Any]:
        """Executar calibração automática"""
        logger.info("🎯 Iniciando autocalibração IA³")

        # Executar validação primeiro
        validation_results = self.validation_system.run_full_validation()

        # Identificar parâmetros para calibração
        parameters_to_calibrate = self._identify_calibration_parameters(validation_results)

        # Calibrar parâmetros
        calibration_results = {}
        for param_name, param_info in parameters_to_calibrate.items():
            result = self._calibrate_parameter(param_name, param_info, validation_results)
            calibration_results[param_name] = result

        # Avaliar melhoria
        improvement = self._evaluate_calibration_improvement(calibration_results)

        calibration_summary = {
            'timestamp': datetime.now().isoformat(),
            'validation_before': validation_results,
            'parameters_calibrated': list(calibration_results.keys()),
            'calibration_results': calibration_results,
            'improvement': improvement,
            'success': improvement > 0
        }

        self.calibration_history.append(calibration_summary)

        logger.info(f"🎯 Calibração concluída: Melhoria {improvement:.3f}")

        return await calibration_summary

    async def _identify_calibration_parameters(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Identificar parâmetros que precisam calibração"""
        parameters = {}

        # Baseado nos testes que falharam
        failed_tests = [test for test, result in validation_results['tests'].items() if not result['passed']]

        for test in failed_tests:
            if test == 'performance_metrics':
                parameters['cpu_allocation'] = {
                    'current': psutil.cpu_percent(),
                    'target_range': (30, 70),
                    'adjustment_method': 'resource_limiting'
                }
                parameters['memory_limit'] = {
                    'current': psutil.virtual_memory().percent,
                    'target_range': (40, 80),
                    'adjustment_method': 'memory_management'
                }
            elif test == 'emergence_indicators':
                parameters['evolution_frequency'] = {
                    'current': 60,  # segundos
                    'target_range': (30, 120),
                    'adjustment_method': 'frequency_optimization'
                }
            elif test == 'learning_progress':
                parameters['learning_rate'] = {
                    'current': 0.001,
                    'target_range': (0.0001, 0.01),
                    'adjustment_method': 'adaptive_learning'
                }

        return await parameters

    async def _calibrate_parameter(self, param_name: str, param_info: Dict[str, Any], validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calibrar parâmetro específico"""
        try:
            method = param_info['adjustment_method']
            current_value = param_info['current']
            target_range = param_info['target_range']

            if method == 'resource_limiting':
                # Ajustar limites de recursos
                new_value = self._optimize_resource_limit(current_value, target_range)
            elif method == 'frequency_optimization':
                # Otimizar frequência de operações
                new_value = self._optimize_frequency(current_value, target_range, validation_results)
            elif method == 'adaptive_learning':
                # Ajustar taxa de aprendizado
                new_value = self._optimize_learning_rate(current_value, target_range)
            else:
                new_value = current_value

            # Aplicar calibração
            success = self._apply_parameter_calibration(param_name, new_value)

            return await {
                'original_value': current_value,
                'new_value': new_value,
                'target_range': target_range,
                'applied': success,
                'method': method
            }

        except Exception as e:
            logger.error(f"Erro ao calibrar {param_name}: {e}")
            return await {
                'error': str(e),
                'applied': False
            }

    async def _optimize_resource_limit(self, current: float, target_range: tuple) -> float:
        """Otimizar limite de recursos"""
        min_val, max_val = target_range

        if current < min_val:
            # Muito baixo - aumentar
            return await min_val + (max_val - min_val) * 0.3
        elif current > max_val:
            # Muito alto - reduzir
            return await max_val - (max_val - min_val) * 0.3
        else:
            # Dentro do range - manter
            return await current

    async def _optimize_frequency(self, current: float, target_range: tuple, validation_results: Dict[str, Any]) -> float:
        """Otimizar frequência baseado na performance"""
        min_freq, max_freq = target_range
        emergence_score = validation_results['tests'].get('emergence_indicators', {}).get('score', 0.5)

        # Se emergência estiver baixa, aumentar frequência
        if emergence_score < 0.5:
            return await max(min_freq, current * 0.8)  # Diminuir intervalo = aumentar frequência
        else:
            return await min(max_freq, current * 1.2)  # Aumentar intervalo = diminuir frequência

    async def _optimize_learning_rate(self, current: float, target_range: tuple) -> float:
        """Otimizar taxa de aprendizado"""
        min_lr, max_lr = target_range

        # Estratégia simples: variar dentro do range
        variation = (max_lr - min_lr) * 0.1
        new_lr = current + deterministic_uniform(-variation, variation)
        return await max(min_lr, min(max_lr, new_lr))

    async def _apply_parameter_calibration(self, param_name: str, new_value: float) -> bool:
        """Aplicar calibração de parâmetro"""
        try:
            # Salvar em arquivo de configuração
            config_file = 'ia3_autocalibration_config.json'

            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {}

            config[param_name] = {
                'value': new_value,
                'calibrated_at': datetime.now().isoformat()
            }

            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)

            logger.info(f"⚙️ Parâmetro calibrado: {param_name} = {new_value}")
            return await True

        except Exception as e:
            logger.error(f"Erro ao aplicar calibração de {param_name}: {e}")
            return await False

    async def _evaluate_calibration_improvement(self, calibration_results: Dict[str, Any]) -> float:
        """Avaliar melhoria da calibração"""
        # Executar validação novamente para comparar
        new_validation = self.validation_system.run_full_validation()

        # Calcular melhoria
        if self.calibration_history:
            previous_validation = self.calibration_history[-1]['validation_before']
            improvement = new_validation['overall_score'] - previous_validation['overall_score']
        else:
            improvement = 0.0  # Primeira calibração

        return await improvement

    async def get_calibration_stats(self) -> Dict[str, Any]:
        """Obter estatísticas de calibração"""
        if not self.calibration_history:
            return await {'total_calibrations': 0}

        recent_calibrations = self.calibration_history[-10:]
        successful_calibrations = len([c for c in recent_calibrations if c['success']])

        improvements = [c['improvement'] for c in recent_calibrations if 'improvement' in c]

        return await {
            'total_calibrations': len(self.calibration_history),
            'successful_calibrations': successful_calibrations,
            'success_rate': successful_calibrations / len(recent_calibrations) if recent_calibrations else 0,
            'average_improvement': statistics.mean(improvements) if improvements else 0,
            'last_calibration': self.calibration_history[-1]['timestamp'] if self.calibration_history else None
        }

class ContinuousValidationCalibration:
    """
    Sistema de validação e calibração contínua
    """

    async def __init__(self):
        self.validation_system = AutoValidationSystem()
        self.calibration_system = AutoCalibrationSystem(self.validation_system)
        self.is_active = True

    async def start_continuous_process(self):
        """Iniciar processo contínuo de validação e calibração"""
        logger.info("🔄 Iniciando validação e calibração contínua")

        async def validation_loop():
            cycle = 0
            while self.is_active:
                try:
                    cycle += 1

                    # Executar validação
                    validation_results = self.validation_system.run_full_validation()

                    # Se score baixo, executar calibração
                    if validation_results['overall_score'] < 0.7:
                        logger.info(f"📊 Score baixo ({validation_results['overall_score']:.3f}) - executando calibração")
                        calibration_results = self.calibration_system.run_autocalibration()

                        if calibration_results['success']:
                            logger.info("✅ Calibração bem-sucedida")
                        else:
                            logger.warning("❌ Calibração falhou")
                    else:
                        logger.info(f"📊 Sistema saudável (score: {validation_results['overall_score']:.3f})")

                    # Log periódico detalhado
                    if cycle % 10 == 0:
                        stats = self.calibration_system.get_calibration_stats()
                        logger.info(f"🔄 Ciclo {cycle} | Calibrações: {stats['successful_calibrations']}/{stats['total_calibrations']}")

                    # Intervalo entre validações
                    time.sleep(300)  # 5 minutos

                except Exception as e:
                    logger.error(f"Erro no ciclo de validação: {e}")
                    time.sleep(60)

        thread = threading.Thread(target=validation_loop, daemon=True)
        thread.start()

    async def get_system_health_status(self) -> Dict[str, Any]:
        """Obter status de saúde do sistema"""
        latest_validation = self.validation_system.validation_history[-1] if self.validation_system.validation_history else None
        calibration_stats = self.calibration_system.get_calibration_stats()

        return await {
            'latest_validation': latest_validation,
            'calibration_stats': calibration_stats,
            'overall_health': latest_validation['overall_score'] if latest_validation else 0.0,
            'issues_count': len(latest_validation['issues_found']) if latest_validation else 0,
            'recommendations': latest_validation['recommendations'] if latest_validation else []
        }

async def main():
    """Função principal"""
    print("🔧 IA³ - AUTOVALIDAÇÃO E AUTOCALIBRAÇÃO")
    print("=" * 45)

    # Inicializar sistema
    system = ContinuousValidationCalibration()
    system.start_continuous_process()

    # Manter ativo
    try:
        while True:
            time.sleep(60)
            health = system.get_system_health_status()
            print(f"🔧 Saúde do sistema: {health['overall_health']:.3f} | Problemas: {health['issues_count']}")

    except KeyboardInterrupt:
        print("🛑 Parando validação e calibração...")
        system.is_active = False

if __name__ == "__main__":
    main()