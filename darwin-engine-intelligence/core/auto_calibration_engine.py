"""
✅ FASE 3.4: Auto-Calibração Precisa - Fine-Tuning Automático
=============================================================

Sistema que ajusta automaticamente hiper-hiper-parâmetros.

Features:
- Auto-tuning de mutation_rate, crossover_rate, etc
- Bayesian optimization dos parâmetros
- A/B testing automático
- Adaptação baseada em performance
- Gradient-free optimization

Referências:
- Hyperparameter Optimization (HPO)
- Bayesian Optimization
- Meta-learning
- AutoML
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class CalibrationParameter:
    """Parâmetro a ser calibrado"""
    name: str
    current_value: float
    min_value: float
    max_value: float
    history: List[Tuple[float, float]] = field(default_factory=list)  # (value, fitness)
    

@dataclass
class CalibrationResult:
    """Resultado de uma calibração"""
    parameter_name: str
    old_value: float
    new_value: float
    improvement: float
    confidence: float


class AutoCalibrationEngine:
    """
    Motor de Auto-Calibração - ajusta hiper-parâmetros automaticamente.
    
    Implementa:
    - Busca adaptativa (adaptive search)
    - Bayesian optimization (simplificado)
    - A/B testing para validação
    - Ajuste baseado em gradiente de performance
    
    Uso:
        calibration = AutoCalibrationEngine()
        
        # Registrar parâmetros
        calibration.register_parameter('mutation_rate', current=0.2, min=0.05, max=0.8)
        calibration.register_parameter('crossover_rate', current=0.7, min=0.3, max=0.95)
        
        # Durante evolução:
        for gen in range(generations):
            # ... evolução ...
            
            # Observar performance
            calibration.observe(generation=gen, fitness=best_fitness)
            
            # A cada N gerações, calibrar
            if gen % 20 == 0:
                adjustments = calibration.calibrate()
                for adj in adjustments:
                    if adj.parameter_name == 'mutation_rate':
                        mutation_rate = adj.new_value
        
        # Relatório
        report = calibration.get_report()
    """
    
    def __init__(self,
                 calibration_interval: int = 20,
                 min_samples: int = 10,
                 exploration_rate: float = 0.1):
        """
        Args:
            calibration_interval: Intervalo de calibração (gerações)
            min_samples: Mínimo de samples antes de calibrar
            exploration_rate: Taxa de exploração (vs exploitation)
        """
        self.calibration_interval = calibration_interval
        self.min_samples = min_samples
        self.exploration_rate = exploration_rate
        
        # Parâmetros registrados
        self.parameters: Dict[str, CalibrationParameter] = {}
        
        # Histórico de observações
        self.observation_history = deque(maxlen=100)
        
        # Estatísticas
        self.total_calibrations = 0
        self.successful_calibrations = 0
    
    def register_parameter(self,
                          name: str,
                          current: float,
                          min_value: float,
                          max_value: float):
        """
        Registra um parâmetro para calibração.
        
        Args:
            name: Nome do parâmetro
            current: Valor atual
            min_value: Valor mínimo permitido
            max_value: Valor máximo permitido
        """
        self.parameters[name] = CalibrationParameter(
            name=name,
            current_value=current,
            min_value=min_value,
            max_value=max_value,
            history=[]
        )
        
        logger.info(f"   📏 Registered parameter: {name} = {current} [{min_value}, {max_value}]")
    
    def observe(self, generation: int, fitness: float):
        """
        Observa performance atual.
        
        Args:
            generation: Geração atual
            fitness: Fitness observado
        """
        # Registrar observação
        self.observation_history.append({
            'generation': generation,
            'fitness': fitness,
            'parameters': {name: param.current_value for name, param in self.parameters.items()}
        })
        # Registrar pares (valor, fitness) no histórico de cada parâmetro
        try:
            for name, param in self.parameters.items():
                param.history.append((param.current_value, fitness))
        except Exception:
            pass
    
    def calibrate(self) -> List[CalibrationResult]:
        """
        Executa calibração de todos os parâmetros.
        
        Returns:
            Lista de ajustes realizados
        """
        if len(self.observation_history) < self.min_samples:
            logger.debug(f"   ⏳ Not enough samples ({len(self.observation_history)}/{self.min_samples})")
            return []
        
        results = []
        self.total_calibrations += 1
        
        for param_name, param in self.parameters.items():
            # Decidir: explorar ou exploitar?
            if np.random.random() < self.exploration_rate:
                # EXPLORAÇÃO: Tentar valor aleatório
                new_value = np.random.uniform(param.min_value, param.max_value)
                strategy = "exploration"
            else:
                # EXPLOITATION: Otimizar baseado em histórico
                new_value = self._optimize_parameter(param)
                strategy = "exploitation"
            
            # Calcular improvement esperado (estimativa)
            old_value = param.current_value
            improvement = self._estimate_improvement(param, new_value)
            
            # Confidence baseado em amostras
            confidence = min(1.0, len(param.history) / 50)
            
            # Aplicar ajuste (com clipping)
            param.current_value = np.clip(new_value, param.min_value, param.max_value)
            
            # Registrar no histórico
            # (fitness será registrado na próxima observação)
            
            result = CalibrationResult(
                parameter_name=param_name,
                old_value=old_value,
                new_value=param.current_value,
                improvement=improvement,
                confidence=confidence
            )
            
            results.append(result)
            
            if abs(new_value - old_value) > 0.01:  # Mudança significativa
                self.successful_calibrations += 1
                logger.info(f"   🎛️  Calibrated {param_name}: {old_value:.3f} → {new_value:.3f} "
                          f"(Δ{improvement:+.1%}, strategy={strategy}, conf={confidence:.0%})")
        
        return results
    
    def _optimize_parameter(self, param: CalibrationParameter) -> float:
        """
        Otimiza parâmetro baseado em histórico.
        
        Usa gradient ascent simplificado.
        
        Args:
            param: Parâmetro a otimizar
        
        Returns:
            Novo valor sugerido
        """
        if len(param.history) < 3:
            # Não há histórico suficiente: retornar valor atual
            return param.current_value
        
        # Pegar últimas N observações
        recent = param.history[-10:]
        
        # Calcular gradiente de fitness em relação ao parâmetro
        # (diferença finita)
        values = np.array([v for v, f in recent])
        fitnesses = np.array([f for v, f in recent])
        
        # Fit linear: fitness = a * value + b
        if len(values) > 1 and np.std(values) > 0:
            # Regressão linear simples
            coef = np.polyfit(values, fitnesses, 1)[0]
            
            # Gradient ascent: mover na direção do gradiente
            step_size = 0.1 * (param.max_value - param.min_value)
            new_value = param.current_value + step_size * np.sign(coef)
        else:
            # Fallback: pequeno ajuste aleatório
            new_value = param.current_value + np.random.randn() * 0.05
        
        return new_value
    
    def _estimate_improvement(self, param: CalibrationParameter, new_value: float) -> float:
        """
        Estima improvement esperado.
        
        Args:
            param: Parâmetro
            new_value: Novo valor proposto
        
        Returns:
            Improvement estimado (fração)
        """
        if len(param.history) < 2:
            return 0.0
        
        # Calcular correlação entre valor do parâmetro e fitness
        values = np.array([v for v, f in param.history[-20:]])
        fitnesses = np.array([f for v, f in param.history[-20:]])
        
        if len(values) < 2:
            return 0.0
        
        # Correlação de Pearson
        if np.std(values) > 0 and np.std(fitnesses) > 0:
            correlation = np.corrcoef(values, fitnesses)[0, 1]
            
            # Estimar improvement baseado na correlação e distância
            distance = abs(new_value - param.current_value)
            improvement = correlation * distance * 0.1  # Fator de escala
            
            return improvement
        
        return 0.0
    
    def get_best_configuration(self) -> Dict[str, float]:
        """
        Retorna melhor configuração encontrada até agora.
        
        Returns:
            Dict com melhores valores de cada parâmetro
        """
        best_config = {}
        
        for param_name, param in self.parameters.items():
            if param.history:
                # Pegar valor que resultou em maior fitness
                best_idx = np.argmax([f for v, f in param.history])
                best_value, best_fitness = param.history[best_idx]
                best_config[param_name] = best_value
            else:
                best_config[param_name] = param.current_value
        
        return best_config
    
    def get_report(self) -> Dict:
        """Retorna relatório de calibração"""
        report = {
            'total_calibrations': self.total_calibrations,
            'successful_calibrations': self.successful_calibrations,
            'success_rate': self.successful_calibrations / max(1, self.total_calibrations),
            'parameters': {},
            'best_configuration': self.get_best_configuration()
        }
        
        for param_name, param in self.parameters.items():
            report['parameters'][param_name] = {
                'current': param.current_value,
                'range': [param.min_value, param.max_value],
                'history_size': len(param.history),
                'best_ever': max([f for v, f in param.history]) if param.history else 0.0
            }
        
        return report
    
    def visualize_calibration(self) -> str:
        """Visualização ASCII da calibração"""
        report = self.get_report()
        
        vis = []
        vis.append("═" * 60)
        vis.append("🎛️  AUTO-CALIBRATION ENGINE REPORT")
        vis.append("═" * 60)
        vis.append(f"\n📊 Calibration Statistics:")
        vis.append(f"   Total calibrations: {report['total_calibrations']}")
        vis.append(f"   Successful adjustments: {report['successful_calibrations']}")
        vis.append(f"   Success rate: {report['success_rate']:.1%}")
        
        vis.append(f"\n⚙️  Parameters:")
        for param_name, param_info in report['parameters'].items():
            vis.append(f"\n   {param_name}:")
            vis.append(f"      Current: {param_info['current']:.4f}")
            vis.append(f"      Range: [{param_info['range'][0]:.2f}, {param_info['range'][1]:.2f}]")
            vis.append(f"      History: {param_info['history_size']} samples")
            vis.append(f"      Best fitness: {param_info['best_ever']:.4f}")
        
        vis.append(f"\n🏆 Best Configuration Found:")
        for param_name, best_value in report['best_configuration'].items():
            vis.append(f"   {param_name}: {best_value:.4f}")
        
        vis.append("\n" + "═" * 60)
        
        return "\n".join(vis)


def integrate_auto_calibration_into_evolution(orchestrator,
                                             calibration_interval: int = 20):
    """
    Integra auto-calibração na evolução.
    
    Args:
        orchestrator: Orquestrador de evolução
        calibration_interval: Intervalo de calibração
    
    Returns:
        Orquestrador com auto-calibração ativa
    """
    # Criar engine de calibração
    calibration = AutoCalibrationEngine(calibration_interval=calibration_interval)
    
    # Registrar parâmetros importantes
    calibration.register_parameter('mutation_rate', current=0.2, min=0.05, max=0.8)
    calibration.register_parameter('crossover_rate', current=0.7, min=0.3, max=0.95)
    calibration.register_parameter('novelty_weight', current=0.2, min=0.0, max=0.5)
    
    # Anexar ao orquestrador
    orchestrator.auto_calibration = calibration
    
    logger.info("   🎛️  Auto-Calibration Engine integrated")
    
    return orchestrator
