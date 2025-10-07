#!/usr/bin/env python3
"""
INCOMPLETUDE GÖDELIANA EVOLUÍDA
================================
Versão melhorada baseada em testes e teoria
Incorpora sugestões hipotéticas das 6 APIs
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
import random
import math

class EvolvedGodelianIncompleteness:
    """
    Incompletude Gödeliana Evoluída - Versão 2.0
    
    Melhorias baseadas no consenso teórico das APIs:
    1. Detecção multi-sinal (loss, gradientes, pesos)
    2. Ações combinadas e criativas
    3. Memória de sucesso/falha
    4. Adaptação dinâmica
    5. Prevenção proativa
    6. Motor de inquietação melhorado
    """
    
    def __init__(self, delta_0: float = 0.05):
        # Parâmetros fundamentais
        self.delta_0 = delta_0  # Piso Gödeliano
        self.stagnation_counter = 0
        self.exploration_intensity = 0.0
        
        # Históricos expandidos
        self.loss_history = deque(maxlen=50)
        self.gradient_history = deque(maxlen=30)
        self.weight_change_history = deque(maxlen=30)
        self.accuracy_history = deque(maxlen=30)
        
        # Memória de intervenções
        self.intervention_memory = {
            'lr_change': {'success': 0, 'failure': 0, 'last_value': None},
            'noise_injection': {'success': 0, 'failure': 0, 'last_scale': None},
            'neuron_reactivation': {'success': 0, 'failure': 0},
            'optimizer_switch': {'success': 0, 'failure': 0},
            'dropout_modulation': {'success': 0, 'failure': 0},
            'batch_perturbation': {'success': 0, 'failure': 0},
            'momentum_adjustment': {'success': 0, 'failure': 0},
            'architecture_expansion': {'success': 0, 'failure': 0}
        }
        
        # Adaptive thresholds
        self.variance_threshold = 0.0001
        self.gradient_threshold = 1e-6
        self.weight_change_threshold = 1e-5
        
        # Estado avançado
        self.total_interventions = 0
        self.successful_interventions = 0
        self.current_strategy = 'conservative'
        self.last_intervention_step = 0
        self.intervention_cooldown = 5
        
        # Motor de inquietação aprimorado
        self.restlessness_level = 0.0
        self.curiosity_bias = 0.1
        
        # Pesos anteriores para detectar mudanças
        self.previous_weights = None
        
    def detect_stagnation_advanced(self, 
                                  loss: float,
                                  model: nn.Module,
                                  accuracy: Optional[float] = None) -> Tuple[bool, Dict[str, float]]:
        """
        Detecção avançada multi-sinal de estagnação
        """
        signals = {}
        
        # 1. Análise de Loss
        self.loss_history.append(loss)
        if len(self.loss_history) >= 10:
            recent_losses = list(self.loss_history)[-10:]
            
            # Variância do loss
            loss_variance = np.var(recent_losses)
            signals['loss_variance'] = loss_variance
            
            # Tendência do loss (slope)
            x = np.arange(len(recent_losses))
            slope, _ = np.polyfit(x, recent_losses, 1)
            signals['loss_slope'] = abs(slope)
            
            # Oscilação (mudanças de direção)
            directions = np.diff(recent_losses)
            oscillations = np.sum(np.diff(np.sign(directions)) != 0)
            signals['loss_oscillations'] = oscillations
        
        # 2. Análise de Gradientes
        total_grad_norm = 0.0
        num_params = 0
        
        for param in model.parameters():
            if param.grad is not None:
                grad_norm = param.grad.data.norm(2).item()
                total_grad_norm += grad_norm
                num_params += 1
        
        avg_grad_norm = total_grad_norm / max(num_params, 1)
        self.gradient_history.append(avg_grad_norm)
        signals['gradient_norm'] = avg_grad_norm
        
        # 3. Análise de Mudança nos Pesos
        if self.previous_weights is not None:
            weight_change = 0.0
            for (name, param), prev_param in zip(model.named_parameters(), self.previous_weights.values()):
                if param.requires_grad:
                    change = (param.data - prev_param).abs().mean().item()
                    weight_change += change
            
            self.weight_change_history.append(weight_change)
            signals['weight_change'] = weight_change
        
        # Salvar pesos atuais
        self.previous_weights = {name: param.data.clone() 
                                for name, param in model.named_parameters()}
        
        # 4. Análise de Accuracy (se disponível)
        if accuracy is not None:
            self.accuracy_history.append(accuracy)
            if len(self.accuracy_history) >= 5:
                recent_acc = list(self.accuracy_history)[-5:]
                acc_variance = np.var(recent_acc)
                signals['accuracy_variance'] = acc_variance
        
        # 5. Detecção de Estagnação Multi-Critério
        is_stagnant = False
        stagnation_score = 0.0
        
        # Adaptar thresholds dinamicamente
        if len(self.loss_history) >= 20:
            historical_variance = np.var(list(self.loss_history)[-20:])
            self.variance_threshold = max(historical_variance * 0.1, 1e-6)
        
        # Verificar cada sinal
        if 'loss_variance' in signals and signals['loss_variance'] < self.variance_threshold:
            stagnation_score += 0.3
            
        if 'loss_slope' in signals and abs(signals['loss_slope']) < 0.001:
            stagnation_score += 0.2
            
        if 'gradient_norm' in signals and signals['gradient_norm'] < self.gradient_threshold:
            stagnation_score += 0.25
            
        if 'weight_change' in signals and signals['weight_change'] < self.weight_change_threshold:
            stagnation_score += 0.25
            
        # Decisão final
        if stagnation_score > 0.5:
            is_stagnant = True
            self.stagnation_counter += 1
        else:
            self.stagnation_counter = max(0, self.stagnation_counter - 1)
        
        # Aumentar inquietação se estagnado
        if is_stagnant:
            self.restlessness_level = min(1.0, self.restlessness_level + 0.1)
        else:
            self.restlessness_level = max(0.0, self.restlessness_level - 0.05)
        
        signals['stagnation_score'] = stagnation_score
        signals['is_stagnant'] = is_stagnant
        
        return is_stagnant, signals
    
    def select_interventions(self, signals: Dict[str, float]) -> List[str]:
        """
        Seleciona intervenções baseado em sinais e memória
        """
        interventions = []
        
        # Calcular scores para cada intervenção baseado em sucesso passado
        intervention_scores = {}
        
        for action, stats in self.intervention_memory.items():
            total = stats['success'] + stats['failure']
            if total > 0:
                success_rate = stats['success'] / total
            else:
                success_rate = 0.5  # Neutro se nunca tentado
            
            # Adicionar bonus de exploração para ações não tentadas
            exploration_bonus = 0.2 if total < 3 else 0
            
            intervention_scores[action] = success_rate + exploration_bonus + random.random() * self.curiosity_bias
        
        # Ordenar por score
        sorted_interventions = sorted(intervention_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Selecionar baseado na intensidade de exploração
        num_interventions = min(
            int(1 + self.exploration_intensity * 3),  # 1-4 intervenções
            len(sorted_interventions)
        )
        
        # Adicionar top intervenções
        for action, score in sorted_interventions[:num_interventions]:
            # Verificar condições específicas
            if action == 'lr_change' and self.stagnation_counter > 2:
                interventions.append(action)
            elif action == 'noise_injection' and self.stagnation_counter > 4:
                interventions.append(action)
            elif action == 'neuron_reactivation' and self.stagnation_counter > 6:
                interventions.append(action)
            elif action == 'dropout_modulation' and signals.get('loss_oscillations', 0) > 5:
                interventions.append(action)
            elif action == 'momentum_adjustment' and signals.get('gradient_norm', 1) < 0.0001:
                interventions.append(action)
            elif action == 'batch_perturbation' and self.restlessness_level > 0.5:
                interventions.append(action)
            elif action == 'optimizer_switch' and self.stagnation_counter > 10:
                interventions.append(action)
            elif action == 'architecture_expansion' and self.stagnation_counter > 15:
                interventions.append(action)
        
        return interventions
    
    def apply_incompleteness_evolved(self,
                                    model: nn.Module,
                                    optimizer: optim.Optimizer,
                                    loss: float,
                                    accuracy: Optional[float] = None,
                                    batch_size: Optional[int] = None) -> Dict[str, Any]:
        """
        Aplicação evoluída da incompletude com múltiplas estratégias
        """
        # Detectar estagnação
        is_stagnant, signals = self.detect_stagnation_advanced(loss, model, accuracy)
        
        # Calcular intensidade de exploração
        self.exploration_intensity = min(1.0, 
                                        self.delta_0 * (1 + self.stagnation_counter * 0.15) + 
                                        self.restlessness_level * 0.3)
        
        actions_taken = []
        intervention_results = {}
        
        # Verificar cooldown (baseado em steps totais, não intervenções)
        self.total_interventions += 1  # Incrementar contador de steps
        steps_since_last = self.total_interventions - self.last_intervention_step
        
        if is_stagnant and steps_since_last >= self.intervention_cooldown:
            # Selecionar intervenções
            selected_interventions = self.select_interventions(signals)
            
            for intervention in selected_interventions:
                if intervention == 'lr_change':
                    result = self._adjust_learning_rate(optimizer)
                    if result:
                        actions_taken.append(result)
                        intervention_results[intervention] = True
                
                elif intervention == 'noise_injection':
                    result = self._inject_noise(model)
                    if result:
                        actions_taken.append(result)
                        intervention_results[intervention] = True
                
                elif intervention == 'neuron_reactivation':
                    result = self._reactivate_neurons(model)
                    if result:
                        actions_taken.append(result)
                        intervention_results[intervention] = True
                
                elif intervention == 'dropout_modulation':
                    result = self._modulate_dropout(model)
                    if result:
                        actions_taken.append(result)
                        intervention_results[intervention] = True
                
                elif intervention == 'momentum_adjustment':
                    result = self._adjust_momentum(optimizer)
                    if result:
                        actions_taken.append(result)
                        intervention_results[intervention] = True
                
                elif intervention == 'batch_perturbation':
                    if batch_size:
                        result = f"Batch size suggestion: {int(batch_size * (1 + random.uniform(-0.3, 0.3)))}"
                        actions_taken.append(result)
                        intervention_results[intervention] = True
                
                elif intervention == 'optimizer_switch':
                    actions_taken.append("REQUEST_OPTIMIZER_CHANGE")
                    intervention_results[intervention] = True
                
                elif intervention == 'architecture_expansion':
                    result = self._suggest_architecture_change(model)
                    if result:
                        actions_taken.append(result)
                        intervention_results[intervention] = True
            
            # Registrar intervenções
            if actions_taken:
                self.total_interventions += 1
                self.last_intervention_step = self.total_interventions
        
        # Prevenção proativa - avisos antes da estagnação completa
        warnings = []
        if signals.get('stagnation_score', 0) > 0.3 and not is_stagnant:
            warnings.append("⚠️ Pre-stagnation detected - consider early intervention")
        
        if self.restlessness_level > 0.7:
            warnings.append("⚠️ High restlessness - system needs major perturbation")
        
        return {
            'stagnant': is_stagnant,
            'stagnation_level': self.stagnation_counter,
            'exploration_intensity': self.exploration_intensity,
            'restlessness': self.restlessness_level,
            'actions': actions_taken,
            'signals': signals,
            'warnings': warnings,
            'intervention_results': intervention_results,
            'total_interventions': self.total_interventions
        }
    
    def _adjust_learning_rate(self, optimizer: optim.Optimizer) -> Optional[str]:
        """Ajuste inteligente de learning rate"""
        old_lr = optimizer.param_groups[0]['lr']
        
        # Estratégia baseada em histórico
        if self.intervention_memory['lr_change']['success'] > self.intervention_memory['lr_change']['failure']:
            # Estratégia bem-sucedida anterior
            factor = 2.0 if self.stagnation_counter > 5 else 1.5
        else:
            # Tentar estratégia diferente
            factor = 0.5 if self.stagnation_counter < 5 else 3.0
        
        # Adicionar aleatoriedade controlada
        factor *= (1 + random.uniform(-0.2, 0.2))
        
        new_lr = old_lr * factor
        new_lr = max(1e-7, min(1.0, new_lr))
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr
        
        self.intervention_memory['lr_change']['last_value'] = new_lr
        
        return f"LR: {old_lr:.6f} → {new_lr:.6f} (factor={factor:.2f})"
    
    def _inject_noise(self, model: nn.Module) -> Optional[str]:
        """Injeção inteligente de ruído"""
        # Escala adaptativa baseada em exploração
        base_scale = self.exploration_intensity * 0.01
        
        # Ajustar baseado em sucesso anterior
        if self.intervention_memory['noise_injection']['success'] > self.intervention_memory['noise_injection']['failure']:
            noise_scale = base_scale * 1.2
        else:
            noise_scale = base_scale * 0.8
        
        # Adicionar ruído direcionado (não totalmente aleatório)
        with torch.no_grad():
            total_params = 0
            noisy_params = 0
            
            for param in model.parameters():
                if param.requires_grad:
                    total_params += 1
                    
                    # Aplicar ruído seletivamente (não em todos os parâmetros)
                    if random.random() < 0.3 + self.restlessness_level * 0.4:
                        # Ruído com viés baseado no gradiente
                        if param.grad is not None:
                            grad_direction = torch.sign(param.grad)
                            noise = torch.randn_like(param) * noise_scale
                            # 70% aleatório, 30% na direção oposta ao gradiente
                            directed_noise = noise * 0.7 - grad_direction * noise_scale * 0.3
                        else:
                            directed_noise = torch.randn_like(param) * noise_scale
                        
                        param.add_(directed_noise)
                        noisy_params += 1
        
        self.intervention_memory['noise_injection']['last_scale'] = noise_scale
        
        return f"Noise injected: {noisy_params}/{total_params} params (σ={noise_scale:.5f})"
    
    def _reactivate_neurons(self, model: nn.Module) -> Optional[str]:
        """Reativação inteligente de neurônios mortos"""
        reactivated = 0
        checked = 0
        
        with torch.no_grad():
            for name, module in model.named_modules():
                if isinstance(module, nn.Linear):
                    checked += 1
                    
                    # Detectar neurônios mortos com threshold adaptativo
                    weight_norms = module.weight.abs().mean(dim=1)
                    dead_threshold = weight_norms.mean() * 0.1  # 10% da média
                    dead_mask = weight_norms < dead_threshold
                    
                    num_dead = dead_mask.sum().item()
                    
                    if num_dead > 0:
                        # Reinicialização inteligente
                        fan_in = module.weight.size(1)
                        fan_out = module.weight.size(0)
                        
                        # Usar inicialização apropriada
                        if self.restlessness_level > 0.5:
                            # Inicialização mais agressiva quando inquieto
                            std = np.sqrt(6.0 / (fan_in + fan_out))  # Xavier
                        else:
                            std = np.sqrt(2.0 / fan_in)  # He
                        
                        module.weight[dead_mask] = torch.randn(num_dead, fan_in) * std
                        
                        if module.bias is not None:
                            module.bias[dead_mask] = torch.randn(num_dead) * 0.01
                        
                        reactivated += num_dead
        
        if reactivated > 0:
            return f"Reactivated {reactivated} neurons in {checked} layers"
        return None
    
    def _modulate_dropout(self, model: nn.Module) -> Optional[str]:
        """Modula dropout dinamicamente"""
        changes = []
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Dropout):
                old_p = module.p
                
                # Ajustar baseado na exploração
                if self.exploration_intensity > 0.7:
                    # Aumentar dropout para forçar robustez
                    new_p = min(0.7, old_p + 0.1)
                elif self.exploration_intensity < 0.3:
                    # Reduzir dropout para acelerar convergência
                    new_p = max(0.1, old_p - 0.1)
                else:
                    # Oscilação controlada
                    new_p = old_p + random.uniform(-0.1, 0.1)
                    new_p = max(0.1, min(0.7, new_p))
                
                module.p = new_p
                changes.append(f"{old_p:.2f}→{new_p:.2f}")
        
        if changes:
            return f"Dropout modulated: [{', '.join(changes)}]"
        return None
    
    def _adjust_momentum(self, optimizer: optim.Optimizer) -> Optional[str]:
        """Ajusta momentum do optimizer"""
        if isinstance(optimizer, optim.SGD):
            for group in optimizer.param_groups:
                if 'momentum' in group:
                    old_momentum = group['momentum']
                    
                    # Momentum adaptativo
                    if self.stagnation_counter > 5:
                        # Reduzir momentum para escapar de mínimo
                        new_momentum = max(0.5, old_momentum - 0.1)
                    else:
                        # Aumentar momentum para acelerar
                        new_momentum = min(0.99, old_momentum + 0.05)
                    
                    group['momentum'] = new_momentum
                    
                    return f"Momentum: {old_momentum:.2f} → {new_momentum:.2f}"
        
        return None
    
    def _suggest_architecture_change(self, model: nn.Module) -> Optional[str]:
        """Sugere mudanças arquiteturais"""
        suggestions = []
        
        # Analisar tamanho das camadas
        layer_sizes = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                layer_sizes.append((module.in_features, module.out_features))
        
        if layer_sizes:
            # Sugerir expansão se muito estagnado
            if self.stagnation_counter > 15:
                suggestions.append("Consider adding neurons to hidden layers (+20%)")
            
            # Sugerir regularização se oscilando
            if self.loss_history and np.var(list(self.loss_history)[-10:]) > 0.1:
                suggestions.append("Consider adding BatchNorm or LayerNorm")
            
            # Sugerir skip connections se muito profundo
            if len(layer_sizes) > 5:
                suggestions.append("Consider adding skip connections")
        
        if suggestions:
            return f"Architecture: {'; '.join(suggestions)}"
        return None
    
    def update_memory(self, intervention_results: Dict[str, bool], 
                     loss_improvement: float):
        """
        Atualiza memória de sucesso/falha das intervenções
        """
        for intervention, was_applied in intervention_results.items():
            if was_applied:
                if loss_improvement > 0.01:  # Melhoria significativa
                    self.intervention_memory[intervention]['success'] += 1
                    self.successful_interventions += 1
                else:
                    self.intervention_memory[intervention]['failure'] += 1
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """
        Retorna resumo da estratégia atual
        """
        success_rates = {}
        for action, stats in self.intervention_memory.items():
            total = stats['success'] + stats['failure']
            if total > 0:
                success_rates[action] = stats['success'] / total
            else:
                success_rates[action] = 0.5
        
        return {
            'current_strategy': self.current_strategy,
            'stagnation_level': self.stagnation_counter,
            'restlessness': self.restlessness_level,
            'exploration_intensity': self.exploration_intensity,
            'total_interventions': self.total_interventions,
            'success_rate': self.successful_interventions / max(self.total_interventions, 1),
            'intervention_success_rates': success_rates,
            'adaptive_thresholds': {
                'variance': self.variance_threshold,
                'gradient': self.gradient_threshold,
                'weight_change': self.weight_change_threshold
            }
        }


# ==================== TESTE DA INCOMPLETUDE EVOLUÍDA ====================

def test_evolved_incompletude():
    """Testa a incompletude evoluída"""
    
    print("="*70)
    print("🧬 TESTANDO INCOMPLETUDE GÖDELIANA EVOLUÍDA")
    print("="*70)
    
    # Criar modelo simples
    model = nn.Sequential(
        nn.Linear(100, 50),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(50, 10)
    )
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Criar incompletude evoluída
    godelian = EvolvedGodelianIncompleteness(delta_0=0.05)
    
    print("\n📊 Simulando 50 steps com estagnação forçada...")
    print("-"*70)
    
    # Simular treinamento com estagnação
    for step in range(50):
        # Dados sintéticos
        x = torch.randn(32, 100)
        y = torch.randint(0, 10, (32,))
        
        # Forward pass
        optimizer.zero_grad()
        output = model(x)
        
        # Forçar estagnação após step 20
        if step < 20:
            loss = criterion(output, y)
        else:
            # Loss artificial estagnado
            loss = torch.tensor(1.5 + random.uniform(-0.001, 0.001), requires_grad=True)
        
        apply_incompletude(loss.item() if hasattr(loss, "item") else loss, model=model if "model" in locals() else None, optimizer=optimizer if "optimizer" in locals() else None)
        loss.backward()
        optimizer.step()
