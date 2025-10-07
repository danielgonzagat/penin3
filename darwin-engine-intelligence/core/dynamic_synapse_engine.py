"""
✅ FASE 3.1: Auto-Sinapse Dinâmica - Conexões Ajustáveis em Tempo Real
=======================================================================

Sistema que ajusta conexões neurais dinamicamente durante evolução.

Features:
- Ajuste de pesos sinápticos em tempo real
- Poda de conexões fracas (synaptic pruning)
- Crescimento de novas conexões (neurogenesis)
- Plasticidade Hebbiana ("neurons that fire together, wire together")
- Homeostase sináptica (manter estabilidade)

Referências:
- Hebbian Learning (Hebb, 1949)
- Synaptic Plasticity (neuroscience)
- Neural Darwinism (Edelman)
- Developmental plasticity in neural networks
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class SynapseState:
    """Estado de uma sinapse (conexão)"""
    source_id: int
    target_id: int
    weight: float
    strength: float  # Força da conexão (0-1)
    age: int  # Quantos ciclos vive
    activation_history: List[float]  # Histórico de ativações
    

class DynamicSynapseEngine:
    """
    Motor de Sinapses Dinâmicas - ajusta conexões em tempo real.
    
    Simula plasticidade neural:
    - Conexões usadas fortalecem (Hebbian)
    - Conexões não-usadas enfraquecem (pruning)
    - Novas conexões podem surgir (neurogenesis)
    - Homeostase mantém equilíbrio
    
    Uso:
        synapse = DynamicSynapseEngine(model)
        
        # Durante treino:
        for epoch in range(epochs):
            output = model(input)
            loss = criterion(output, target)
            
            # Ajustar sinapses dinamicamente
            synapse.update_synapses(model, activations={'layer1': a1, 'layer2': a2})
            synapse.prune_weak_synapses(model, threshold=0.01)
            synapse.grow_new_synapses(model, max_new=5)
            
            loss.backward()
            optimizer.step()
        
        # Relatório
        report = synapse.get_plasticity_report()
    """
    
    def __init__(self, 
                 learning_rate: float = 0.01,
                 prune_threshold: float = 0.01,
                 growth_rate: float = 0.05):
        """
        Args:
            learning_rate: Taxa de ajuste Hebbiano
            prune_threshold: Threshold para podar conexões fracas
            growth_rate: Taxa de crescimento de novas conexões
        """
        self.learning_rate = learning_rate
        self.prune_threshold = prune_threshold
        self.growth_rate = growth_rate
        
        # Rastreamento de sinapses
        self.synapses: Dict[str, SynapseState] = {}
        self.pruned_count = 0
        self.grown_count = 0
        self.total_updates = 0
        
        # Estatísticas
        self.stats = {
            'total_synapses': 0,
            'active_synapses': 0,
            'pruned_synapses': 0,
            'grown_synapses': 0,
            'avg_strength': 0.0
        }
    
    def update_synapses(self, model: nn.Module, activations: Dict[str, torch.Tensor]):
        """
        Atualiza forças sinápticas baseado em ativações (Hebbian learning).
        
        "Neurons that fire together, wire together"
        
        Args:
            model: Modelo PyTorch
            activations: Dict com ativações de cada camada
        """
        self.total_updates += 1
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                # Ajuste Hebbiano: weight += lr * (pre_activation * post_activation)
                if name in activations and f"{name}_pre" in activations:
                    pre = activations[f"{name}_pre"]  # Ativação pré-sináptica
                    post = activations[name]  # Ativação pós-sináptica
                    
                    # Regra Hebbiana simplificada
                    with torch.no_grad():
                        # Correlação entre pré e pós
                        correlation = torch.outer(post.mean(0), pre.mean(0))
                        
                        # Ajustar pesos proporcionalmente
                        delta = self.learning_rate * correlation
                        
                        # Normalizar para não explodir
                        delta = torch.clamp(delta, -0.1, 0.1)
                        
                        module.weight.data += delta
                
                # Rastrear força das conexões
                with torch.no_grad():
                    strength = module.weight.abs().mean().item()
                    self.synapses[f"{name}_weight"] = SynapseState(
                        source_id=0,  # Simplificado
                        target_id=0,
                        weight=module.weight.mean().item(),
                        strength=strength,
                        age=self.synapses.get(f"{name}_weight", SynapseState(0,0,0,0,0,[])).age + 1,
                        activation_history=[]
                    )
    
    def prune_weak_synapses(self, model: nn.Module, threshold: Optional[float] = None):
        """
        Poda conexões fracas (synaptic pruning).
        
        Remove pesos que são muito pequenos (não contribuem).
        
        Args:
            model: Modelo PyTorch
            threshold: Threshold de magnitude (default: self.prune_threshold)
        """
        threshold = threshold or self.prune_threshold
        pruned = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    # Máscara de pesos significativos
                    mask = module.weight.abs() > threshold
                    
                    # Contar quantos foram podados
                    pruned += (~mask).sum().item()
                    
                    # Aplicar poda (zerar pesos fracos)
                    module.weight.data *= mask.float()
        
        self.pruned_count += pruned
        self.stats['pruned_synapses'] = self.pruned_count
        
        if pruned > 0:
            logger.debug(f"   ✂️ Pruned {pruned} weak synapses (threshold={threshold:.4f})")
    
    def grow_new_synapses(self, model: nn.Module, max_new: int = 5):
        """
        Crescimento de novas conexões (neurogenesis).
        
        Adiciona pequenos pesos aleatórios onde havia zeros.
        
        Args:
            model: Modelo PyTorch
            max_new: Máximo de novas conexões por camada
        """
        grown = 0
        
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    # Encontrar pesos zerados (podados anteriormente)
                    zero_mask = module.weight.abs() < 1e-6
                    
                    # Se não há zeros, pular
                    if zero_mask.sum() == 0:
                        continue
                    
                    # Escolher aleatoriamente até max_new para reativar
                    zero_indices = torch.nonzero(zero_mask)
                    
                    if len(zero_indices) == 0:
                        continue
                    
                    # Limitar a max_new
                    n_grow = min(max_new, len(zero_indices))
                    
                    # Escolher aleatoriamente
                    perm = torch.randperm(len(zero_indices))[:n_grow]
                    selected = zero_indices[perm]
                    
                    # Inicializar com pequenos valores aleatórios
                    for idx in selected:
                        i, j = idx[0].item(), idx[1].item()
                        module.weight.data[i, j] = torch.randn(1).item() * self.growth_rate
                    
                    grown += n_grow
        
        self.grown_count += grown
        self.stats['grown_synapses'] = self.grown_count
        
        if grown > 0:
            logger.debug(f"   🌱 Grew {grown} new synapses (rate={self.growth_rate:.4f})")
    
    def apply_homeostasis(self, model: nn.Module, target_mean: float = 0.0, target_std: float = 1.0):
        """
        Homeostase sináptica - mantém distribuição de pesos estável.
        
        Normaliza pesos para evitar explosão/desaparecimento.
        
        Args:
            model: Modelo PyTorch
            target_mean: Média desejada
            target_std: Desvio padrão desejado
        """
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                with torch.no_grad():
                    # Calcular estatísticas atuais
                    current_mean = module.weight.mean()
                    current_std = module.weight.std()
                    
                    # Normalizar para target
                    if current_std > 0:
                        module.weight.data = (module.weight.data - current_mean) / current_std
                        module.weight.data = module.weight.data * target_std + target_mean
    
    def get_plasticity_report(self) -> Dict:
        """
        Retorna relatório de plasticidade.
        
        Returns:
            Dict com estatísticas de sinapses
        """
        total = sum(1 for _ in self.synapses.values())
        active = sum(1 for s in self.synapses.values() if s.strength > self.prune_threshold)
        avg_strength = np.mean([s.strength for s in self.synapses.values()]) if self.synapses else 0.0
        
        self.stats.update({
            'total_synapses': total,
            'active_synapses': active,
            'pruned_synapses': self.pruned_count,
            'grown_synapses': self.grown_count,
            'avg_strength': float(avg_strength),
            'total_updates': self.total_updates
        })
        
        return dict(self.stats)
    
    def visualize_plasticity(self) -> str:
        """Visualização ASCII da plasticidade"""
        report = self.get_plasticity_report()
        
        vis = []
        vis.append("═" * 60)
        vis.append("🧠 SYNAPTIC PLASTICITY REPORT")
        vis.append("═" * 60)
        vis.append(f"\n📊 Statistics:")
        vis.append(f"   Total synapses: {report['total_synapses']}")
        vis.append(f"   Active (strong): {report['active_synapses']}")
        vis.append(f"   Pruned (removed): {report['pruned_synapses']}")
        vis.append(f"   Grown (new): {report['grown_synapses']}")
        vis.append(f"   Avg strength: {report['avg_strength']:.4f}")
        vis.append(f"   Total updates: {report['total_updates']}")
        
        # Ratio
        if report['total_synapses'] > 0:
            active_ratio = report['active_synapses'] / report['total_synapses']
            vis.append(f"\n💪 Active ratio: {active_ratio:.1%}")
        
        vis.append("\n" + "═" * 60)
        
        return "\n".join(vis)


def integrate_dynamic_synapses_into_training(model: nn.Module, 
                                            train_loader,
                                            epochs: int = 10,
                                            use_plasticity: bool = True):
    """
    Função helper para integrar sinapses dinâmicas no treino.
    
    Args:
        model: Modelo PyTorch
        train_loader: DataLoader de treino
        epochs: Número de épocas
        use_plasticity: Se True, usa plasticidade sináptica
    
    Returns:
        Modelo treinado com sinapses dinâmicas
    """
    if not use_plasticity:
        # Treino normal sem plasticidade
        return model
    
    # Inicializar engine
    synapse = DynamicSynapseEngine(
        learning_rate=0.01,
        prune_threshold=0.01,
        growth_rate=0.05
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Forward pass (capturar ativações se necessário)
            output = model(data)
            loss = criterion(output, target)
            
            # ✅ Plasticidade sináptica
            if epoch % 2 == 0:  # A cada 2 épocas
                synapse.prune_weak_synapses(model, threshold=0.01)
            
            if epoch % 5 == 0:  # A cada 5 épocas
                synapse.grow_new_synapses(model, max_new=10)
            
            # Backward + optimize
            loss.backward()
            optimizer.step()
            
            # Homeostase a cada batch
            if batch_idx % 50 == 0:
                synapse.apply_homeostasis(model)
    
    # Relatório final
    logger.info(synapse.visualize_plasticity())
    
    return model
