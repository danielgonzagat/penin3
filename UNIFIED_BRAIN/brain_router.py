#!/usr/bin/env python3
"""
🎯 IA³ ROUTER
Sistema de roteamento inteligente que escolhe quais neurônios ativar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
import numpy as np

class IA3Router(nn.Module):
    """
    Router que escolhe top-k neurônios por passo
    baseado em competência, novidade e diversidade
    """
    def __init__(
        self,
        H: int = 1024,
        num_neurons: int = 1000,
        top_k: int = 64,
        temperature: float = 1.0
    ):
        super().__init__()
        self.H = H
        self.num_neurons = num_neurons
        self.top_k = top_k
        self.temperature = temperature
        
        # Scoring network
        self.scorer = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.LayerNorm(H // 2),
            nn.GELU(),
            nn.Linear(H // 2, num_neurons)
        )
        
        # Competence scores (aprendidos)
        # Ensure gradients can exist when desired
        self.competence = nn.Parameter(torch.ones(num_neurons), requires_grad=True)
        
        # Pending competence updates (accumulated during episode, applied during training)
        self.pending_updates = []  # List of (neuron_idx, reward, lr)
        
        # Novelty tracking (EMA de ativações)
        self.register_buffer('activation_ema', torch.zeros(num_neurons))
        self.ema_decay = 0.99
        
        # Diversity penalty
        self.diversity_weight = 0.1
        
    def forward(
        self,
        z: torch.Tensor,  # [B, H]
        neuron_ids: List[str] = None,
        return_scores: bool = False
    ) -> Tuple[torch.Tensor, List[int]]:
        """
        Escolhe top-k neurônios para ativar
        
        Args:
            z: estado latente atual
            neuron_ids: lista de IDs disponíveis
            return_scores: se True, retorna scores também
            
        Returns:
            mask: [B, num_neurons] máscara binária
            selected_indices: lista de índices selecionados
        """
        B = z.shape[0]
        
        # Score contextual (baseado em z)
        context_scores = self.scorer(z)  # [B, num_neurons]
        
        # Competence score (estático, mas treináve l)
        comp_scores = self.competence.unsqueeze(0).expand(B, -1)  # [B, num_neurons]
        
        # Novelty bonus (inversamente proporcional à frequência)
        novelty_scores = 1.0 / (self.activation_ema + 1e-6)
        novelty_scores = novelty_scores.unsqueeze(0).expand(B, -1)
        
        # Score final combinado
        final_scores = (
            context_scores +
            comp_scores * 0.3 +
            novelty_scores * self.diversity_weight
        )
        
        # Temperature scaling
        final_scores = final_scores / self.temperature
        
        # Top-k selection (clamped to avoid out of range)
        k_actual = min(self.top_k, self.num_neurons)
        if k_actual == 0:
            # Fallback: retorna índice 0
            top_indices = torch.zeros(B, 1, dtype=torch.long, device=z.device)
        else:
            _, top_indices = torch.topk(final_scores, k_actual, dim=1)
        
        # Cria máscara
        mask = torch.zeros(B, self.num_neurons, device=z.device)
        mask.scatter_(1, top_indices, 1.0)
        
        # Atualiza EMA de ativações
        with torch.no_grad():
            activation_count = mask.sum(dim=0)  # [num_neurons]
            self.activation_ema = (
                self.ema_decay * self.activation_ema +
                (1 - self.ema_decay) * activation_count
            )
        
        selected_indices = top_indices[0].tolist()
        
        if return_scores:
            return mask, selected_indices, final_scores
        return mask, selected_indices
    
    def update_competence(
        self,
        neuron_idx: int,
        reward: float,
        lr: float = 0.01,
        allow_gradients: bool = False  # ✅ CORREÇÃO #7: Novo parâmetro
    ):
        """
        Atualiza score de competência de um neurônio
        ✅ CORREÇÃO #7bis: Acumula updates pendentes para aplicar no training
        
        Args:
            neuron_idx: índice do neurônio (0 a num_neurons-1)
            reward: recompensa recebida (-1.0 a 1.0)
            lr: learning rate para update
            allow_gradients: se True, acumula para backward posterior
        """
        if allow_gradients and self.training:
            # Accumulate pending update to apply during training backward
            self.pending_updates.append((int(neuron_idx), float(reward), float(lr)))
        else:
            # EMA-like no-grad update for inference
            with torch.no_grad():
                self.competence[neuron_idx] += lr * reward
                self.competence.clamp_(min=0.0, max=10.0)
    
    def get_competence_loss(self):
        """
        Gera loss de competence baseado em pending updates
        Deve ser chamado DURANTE training, ANTES do backward principal
        """
        if not self.pending_updates:
            return None
        
        # Build loss from all pending updates
        total_loss = torch.tensor(0.0, device=self.competence.device, requires_grad=True)
        
        for idx, reward, lr in self.pending_updates:
            # Maximize competence for positive rewards
            # Loss = -reward * lr * competence[idx]
            coef = reward * lr
            total_loss = total_loss + (-coef * self.competence[idx])
        
        # Clear pending updates
        self.pending_updates.clear()
        
        return total_loss
    
    def apply_pending_updates_ema(self):
        """Fallback: aplica updates via EMA se não houver backward"""
        if not self.pending_updates:
            return
        
        with torch.no_grad():
            for idx, reward, lr in self.pending_updates:
                self.competence[idx] += lr * reward
            self.competence.clamp_(min=0.0, max=10.0)
        
        self.pending_updates.clear()
    
    def get_activation_stats(self) -> Dict:
        """Retorna estatísticas de ativação"""
        # Clamp k para não exceder num_neurons
        k_top = min(10, self.num_neurons)
        
        return {
            'mean_activations': self.activation_ema.mean().item(),
            'std_activations': self.activation_ema.std().item(),
            'max_activations': self.activation_ema.max().item(),
            'min_activations': self.activation_ema.min().item(),
            'active_neurons': (self.activation_ema > 0).sum().item(),
            'top_10_indices': torch.topk(self.activation_ema, k_top).indices.tolist() if k_top > 0 else [],
        }


class AdaptiveRouter(IA3Router):
    """
    Router adaptativo que ajusta top_k e temperature dinamicamente
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Parâmetros adaptativos (como tensors escalares para state_dict)
        self.register_buffer('performance_ema', torch.tensor(0.5))
        self.register_buffer('exploration_phase', torch.tensor(1.0))
        
        self.min_k = max(8, self.top_k // 4)
        self.max_k = min(self.num_neurons, self.top_k * 2)
        
        self.min_temp = 0.5
        self.max_temp = 2.0
    
    def adapt_parameters(self, reward: float, chaos_signal: float = 0.0):
        """
        Adapta top_k e temperature baseado em performance
        
        Args:
            reward: recompensa atual (0-1)
            chaos_signal: sinal de CAOS+ do PENIN-Ω (0-1)
        """
        with torch.no_grad():
            # EMA de performance (FIX Bug #27: use .fill_() para tensor buffer)
            new_perf = 0.9 * self.performance_ema.item() + 0.1 * reward
            self.performance_ema.fill_(new_perf)
            
            perf = self.performance_ema.item()
            
            # Se performance alta, diminui exploration
            # Se performance baixa, aumenta exploration
            if perf > 0.7:
                # Exploit: menos neurônios, menos temperatura
                self.top_k = max(self.min_k, int(self.top_k * 0.95))
                self.top_k = min(self.top_k, self.num_neurons)  # Clamp
                self.temperature = max(self.min_temp, self.temperature * 0.98)
                
                # FIX Bug #27: atualizar tensor buffer corretamente
                new_phase = self.exploration_phase.item() * 0.99
                self.exploration_phase.fill_(new_phase)
                
            elif perf < 0.3:
                # Explore: mais neurônios, mais temperatura
                self.top_k = min(self.max_k, int(self.top_k * 1.05))
                self.top_k = min(self.top_k, self.num_neurons)  # Clamp
                self.temperature = min(self.max_temp, self.temperature * 1.02)
                
                # FIX Bug #27: atualizar tensor buffer corretamente
                new_phase = min(1.0, self.exploration_phase.item() * 1.01)
                self.exploration_phase.fill_(new_phase)
            
            # CAOS+ force exploration
            if chaos_signal > 0.7:
                self.temperature = min(self.max_temp, self.temperature + 0.1)
                self.top_k = min(self.max_k, self.top_k + 4)
                self.top_k = min(self.top_k, self.num_neurons)  # Clamp
            
            # Adapta EMA decay também (Bug #13 fix)
            self.adapt_ema_decay(perf)

            # Final safety clamp (rigorous)
            self.top_k = max(1, min(int(self.top_k), int(self.num_neurons)))
            self.temperature = max(self.min_temp, min(float(self.temperature), self.max_temp))
    
    def adapt_ema_decay(self, performance: float):
        """
        Adapta EMA decay baseado em performance (Bug #13)
        
        Args:
            performance: performance atual (0-1)
        """
        with torch.no_grad():
            if performance > 0.7:
                # Performance alta: memória mais curta (exploita)
                self.ema_decay = max(0.90, self.ema_decay * 0.995)
            elif performance < 0.3:
                # Performance baixa: memória mais longa (explora)
                self.ema_decay = min(0.99, self.ema_decay * 1.005)


if __name__ == "__main__":
    print("🎯 IA³ Router Module")
    
    # Test
    router = AdaptiveRouter(H=1024, num_neurons=1000, top_k=64)
    z = torch.randn(4, 1024)
    
    mask, selected = router(z)
    print(f"Selected {len(selected)} neurons: {selected[:10]}...")
    print(f"Stats: {router.get_activation_stats()}")
