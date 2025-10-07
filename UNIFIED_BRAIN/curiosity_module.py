#!/usr/bin/env python3
"""
🎯 CURIOSITY MODULE
Intrinsic Curiosity Module (ICM) - Drive interno
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from brain_logger import brain_logger

class CuriosityModule(nn.Module):
    """
    ICM: Reward intrínseco baseado em surpresa
    Cérebro é recompensado por encontrar situações novas/surpreendentes
    """
    def __init__(self, H=1024, action_dim=2):
        super().__init__()
        self.H = H
        self.action_dim = action_dim
        
        # Forward model: prediz próximo estado dado (estado, ação)
        self.forward_model = nn.Sequential(
            nn.Linear(H + action_dim, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Linear(H, H)
        )
        
        # Inverse model: prediz ação dado (estado, próximo_estado)
        self.inverse_model = nn.Sequential(
            nn.Linear(H * 2, H),
            nn.LayerNorm(H),
            nn.GELU(),
            nn.Linear(H, action_dim)
        )
        
        # Optimizer para treinar modelos
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        
        # Estado anterior
        self.last_z = None
        self.last_action = None
        
        # Stats
        self.total_curiosity = 0.0
        self.surprises = []
    
    def compute_curiosity(self, z_current: torch.Tensor, action: int) -> float:
        """
        Calcula reward de curiosity (surpresa)
        
        Args:
            z_current: estado atual no espaço latente
            action: ação tomada
            
        Returns:
            curiosity_reward: quanto mais surpreso, mais reward
        """
        if self.last_z is None:
            self.last_z = z_current.detach()
            self.last_action = action
            return 0.0
        
        # Action one-hot
        action_onehot = torch.zeros(1, self.action_dim)
        action_onehot[0, action] = 1.0
        
        # 1. FORWARD MODEL: prediz próximo estado
        input_forward = torch.cat([self.last_z, action_onehot], dim=1)
        z_predicted = self.forward_model(input_forward)
        
        # ✅ CORREÇÃO #5: Normalizar predição para estabilidade
        z_predicted = F.layer_norm(z_predicted, (z_predicted.shape[-1],))
        
        # 2. SURPRESA = erro de predição
        surprise = F.mse_loss(z_predicted, z_current).item()
        
        # 3. INVERSE MODEL: prediz ação
        input_inverse = torch.cat([self.last_z, z_current], dim=1)
        action_predicted = self.inverse_model(input_inverse)
        
        # 4. TREINA ambos modelos
        self.optimizer.zero_grad()
        
        # Loss forward
        loss_forward = F.mse_loss(z_predicted, z_current)
        
        # Loss inverse
        action_target = torch.LongTensor([self.last_action])
        loss_inverse = F.cross_entropy(action_predicted, action_target)
        
        # Total loss
        loss = loss_forward + loss_inverse
        loss.backward()
        self.optimizer.step()
        
        # Atualiza estado
        self.last_z = z_current.detach()
        self.last_action = action
        
        # Normaliza surprise para [0, 1]
        curiosity_reward = min(1.0, surprise * 10.0)
        
        self.total_curiosity += curiosity_reward
        self.surprises.append(surprise)
        if len(self.surprises) > 1000:
            self.surprises = self.surprises[-1000:]
        
        return curiosity_reward
    
    def get_stats(self):
        """Estatísticas de curiosity"""
        if not self.surprises:
            return {}
        
        return {
            'total_curiosity': self.total_curiosity,
            'avg_surprise': sum(self.surprises) / len(self.surprises),
            'recent_surprise': self.surprises[-1] if self.surprises else 0,
            'num_predictions': len(self.surprises)
        }


if __name__ == "__main__":
    # Teste
    print("Testing Curiosity Module...")
    
    curiosity = CuriosityModule(H=1024, action_dim=2)
    
    z1 = torch.randn(1, 1024)
    z2 = torch.randn(1, 1024)
    z3 = torch.randn(1, 1024)
    
    # Step 1
    c1 = curiosity.compute_curiosity(z1, action=0)
    print(f"Step 1: curiosity={c1:.4f}")
    
    # Step 2
    c2 = curiosity.compute_curiosity(z2, action=1)
    print(f"Step 2: curiosity={c2:.4f}")
    
    # Step 3
    c3 = curiosity.compute_curiosity(z3, action=0)
    print(f"Step 3: curiosity={c3:.4f}")
    
    print(f"\nStats: {curiosity.get_stats()}")
    print("✅ Curiosity Module OK!")
