#!/usr/bin/env python3
"""
🤔 META CURIOSITY MODULE - P3.3
Aprende O QUE ser curioso (meta-aprendizado de curiosidade)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import logging
from collections import deque
from typing import Dict, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MetaCuriosity')

class MetaCuriosityModule(nn.Module):
    """Aprende O QUE ser curioso - curiosidade auto-aprendida"""
    
    def __init__(self, H: int = 1024):
        super().__init__()
        
        # Predictor (forward model - prediz próximo estado)
        self.predictor = nn.Sequential(
            nn.Linear(H + 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, H)
        )
        
        # META: Curiosity Reward Learner
        # Aprende a atribuir valor intrínseco a diferentes tipos de surprise
        self.curiosity_reward_learner = nn.Sequential(
            nn.Linear(H * 2 + 1, 64),  # obs, next_obs, surprise
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # 0-1 curiosity reward
        )
        
        # Optimizer para o learner
        self.learner_optimizer = torch.optim.Adam(
            self.curiosity_reward_learner.parameters(),
            lr=1e-3
        )
        
        # Predictor optimizer
        self.predictor_optimizer = torch.optim.Adam(
            self.predictor.parameters(),
            lr=3e-4
        )
        
        # História para meta-learning
        self.curiosity_history = deque(maxlen=10000)
        
        # Métricas
        self.total_predictions = 0
        self.meta_learning_steps = 0
        
        logger.info("🤔 Meta Curiosity Module initialized")
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Computa surprise E curiosity reward meta-aprendido
        
        Args:
            obs: Estado atual [batch, H]
            action: Ação tomada [batch, 2]
            next_obs: Próximo estado [batch, H]
        
        Returns:
            {
                'surprise': tensor,
                'curiosity_reward': tensor,
                'total_intrinsic': tensor
            }
        """
        # Garantir batch dimension
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        if next_obs.dim() == 1:
            next_obs = next_obs.unsqueeze(0)
        
        # 1. Predizer próximo estado
        pred_input = torch.cat([obs, action], dim=-1)
        pred_next = self.predictor(pred_input)
        
        # 2. Surprise = prediction error
        surprise = F.mse_loss(pred_next, next_obs, reduction='none').mean(dim=-1, keepdim=True)
        
        # 3. META: Curiosity reward baseado em contexto
        # Quanto reward dar para ESTA surprise neste contexto?
        context = torch.cat([obs, next_obs, surprise], dim=-1)
        curiosity_reward = self.curiosity_reward_learner(context)
        
        # 4. Intrinsic total = surprise * curiosity_reward
        total_intrinsic = surprise * curiosity_reward
        
        # 5. Registrar para meta-learning
        self._record_for_meta_learning(obs, next_obs, surprise, curiosity_reward)
        
        self.total_predictions += 1
        
        return {
            'surprise': surprise,
            'curiosity_reward': curiosity_reward,
            'total_intrinsic': total_intrinsic
        }
    
    def _record_for_meta_learning(
        self, 
        obs: torch.Tensor, 
        next_obs: torch.Tensor, 
        surprise: torch.Tensor, 
        curiosity_reward: torch.Tensor
    ):
        """Registra experiência para meta-learning posterior"""
        self.curiosity_history.append({
            'obs': obs.detach().cpu(),
            'next_obs': next_obs.detach().cpu(),
            'surprise': surprise.detach().cpu().item(),
            'curiosity_reward': curiosity_reward.detach().cpu().item(),
            'actual_reward': None  # Será preenchido depois
        })
    
    def learn_curiosity(self, actual_reward: float):
        """
        Meta-aprendizado: aprende O QUE curiosidade é valiosa
        
        Objetivo: curiosity_reward deve predizer actual_reward futuro
        
        Args:
            actual_reward: Reward real obtido após exploração curiosa
        """
        # Atualizar última entry com actual_reward
        if self.curiosity_history:
            self.curiosity_history[-1]['actual_reward'] = actual_reward
        
        # Treinar se temos batch suficiente
        if len(self.curiosity_history) < 64:
            return
        
        # Sample batch
        batch = random.sample([h for h in self.curiosity_history if h['actual_reward'] is not None], 
                              min(32, len(self.curiosity_history)))
        
        if len(batch) < 10:
            return
        
        # Preparar dados
        obs_batch = torch.stack([b['obs'] for b in batch]).squeeze(1)
        next_obs_batch = torch.stack([b['next_obs'] for b in batch]).squeeze(1)
        surprise_batch = torch.tensor([[b['surprise']] for b in batch])
        actual_rewards = torch.tensor([[b['actual_reward']] for b in batch])
        
        # Normalizar rewards [-1, 1]
        actual_rewards = torch.tanh(actual_rewards / 100.0)
        
        # Forward
        context = torch.cat([obs_batch, next_obs_batch, surprise_batch], dim=-1)
        predicted_curiosity = self.curiosity_reward_learner(context)
        
        # Loss: curiosity_reward deve predizer actual_reward
        loss = F.mse_loss(predicted_curiosity, actual_rewards)
        
        # Backward
        self.learner_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.curiosity_reward_learner.parameters(), 1.0)
        self.learner_optimizer.step()
        
        self.meta_learning_steps += 1
        
        if self.meta_learning_steps % 100 == 0:
            logger.info(f"🎓 Meta-learning step {self.meta_learning_steps}: loss={loss.item():.4f}")
    
    def train_predictor(self, obs: torch.Tensor, action: torch.Tensor, next_obs: torch.Tensor):
        """Treina forward model (predictor)"""
        # Garantir batch
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            action = action.unsqueeze(0)
            next_obs = next_obs.unsqueeze(0)
        
        # Predict
        pred_input = torch.cat([obs, action], dim=-1)
        pred_next = self.predictor(pred_input)
        
        # Loss
        loss = F.mse_loss(pred_next, next_obs)
        
        # Backward
        self.predictor_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.predictor_optimizer.step()
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do módulo"""
        if not self.curiosity_history:
            return {}
        
        recent = list(self.curiosity_history)[-1000:]
        
        avg_surprise = sum(h['surprise'] for h in recent) / len(recent)
        avg_curiosity_reward = sum(h['curiosity_reward'] for h in recent) / len(recent)
        
        rewards_filled = [h for h in recent if h['actual_reward'] is not None]
        avg_actual_reward = sum(h['actual_reward'] for h in rewards_filled) / max(len(rewards_filled), 1)
        
        return {
            'total_predictions': self.total_predictions,
            'meta_learning_steps': self.meta_learning_steps,
            'avg_surprise': avg_surprise,
            'avg_curiosity_reward': avg_curiosity_reward,
            'avg_actual_reward': avg_actual_reward,
            'history_size': len(self.curiosity_history)
        }


if __name__ == "__main__":
    # Teste
    H = 1024
    meta_curiosity = MetaCuriosityModule(H)
    
    # Simular episódio
    for step in range(100):
        obs = torch.randn(H)
        action = torch.tensor([0.5, 0.3])
        next_obs = torch.randn(H)
        
        # Forward
        result = meta_curiosity(obs, action, next_obs)
        
        print(f"Step {step}: surprise={result['surprise'].item():.3f}, "
              f"curiosity={result['curiosity_reward'].item():.3f}, "
              f"intrinsic={result['total_intrinsic'].item():.3f}")
        
        # Treinar predictor
        meta_curiosity.train_predictor(obs, action, next_obs)
        
        # Meta-learning (simular reward)
        if step > 10:
            actual_reward = random.uniform(-10, 10)
            meta_curiosity.learn_curiosity(actual_reward)
    
    # Stats
    stats = meta_curiosity.get_stats()
    print("\nStats:", stats)