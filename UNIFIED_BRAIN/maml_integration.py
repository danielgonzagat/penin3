#!/usr/bin/env python3
"""
✅ MAML INTEGRATION - Implementação completa para UNIFIED_BRAIN
Meta-learning para fast adaptation
"""

import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')
sys.path.insert(0, '/root')

import torch
import torch.nn as nn
from typing import List, Dict, Tuple
from brain_logger import brain_logger

# Import MAML components
try:
    from intelligence_system.extracted_algorithms.maml_engine import MAMLEngine, Task
    MAML_AVAILABLE = True
except Exception as e:
    brain_logger.error(f"MAML import failed: {e}")
    MAML_AVAILABLE = False

class BrainMAMLWrapper(nn.Module):
    """Wrapper do brain para MAML"""
    
    def __init__(self, brain_core, obs_dim: int = 4, act_dim: int = 2):
        super().__init__()
        self.brain = brain_core
        H = brain_core.H
        
        # Encoder/decoder simplificados
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, H),
            nn.LayerNorm(H),
            nn.GELU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(H, act_dim)
        )
    
    def forward(self, x):
        """x: [batch, obs_dim] → [batch, act_dim]"""
        z = self.encoder(x)
        z_out = self.brain.step(z, reward=None, chaos_signal=0.0)[0]
        return self.decoder(z_out)

class MAMLBrainAdapter:
    """
    Adapta MAML para funcionar com UNIFIED_BRAIN
    Permite fast adaptation a novas tasks
    """
    
    def __init__(self, brain_core, inner_lr=0.01, outer_lr=0.001):
        if not MAML_AVAILABLE:
            brain_logger.error("❌ MAML não disponível")
            self.enabled = False
            return
        
        self.brain = brain_core
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        
        try:
            # Wrapper MAML-compatible
            self.wrapper = BrainMAMLWrapper(brain_core)
            
            # MAML engine
            self.maml_engine = MAMLEngine(
                model=self.wrapper,
                inner_lr=inner_lr,
                outer_lr=outer_lr,
                inner_steps=5,
                first_order=True  # Mais rápido
            )
            
            self.adaptation_count = 0
            self.meta_updates = 0
            
            self.enabled = True
            brain_logger.info("✅ MAML Integration ACTIVE")
            
        except Exception as e:
            brain_logger.error(f"MAML init failed: {e}")
            self.enabled = False
    
    def generate_synthetic_task(self) -> Task:
        """Gera task sintética para meta-training"""
        batch_size = 16
        obs_dim = 4
        act_dim = 2
        
        # Support set
        support_x = torch.randn(batch_size, obs_dim)
        support_y = torch.randint(0, act_dim, (batch_size,))
        
        # Query set
        query_x = torch.randn(batch_size, obs_dim)
        query_y = torch.randint(0, act_dim, (batch_size,))
        
        return Task(
            support_x=support_x,
            support_y=support_y,
            query_x=query_x,
            query_y=query_y,
            task_id=f"synthetic_{self.adaptation_count}"
        )
    
    def meta_train_step(self, num_tasks: int = 4) -> Dict:
        """
        Um passo de meta-training
        
        Args:
            num_tasks: Número de tasks para meta-batch
            
        Returns:
            Métricas do meta-update
        """
        if not self.enabled:
            return {'success': False, 'reason': 'disabled'}
        
        try:
            # Gerar tasks
            tasks = [self.generate_synthetic_task() for _ in range(num_tasks)]
            
            # Meta-train
            metrics = self.maml_engine.outer_loop(tasks)
            
            self.meta_updates += 1
            
            brain_logger.info(
                f"🧠 MAML meta-update {self.meta_updates}: "
                f"loss={metrics['meta_loss']:.4f}, "
                f"acc={metrics['query_accuracy']:.2%}"
            )
            
            return {
                'success': True,
                'meta_loss': metrics['meta_loss'],
                'query_accuracy': metrics['query_accuracy'],
                'meta_updates': self.meta_updates
            }
            
        except Exception as e:
            brain_logger.error(f"MAML meta-train failed: {e}")
            return {'success': False, 'error': str(e)}
    
    def fast_adapt(self, support_data: Tuple) -> Dict:
        """
        Fast adaptation a nova task
        
        Args:
            support_data: (support_x, support_y)
            
        Returns:
            Métricas de adaptação
        """
        if not self.enabled:
            return {'success': False, 'reason': 'disabled'}
        
        try:
            support_x, support_y = support_data
            
            # Criar task com support set
            # (query set será ignorado para fast adapt)
            task = Task(
                support_x=support_x,
                support_y=support_y,
                query_x=support_x,  # Dummy
                query_y=support_y,  # Dummy
                task_id=f"adapt_{self.adaptation_count}"
            )
            
            # Inner loop (fast adaptation)
            adapted_model, support_loss = self.maml_engine.inner_loop(task)
            
            self.adaptation_count += 1
            
            brain_logger.info(
                f"⚡ Fast adapted: loss={support_loss:.4f}, "
                f"adaptations={self.adaptation_count}"
            )
            
            return {
                'success': True,
                'support_loss': support_loss,
                'adaptation_count': self.adaptation_count
            }
            
        except Exception as e:
            brain_logger.error(f"Fast adapt failed: {e}")
            return {'success': False, 'error': str(e)}

# Test if can be imported
if __name__ == "__main__":
    brain_logger.info("🧠 MAML Integration - standalone test")
    
    if MAML_AVAILABLE:
        print("✅ MAML disponível e pronto para integração")
    else:
        print("❌ MAML não disponível")