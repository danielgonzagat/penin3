"""
higher Integration - COMPLETO - Meta-learning for PyTorch
Merge REAL do /root/higher com MAML e few-shot learning
"""
import logging
import sys
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

logger = logging.getLogger(__name__)

# Try to import from installed higher
try:
    sys.path.insert(0, '/root/higher')
    import higher
    HIGHER_AVAILABLE = True
    logger.info("✅ higher imported successfully from /root/higher")
except ImportError as e:
    logger.warning(f"higher not available: {e}")
    HIGHER_AVAILABLE = False

class HigherMetaLearner:
    """
    higher-based meta-learning - PRODUCTION READY
    MAML (Model-Agnostic Meta-Learning) and few-shot learning
    """
    
    def __init__(self, model: Optional[nn.Module] = None,
                 meta_lr: float = 0.001,
                 inner_lr: float = 0.01,
                 inner_steps: int = 5):
        self.higher_available = HIGHER_AVAILABLE
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.inner_steps = inner_steps
        self.meta_steps = 0
        
        if model and HIGHER_AVAILABLE:
            self.meta_optimizer = optim.Adam(model.parameters(), lr=meta_lr)
        else:
            self.meta_optimizer = None
        
        logger.info(f"🎓 higher Meta-Learner initialized (available: {HIGHER_AVAILABLE})")
        logger.info(f"   Meta LR: {meta_lr}, Inner LR: {inner_lr}, Inner steps: {inner_steps}")
    
    def maml_step(self, tasks: List[Dict[str, torch.Tensor]], 
                  loss_fn: Callable) -> Dict[str, float]:
        """
        Execute one MAML meta-learning step
        
        Args:
            tasks: List of tasks, each with 'support' and 'query' data
                   {'support': (X, y), 'query': (X_q, y_q)}
            loss_fn: Loss function to use
        
        Returns:
            Meta-learning metrics
        """
        if not self.higher_available or self.model is None:
            return {'error': 'higher not available', 'meta_loss': 0.0}
        
        try:
            self.meta_optimizer.zero_grad()
            meta_loss = 0.0
            task_losses = []
            
            for task in tasks:
                # Get support and query data
                X_support, y_support = task['support']
                X_query, y_query = task['query']
                
                # Inner loop: adapt to task with higher
                with higher.innerloop_ctx(
                    self.model, 
                    self.meta_optimizer,
                    copy_initial_weights=True,
                    track_higher_grads=True
                ) as (fmodel, diffopt):
                    
                    # Inner loop optimization
                    for _ in range(self.inner_steps):
                        support_pred = fmodel(X_support)
                        support_loss = loss_fn(support_pred, y_support)
                        diffopt.step(support_loss)
                    
                    # Evaluate on query set (meta-loss)
                    query_pred = fmodel(X_query)
                    query_loss = loss_fn(query_pred, y_query)
                    task_losses.append(query_loss.item())
                    
                    # Accumulate meta-gradient
                    query_loss.backward()
                    meta_loss += query_loss.item()
            
            # Meta-optimization step
            self.meta_optimizer.step()
            self.meta_steps += 1
            
            meta_loss /= len(tasks)
            
            metrics = {
                'meta_loss': meta_loss,
                'task_losses': task_losses,
                'meta_steps': self.meta_steps,
                'num_tasks': len(tasks)
            }
            
            logger.info(f"✅ MAML step complete: meta_loss={meta_loss:.4f}, tasks={len(tasks)}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"MAML step failed: {e}")
            return {'error': str(e), 'meta_loss': 0.0}
    
    def few_shot_adapt(self, support_data: tuple, 
                       loss_fn: Callable,
                       inner_steps: Optional[int] = None) -> nn.Module:
        """
        Adapt model to new task with few-shot learning
        
        Args:
            support_data: (X, y) support set for adaptation
            loss_fn: Loss function
            inner_steps: Number of adaptation steps (default: self.inner_steps)
        
        Returns:
            Adapted model
        """
        if not self.higher_available or self.model is None:
            logger.warning("Cannot adapt: higher not available")
            return self.model
        
        try:
            X_support, y_support = support_data
            steps = inner_steps or self.inner_steps
            
            # Create temporary optimizer for adaptation
            temp_optimizer = optim.SGD(self.model.parameters(), lr=self.inner_lr)
            
            # Adapt with higher
            with higher.innerloop_ctx(
                self.model,
                temp_optimizer,
                copy_initial_weights=True,
                track_higher_grads=False
            ) as (fmodel, diffopt):
                
                for step in range(steps):
                    pred = fmodel(X_support)
                    loss = loss_fn(pred, y_support)
                    diffopt.step(loss)
                    
                    if step % 10 == 0:
                        logger.debug(f"   Adapt step {step}: loss={loss.item():.4f}")
                
                logger.info(f"✅ Few-shot adaptation complete ({steps} steps)")
                
                # Return adapted model
                return fmodel
        
        except Exception as e:
            logger.error(f"Few-shot adaptation failed: {e}")
            return self.model
    
    def reptile_step(self, tasks: List[Dict[str, torch.Tensor]],
                     loss_fn: Callable,
                     epsilon: float = 1.0) -> Dict[str, float]:
        """
        Execute Reptile meta-learning step (simpler alternative to MAML)
        
        Args:
            tasks: List of tasks with training data
            loss_fn: Loss function
            epsilon: Reptile step size
        
        Returns:
            Metrics
        """
        if not self.higher_available or self.model is None:
            return {'error': 'higher not available'}
        
        try:
            # Store original weights
            original_weights = [p.clone() for p in self.model.parameters()]
            meta_loss = 0.0
            
            for task in tasks:
                X_train, y_train = task['train']
                
                # Reset to original weights
                for p, orig in zip(self.model.parameters(), original_weights):
                    p.data.copy_(orig.data)
                
                # Adapt to task
                temp_opt = optim.SGD(self.model.parameters(), lr=self.inner_lr)
                
                for _ in range(self.inner_steps):
                    temp_opt.zero_grad()
                    pred = self.model(X_train)
                    loss = loss_fn(pred, y_train)
                    loss.backward()
                    temp_opt.step()
                    meta_loss += loss.item()
                
                # Update original weights towards adapted weights
                for p, orig in zip(self.model.parameters(), original_weights):
                    orig.data.add_(p.data - orig.data, alpha=epsilon / len(tasks))
            
            # Apply meta-update
            for p, orig in zip(self.model.parameters(), original_weights):
                p.data.copy_(orig.data)
            
            self.meta_steps += 1
            meta_loss /= (len(tasks) * self.inner_steps)
            
            logger.info(f"✅ Reptile step complete: meta_loss={meta_loss:.4f}")
            
            return {
                'meta_loss': meta_loss,
                'meta_steps': self.meta_steps,
                'num_tasks': len(tasks)
            }
            
        except Exception as e:
            logger.error(f"Reptile step failed: {e}")
            return {'error': str(e)}
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive higher statistics"""
        return {
            'higher_available': self.higher_available,
            'model_set': self.model is not None,
            'meta_optimizer_set': self.meta_optimizer is not None,
            'meta_steps': self.meta_steps,
            'meta_lr': self.meta_lr,
            'inner_lr': self.inner_lr,
            'inner_steps': self.inner_steps
        }
    
    def is_ready(self) -> bool:
        """Check if meta-learner is ready"""
        return self.higher_available and self.model is not None

