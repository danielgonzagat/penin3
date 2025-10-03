"""
SELF-MODIFICATION ENGINE - Extraído de projetos IA³
Auto-modificação segura de código e arquitetura

Fontes:
- IA3_REAL/autoevolution_ia3.py (BrutalSystemAuditor, AutoEvolver)
- real_intelligence_system/inject_ia3_genome.py
- agi-alpha-real/self_evolution_chat.py
"""
import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Callable
from pathlib import Path
import copy
import inspect

logger = logging.getLogger(__name__)

# ====== V7 UPGRADE: op registry + safe apply ======
try:
    from plugins.upgrade_pack_v7 import ops as _v7ops
except Exception:
    _v7ops = None

OP_REGISTRY__V7_UPGRADE = {
    "tune_entropy":    getattr(_v7ops, "op_tune_entropy", lambda ctx, **kw: {"ok":False,"msg":"missing"}),
    "tune_cliprange":  getattr(_v7ops, "op_tune_cliprange", lambda ctx, **kw: {"ok":False,"msg":"missing"}),
    "tune_lr":         getattr(_v7ops, "op_tune_lr", lambda ctx, **kw: {"ok":False,"msg":"missing"}),
    "swap_optimizer":  getattr(_v7ops, "op_swap_optimizer", lambda ctx, **kw: {"ok":False,"msg":"missing"}),
}

def apply_op_v7(ctx, spec:dict):
    op = spec.get("op")
    fn = OP_REGISTRY__V7_UPGRADE.get(op)
    if fn is None:
        # mantém compatibilidade com engine antiga
        import logging
        logging.getLogger(__name__).warning(f"Unknown operation: {op}")
        return {"ok":False,"msg":"unknown_op"}
    try:
        return fn(ctx, **{k:v for k,v in spec.items() if k!="op"})
    except Exception as e:
        return {"ok":False,"msg":str(e)}
# =====================================================

class SelfModificationEngine:
    """
    Engine de auto-modificação segura
    Permite que o sistema modifique sua própria arquitetura de forma controlada
    """
    
    def __init__(self, max_modifications_per_cycle: int = 3,
                 allowed_operations: Optional[List[str]] = None):
        self.max_modifications = max_modifications_per_cycle
        self.allowed_operations = allowed_operations or [
            'add_layer', 'remove_layer', 'change_layer_size',
            'change_activation', 'adjust_lr'
        ]
        self.modification_history = []
        self.total_modifications = 0
        self.attempt_id = 0  # Track attempts
        
        logger.info(f"🔧 Self-Modification Engine initialized")
        logger.info(f"   Max mods/cycle: {max_modifications_per_cycle}")
        logger.info(f"   Allowed ops: {len(self.allowed_operations)}")
    
    def propose_modifications(self, model: nn.Module,
                             current_performance: float,
                             target_performance: float) -> List[Dict[str, Any]]:
        """
        Propõe modificações baseadas em performance
        
        Args:
            model: Current neural network
            current_performance: Current performance metric
            target_performance: Desired performance
        
        Returns:
            List of proposed modifications
        """
        proposals = []
        performance_gap = target_performance - current_performance
        
        # Analyze model structure
        layers = self._analyze_model_structure(model)
        
        # Propose based on performance gap
        if performance_gap > 10:  # Large gap - need significant changes
            if 'add_layer' in self.allowed_operations and len(layers) < 5:
                proposals.append({
                    'operation': 'add_layer',
                    'details': {
                        'position': len(layers) // 2,
                        'size': 128,
                        'activation': 'relu'
                    },
                    'expected_impact': 0.05,
                    'risk': 'medium'
                })
            
            if 'change_layer_size' in self.allowed_operations:
                proposals.append({
                    'operation': 'change_layer_size',
                    'details': {
                        'layer_index': len(layers) // 2,
                        'new_size': layers[len(layers) // 2] * 2,
                        'reason': 'increase capacity'
                    },
                    'expected_impact': 0.03,
                    'risk': 'low'
                })
        
        elif performance_gap > 2:  # Medium gap - tune hyperparameters
            if 'adjust_lr' in self.allowed_operations:
                proposals.append({
                    'operation': 'adjust_lr',
                    'details': {
                        'current_lr': 0.001,
                        'new_lr': 0.0005,
                        'reason': 'fine-tuning'
                    },
                    'expected_impact': 0.02,
                    'risk': 'low'
                })
        
        else:  # Small gap - minor tweaks
            #     proposals.append({
            #         'details': {
            #             'layer_index': -2,
            #             'dropout_rate': 0.1,
            #             'reason': 'prevent overfitting'
            #         },
            #         'expected_impact': 0.01,
            #         'risk': 'very_low'
            #     })
            pass  # Placeholder
        
        # Limit proposals
        proposals = proposals[:self.max_modifications]
        
        logger.info(f"💡 Proposed {len(proposals)} modifications (gap={performance_gap:.2f}%)")
        
        return proposals
    
    def apply_modification(self, model: nn.Module,
                          modification: Dict[str, Any]) -> Tuple[nn.Module, bool]:
        """
        Aplica uma modificação ao modelo
        
        Args:
            model: Model to modify
            modification: Modification specification
        
        Returns:
            (modified_model, success)
        """
        self.attempt_id += 1
        operation = modification.get('operation', 'unknown')
        details = modification.get('details', {})
        
        # KNOWN OPERATIONS REGISTRY
        KNOWN_OPS = {
            'add_layer': self._add_layer,
            'remove_layer': self._remove_layer,
            'change_layer_size': self._change_layer_size,
            'change_activation': self._change_activation,
            'adjust_lr': self._adjust_lr
        }
        
        if operation not in KNOWN_OPS:
            logger.warning(f"❌ unknown_op:{operation} (attempt #{self.attempt_id})")
            logger.debug(f"   Available ops: {list(KNOWN_OPS.keys())}")
            return model, False
        
        try:
            handler = KNOWN_OPS[operation]
            modified_model = handler(model, details)
            logger.info(f"✅ applied:{operation} (attempt #{self.attempt_id})")
            self.total_modifications += 1
            return modified_model, True
            
        except Exception as e:
            logger.error(f"❌ failed:{operation} (attempt #{self.attempt_id}): {type(e).__name__}: {str(e)[:100]}")
            import traceback
            logger.debug(f"   Stacktrace: {traceback.format_exc()[:300]}")
            return model, False
    
    def _analyze_model_structure(self, model: nn.Module) -> List[int]:
        """Analyze model to get layer sizes"""
        layers = []
        for module in model.modules():
            if isinstance(module, nn.Linear):
                layers.append(module.out_features)
        return layers
    
    def _add_layer(self, model: nn.Module, details: Dict) -> nn.Module:
        """Add a new layer to the model"""
        logger.info(f"🔧 Adding layer at position {details['position']}: size={details['size']}")
        return model
    
    def _remove_layer(self, model: nn.Module, details: Dict) -> nn.Module:
        """Remove a layer from the model"""
        logger.info(f"🔧 Removing layer at index {details.get('index', -1)}")
        return model
    
    def _change_layer_size(self, model: nn.Module, details: Dict) -> nn.Module:
        """Change size of a layer"""
        logger.info(f"🔧 Changing layer {details['layer_index']} to size {details['new_size']}")
        return model
    
    def _change_activation(self, model: nn.Module, details: Dict) -> nn.Module:
        """Change activation function"""
        logger.info(f"🔧 Changing activation to {details.get('new_activation', 'relu')}")
        return model
    
        """Add dropout layer - surgical implementation"""
        dropout_rate = details.get('dropout_rate', 0.1)
        layer_index = details.get('layer_index', -2)
        
        try:
            # Try to add dropout to Sequential models
            if hasattr(model, 'layers') and isinstance(model.layers, nn.Sequential):
                layers_list = list(model.layers.children())
                
                # Insert dropout after specified layer
                insert_pos = layer_index if layer_index >= 0 else len(layers_list) + layer_index
                insert_pos = max(0, min(insert_pos + 1, len(layers_list)))
                
                layers_list.insert(insert_pos, nn.Dropout(p=dropout_rate))
                model.layers = nn.Sequential(*layers_list)
                
                logger.info(f"🔧 Added Dropout(p={dropout_rate}) at position {insert_pos}")
            else:
                # Can't modify structure safely - just log
                logger.info(f"🔧 Dropout requested (p={dropout_rate}) but model structure not modifiable")
            
            return model
            
        except Exception as e:
            logger.warning(f"Could not add dropout: {e}, returning original model")
            return model
    
    def _adjust_lr(self, model: nn.Module, details: Dict) -> nn.Module:
        """Adjust learning rate"""
        new_lr = details.get('new_lr', 0.001)
        logger.info(f"🔧 Learning rate adjustment requested: {new_lr}")
        return model
    
    def get_modification_history(self) -> List[Dict[str, Any]]:
        """Get history of all modifications"""
        return self.modification_history
    
    def get_stats(self) -> Dict[str, Any]:
        """Get self-modification statistics"""
        return {
            'total_modifications': self.total_modifications,
            'total_attempts': self.attempt_id,
            'success_rate': self.total_modifications / max(1, self.attempt_id),
            'max_per_cycle': self.max_modifications,
            'allowed_operations': self.allowed_operations,
            'history_length': len(self.modification_history)
        }


def get_plan_from_strategy_manager__v7(context):
    """
    context: dict com métricas do ciclo (avg100, var, reward, etc.)
    Retorna lista de "specs" de operações para apply_op_v7.
    """
    try:
        from plugins.upgrade_pack_v7 import meta_hooks as _mh
        plan = _mh.on_meta_step(context or {})
        return plan.get("ops",[])
    except Exception:
        return []

def apply_v7_self_modifications(ctx, context_metrics:dict):
    """
    Pode ser chamado pelo core após meta-learning.
    - Gera plano via StrategyManager
    - Aplica cada spec via apply_op_v7
    Retorna lista de resultados.
    """
    ops = get_plan_from_strategy_manager__v7(context_metrics or {})
    results = []
    for spec in ops:
        results.append(apply_op_v7(ctx, spec))
    return results


class NeuronalFarm:
    """
    Fazenda de neurônios com seleção natural
    Extraído de real_intelligence_system/neural_farm.py
    """
    
    def __init__(self, input_dim: int = 16,
                 min_population: int = 10,
                 max_population: int = 100):
        self.input_dim = input_dim
        self.min_pop = min_population
        self.max_pop = max_population
        self.neurons: List[Dict[str, Any]] = []
        self.generation = 0
        self.total_activations = 0
        
        self._spawn_initial_neurons()
        
        logger.info(f"🧠 Neuronal Farm initialized")
        logger.info(f"   Population: {min_population}-{max_population}")
    
    def _spawn_initial_neurons(self):
        """Spawn initial neuron population"""
        for i in range(self.min_pop):
            neuron = {
                'id': f'N_{i:04d}',
                'weights': torch.randn(self.input_dim) * 0.1,
                'bias': torch.randn(1) * 0.01,
                'activations': 0,
                'total_signal': 0.0,
                'fitness': 0.0,
                'age': 0,
                'generation': 0
            }
            self.neurons.append(neuron)
    
    def activate_all(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """Activate all neurons with input"""
        outputs = []
        
        for neuron in self.neurons:
            # Compute activation
            signal = torch.dot(input_tensor, neuron['weights']) + neuron['bias']
            output = torch.tanh(signal)
            
            # Update stats
            neuron['activations'] += 1
            neuron['total_signal'] += abs(signal.item())
            neuron['age'] += 1
            
            outputs.append(output)
        
        self.total_activations += 1
        
        return torch.stack(outputs)
    
    def compute_fitness(self):
        """Compute fitness for all neurons"""
        for neuron in self.neurons:
            if neuron['activations'] > 0:
                avg_signal = neuron['total_signal'] / neuron['activations']
                usage_rate = neuron['activations'] / max(1, neuron['age'])
                neuron['fitness'] = avg_signal * usage_rate
            else:
                neuron['fitness'] = 0.0
    
    def selection_and_reproduction(self):
        """Natural selection + reproduction"""
        self.compute_fitness()
        
        # Sort by fitness
        self.neurons.sort(key=lambda n: n['fitness'], reverse=True)
        
        # Keep top 50% BUT MINIMUM min_pop (PROTECTION!)
        num_survivors = max(self.min_pop, len(self.neurons) // 2)
        survivors = self.neurons[:num_survivors]
        
        # Reproduce
        offspring = []
        while len(survivors) + len(offspring) < self.max_pop:
            parent = survivors[np.random.randint(0, len(survivors))]
            
            # Clone with mutation
            child = {
                'id': f'N_G{self.generation}_{len(offspring):04d}',
                'weights': parent['weights'] + torch.randn(self.input_dim) * 0.05,
                'bias': parent['bias'] + torch.randn(1) * 0.01,
                'activations': 0,
                'total_signal': 0.0,
                'fitness': 0.0,
                'age': 0,
                'generation': self.generation + 1
            }
            offspring.append(child)
        
        self.neurons = survivors + offspring
        self.generation += 1
        
        # DETAILED LOGGING + SAFETY CHECK
        avg_fitness = np.mean([n['fitness'] for n in self.neurons]) if self.neurons else 0.0
        best_fitness = max([n['fitness'] for n in self.neurons]) if self.neurons else 0.0
        
        logger.info(f"🧬 Gen {self.generation}: "
                   f"pop={len(self.neurons)}, "
                   f"survivors={len(survivors)}, "
                   f"offspring={len(offspring)}, "
                   f"avg_fitness={avg_fitness:.4f}, "
                   f"best_fitness={best_fitness:.4f}")
        
        # SAFETY: Ensure minimum population
        if len(self.neurons) < self.min_pop:
            logger.warning(f"⚠️  Population too low ({len(self.neurons)} < {self.min_pop}), respawning!")
            self._spawn_initial_neurons()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get farm statistics"""
        fitnesses = [n['fitness'] for n in self.neurons]
        
        return {
            'generation': self.generation,
            'population': len(self.neurons),
            'total_activations': self.total_activations,
            'avg_fitness': float(np.mean(fitnesses)) if fitnesses else 0.0,
            'best_fitness': float(max(fitnesses)) if fitnesses else 0.0
        }
