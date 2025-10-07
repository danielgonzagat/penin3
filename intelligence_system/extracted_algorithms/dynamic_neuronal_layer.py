"""
DYNAMIC NEURONAL LAYER - Extraído de IA3_SUPREME
Neurônios que crescem, morrem e se replicam dinamicamente

Fonte: IA3_SUPREME/ia3_supreme.py
"""
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import hashlib
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

class DynamicNeuron:
    """Neurônio individual com vida própria"""
    
    def __init__(self, neuron_id: str, input_dim: int):
        self.id = neuron_id
        self.weight = torch.randn(input_dim) * 0.1
        self.bias = torch.randn(1) * 0.01
        self.activation_count = 0
        self.total_signal = 0.0
        self.birth_time = time.time()
        self.importance = 1.0
    
    def fire(self, x: torch.Tensor) -> torch.Tensor:
        """Fire neuron and track activity"""
        signal = torch.sum(x * self.weight) + self.bias
        output = torch.tanh(signal)
        
        self.activation_count += 1
        self.total_signal += abs(signal.item())
        # Track last activation for selection heuristics
        try:
            self.last_activation = float(output.detach().cpu().item())
        except Exception:
            self.last_activation = 0.0
        
        return output * self.importance
    
    def should_die(self, min_age: float = 10.0, signal_threshold: float = 0.01) -> bool:
        """Determine if neuron should die"""
        age = time.time() - self.birth_time
        if age < min_age:  # Protection period
            return False
        
        avg_signal = self.total_signal / max(1, self.activation_count)
        return avg_signal < signal_threshold and self.importance < 0.1
    
    def should_replicate(self, min_activations: int = 100, signal_threshold: float = 1.0) -> bool:
        """Determine if neuron should replicate"""
        if self.activation_count < min_activations:
            return False
        
        avg_signal = self.total_signal / max(1, self.activation_count)
        return avg_signal > signal_threshold and self.importance > 0.8 and np.random.random() < 0.1


class DynamicNeuronalLayer:
    """Layer with dynamic neurons that can grow/die"""
    
    def __init__(self, input_dim: int, initial_neurons: int, layer_id: str = "DYN",
                 min_neurons: int = 3, max_neurons: int = 1000):
        self.layer_id = layer_id
        self.input_dim = input_dim
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        
        # Create initial neurons
        self.neurons = {}
        for i in range(initial_neurons):
            neuron_id = f"{layer_id}_N{i}_{hashlib.md5(str(time.time() + i).encode()).hexdigest()[:4]}"
            self.neurons[neuron_id] = DynamicNeuron(neuron_id, input_dim)
        
        self.stats = {
            'neurons_created': initial_neurons,
            'neurons_killed': 0,
            'replications': 0
        }
        
        logger.info(f"🧠 Dynamic Layer {layer_id} initialized with {initial_neurons} neurons")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dynamic neuron management"""
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        # Get all neuron outputs
        outputs = []
        to_kill = []
        to_replicate = []
        
        for neuron_id, neuron in self.neurons.items():
            output = neuron.fire(x.squeeze())
            outputs.append(output)
            
            # Check life/death/replication
            if neuron.should_die():
                to_kill.append(neuron_id)
            elif neuron.should_replicate():
                to_replicate.append(neuron_id)
        
        # Population management
        for neuron_id in to_kill:
            if len(self.neurons) > self.min_neurons:
                self._kill_neuron(neuron_id)
        
        for neuron_id in to_replicate:
            if len(self.neurons) < self.max_neurons:
                self._replicate_neuron(neuron_id)
        
        # Ensure valid output
        if not outputs:
            self._create_neuron()
            return torch.zeros(1, 1)
        
        # Stack outputs
        if len(outputs) == 1:
            return outputs[0].unsqueeze(0)
        else:
            return torch.stack(outputs).unsqueeze(0)
    
    def _create_neuron(self) -> str:
        """Create new neuron"""
        neuron_id = f"{self.layer_id}_N{len(self.neurons)}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:4]}"
        self.neurons[neuron_id] = DynamicNeuron(neuron_id, self.input_dim)
        self.stats['neurons_created'] += 1
        
        logger.debug(f"🌱 Neuron {neuron_id} born")
        
        return neuron_id
    
    def _kill_neuron(self, neuron_id: str):
        """Kill neuron"""
        if neuron_id in self.neurons:
            del self.neurons[neuron_id]
            self.stats['neurons_killed'] += 1
            logger.debug(f"💀 Neuron {neuron_id} died")
    
    def _replicate_neuron(self, parent_id: str):
        """Replicate successful neuron"""
        if parent_id not in self.neurons:
            return
        
        parent = self.neurons[parent_id]
        child_id = f"{self.layer_id}_N{len(self.neurons)}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:4]}"
        child = DynamicNeuron(child_id, self.input_dim)
        
        # Inherit with mutation
        child.weight = parent.weight.clone() + torch.randn_like(parent.weight) * 0.01
        child.bias = parent.bias.clone() + torch.randn(1) * 0.001
        child.importance = parent.importance * 0.9
        
        self.neurons[child_id] = child
        self.stats['replications'] += 1
        
        logger.debug(f"🧬 Neuron {child_id} replicated from {parent_id}")
    
    def get_stats(self) -> Dict:
        """Get layer statistics"""
        return {
            'layer_id': self.layer_id,
            'current_neurons': len(self.neurons),
            'neurons_created': self.stats['neurons_created'],
            'neurons_killed': self.stats['neurons_killed'],
            'replications': self.stats['replications']
        }

    # --- Public wrappers for safe evolve operations ---
    def replicate_best_neurons(self, k: int = 1):
        try:
            if not self.neurons:
                return
            scored = [
                (nid, getattr(n, 'last_activation', 0.0))
                for nid, n in self.neurons.items()
            ]
            scored.sort(key=lambda t: t[1], reverse=True)
            for nid, _ in scored[:max(1, k)]:
                if len(self.neurons) < self.max_neurons:
                    self._replicate_neuron(nid)
        except Exception:
            return

    def prune_weak_neurons(self, k: int = 1):
        try:
            if len(self.neurons) <= self.min_neurons:
                return
            scored = [
                (nid, getattr(n, 'last_activation', 0.0))
                for nid, n in self.neurons.items()
            ]
            scored.sort(key=lambda t: t[1])
            for nid, _ in scored[:max(1, k)]:
                if len(self.neurons) > self.min_neurons:
                    self._kill_neuron(nid)
        except Exception:
            return


if __name__ == "__main__":
    logger.info("🧪 Testing Dynamic Neuronal Layer...")

    # Test dynamic layer
    layer = DynamicNeuronalLayer(input_dim=10, initial_neurons=5, layer_id="TEST")
    
    # Simulate some activations
    for i in range(20):
        x = torch.randn(10)
        output = layer.forward(x)
    
    stats = layer.get_stats()
    logger.info(f"✅ Dynamic Layer: {stats['current_neurons']} neurons, created={stats['neurons_created']}, killed={stats['neurons_killed']}")
    
    logger.info("✅ All components working!")
