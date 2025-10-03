"""
Neural Genesis - Evolving Neural Networks
Extracted from NEURAL_GENESIS_IA3.py
"""
import torch
import torch.nn as nn
import random
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DynamicNeuron(nn.Module):
    """Neurônio dinâmico com fitness individual"""
    
    def __init__(self, neuron_id: str, input_size: int):
        super().__init__()
        self.neuron_id = neuron_id
        self.input_size = input_size
        self.weights = nn.Parameter(torch.randn(input_size) * 0.1)
        self.bias = nn.Parameter(torch.zeros(1))
        self.activation = nn.ReLU()
        self.fitness = 0.0
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Ensure correct shape
        if x.shape[-1] != self.weights.shape[0]:
            # Pad or trim
            if x.shape[-1] < self.weights.shape[0]:
                padding = torch.zeros(*x.shape[:-1], self.weights.shape[0] - x.shape[-1])
                x = torch.cat([x, padding], dim=-1)
            else:
                x = x[..., :self.weights.shape[0]]
        
        if x.dim() == 1:
            x = x.view(1, -1)
        
        signal = (x * self.weights).sum(dim=-1, keepdim=True) + self.bias
        return self.activation(signal)


class EvolvingNeuralNetwork(nn.Module):
    """
    Rede neural que evolui sua arquitetura
    
    Features REAIS:
    - Adiciona neurônios baseado em performance
    - Conexões inteligentes entre neurônios
    - Arquitetura evolui com feedback
    - Agregação por attention
    """
    
    def __init__(self, input_dim: int = 10, output_dim: int = 10):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.neurons = nn.ModuleDict()
        self.connections: Dict[str, List[str]] = {}
        self.input_layer = nn.Linear(input_dim, input_dim)
        self.output_layer = nn.Linear(input_dim, output_dim)
        self.evolution_history = []
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.input_layer(x))
        
        if not self.neurons:
            return self.output_layer(x)
        
        # Process through evolved neurons
        neuron_outputs = {}
        for nid, neuron in self.neurons.items():
            # Get inputs for this neuron
            inputs = []
            for src in self.connections.get(nid, ['input']):
                if src == 'input':
                    inputs.append(x)
                elif src in neuron_outputs:
                    inputs.append(neuron_outputs[src])
            
            if inputs:
                combined_input = torch.cat(inputs, dim=-1)
                neuron_outputs[nid] = neuron(combined_input)
        
        if neuron_outputs:
            # Attention-like aggregation
            stacked = torch.stack(list(neuron_outputs.values()), dim=0)
            weights = torch.softmax(torch.randn(len(neuron_outputs)), dim=0)
            weighted_output = torch.sum(stacked * weights.unsqueeze(-1), dim=0)
            
            # Ensure correct shape for output layer
            if weighted_output.shape[-1] < self.input_dim:
                padding = torch.zeros(*weighted_output.shape[:-1], 
                                    self.input_dim - weighted_output.shape[-1])
                weighted_output = torch.cat([weighted_output, padding], dim=-1)
            elif weighted_output.shape[-1] > self.input_dim:
                weighted_output = weighted_output[..., :self.input_dim]
            
            return self.output_layer(weighted_output)
        
        return self.output_layer(x)
    
    def evolve_architecture(self, performance_feedback: Dict = None):
        """Evolve network architecture based on performance"""
        if performance_feedback is None:
            performance_feedback = {}
        
        current_complexity = len(self.neurons)
        target_complexity = performance_feedback.get('target_complexity', current_complexity + 1)
        
        if current_complexity < target_complexity:
            # Add new neuron
            new_id = f"evolved_{len(self.neurons)}"
            input_size = self.input_dim + len(self.neurons)
            new_neuron = DynamicNeuron(new_id, input_size)
            self.neurons[new_id] = new_neuron
            
            # Create intelligent connections
            if len(self.neurons) > 1:
                existing_neurons = list(self.neurons.keys())[:-1]
                if existing_neurons:
                    num_connections = min(3, len(existing_neurons))
                    connections = random.sample(existing_neurons, num_connections)
                    connections.append('input')
                    self.connections[new_id] = connections
            else:
                self.connections[new_id] = ['input']
            
            self.evolution_history.append({
                'action': 'add_neuron',
                'neuron_id': new_id,
                'connections': self.connections[new_id]
            })
            
            logger.info(f"🧬 Added neuron {new_id} with {len(self.connections[new_id])} connections")
    
    def get_stats(self) -> Dict:
        """Get network statistics"""
        return {
            'num_neurons': len(self.neurons),
            'num_connections': sum(len(conns) for conns in self.connections.values()),
            'evolution_steps': len(self.evolution_history),
            'avg_connections_per_neuron': sum(len(conns) for conns in self.connections.values()) / max(len(self.neurons), 1)
        }


__all__ = ['DynamicNeuron', 'EvolvingNeuralNetwork']
