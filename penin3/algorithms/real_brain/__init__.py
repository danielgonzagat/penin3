"""
Real Brain - Dynamic Neural Growth
Extracted from REAL_INTELLIGENCE_SYSTEM.py
"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class RealBrain(nn.Module):
    """
    Cérebro que REALMENTE cresce neurônios baseado em necessidade
    
    Features REAIS:
    - Adiciona neurônios quando sistema fica complexo
    - Remove neurônios inativos
    - Rastreia ativações REAIS
    - Crescimento não-supervisionado
    """
    
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=5):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Rede neural
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
        # Estatísticas REAIS
        self.neuron_activations = torch.zeros(hidden_dim)
        self.total_forward_passes = 0
        self.neurons_added = 0
        self.neurons_removed = 0
        
    def forward(self, x):
        if isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32)
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
            
        h = self.fc1(x)
        
        # REALMENTE rastrear ativações
        with torch.no_grad():
            self.neuron_activations += torch.abs(h.mean(0))
            self.total_forward_passes += 1
            
        h = self.act(h)
        out = self.fc2(h)
        
        # Verificar crescimento
        if self.total_forward_passes % 100 == 0:
            self.check_growth()
        
        return out
    
    def check_growth(self):
        """Verificar se precisa crescer/podar neurônios"""
        if self.total_forward_passes < 200:
            return
        
        # Calcular ativação média
        avg_activation = self.neuron_activations / self.total_forward_passes
        
        # Crescer se muitos neurônios muito ativos
        high_activation_ratio = (avg_activation > 0.5).float().mean()
        if high_activation_ratio > 0.8 and self.hidden_dim < 100:
            self.add_neuron()
            logger.info(f"🌱 Neurônio adicionado (hidden_dim: {self.hidden_dim})")
        
        # Podar se muitos neurônios inativos
        low_activation_ratio = (avg_activation < 0.1).float().mean()
        if low_activation_ratio > 0.3 and self.hidden_dim > 15:
            self.remove_neuron()
            logger.info(f"💀 Neurônio removido (hidden_dim: {self.hidden_dim})")
    
    def add_neuron(self):
        """Adicionar neurônio REAL"""
        new_hidden = self.hidden_dim + 1
        
        # Expandir camadas
        new_fc1 = nn.Linear(self.input_dim, new_hidden)
        new_fc2 = nn.Linear(new_hidden, self.output_dim)
        
        # Copiar pesos existentes
        with torch.no_grad():
            new_fc1.weight[:self.hidden_dim] = self.fc1.weight
            new_fc1.bias[:self.hidden_dim] = self.fc1.bias
            new_fc2.weight[:, :self.hidden_dim] = self.fc2.weight
            
            # Novo neurônio com pesos aleatórios pequenos
            nn.init.xavier_uniform_(new_fc1.weight[self.hidden_dim:])
            nn.init.xavier_uniform_(new_fc2.weight[:, self.hidden_dim:])
        
        self.fc1 = new_fc1
        self.fc2 = new_fc2
        self.hidden_dim = new_hidden
        
        # Expandir estatísticas
        self.neuron_activations = torch.cat([self.neuron_activations, torch.zeros(1)])
        self.neurons_added += 1
    
    def remove_neuron(self):
        """Remover neurônio morto"""
        if self.hidden_dim <= 10:
            return
        
        # Encontrar neurônio menos ativo
        avg_activation = self.neuron_activations / self.total_forward_passes
        victim = torch.argmin(avg_activation)
        
        # Criar máscara
        mask = torch.ones(self.hidden_dim, dtype=torch.bool)
        mask[victim] = False
        
        new_hidden = self.hidden_dim - 1
        
        # Contrair camadas
        new_fc1 = nn.Linear(self.input_dim, new_hidden)
        new_fc2 = nn.Linear(new_hidden, self.output_dim)
        
        with torch.no_grad():
            new_fc1.weight = nn.Parameter(self.fc1.weight[mask])
            new_fc1.bias = nn.Parameter(self.fc1.bias[mask])
            new_fc2.weight = nn.Parameter(self.fc2.weight[:, mask])
            new_fc2.bias = self.fc2.bias
        
        self.fc1 = new_fc1
        self.fc2 = new_fc2
        self.hidden_dim = new_hidden
        
        # Contrair estatísticas
        self.neuron_activations = self.neuron_activations[mask]
        self.neurons_removed += 1
    
    def get_stats(self):
        """Get brain statistics"""
        return {
            'hidden_dim': self.hidden_dim,
            'neurons_added': self.neurons_added,
            'neurons_removed': self.neurons_removed,
            'total_forward_passes': self.total_forward_passes,
            'avg_activation': (self.neuron_activations / max(self.total_forward_passes, 1)).mean().item()
        }


__all__ = ['RealBrain']
