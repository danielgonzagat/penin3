#!/usr/bin/env python3
"""
🧬 DARWIN SURVIVORS ADAPTER - V7 Integration
Adaptador que injeta os 254 neurônios sobreviventes do Darwin Gen45 no sistema V7
"""

import torch
import torch.nn as nn
from pathlib import Path
import json

class DarwinSurvivorsAdapter(nn.Module):
    """Adaptador que integra neurônios sobreviventes do Darwin no V7"""
    
    def __init__(self):
        super().__init__()
        self.survivors_dir = Path('/root/INJECTED_SURVIVORS')
        self.survivors = []
        self._load_survivors()
    
    def _load_survivors(self):
        """Carrega índice de sobreviventes"""
        index_path = self.survivors_dir / 'survivors_index.json'
        if index_path.exists():
            with open(index_path, 'r') as f:
                data = json.load(f)
                self.survivor_count = data['count']
                self.survivors_metadata = data['neurons']
                print(f"🧬 Darwin Adapter: {self.survivor_count} neurônios sobreviventes carregados")
    
    def get_survivor_weights(self, neuron_id: str):
        """Retorna pesos de um neurônio sobrevivente específico"""
        weights_path = self.survivors_dir / f"{neuron_id}.pt"
        if weights_path.exists():
            return torch.load(weights_path, map_location='cpu')
        return None
    
    def inject_into_layer(self, layer: nn.Module, neuron_indices: list):
        """Injeta neurônios sobreviventes em uma camada do V7"""
        if not hasattr(layer, 'weight'):
            return False
        
        injected = 0
        for idx in neuron_indices:
            if idx < len(self.survivors_metadata):
                neuron_meta = self.survivors_metadata[idx]
                weights = self.get_survivor_weights(neuron_meta['id'])
                
                if weights is not None:
                    # Adaptar dimensões se necessário
                    target_shape = layer.weight.shape
                    if weights.shape == target_shape:
                        layer.weight.data = weights
                        injected += 1
                    else:
                        # Tentar adaptar dimensões
                        adapted = self._adapt_weights(weights, target_shape)
                        if adapted is not None:
                            layer.weight.data = adapted
                            injected += 1
        
        return injected > 0
    
    def _adapt_weights(self, source: torch.Tensor, target_shape: tuple):
        """Adapta pesos de uma forma para outra"""
        try:
            if len(source.shape) != len(target_shape):
                # Tentar reshape
                if source.numel() == torch.prod(torch.tensor(target_shape)):
                    return source.reshape(target_shape)
            
            # Truncar ou expandir dimensões
            adapted = torch.zeros(target_shape)
            min_dims = [min(s, t) for s, t in zip(source.shape, target_shape)]
            
            slices_src = tuple(slice(0, d) for d in min_dims)
            slices_tgt = tuple(slice(0, d) for d in min_dims)
            
            adapted[slices_tgt] = source[slices_src]
            return adapted
            
        except Exception as e:
            return None

# Singleton global
_DARWIN_ADAPTER = None

def get_darwin_adapter():
    """Retorna instância singleton do adaptador"""
    global _DARWIN_ADAPTER
    if _DARWIN_ADAPTER is None:
        _DARWIN_ADAPTER = DarwinSurvivorsAdapter()
    return _DARWIN_ADAPTER
