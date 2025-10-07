#!/usr/bin/env python3
"""
🧬 DARWIN SURVIVORS - PENIN Integration
254 neurônios sobreviventes injetados em PENIN
"""

import torch
from pathlib import Path
import json

class DarwinNeuronsPENIN:
    """Neurônios Darwin integrados ao PENIN"""
    
    def __init__(self):
        self.survivors_dir = Path('/root/INJECTED_SURVIVORS')
        self.load_survivors()
    
    def load_survivors(self):
        index_path = self.survivors_dir / 'survivors_index.json'
        if index_path.exists():
            with open(index_path, 'r') as f:
                self.data = json.load(f)
                print(f"🧬 PENIN-Darwin: {self.data['count']} neurônios carregados")
    
    def get_neuron(self, idx: int):
        if idx < len(self.data['neurons']):
            neuron_meta = self.data['neurons'][idx]
            weights_path = self.survivors_dir / f"{neuron_meta['id']}.pt"
            return torch.load(weights_path, map_location='cpu')
        return None

# Instância global
DARWIN_NEURONS = DarwinNeuronsPENIN()
