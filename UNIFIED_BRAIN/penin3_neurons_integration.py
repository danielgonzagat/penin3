#!/usr/bin/env python3
"""
✅ PENIN3 NEURONS INTEGRATION
Integra EvolvingNeuralNetwork como neurônios do UNIFIED_BRAIN
"""

import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')
sys.path.insert(0, '/root/penin3')
sys.path.insert(0, '/root')

import torch
import torch.nn as nn
from typing import Dict

# Import logger first, before other imports that might fail
try:
    from brain_logger import brain_logger
except:
    import logging
    brain_logger = logging.getLogger(__name__)

# Import PENIN3 components
try:
    from penin3.algorithms.neural_genesis.evolving_network import EvolvingNeuralNetwork
    PENIN3_AVAILABLE = True
except Exception as e:
    brain_logger.error(f"PENIN3 import failed: {e}")
    PENIN3_AVAILABLE = False

class EvolvingNeuronWrapper(nn.Module):
    """Wrapper de EvolvingNeuralNetwork para UNIFIED_BRAIN"""
    
    def __init__(self, evolving_network):
        super().__init__()
        self.evolving_network = evolving_network
        self.performance_history = []
        self.evolution_count = 0
    
    def forward(self, x):
        """Forward compatível com brain (tensor [B, H] → [B, H])"""
        try:
            # EvolvingNetwork espera entrada com dim correto
            if x.dim() == 1:
                x = x.unsqueeze(0)
            
            # Forward
            output = self.evolving_network(x)
            
            # Garantir shape consistente
            if output.dim() == 1:
                output = output.unsqueeze(0)
            
            return output
            
        except Exception as e:
            brain_logger.debug(f"EvolvingNeuron forward error: {e}")
            # Fallback: retornar input
            return x
    
    def evolve_if_needed(self, performance: float):
        """Evolve arquitetura se performance sugerir"""
        self.performance_history.append(performance)
        
        # Manter apenas últimos 20
        if len(self.performance_history) > 20:
            self.performance_history = self.performance_history[-20:]
        
        # Evolve se performance baixa consistentemente
        if len(self.performance_history) >= 10:
            recent = self.performance_history[-10:]
            
            if all(p < 0.3 for p in recent[-5:]):
                # Performance baixa → evolve!
                try:
                    self.evolving_network.evolve_architecture({
                        'target_complexity': len(self.evolving_network.neurons) + 1,
                        'performance': performance
                    })
                    
                    self.evolution_count += 1
                    
                    brain_logger.warning(
                        f"🧬 PENIN3 neuron evolved! "
                        f"(count={self.evolution_count}, "
                        f"neurons={len(self.evolving_network.neurons)})"
                    )
                    
                    return True
                    
                except Exception as e:
                    brain_logger.debug(f"Evolution failed: {e}")
        
        return False

def create_penin3_evolving_neuron(neuron_id: str, H: int = 1024):
    """
    Cria neurônio evoluível baseado em PENIN3
    
    Returns:
        RegisteredNeuron com capacidade de evolução
    """
    if not PENIN3_AVAILABLE:
        brain_logger.error("❌ PENIN3 não disponível")
        return None
    
    try:
        from brain_spec import RegisteredNeuron, NeuronMeta, NeuronStatus
        from datetime import datetime
        
        # Criar EvolvingNetwork
        evolving_net = EvolvingNeuralNetwork(
            input_dim=H,
            output_dim=H
        )
        
        # Wrapper
        wrapper = EvolvingNeuronWrapper(evolving_net)
        
        # Metadata
        meta = NeuronMeta(
            id=neuron_id,
            source='penin3_evolving',
            status=NeuronStatus.ACTIVE,
            params_count=sum(p.numel() for p in wrapper.parameters()),
            generation=0,
            created_at=datetime.now().isoformat()
        )
        
        # Registered neuron
        reg_neuron = RegisteredNeuron(
            forward_fn=wrapper,
            meta=meta
        )
        
        brain_logger.info(f"✨ Created PENIN3 evolving neuron: {neuron_id}")
        
        return reg_neuron
        
    except Exception as e:
        brain_logger.error(f"PENIN3 neuron creation failed: {e}")
        return None

def add_penin3_neurons_to_brain(brain, count: int = 3) -> int:
    """
    Adiciona múltiplos neurônios PENIN3 ao brain
    
    Returns:
        Número de neurônios adicionados com sucesso
    """
    if not PENIN3_AVAILABLE:
        return 0
    
    added = 0
    H = brain.H
    
    for i in range(count):
        neuron_id = f"penin3_evol_{i}"
        neuron = create_penin3_evolving_neuron(neuron_id, H)
        
        if neuron:
            success = brain.register_neuron(neuron)
            if success:
                added += 1
                brain_logger.info(f"✅ Registered: {neuron_id}")
    
    if added > 0:
        # Rebuild router com novos neurônios
        brain.initialize_router()
        brain_logger.warning(f"🧬 Added {added} PENIN3 evolving neurons!")
    
    return added

# Test
if __name__ == "__main__":
    brain_logger.info("🧬 PENIN3 Neurons Integration - standalone test")
    
    if PENIN3_AVAILABLE:
        print("✅ PENIN3 disponível e pronto")
    else:
        print("❌ PENIN3 não disponível")