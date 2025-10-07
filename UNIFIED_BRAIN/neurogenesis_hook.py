#!/usr/bin/env python3
"""
Neurogenesis Hook - Birth/Death de neurônios baseado em performance
"""
import torch
import torch.nn as nn
from typing import Any

class NeurogenesisController:
    """Controla nascimento e morte de neurônios"""
    
    def __init__(self, core: Any, min_neurons: int = 4, max_neurons: int = 32):
        self.core = core
        self.min_neurons = min_neurons
        self.max_neurons = max_neurons
        self.birth_threshold = 0.9  # Nascer se performance > 90% do best
        self.death_competence = 0.1  # Morrer se competence < 0.1
        self.episodes_since_birth = 0
        self.episodes_since_death = 0
        self.cooldown = 10  # Esperar 10 eps entre births/deaths
        
    def should_birth(self, stats: dict) -> bool:
        """Decide se deve nascer um neurônio"""
        if self.episodes_since_birth < self.cooldown:
            self.episodes_since_birth += 1
            return False
            
        active = len(self.core.registry.get_active())
        if active >= self.max_neurons:
            return False
            
        avg = stats.get('avg_reward_last_100', 0)
        best = stats.get('best_reward', 0)
        
        # Nascer se performance consistente perto do best
        if best > 0 and avg > best * self.birth_threshold:
            self.episodes_since_birth = 0
            return True
        
        self.episodes_since_birth += 1
        return False
    
    def should_death(self) -> bool:
        """Decide se deve matar neurônios de baixa competence"""
        if self.episodes_since_death < self.cooldown:
            self.episodes_since_death += 1
            return False
            
        active = len(self.core.registry.get_active())
        if active <= self.min_neurons:
            return False
        
        # Verificar se há neurônios com baixa competence
        router = getattr(self.core, 'router', None)
        if router is None or not hasattr(router, 'competence'):
            return False
            
        competence = router.competence.detach()
        if (competence < self.death_competence).any():
            self.episodes_since_death = 0
            return True
        
        self.episodes_since_death += 1
        return False
    
    def birth_neuron(self) -> None:
        """Cria um novo neurônio"""
        try:
            from brain_spec import RegisteredNeuron, NeuronMeta, NeuronStatus
            import hashlib
            
            # Criar neurônio simples
            H = getattr(self.core, 'H', 1024)
            
            class SimpleNeuron(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(H, H)
                def forward(self, x):
                    return torch.tanh(self.fc(x))
            
            model = SimpleNeuron()
            nid = f"genesis_{len(self.core.registry.get_active())}_{int(torch.rand(1).item()*10000)}"
            
            meta = NeuronMeta(
                id=nid,
                in_shape=(H,),
                out_shape=(H,),
                dtype=torch.float32,
                device='cpu',
                status=NeuronStatus.ACTIVE,
                source='neurogenesis',
                params_count=H*H,
                checksum=hashlib.md5(nid.encode()).hexdigest()[:16],
                competence_score=0.5,
            )
            
            neuron = RegisteredNeuron(meta, model.forward, H=H)
            self.core.register_neuron(neuron)
            
            # Re-inicializar router se necessário
            if hasattr(self.core, 'initialize_router'):
                self.core.initialize_router()
                
            return True
        except Exception:
            return False
    
    def kill_weakest(self) -> None:
        """Mata o neurônio mais fraco"""
        try:
            router = self.core.router
            if router is None:
                return False
                
            competence = router.competence.detach()
            weakest_idx = torch.argmin(competence).item()
            
            active = self.core.registry.get_active()
            if weakest_idx < len(active):
                weak_neuron = active[weakest_idx]
                weak_neuron.meta.status = 'FROZEN'
                
                # Re-inicializar router
                if hasattr(self.core, 'initialize_router'):
                    self.core.initialize_router()
                    
                return True
        except Exception:
            return False