#!/usr/bin/env python3
"""
🔧 MODULE SYNTHESIS - FASE 3
Auto-construção de neurônios e arquiteturas
"""

import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional
from datetime import datetime
from brain_logger import brain_logger

class NeuronTemplate:
    """Template para gerar novos neurônios"""
    def __init__(self, architecture: str, H: int = 1024):
        self.architecture = architecture
        self.H = H
    
    def build(self) -> nn.Module:
        """Constrói neurônio baseado no template"""
        if self.architecture == "simple_mlp":
            return nn.Sequential(
                nn.Linear(self.H, self.H),
                nn.GELU(),
                nn.Linear(self.H, self.H)
            )
        elif self.architecture == "deep_mlp":
            return nn.Sequential(
                nn.Linear(self.H, self.H),
                nn.LayerNorm(self.H),
                nn.GELU(),
                nn.Linear(self.H, self.H),
                nn.LayerNorm(self.H),
                nn.GELU(),
                nn.Linear(self.H, self.H)
            )
        elif self.architecture == "residual":
            class ResidualBlock(nn.Module):
                def __init__(self, H):
                    super().__init__()
                    self.block = nn.Sequential(
                        nn.Linear(H, H),
                        nn.GELU(),
                        nn.Linear(H, H)
                    )
                
                def forward(self, x):
                    return x + self.block(x)
            
            return ResidualBlock(self.H)
        elif self.architecture == "attention":
            class AttentionNeuron(nn.Module):
                def __init__(self, H):
                    super().__init__()
                    self.query = nn.Linear(H, H)
                    self.key = nn.Linear(H, H)
                    self.value = nn.Linear(H, H)
                    self.out = nn.Linear(H, H)
                
                def forward(self, x):
                    q = self.query(x)
                    k = self.key(x)
                    v = self.value(x)
                    attn = torch.softmax(q @ k.T / (self.H ** 0.5), dim=-1)
                    return self.out(attn @ v)
            
            return AttentionNeuron(self.H)
        else:
            # Default: simple MLP
            return nn.Sequential(
                nn.Linear(self.H, self.H),
                nn.GELU(),
                nn.Linear(self.H, self.H)
            )


class ModuleSynthesizer:
    """
    Sintetiza novos módulos/neurônios dinamicamente
    Baseado em performance e necessidades
    """
    def __init__(self, H: int = 1024):
        self.H = H
        self.templates = {
            'simple_mlp': NeuronTemplate('simple_mlp', H),
            'deep_mlp': NeuronTemplate('deep_mlp', H),
            'residual': NeuronTemplate('residual', H),
            'attention': NeuronTemplate('attention', H)
        }
        self.synthesis_history = []
        self.performance_tracker = {}
    
    def synthesize_and_register(self, brain, architecture: str, reason: str):
        """
        ✅ CORREÇÃO P2.2: Sintetiza E registra novo neurônio
        
        Returns:
            RegisteredNeuron ou None
        """
        try:
            from brain_spec import RegisteredNeuron, NeuronMeta, NeuronStatus
            from datetime import datetime
            
            # Build neurônio
            template = self.templates.get(architecture, self.templates['simple_mlp'])
            neuron_module = template.build()
            
            # Metadata
            neuron_id = f"synth_{len(self.synthesis_history)}_{datetime.now().strftime('%H%M%S')}"
            meta = NeuronMeta(
                id=neuron_id,
                source='synthesis',
                status=NeuronStatus.ACTIVE,
                params_count=sum(p.numel() for p in neuron_module.parameters()),
                generation=0,
                created_at=datetime.now().isoformat()
            )
            
            # Wrap
            reg_neuron = RegisteredNeuron(
                forward_fn=neuron_module,
                meta=meta
            )
            
            # Register
            success = brain.register_neuron(reg_neuron)
            
            if success:
                self.synthesis_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'neuron_id': neuron_id,
                    'architecture': architecture,
                    'reason': reason
                })
                
                brain_logger.info(f"✨ Synthesized: {neuron_id} ({architecture})")
                return reg_neuron
            else:
                brain_logger.error("❌ Registration failed")
                return None
                
        except Exception as e:
            brain_logger.error(f"Synthesis failed: {e}")
            return None
    
    def should_synthesize(self, bottleneck_report: Dict[str, Any], 
                          performance_stats: Dict[str, Any]) -> bool:
        """
        Decide se deve sintetizar novo neurônio
        
        Args:
            bottleneck_report: Relatório de bottlenecks
            performance_stats: Stats de performance
            
        Returns:
            True se deve sintetizar
        """
        # Critério 1: Bottleneck crítico
        if bottleneck_report.get('bottlenecks', []):
            critical = [b for b in bottleneck_report['bottlenecks'] if b['severity'] == 'HIGH']
            if critical:
                return True
        
        # Critério 2: Performance degradando
        avg_reward = performance_stats.get('avg_reward_100', 0)
        if len(self.performance_tracker) > 10:
            recent = list(self.performance_tracker.values())[-10:]
            if all(r < avg_reward * 0.8 for r in recent[-5:]):
                return True
        
        # Critério 3: Stagnação prolongada
        if len(self.synthesis_history) > 0:
            last_synthesis = self.synthesis_history[-1]
            episodes_since = performance_stats.get('episode', 0) - last_synthesis['episode']
            if episodes_since > 100 and avg_reward < 50:  # Stagnado há 100 eps
                return True
        
        return False
    
    def select_architecture(self, bottleneck_report: Dict[str, Any]) -> str:
        """
        Seleciona arquitetura baseada em bottlenecks
        
        Args:
            bottleneck_report: Relatório de bottlenecks
            
        Returns:
            Nome da arquitetura
        """
        bottlenecks = bottleneck_report.get('bottlenecks', [])
        
        if not bottlenecks:
            return 'simple_mlp'
        
        top_bottleneck = bottlenecks[0]['component']
        
        # Heurísticas
        if 'forward' in top_bottleneck:
            return 'residual'  # Residual para acelerar forward
        elif 'training' in top_bottleneck:
            return 'simple_mlp'  # Simple para treinar rápido
        elif 'router' in top_bottleneck:
            return 'attention'  # Attention para melhor routing
        else:
            return 'deep_mlp'  # Default: deep para capacidade
    
    def synthesize_neuron(self, architecture: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sintetiza novo neurônio
        
        Args:
            architecture: Nome da arquitetura
            metadata: Metadados do neurônio
            
        Returns:
            Dict com neurônio e info
        """
        template = self.templates.get(architecture, self.templates['simple_mlp'])
        neuron = template.build()
        
        synthesis_info = {
            'timestamp': datetime.now().isoformat(),
            'architecture': architecture,
            'episode': metadata.get('episode', 0),
            'reason': metadata.get('reason', 'unknown'),
            'neuron_id': f"synthesized_{architecture}_{len(self.synthesis_history)}",
            'parameters': sum(p.numel() for p in neuron.parameters())
        }
        
        self.synthesis_history.append(synthesis_info)
        
        brain_logger.info(
            f"🔧 Synthesized neuron: {synthesis_info['neuron_id']} "
            f"({architecture}, {synthesis_info['parameters']} params)"
        )
        
        return {
            'neuron': neuron,
            'info': synthesis_info
        }
    
    def auto_synthesize(self, analysis_report: Dict[str, Any], 
                       performance_stats: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Pipeline completo de auto-síntese
        
        Args:
            analysis_report: Relatório de self-analysis
            performance_stats: Stats de performance
            
        Returns:
            Neurônio sintetizado ou None
        """
        # Track performance
        episode = performance_stats.get('episode', 0)
        self.performance_tracker[episode] = performance_stats.get('avg_reward_100', 0)
        
        # Decide se sintetiza
        if not self.should_synthesize(analysis_report, performance_stats):
            return None
        
        # Seleciona arquitetura
        architecture = self.select_architecture(analysis_report)
        
        # Sintetiza
        metadata = {
            'episode': episode,
            'reason': 'bottleneck' if analysis_report.get('bottlenecks') else 'stagnation'
        }
        
        result = self.synthesize_neuron(architecture, metadata)
        
        brain_logger.warning(
            f"🔧 AUTO-SYNTHESIS triggered at ep {episode}: "
            f"{architecture} neuron created"
        )
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de síntese"""
        return {
            'total_synthesized': len(self.synthesis_history),
            'architectures_used': {
                arch: sum(1 for s in self.synthesis_history if s['architecture'] == arch)
                for arch in self.templates.keys()
            },
            'last_synthesis': self.synthesis_history[-1] if self.synthesis_history else None
        }


class ArchitectureSearch:
    """
    Busca automática de arquiteturas
    Testa diferentes configurações e mantém as melhores
    """
    def __init__(self):
        self.candidates = []
        self.test_results = {}
        self.best_architectures = []
    
    def generate_candidates(self, num_candidates: int = 5) -> List[Dict[str, Any]]:
        """
        Gera candidatos de arquitetura
        
        Args:
            num_candidates: Número de candidatos
            
        Returns:
            Lista de configurações
        """
        candidates = []
        
        # Variações de depth
        for depth in [2, 3, 4]:
            candidates.append({
                'type': 'mlp',
                'depth': depth,
                'hidden_size': 1024,
                'activation': 'gelu'
            })
        
        # Variações de width
        for width in [512, 1024, 2048]:
            candidates.append({
                'type': 'mlp',
                'depth': 3,
                'hidden_size': width,
                'activation': 'gelu'
            })
        
        # Arquiteturas especiais
        candidates.append({
            'type': 'residual',
            'num_blocks': 2,
            'hidden_size': 1024
        })
        
        candidates.append({
            'type': 'attention',
            'num_heads': 4,
            'hidden_size': 1024
        })
        
        self.candidates = candidates[:num_candidates]
        return self.candidates
    
    def evaluate_candidate(self, candidate: Dict[str, Any], 
                          performance_metric: float) -> Dict[str, Any]:
        """
        Avalia candidato
        
        Args:
            candidate: Configuração do candidato
            performance_metric: Métrica de performance (reward)
            
        Returns:
            Resultado da avaliação
        """
        candidate_id = str(candidate)
        
        if candidate_id not in self.test_results:
            self.test_results[candidate_id] = {
                'candidate': candidate,
                'trials': [],
                'avg_performance': 0
            }
        
        self.test_results[candidate_id]['trials'].append(performance_metric)
        self.test_results[candidate_id]['avg_performance'] = (
            sum(self.test_results[candidate_id]['trials']) / 
            len(self.test_results[candidate_id]['trials'])
        )
        
        return self.test_results[candidate_id]
    
    def select_best(self, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Seleciona melhores arquiteturas
        
        Args:
            top_k: Número de melhores
            
        Returns:
            Lista das melhores
        """
        sorted_results = sorted(
            self.test_results.values(),
            key=lambda x: x['avg_performance'],
            reverse=True
        )
        
        self.best_architectures = [r['candidate'] for r in sorted_results[:top_k]]
        return self.best_architectures


if __name__ == "__main__":
    print("🔧 Module Synthesis - FASE 3")
    
    # Test synthesizer
    synthesizer = ModuleSynthesizer(H=1024)
    
    # Simula bottleneck
    bottleneck_report = {
        'bottlenecks': [{
            'component': 'forward',
            'severity': 'HIGH',
            'percentage': 45
        }]
    }
    
    performance_stats = {
        'episode': 50,
        'avg_reward_100': 25.0
    }
    
    # Auto-synthesize
    result = synthesizer.auto_synthesize(bottleneck_report, performance_stats)
    
    if result:
        print(f"✅ Synthesized: {result['info']['neuron_id']}")
        print(f"   Architecture: {result['info']['architecture']}")
        print(f"   Parameters: {result['info']['parameters']}")
    
    # Test architecture search
    print("\n🔍 Architecture Search")
    search = ArchitectureSearch()
    candidates = search.generate_candidates(5)
    print(f"Generated {len(candidates)} candidates")
    
    # Simula avaliações
    for i, candidate in enumerate(candidates):
        perf = 20 + i * 5  # Simula performance crescente
        search.evaluate_candidate(candidate, perf)
    
    best = search.select_best(3)
    print(f"Best architectures: {len(best)}")
    for b in best:
        print(f"  {b['type']}")
    
    print("\n✅ Module Synthesis OK")
