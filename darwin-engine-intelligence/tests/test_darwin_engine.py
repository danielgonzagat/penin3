"""
Testes Unitários - Darwin Engine
Baseados nos 8 testes empíricos da auditoria
"""

import pytest
import sys
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.adapter_darwin_core import DarwinEngine, ReproductionEngine
from core.darwin_evolution_system_FIXED import EvolvableMNIST, DarwinEvolutionOrchestrator


class TestDarwinEngine:
    """Testes do Darwin Engine Base"""
    
    def test_darwin_init(self):
        """Teste #1: Inicialização"""
        darwin = DarwinEngine(survival_rate=0.4)
        assert darwin.survival_rate == 0.4
    
    def test_reproduction_engine(self):
        """Teste #2: Reproduction Engine"""
        repro = ReproductionEngine(sexual_rate=0.8)
        assert repro.sexual_rate == 0.8


class TestEvolvableMNIST:
    """Testes do MNIST Evoluível"""
    
    def test_mnist_init(self):
        """Teste #3: Inicialização MNIST"""
        individual = EvolvableMNIST()
        assert individual.genome is not None
        assert 'hidden_size' in individual.genome
    
    @pytest.mark.timeout(120)
    def test_mnist_build(self, monkeypatch):
        """Teste #4: Construção de modelo"""
        # Forçar atributos mínimos e fallback de dropout
        individual = EvolvableMNIST({'hidden_size': 128, 'num_layers': 2, 'dropout': 0.1})
        # Opcionalmente garantir tamanho mínimo por env
        monkeypatch.setenv('DARWIN_ENSURE_LARGE_MODEL', '1')
        monkeypatch.setenv('DARWIN_MIN_PARAMS', '100000')
        model = individual.build()
        assert model is not None
        
        # Contar parâmetros
        params = sum(p.numel() for p in model.parameters())
        # Cálculo esperado: 784*h + h + (L-1)*(h*h + h) + (h*10 + 10)
        h = int(individual.genome.get('hidden_size', 128))
        L = int(individual.genome.get('num_layers', 2))
        expected_min = (784*h + h) + max(0, (L-1))*(h*h + h) + (h*10 + 10)
        assert params >= min(100000, expected_min)
    
    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_mnist_fitness(self):
        """Teste #5: Fitness (baseado em evidência empírica)
        
        Evidência: 8 testes mostraram fitness 0.87-0.96
        """
        individual = EvolvableMNIST({
            'hidden_size': 128,
            'learning_rate': 0.001,
            'batch_size': 64,
            'dropout': 0.1,
            'num_layers': 2
        })
        
        fitness = individual.evaluate_fitness()
        
        # Baseado em evidência empírica
        assert fitness > 0.85, f"Fitness {fitness} abaixo do esperado (>0.85)"
        assert fitness < 1.0, f"Fitness {fitness} suspeito (>1.0)"


class TestOrchestrator:
    """Testes do Orchestrator"""
    
    def test_orchestrator_init(self):
        """Teste #6: Inicialização"""
        orch = DarwinEvolutionOrchestrator()
        assert orch is not None


class TestViralContamination:
    """Testes da Contaminação Viral"""
    
    def test_contaminator_import(self):
        """Teste #7: Import"""
        from contamination.darwin_viral_contamination import DarwinViralContamination
        assert DarwinViralContamination is not None
    
    def test_contaminator_init(self):
        """Teste #8: Inicialização"""
        from contamination.darwin_viral_contamination import DarwinViralContamination
        c = DarwinViralContamination()
        assert c is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
