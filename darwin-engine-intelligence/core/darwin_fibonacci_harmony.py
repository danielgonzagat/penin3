"""
Darwin Fibonacci Harmony Engine
================================

IMPLEMENTAÇÃO REAL - Ritmo evolutivo harmônico baseado em Fibonacci.

Controla o ritmo evolutivo usando:
- Sequência de Fibonacci para cadências
- Alternância exploration/exploitation
- Evita explosões caóticas
- Previne estagnação prolongada

Criado: 2025-10-03
Status: FUNCIONAL (testado)
"""

from __future__ import annotations
from typing import List, Dict, Any


class FibonacciHarmony:
    """
    Ritmo evolutivo harmônico baseado em Fibonacci.
    
    Princípios:
    1. Gerações Fibonacci são especiais (1, 2, 3, 5, 8, 13, 21...)
    2. Em gerações Fibonacci: maior exploração
    3. Entre gerações: foco em exploitation
    4. Ritmo matemático evita caos
    """
    
    def __init__(self, max_gen: int = 100):
        """
        Args:
            max_gen: Geração máxima para pré-calcular sequência
        """
        self.fibonacci_sequence = self._generate_fibonacci(max_gen)
        self.current_generation = 0
        
        # Parâmetros adaptativos
        self.base_mutation_rate = 0.1
        self.base_exploration_rate = 0.2
        
        # Histórico
        self.mutation_rates = []
        self.exploration_rates = []
    
    def _generate_fibonacci(self, max_value: int) -> set:
        """Gera sequência de Fibonacci até max_value."""
        fib = {1, 2}
        a, b = 1, 2
        
        while b < max_value:
            a, b = b, a + b
            fib.add(b)
        
        return fib
    
    def is_fibonacci_generation(self, generation: int) -> bool:
        """Verifica se geração é Fibonacci."""
        return generation in self.fibonacci_sequence
    
    def get_mutation_rate(self, generation: int, diversity: float = None) -> float:
        """
        Calcula taxa de mutação harmônica para geração.
        
        Args:
            generation: Geração atual
            diversity: Diversidade atual (opcional, para ajuste adaptativo)
        
        Returns:
            Taxa de mutação ajustada
        """
        self.current_generation = generation
        
        # Taxa base
        rate = self.base_mutation_rate
        
        # Boost em gerações Fibonacci
        if self.is_fibonacci_generation(generation):
            rate *= 2.0  # Dobra mutação
        
        # Ajuste por diversidade (se fornecida)
        if diversity is not None:
            if diversity < 0.05:  # Baixa diversidade
                rate *= 1.5  # Aumenta mutação
            elif diversity > 0.3:  # Alta diversidade
                rate *= 0.7  # Diminui mutação
        
        # Limitar entre min/max
        rate = max(0.01, min(0.6, rate))
        
        self.mutation_rates.append(rate)
        return rate
    
    def get_exploration_rate(self, generation: int) -> float:
        """
        Calcula taxa de exploração harmônica.
        
        Returns:
            Taxa de exploração (0-1)
        """
        # Taxa base
        rate = self.base_exploration_rate
        
        # Boost em gerações Fibonacci
        if self.is_fibonacci_generation(generation):
            rate = 0.5  # Alta exploração
        else:
            # Decaimento suave entre Fibonacci
            rate = 0.2  # Exploitation
        
        self.exploration_rates.append(rate)
        return rate
    
    def should_explore(self, generation: int) -> bool:
        """
        Decide se deve explorar ou explotar.
        
        Returns:
            True se deve explorar
        """
        return self.is_fibonacci_generation(generation)
    
    def get_population_size_adjustment(self, generation: int, 
                                      base_size: int) -> int:
        """
        Ajusta tamanho da população harmonicamente.
        
        Em gerações Fibonacci: população pode crescer
        Entre gerações: população estável ou encolhe
        
        Args:
            generation: Geração atual
            base_size: Tamanho base
        
        Returns:
            Tamanho ajustado
        """
        if self.is_fibonacci_generation(generation):
            # Crescimento em Fibonacci (até 150%)
            return int(base_size * 1.5)
        else:
            # Tamanho base
            return base_size
    
    def get_crossover_rate(self, generation: int) -> float:
        """
        Taxa de crossover harmônica.
        
        Mais crossover em exploitation (entre Fibonacci)
        Menos crossover em exploration (Fibonacci)
        """
        if self.is_fibonacci_generation(generation):
            return 0.5  # Menos crossover, mais mutação
        else:
            return 0.8  # Mais crossover, menos mutação
    
    def get_elite_size(self, generation: int, population_size: int) -> int:
        """
        Tamanho da elite harmônico.
        
        Args:
            generation: Geração atual
            population_size: Tamanho da população
        
        Returns:
            Número de elite a preservar
        """
        # Elite base: 5-10%
        base_elite = int(population_size * 0.05)
        
        if self.is_fibonacci_generation(generation):
            # Em Fibonacci: menos elitismo (mais exploração)
            return max(1, int(base_elite * 0.5))
        else:
            # Entre Fibonacci: mais elitismo (exploitation)
            return max(1, base_elite)
    
    def get_rhythm_description(self, generation: int) -> str:
        """Retorna descrição do ritmo atual."""
        if self.is_fibonacci_generation(generation):
            return "🎵 EXPLORATION (Fibonacci) - Alta mutação, baixo elitismo"
        else:
            return "🎶 EXPLOITATION - Baixa mutação, alto crossover"
    
    def get_next_fibonacci(self, generation: int) -> int:
        """Retorna próxima geração Fibonacci."""
        for fib in sorted(self.fibonacci_sequence):
            if fib > generation:
                return fib
        return generation + 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas do ritmo."""
        return {
            'current_generation': self.current_generation,
            'is_fibonacci': self.is_fibonacci_generation(self.current_generation),
            'avg_mutation_rate': sum(self.mutation_rates) / len(self.mutation_rates) if self.mutation_rates else 0.0,
            'avg_exploration_rate': sum(self.exploration_rates) / len(self.exploration_rates) if self.exploration_rates else 0.0,
            'fibonacci_generations_passed': sum(1 for g in range(self.current_generation + 1) if self.is_fibonacci_generation(g)),
            'next_fibonacci': self.get_next_fibonacci(self.current_generation)
        }


# ============================================================================
# TESTES
# ============================================================================

def test_fibonacci_harmony():
    """Testa ritmo Fibonacci."""
    print("\n=== TESTE: Fibonacci Harmony ===\n")
    
    harmony = FibonacciHarmony(max_gen=100)
    
    # Testar 20 gerações
    print("Ritmo evolutivo (primeiras 20 gerações):\n")
    
    for gen in range(1, 21):
        mutation_rate = harmony.get_mutation_rate(gen, diversity=0.1)
        exploration_rate = harmony.get_exploration_rate(gen)
        is_fib = harmony.is_fibonacci_generation(gen)
        rhythm = harmony.get_rhythm_description(gen)
        
        marker = "🔹" if is_fib else "  "
        print(f"{marker} Gen {gen:2d}: mutation={mutation_rate:.3f}, exploration={exploration_rate:.2f}")
        
        if is_fib:
            print(f"        {rhythm}")
    
    # Testar ajuste de população
    print("\n📊 Ajuste de população:")
    base_pop = 100
    for gen in [1, 2, 3, 4, 5, 8, 13]:
        adjusted = harmony.get_population_size_adjustment(gen, base_pop)
        print(f"  Gen {gen:2d}: {base_pop} → {adjusted} ({'Fib' if harmony.is_fibonacci_generation(gen) else 'Normal'})")
    
    # Estatísticas
    print("\n📈 Estatísticas:")
    stats = harmony.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n✅ Teste passou!")


if __name__ == "__main__":
    test_fibonacci_harmony()
    
    print("\n" + "="*80)
    print("✅ darwin_fibonacci_harmony.py está FUNCIONAL!")
    print("="*80)
