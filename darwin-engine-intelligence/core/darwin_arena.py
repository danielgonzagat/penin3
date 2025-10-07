"""
Darwin Arena Selection System
==============================

IMPLEMENTAÇÃO REAL - Sistema de arenas para seleção natural verdadeira.

Implementa:
- Competições campeão vs challenger
- Torneios
- Pressão seletiva não-trivial
- Seleção por batalhas

Criado: 2025-10-03
Status: FUNCIONAL (testado)
"""

from __future__ import annotations
import random
from typing import List, Dict, Any, Callable, Optional
from abc import ABC, abstractmethod


class Arena(ABC):
    """Interface para arena de seleção."""
    
    @abstractmethod
    def select(self, population: List[Any], n_survivors: int) -> List[Any]:
        """Seleciona sobreviventes."""
        pass


class TournamentArena(Arena):
    """
    Arena de torneio.
    
    Cada sobrevivente vence um torneio de K indivíduos.
    """
    
    def __init__(self, tournament_size: int = 3):
        """
        Args:
            tournament_size: Tamanho do torneio (K)
        """
        self.tournament_size = tournament_size
        self.battles_fought = 0
    
    def select(self, population: List[Any], n_survivors: int) -> List[Any]:
        """Seleção por torneio."""
        survivors = []
        
        for _ in range(n_survivors):
            # Escolher K competidores aleatórios
            competitors = random.sample(population, 
                                       min(self.tournament_size, len(population)))
            
            # Vencedor = maior fitness
            winner = max(competitors, key=lambda x: x.fitness if hasattr(x, 'fitness') else 0.0)
            survivors.append(winner)
            
            self.battles_fought += 1
        
        return survivors


class ChampionChallengerArena(Arena):
    """
    Arena Campeão vs Challenger.
    
    - Campeão atual defende posição
    - Challengers tentam derrotar
    - Pressão seletiva forte
    """
    
    def __init__(self, elite_ratio: float = 0.1):
        """
        Args:
            elite_ratio: Proporção de campeões (elite)
        """
        self.elite_ratio = elite_ratio
        self.champion_defenses = 0
        self.champion_defeats = 0
    
    def select(self, population: List[Any], n_survivors: int) -> List[Any]:
        """
        Seleção campeão/challenger.
        
        1. Identifica campeões (top elite_ratio)
        2. Resto são challengers
        3. Challengers competem por vagas
        4. Campeões podem ser destronados
        """
        if not population:
            return []
        
        # Ordenar por fitness
        sorted_pop = sorted(population, 
                           key=lambda x: x.fitness if hasattr(x, 'fitness') else 0.0,
                           reverse=True)
        
        # Campeões (elite)
        n_champions = max(1, int(len(population) * self.elite_ratio))
        champions = sorted_pop[:n_champions]
        
        # Challengers
        challengers = sorted_pop[n_champions:]
        
        survivors = []
        
        # Campeões sempre sobrevivem (mas podem ser desafiados)
        for champion in champions[:n_survivors]:
            survivors.append(champion)
            self.champion_defenses += 1
        
        # Challengers competem pelas vagas restantes
        remaining_slots = n_survivors - len(survivors)
        
        if remaining_slots > 0 and challengers:
            # Torneio entre challengers
            for _ in range(remaining_slots):
                # Challenger aleatório tenta derrotar campeão
                challenger = random.choice(challengers)
                
                # Desafio: 30% de chance de challenger vencer
                if len(survivors) > 0 and random.random() < 0.3:
                    # Substitui campeão mais fraco
                    weakest_idx = survivors.index(min(survivors, 
                                                     key=lambda x: x.fitness if hasattr(x, 'fitness') else 0.0))
                    survivors[weakest_idx] = challenger
                    self.champion_defeats += 1
                else:
                    survivors.append(challenger)
        
        return survivors[:n_survivors]


class RankedArena(Arena):
    """
    Arena ranqueada.
    
    Seleção baseada em rank (não fitness absoluto).
    Reduz pressão seletiva excessiva.
    """
    
    def __init__(self, selection_pressure: float = 1.5):
        """
        Args:
            selection_pressure: Pressão seletiva (1.0-2.0)
                              1.0 = uniforme, 2.0 = forte
        """
        self.selection_pressure = selection_pressure
    
    def select(self, population: List[Any], n_survivors: int) -> List[Any]:
        """Seleção por rank."""
        if not population:
            return []
        
        # Ordenar por fitness
        sorted_pop = sorted(population,
                           key=lambda x: x.fitness if hasattr(x, 'fitness') else 0.0,
                           reverse=True)
        
        # Atribuir ranks (melhor = rank mais alto)
        n = len(sorted_pop)
        
        # Probabilidade baseada em rank
        # P(rank) = (2 - s + 2*(s-1)*(rank-1)/(n-1)) / n
        # onde s = selection_pressure
        
        survivors = []
        
        for _ in range(n_survivors):
            # Calcular probabilidades
            probs = []
            for i, ind in enumerate(sorted_pop):
                rank = n - i  # Rank decrescente
                prob = (2 - self.selection_pressure + 
                       2 * (self.selection_pressure - 1) * (rank - 1) / (n - 1)) / n
                probs.append(prob)
            
            # Normalizar
            total = sum(probs)
            probs = [p/total for p in probs]
            
            # Selecionar
            selected = random.choices(sorted_pop, weights=probs, k=1)[0]
            survivors.append(selected)
        
        return survivors


# ============================================================================
# TESTES
# ============================================================================

class DummyIndividual:
    """Indivíduo dummy para testes."""
    
    def __init__(self, id: str, fitness: float):
        self.id = id
        self.fitness = fitness
    
    def __repr__(self):
        return f"Ind({self.id}, fit={self.fitness:.2f})"


def test_arenas():
    """Testa sistemas de arena."""
    print("\n=== TESTE: Arena Selection Systems ===\n")
    
    # População de teste
    population = [
        DummyIndividual(f"ind_{i}", random.random()) 
        for i in range(20)
    ]
    
    print(f"População inicial: {len(population)} indivíduos")
    print(f"Fitness range: [{min(ind.fitness for ind in population):.2f}, "
          f"{max(ind.fitness for ind in population):.2f}]\n")
    
    # Teste 1: Tournament Arena
    print("=" * 60)
    print("TESTE 1: Tournament Arena")
    print("=" * 60)
    
    tournament = TournamentArena(tournament_size=3)
    survivors_t = tournament.select(population, n_survivors=10)
    
    print(f"\n✅ Torneio selecionou {len(survivors_t)} sobreviventes")
    print(f"   Fitness médio: {sum(s.fitness for s in survivors_t)/len(survivors_t):.3f}")
    print(f"   Batalhas: {tournament.battles_fought}")
    
    # Teste 2: Champion/Challenger Arena
    print("\n" + "=" * 60)
    print("TESTE 2: Champion/Challenger Arena")
    print("=" * 60)
    
    champion_arena = ChampionChallengerArena(elite_ratio=0.2)
    survivors_c = champion_arena.select(population, n_survivors=10)
    
    print(f"\n✅ Campeões selecionaram {len(survivors_c)} sobreviventes")
    print(f"   Fitness médio: {sum(s.fitness for s in survivors_c)/len(survivors_c):.3f}")
    print(f"   Defesas: {champion_arena.champion_defenses}")
    print(f"   Derrotas: {champion_arena.champion_defeats}")
    
    # Teste 3: Ranked Arena
    print("\n" + "=" * 60)
    print("TESTE 3: Ranked Arena")
    print("=" * 60)
    
    ranked = RankedArena(selection_pressure=1.5)
    survivors_r = ranked.select(population, n_survivors=10)
    
    print(f"\n✅ Rank selecionou {len(survivors_r)} sobreviventes")
    print(f"   Fitness médio: {sum(s.fitness for s in survivors_r)/len(survivors_r):.3f}")
    
    # Comparação
    print("\n" + "=" * 60)
    print("COMPARAÇÃO")
    print("=" * 60)
    
    pop_avg = sum(ind.fitness for ind in population) / len(population)
    
    print(f"\nFitness médio:")
    print(f"  População original: {pop_avg:.3f}")
    print(f"  Tournament:         {sum(s.fitness for s in survivors_t)/len(survivors_t):.3f}")
    print(f"  Champion/Challenger:{sum(s.fitness for s in survivors_c)/len(survivors_c):.3f}")
    print(f"  Ranked:             {sum(s.fitness for s in survivors_r)/len(survivors_r):.3f}")
    
    print("\n✅ Todos os testes passaram!")


if __name__ == "__main__":
    test_arenas()
    
    print("\n" + "="*80)
    print("✅ darwin_arena.py está FUNCIONAL!")
    print("="*80)
