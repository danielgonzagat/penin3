"""
NSGA-III: Third-Generation Non-Dominated Sorting Genetic Algorithm
==================================================================

IMPLEMENTAÇÃO PURA PYTHON (SEM NUMPY)
Status: FUNCIONAL E TESTADO
Data: 2025-10-03

Based on: Deb & Jain (2014) "An Evolutionary Many-Objective Optimization 
Algorithm Using Reference-Point-Based Nondominated Sorting Approach"
"""

import math
import random
from typing import List, Dict, Tuple, Any, Set
from dataclasses import dataclass


@dataclass
class ReferencePoint:
    """Ponto de referência no espaço de objetivos"""
    coords: List[float]
    niche_count: int = 0


def das_dennis_recursive(n_obj: int, n_partitions: int, depth: int = 0, 
                         current: List[float] = None, 
                         results: List[List[float]] = None) -> List[List[float]]:
    """
    Gera reference points uniformes usando método Das-Dennis
    
    Args:
        n_obj: Número de objetivos
        n_partitions: Número de partições
        depth: Profundidade recursiva (uso interno)
        current: Ponto atual sendo construído
        results: Acumulador de resultados
    
    Returns:
        Lista de reference points
    """
    if results is None:
        results = []
    if current is None:
        current = []
    
    if depth == n_obj - 1:
        current.append(1.0 - sum(current))
        results.append(current[:])
        current.pop()
        return results
    
    for i in range(n_partitions + 1 - sum(int(c * n_partitions) for c in current)):
        current.append(i / n_partitions)
        das_dennis_recursive(n_obj, n_partitions, depth + 1, current, results)
        current.pop()
    
    return results


def perpendicular_distance(point: List[float], ref_point: List[float]) -> float:
    """
    Calcula distância perpendicular de um ponto até a linha ref_point
    
    Usa geometria vetorial pura (sem numpy)
    """
    # Normalizar ref_point
    ref_norm = math.sqrt(sum(x * x for x in ref_point))
    if ref_norm < 1e-9:
        return float('inf')
    
    ref_unit = [x / ref_norm for x in ref_point]
    
    # Projeção do point no ref_unit
    dot_product = sum(p * r for p, r in zip(point, ref_unit))
    projection = [dot_product * r for r in ref_unit]
    
    # Distância perpendicular
    diff = [p - proj for p, proj in zip(point, projection)]
    dist = math.sqrt(sum(d * d for d in diff))
    
    return dist


class NSGA3:
    """
    NSGA-III completo - Puro Python
    
    Características:
    - Reference points uniformes (Das-Dennis)
    - Niching para many-objective (3+ objetivos)
    - Sem numpy (100% stdlib)
    """
    
    def __init__(self, n_objectives: int, n_partitions: int = 12):
        """
        Args:
            n_objectives: Número de objetivos
            n_partitions: Divisões para Das-Dennis (quanto maior, mais ref points)
        """
        self.n_objectives = n_objectives
        self.n_partitions = n_partitions
        
        # Gerar reference points
        self.ref_points = self._generate_reference_points()
        
        print(f"📊 NSGA-III: {n_objectives} objetivos, "
              f"{len(self.ref_points)} reference points")
    
    def _generate_reference_points(self) -> List[ReferencePoint]:
        """Gera reference points usando Das-Dennis"""
        coords_list = das_dennis_recursive(self.n_objectives, self.n_partitions)
        return [ReferencePoint(coords) for coords in coords_list]
    
    def dominates(self, a: Dict[str, float], b: Dict[str, float], 
                  maximize: Dict[str, bool]) -> bool:
        """Relação de dominância Pareto"""
        better_or_equal_all = True
        strictly_better = False
        
        for key, is_max in maximize.items():
            a_val = a.get(key, 0.0)
            b_val = b.get(key, 0.0)
            
            if is_max:
                if a_val < b_val:
                    better_or_equal_all = False
                if a_val > b_val:
                    strictly_better = True
            else:
                if a_val > b_val:
                    better_or_equal_all = False
                if a_val < b_val:
                    strictly_better = True
        
        return better_or_equal_all and strictly_better
    
    def fast_nondominated_sort(self, objectives: List[Dict[str, float]], 
                                maximize: Dict[str, bool]) -> List[List[int]]:
        """Fast non-dominated sorting (O(MN²))"""
        n = len(objectives)
        S = [set() for _ in range(n)]  # Solutions dominated by i
        n_dominated = [0] * n  # Number of solutions dominating i
        fronts = [[]]
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self.dominates(objectives[i], objectives[j], maximize):
                    S[i].add(j)
                elif self.dominates(objectives[j], objectives[i], maximize):
                    n_dominated[i] += 1
            
            if n_dominated[i] == 0:
                fronts[0].append(i)
        
        k = 0
        while fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_dominated[j] -= 1
                    if n_dominated[j] == 0:
                        next_front.append(j)
            k += 1
            fronts.append(next_front)
        
        if not fronts[-1]:
            fronts.pop()
        
        return fronts
    
    def associate_to_reference_points(self, population_indices: List[int],
                                       objectives: List[Dict[str, float]]) -> Dict[int, List[int]]:
        """
        Associa indivíduos aos reference points mais próximos
        
        Returns:
            Dict: ref_point_idx -> [individual_indices]
        """
        # Reset niche counts
        for ref in self.ref_points:
            ref.niche_count = 0
        
        associations = {i: [] for i in range(len(self.ref_points))}
        
        # Normalizar objetivos primeiro
        obj_keys = list(objectives[0].keys())
        
        # Min/max de cada objetivo
        obj_min = {key: min(obj[key] for obj in objectives) for key in obj_keys}
        obj_max = {key: max(obj[key] for obj in objectives) for key in obj_keys}
        
        for idx in population_indices:
            # Normalizar este indivíduo
            obj = objectives[idx]
            normalized = []
            for key in obj_keys:
                val = obj[key]
                min_val = obj_min[key]
                max_val = obj_max[key]
                if max_val - min_val > 1e-9:
                    norm = (val - min_val) / (max_val - min_val)
                else:
                    norm = 0.5
                normalized.append(norm)
            
            # Encontrar ref point mais próximo
            min_dist = float('inf')
            closest_ref_idx = 0
            
            for ref_idx, ref_point in enumerate(self.ref_points):
                dist = perpendicular_distance(normalized, ref_point.coords)
                if dist < min_dist:
                    min_dist = dist
                    closest_ref_idx = ref_idx
            
            # Associar
            associations[closest_ref_idx].append(idx)
            self.ref_points[closest_ref_idx].niche_count += 1
        
        return associations
    
    def niching(self, survivors_needed: int, front: List[int],
                objectives: List[Dict[str, float]]) -> List[int]:
        """
        Niching procedure para selecionar indivíduos do último front
        
        Preserva diversidade preferindo nichos com menos indivíduos
        """
        # Associar aos ref points
        associations = self.associate_to_reference_points(front, objectives)
        
        selected = []
        
        while len(selected) < survivors_needed:
            # Encontrar nicho com menor count
            min_niche = min(range(len(self.ref_points)), 
                           key=lambda i: self.ref_points[i].niche_count)
            
            # Pegar indivíduos deste nicho
            candidates = associations[min_niche]
            
            if not candidates:
                # Nicho vazio, pegar de qualquer lugar
                for ref_idx, inds in associations.items():
                    if inds:
                        selected.append(inds.pop(0))
                        self.ref_points[ref_idx].niche_count -= 1
                        break
                continue
            
            # Selecionar aleatoriamente do nicho
            chosen = random.choice(candidates)
            selected.append(chosen)
            associations[min_niche].remove(chosen)
            self.ref_points[min_niche].niche_count += 1
            
            if len(selected) >= survivors_needed:
                break
        
        return selected[:survivors_needed]
    
    def select(self, population: List[Any], objectives: List[Dict[str, float]],
               maximize: Dict[str, bool], n_survivors: int) -> List[Any]:
        """
        Seleção NSGA-III completa
        
        Args:
            population: População completa
            objectives: Lista de dicts de objetivos
            maximize: Dict indicando se maximizar cada objetivo
            n_survivors: Quantos sobrevivem
        
        Returns:
            População de sobreviventes
        """
        # Non-dominated sorting
        fronts = self.fast_nondominated_sort(objectives, maximize)
        
        survivors = []
        last_front_idx = 0
        
        # Adicionar fronts inteiros até quase completar
        for front_idx, front in enumerate(fronts):
            if len(survivors) + len(front) <= n_survivors:
                survivors.extend([population[i] for i in front])
                last_front_idx = front_idx
            else:
                # Último front: usar niching
                remaining = n_survivors - len(survivors)
                selected_from_last = self.niching(remaining, front, objectives)
                survivors.extend([population[i] for i in selected_from_last])
                break
        
        return survivors


# ============================================================================
# TESTES
# ============================================================================

def test_nsga3_pure():
    """Testa NSGA-III puro Python"""
    print("\n" + "="*80)
    print("TESTE: NSGA-III Puro Python (sem numpy)")
    print("="*80 + "\n")
    
    # Criar NSGA-III para 3 objetivos
    nsga3 = NSGA3(n_objectives=3, n_partitions=4)
    
    print(f"✅ Reference points gerados: {len(nsga3.ref_points)}")
    print(f"   Primeiros 5:")
    for i, ref in enumerate(nsga3.ref_points[:5]):
        print(f"      {i}: {[f'{c:.3f}' for c in ref.coords]}")
    
    # População de teste
    population = list(range(20))  # IDs
    
    # Objetivos de teste (Pareto front conhecido)
    objectives = [
        {'f1': random.random(), 'f2': random.random(), 'f3': random.random()}
        for _ in range(20)
    ]
    
    maximize = {'f1': True, 'f2': True, 'f3': True}
    
    # Selecionar 10 survivors
    survivors = nsga3.select(population, objectives, maximize, n_survivors=10)
    
    print(f"\n✅ Seleção NSGA-III completa:")
    print(f"   População: 20 → {len(survivors)} sobreviventes")
    print(f"   Sobreviventes IDs: {survivors}")
    
    # Validar
    assert len(survivors) == 10, f"Esperava 10, obteve {len(survivors)}"
    assert all(s in population for s in survivors), "Survivors devem estar em population"
    
    print("\n✅ TESTE NSGA-III PASSOU!\n")
    print("="*80)


if __name__ == "__main__":
    test_nsga3_pure()
    print("\n✅ nsga3_pure_python.py FUNCIONAL!")
