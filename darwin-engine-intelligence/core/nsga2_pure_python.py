"""
NSGA-II: Non-dominated Sorting Genetic Algorithm II

Pure Python implementation of the classic multi-objective optimization algorithm.
Simpler than NSGA-III but still very effective for 2-3 objectives.

Features:
- Fast non-dominated sorting
- Crowding distance calculation
- Elitist selection
- Pure Python (no numpy)

Reference:
- Deb, K., et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
"""

import random
from typing import List, Dict, Any, Set


class NSGA2:
    """NSGA-II implementation."""
    
    @staticmethod
    def dominates(obj_a: Dict[str, float], obj_b: Dict[str, float], maximize: Dict[str, bool]) -> bool:
        """
        Check if obj_a dominates obj_b.
        
        A dominates B if A is >= B in all objectives and > in at least one.
        """
        at_least_one_better = False
        
        for key in obj_a.keys():
            is_max = maximize.get(key, True)
            
            if is_max:
                if obj_a[key] < obj_b[key]:
                    return False  # A is worse in this objective
                if obj_a[key] > obj_b[key]:
                    at_least_one_better = True
            else:
                if obj_a[key] > obj_b[key]:
                    return False
                if obj_a[key] < obj_b[key]:
                    at_least_one_better = True
        
        return at_least_one_better
    
    @staticmethod
    def fast_nondominated_sort(
        population: List[Any],
        objectives: List[Dict[str, float]],
        maximize: Dict[str, bool]
    ) -> List[List[int]]:
        """
        Fast non-dominated sorting.
        
        Returns list of fronts, where each front is a list of indices.
        """
        n = len(population)
        
        # For each individual, count how many dominate it
        domination_count = [0] * n
        # For each individual, track which it dominates
        dominated_by: List[Set[int]] = [set() for _ in range(n)]
        
        # Compute domination relationships
        for i in range(n):
            for j in range(i + 1, n):
                if NSGA2.dominates(objectives[i], objectives[j], maximize):
                    # i dominates j
                    dominated_by[i].add(j)
                    domination_count[j] += 1
                elif NSGA2.dominates(objectives[j], objectives[i], maximize):
                    # j dominates i
                    dominated_by[j].add(i)
                    domination_count[i] += 1
        
        # Build fronts
        fronts = []
        current_front = []
        
        # First front: individuals with domination_count = 0
        for i in range(n):
            if domination_count[i] == 0:
                current_front.append(i)
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            
            for i in current_front:
                for j in dominated_by[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            
            current_front = next_front
        
        return fronts
    
    @staticmethod
    def crowding_distance(
        front_indices: List[int],
        objectives: List[Dict[str, float]]
    ) -> List[float]:
        """
        Calculate crowding distance for individuals in a front.
        
        Returns list of distances (same order as front_indices).
        """
        n = len(front_indices)
        if n == 0:
            return []
        if n == 1:
            return [float('inf')]
        if n == 2:
            return [float('inf'), float('inf')]
        
        # Initialize distances
        distances = [0.0] * n
        
        # Get objective names
        obj_names = list(objectives[front_indices[0]].keys())
        
        # For each objective
        for obj_name in obj_names:
            # Sort by this objective
            sorted_indices = sorted(
                range(n),
                key=lambda i: objectives[front_indices[i]][obj_name]
            )
            
            # Boundary points get infinite distance
            distances[sorted_indices[0]] = float('inf')
            distances[sorted_indices[-1]] = float('inf')
            
            # Get objective range
            obj_min = objectives[front_indices[sorted_indices[0]]][obj_name]
            obj_max = objectives[front_indices[sorted_indices[-1]]][obj_name]
            obj_range = obj_max - obj_min
            
            if obj_range < 1e-10:
                continue
            
            # Calculate crowding distance for middle points
            for i in range(1, n - 1):
                idx = sorted_indices[i]
                idx_prev = sorted_indices[i - 1]
                idx_next = sorted_indices[i + 1]
                
                distance = (
                    objectives[front_indices[idx_next]][obj_name] -
                    objectives[front_indices[idx_prev]][obj_name]
                ) / obj_range
                
                distances[idx] += distance
        
        return distances
    
    @staticmethod
    def select(
        population: List[Any],
        objectives: List[Dict[str, float]],
        maximize: Dict[str, bool],
        n_survivors: int
    ) -> List[Any]:
        """
        NSGA-II selection.
        
        Args:
            population: List of individuals
            objectives: List of objective dicts (one per individual)
            maximize: Dict specifying whether to maximize each objective
            n_survivors: Number of survivors to select
        
        Returns:
            List of selected individuals
        """
        n = len(population)
        if n <= n_survivors:
            return population
        
        # Fast non-dominated sorting
        fronts = NSGA2.fast_nondominated_sort(population, objectives, maximize)
        
        # Select individuals front by front
        survivors = []
        for front in fronts:
            if len(survivors) + len(front) <= n_survivors:
                # Add entire front
                survivors.extend([population[i] for i in front])
            else:
                # Need to select from this front using crowding distance
                remaining = n_survivors - len(survivors)
                
                # Calculate crowding distances
                distances = NSGA2.crowding_distance(front, objectives)
                
                # Sort by crowding distance (descending)
                sorted_pairs = sorted(
                    zip(front, distances),
                    key=lambda x: x[1],
                    reverse=True
                )
                
                # Select top remaining individuals
                selected_indices = [idx for idx, _ in sorted_pairs[:remaining]]
                survivors.extend([population[i] for i in selected_indices])
                break
        
        return survivors


# ============================================================================
# TEST
# ============================================================================

def test_nsga2():
    """Test NSGA-II."""
    print("\n" + "="*80)
    print("🧪 TESTE: NSGA-II")
    print("="*80)
    
    # Create test population
    random.seed(42)
    population = list(range(20))
    
    # Generate random objectives (2 objectives)
    objectives = []
    for _ in range(20):
        objectives.append({
            'f1': random.random() * 10,
            'f2': random.random() * 10
        })
    
    maximize = {'f1': True, 'f2': True}
    
    print(f"📦 População: {len(population)} indivíduos")
    print(f"🎯 Objetivos: {list(objectives[0].keys())}")
    
    # Fast non-dominated sort
    fronts = NSGA2.fast_nondominated_sort(population, objectives, maximize)
    
    print(f"\n📊 Fronts de Pareto:")
    for i, front in enumerate(fronts):
        print(f"   Front {i+1}: {len(front)} indivíduos")
    
    # Selection
    survivors = NSGA2.select(population, objectives, maximize, n_survivors=10)
    
    print(f"\n✅ Selecionados: {len(survivors)} sobreviventes")
    
    # Validate
    assert len(fronts) > 0, "No fronts generated"
    assert len(survivors) == 10, f"Expected 10 survivors, got {len(survivors)}"
    
    # Check that survivors are from first fronts
    front1_indices = fronts[0]
    survivors_in_front1 = sum(1 for s in survivors if s in front1_indices)
    print(f"   {survivors_in_front1} do Front 1 (Pareto optimal)")
    
    print("\n✅ NSGA-II: PASS")
    print("="*80)


if __name__ == "__main__":
    test_nsga2()
    print("\n" + "="*80)
    print("✅ nsga2_pure_python.py está FUNCIONAL!")
    print("="*80)
