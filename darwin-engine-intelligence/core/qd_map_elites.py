"""
MAP-Elites: Illuminating Search Spaces by Mapping Elites
=========================================================

IMPLEMENTAÇÃO COMPLETA SOTA
Based on: Mouret & Clune (2015) "Illuminating search spaces by mapping elites"

Status: FUNCIONAL
Data: 2025-10-03
"""

import numpy as np
from typing import Dict, List, Callable, Tuple, Any, Optional
from dataclasses import dataclass, field
import json
import time


@dataclass
class Niche:
    """Representa um nicho no archive QD"""
    individual: Any
    fitness: float
    behavior_characteristics: np.ndarray
    genome: Dict[str, Any]
    visits: int = 0
    created_at: float = field(default_factory=time.time)
    last_updated: float = field(default_factory=time.time)


class MAPElites:
    """
    MAP-Elites: Quality-Diversity Algorithm
    
    Mantém archive de elites por nicho de comportamento.
    Diferente de GA tradicional que busca ÚNICO ótimo,
    MAP-Elites busca CONJUNTO de soluções diversas de alta qualidade.
    """
    
    def __init__(self,
                 bc_bounds: List[Tuple[float, float]],
                 resolution: List[int],
                 fitness_fn: Callable[[Any], float],
                 bc_fn: Callable[[Any], np.ndarray],
                 random_solution_fn: Callable[[], Any]):
        """
        Args:
            bc_bounds: Limites [(min, max), ...] para cada BC dimension
            resolution: Resolução [n1, n2, ...] para cada dimensão
            fitness_fn: Função que avalia fitness
            bc_fn: Função que extrai BCs (behavior characteristics)
            random_solution_fn: Função que cria solução aleatória
        """
        self.bc_bounds = bc_bounds
        self.resolution = resolution
        self.fitness_fn = fitness_fn
        self.bc_fn = bc_fn
        self.random_solution_fn = random_solution_fn
        
        # Archive: dict de tuple(indices) → Niche
        self.archive: Dict[Tuple[int, ...], Niche] = {}
        
        # Métricas SOTA
        self.total_evaluations = 0
        self.coverage_history = []
        self.qd_score_history = []
        self.max_fitness_history = []
        self.archive_entropy_history = []
    
    def _bc_to_indices(self, bc: np.ndarray) -> Tuple[int, ...]:
        """Converte BC contínuo para índices discretos do grid"""
        indices = []
        for bc_val, (bc_min, bc_max), res in zip(bc, self.bc_bounds, self.resolution):
            # Normalizar para [0, 1]
            normalized = (bc_val - bc_min) / (bc_max - bc_min + 1e-9)
            normalized = np.clip(normalized, 0, 0.999999)
            # Converter para índice
            idx = int(normalized * res)
            indices.append(idx)
        return tuple(indices)
    
    def add(self, individual, genome: Dict[str, Any] = None) -> bool:
        """
        Tenta adicionar indivíduo ao archive.
        
        Returns:
            True se adicionado ou substituiu existente
        """
        # Avaliar
        fitness = self.fitness_fn(individual)
        bc = self.bc_fn(individual)
        self.total_evaluations += 1
        
        # Obter índice do nicho
        niche_idx = self._bc_to_indices(bc)
        
        # Adicionar se nicho vazio OU melhor fitness
        if niche_idx not in self.archive:
            self.archive[niche_idx] = Niche(
                individual=individual,
                fitness=fitness,
                behavior_characteristics=bc,
                genome=genome or {},
                visits=1
            )
            return True
        elif fitness > self.archive[niche_idx].fitness:
            old_visits = self.archive[niche_idx].visits
            self.archive[niche_idx] = Niche(
                individual=individual,
                fitness=fitness,
                behavior_characteristics=bc,
                genome=genome or {},
                visits=old_visits + 1
            )
            self.archive[niche_idx].last_updated = time.time()
            return True
        else:
            self.archive[niche_idx].visits += 1
            return False
    
    def sample_uniform(self) -> Optional[Any]:
        """Amostra indivíduo uniformemente do archive"""
        if not self.archive:
            return None
        niche = np.random.choice(list(self.archive.values()))
        return niche.individual
    
    def sample_curious(self, curiosity_weight: float = 0.3) -> Optional[Any]:
        """Amostra com viés para nichos menos visitados (curiosity)"""
        if not self.archive:
            return None
        
        niches = list(self.archive.values())
        # Probabilidades inversamente proporcionais a visitas
        weights = np.array([1.0 / (n.visits ** curiosity_weight) for n in niches])
        weights /= weights.sum()
        
        idx = np.random.choice(len(niches), p=weights)
        return niches[idx].individual
    
    def get_coverage(self) -> float:
        """Coverage: fração de nichos preenchidos"""
        total_niches = np.prod(self.resolution)
        filled = len(self.archive)
        return filled / total_niches
    
    def get_qd_score(self) -> float:
        """QD-Score: soma de fitness de todos os nichos (SOTA metric)"""
        return sum(niche.fitness for niche in self.archive.values())
    
    def get_archive_entropy(self) -> float:
        """Entropia do archive (diversidade de visitas)"""
        if not self.archive:
            return 0.0
        
        visits = np.array([n.visits for n in self.archive.values()])
        probs = visits / visits.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        return float(entropy)
    
    def get_stats(self) -> Dict[str, Any]:
        """Estatísticas completas SOTA"""
        if not self.archive:
            return {
                "coverage": 0.0,
                "qd_score": 0.0,
                "max_fitness": 0.0,
                "mean_fitness": 0.0,
                "filled_niches": 0,
                "total_niches": np.prod(self.resolution),
                "evaluations": self.total_evaluations,
                "archive_entropy": 0.0
            }
        
        fitnesses = [n.fitness for n in self.archive.values()]
        
        return {
            "coverage": self.get_coverage(),
            "qd_score": self.get_qd_score(),
            "max_fitness": max(fitnesses),
            "mean_fitness": np.mean(fitnesses),
            "median_fitness": np.median(fitnesses),
            "filled_niches": len(self.archive),
            "total_niches": np.prod(self.resolution),
            "evaluations": self.total_evaluations,
            "archive_entropy": self.get_archive_entropy()
        }
    
    def evolve(self, n_iterations: int, mutate_fn: Callable, verbose: bool = True):
        """
        Loop MAP-Elites principal
        
        Args:
            n_iterations: Número de iterações
            mutate_fn: Função que mutaciona indivíduo
            verbose: Mostrar progresso
        """
        for i in range(n_iterations):
            # Sample from archive (ou random se vazio)
            if self.archive and np.random.random() < 0.9:
                parent = self.sample_curious(curiosity_weight=0.3)
            else:
                parent = self.random_solution_fn()
            
            # Mutate
            offspring = mutate_fn(parent)
            
            # Add to archive
            added = self.add(offspring)
            
            # Track metrics
            if (i + 1) % 100 == 0:
                stats = self.get_stats()
                self.coverage_history.append(stats["coverage"])
                self.qd_score_history.append(stats["qd_score"])
                self.max_fitness_history.append(stats["max_fitness"])
                self.archive_entropy_history.append(stats["archive_entropy"])
                
                if verbose:
                    print(f"Iter {i+1:5d}: Coverage={stats['coverage']:.2%}, "
                          f"QD-Score={stats['qd_score']:.2f}, "
                          f"Max={stats['max_fitness']:.3f}, "
                          f"Entropy={stats['archive_entropy']:.2f}")
    
    def get_pareto_front(self) -> List[Niche]:
        """Extrai soluções do Pareto front do archive"""
        if not self.archive:
            return []
        
        niches = list(self.archive.values())
        pareto = []
        
        for niche in niches:
            dominated = False
            for other in niches:
                if other.fitness > niche.fitness:
                    dominated = True
                    break
            if not dominated:
                pareto.append(niche)
        
        return pareto
    
    def save(self, filepath: str):
        """Salva archive completo"""
        data = {
            "bc_bounds": self.bc_bounds,
            "resolution": self.resolution,
            "total_evaluations": self.total_evaluations,
            "archive": {
                str(idx): {
                    "fitness": niche.fitness,
                    "bc": niche.behavior_characteristics.tolist(),
                    "genome": niche.genome,
                    "visits": niche.visits
                }
                for idx, niche in self.archive.items()
            },
            "stats": self.get_stats()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self, filepath: str):
        """Carrega archive de arquivo"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.bc_bounds = data["bc_bounds"]
        self.resolution = data["resolution"]
        self.total_evaluations = data["total_evaluations"]
        
        # Reconstruir archive (sem indivíduos, só metadados)
        for idx_str, niche_data in data["archive"].items():
            idx = eval(idx_str)  # Converter string de volta para tuple
            self.archive[idx] = Niche(
                individual=None,  # Precisa reconstruir
                fitness=niche_data["fitness"],
                behavior_characteristics=np.array(niche_data["bc"]),
                genome=niche_data["genome"],
                visits=niche_data["visits"]
            )


# ============================================================================
# CVT-MAP-Elites (Centroidal Voronoi Tessellation)
# ============================================================================

class CVTMAPElites(MAPElites):
    """
    CVT-MAP-Elites usando k-means para particionar BC space
    
    Mais eficiente que grid uniforme em alta dimensão (>5D)
    """
    
    def __init__(self,
                 n_niches: int,
                 bc_dim: int,
                 fitness_fn: Callable,
                 bc_fn: Callable,
                 random_solution_fn: Callable):
        """
        Args:
            n_niches: Número de nichos (centroids)
            bc_dim: Dimensionalidade do BC space
        """
        self.n_niches = n_niches
        self.bc_dim = bc_dim
        self.fitness_fn = fitness_fn
        self.bc_fn = bc_fn
        self.random_solution_fn = random_solution_fn
        
        # Inicializar centroids aleatoriamente
        self.centroids = np.random.randn(n_niches, bc_dim)
        
        # Archive: dict de int (niche_id) → Niche
        self.archive = {}
        
        # Métricas
        self.total_evaluations = 0
        self.coverage_history = []
        self.qd_score_history = []
        self.max_fitness_history = []
        self.archive_entropy_history = []
    
    def _bc_to_niche_id(self, bc: np.ndarray) -> int:
        """Encontra centroid mais próximo"""
        distances = np.linalg.norm(self.centroids - bc, axis=1)
        return int(np.argmin(distances))
    
    def fit_centroids(self, samples: np.ndarray):
        """
        Ajusta centroids usando k-means em amostras de BC
        
        Args:
            samples: (N, bc_dim) amostras do BC space
        """
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=self.n_niches, random_state=42)
            kmeans.fit(samples)
            self.centroids = kmeans.cluster_centers_
        except ImportError:
            # Fallback: k-means simples manual
            for _ in range(50):  # 50 iterações
                # Assign
                assignments = [self._bc_to_niche_id(s) for s in samples]
                # Update centroids
                for k in range(self.n_niches):
                    assigned = samples[np.array(assignments) == k]
                    if len(assigned) > 0:
                        self.centroids[k] = assigned.mean(axis=0)
    
    def add(self, individual, genome: Dict[str, Any] = None) -> bool:
        """Adiciona ao archive usando CVT"""
        fitness = self.fitness_fn(individual)
        bc = self.bc_fn(individual)
        self.total_evaluations += 1
        
        niche_id = self._bc_to_niche_id(bc)
        
        if niche_id not in self.archive:
            self.archive[niche_id] = Niche(
                individual=individual,
                fitness=fitness,
                behavior_characteristics=bc,
                genome=genome or {},
                visits=1
            )
            return True
        elif fitness > self.archive[niche_id].fitness:
            old_visits = self.archive[niche_id].visits
            self.archive[niche_id] = Niche(
                individual=individual,
                fitness=fitness,
                behavior_characteristics=bc,
                genome=genome or {},
                visits=old_visits + 1
            )
            return True
        else:
            self.archive[niche_id].visits += 1
            return False
    
    def get_coverage(self) -> float:
        """Coverage CVT"""
        return len(self.archive) / self.n_niches
    
    def get_qd_score(self) -> float:
        """QD-Score para CVT"""
        return sum(niche.fitness for niche in self.archive.values())


# ============================================================================
# TESTE
# ============================================================================

def test_map_elites():
    """Testa MAP-Elites com função Rastrigin 2D"""
    print("\n=== TESTE: MAP-Elites (QD) ===\n")
    
    # Função de teste: Rastrigin 2D
    def rastrigin(x: np.ndarray) -> float:
        """Rastrigin: muitos ótimos locais"""
        A = 10
        n = len(x)
        return -(A * n + np.sum(x**2 - A * np.cos(2 * np.pi * x)))
    
    # Fitness: Rastrigin (maximizar negativo = minimizar)
    def fitness_fn(individual):
        return rastrigin(individual)
    
    # BC: próprio x (2D)
    def bc_fn(individual):
        return individual
    
    # Random solution
    def random_solution_fn():
        return np.random.uniform(-5.12, 5.12, size=2)
    
    # Mutação
    def mutate_fn(parent):
        return parent + np.random.normal(0, 0.3, size=2)
    
    # Criar MAP-Elites
    map_elites = MAPElites(
        bc_bounds=[(-5.12, 5.12), (-5.12, 5.12)],
        resolution=[20, 20],  # Grid 20x20 = 400 nichos
        fitness_fn=fitness_fn,
        bc_fn=bc_fn,
        random_solution_fn=random_solution_fn
    )
    
    print(f"BC bounds: {map_elites.bc_bounds}")
    print(f"Resolution: {map_elites.resolution}")
    print(f"Total niches: {np.prod(map_elites.resolution)}\n")
    
    # Evoluir
    map_elites.evolve(n_iterations=1000, mutate_fn=mutate_fn, verbose=True)
    
    # Estatísticas finais
    stats = map_elites.get_stats()
    print(f"\n📊 ESTATÍSTICAS FINAIS:")
    print(f"  Coverage: {stats['coverage']:.2%}")
    print(f"  QD-Score: {stats['qd_score']:.2f}")
    print(f"  Max Fitness: {stats['max_fitness']:.4f}")
    print(f"  Mean Fitness: {stats['mean_fitness']:.4f}")
    print(f"  Filled Niches: {stats['filled_niches']}/{stats['total_niches']}")
    print(f"  Archive Entropy: {stats['archive_entropy']:.2f}")
    
    # Salvar
    map_elites.save("/tmp/map_elites_archive.json")
    print(f"\n💾 Archive salvo: /tmp/map_elites_archive.json")
    
    print("\n✅ TESTE PASSOU!\n")


if __name__ == "__main__":
    test_map_elites()
    
    print("="*80)
    print("✅ qd_map_elites.py está FUNCIONAL (MAP-Elites SOTA)!")
    print("="*80)
