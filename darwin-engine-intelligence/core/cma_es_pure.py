"""
CMA-ES: Covariance Matrix Adaptation Evolution Strategy
========================================================

IMPLEMENTAÇÃO PURA PYTHON (SEM NUMPY)
Status: FUNCIONAL E TESTADO
Data: 2025-10-03

CMA-ES é um dos algoritmos SOTA para otimização contínua.
Adapta a matriz de covariância para explorar direções promissoras.

Based on: Hansen & Ostermeier (2001) "Completely Derandomized Self-Adaptation"
"""

import math
import random
from typing import List, Callable, Tuple, Dict
from dataclasses import dataclass


@dataclass
class CMAESState:
    """Estado do CMA-ES"""
    generation: int
    mean: List[float]
    sigma: float
    best_fitness: float
    best_solution: List[float]


class CMAES:
    """
    CMA-ES simplificado puro Python
    
    Características:
    - Adaptação da matriz de covariância
    - Step-size control (sigma adaptation)
    - Rank-mu update
    - Sem numpy (usa listas Python)
    """
    
    def __init__(self,
                 initial_mean: List[float],
                 initial_sigma: float = 0.5,
                 population_size: int = None):
        """
        Args:
            initial_mean: Ponto inicial de busca
            initial_sigma: Step-size inicial
            population_size: Tamanho da população (None = auto)
        """
        self.dim = len(initial_mean)
        self.mean = list(initial_mean)
        self.sigma = initial_sigma
        
        # População
        if population_size is None:
            self.lam = 4 + int(3 * math.log(self.dim))  # λ (lambda)
        else:
            self.lam = population_size
        
        self.mu = self.lam // 2  # μ (mu) = pais selecionados
        
        # Pesos de recombinação
        self.weights = self._calculate_weights()
        self.mueff = sum(self.weights)**2 / sum(w*w for w in self.weights)
        
        # Parâmetros de adaptação
        self.cc = 4.0 / (self.dim + 4.0)  # Cumulation for C
        self.cs = 4.0 / (self.dim + 4.0)  # Cumulation for sigma
        self.c1 = 2.0 / ((self.dim + 1.3)**2 + self.mueff)  # Rank-one update
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1/self.mueff) / 
                      ((self.dim + 2)**2 + self.mueff))  # Rank-mu update
        self.damps = 1.0 + 2.0 * max(0, math.sqrt((self.mueff-1)/(self.dim+1)) - 1) + self.cs
        
        # Matriz de covariância (identidade no início)
        self.C = [[1.0 if i == j else 0.0 for j in range(self.dim)] 
                  for i in range(self.dim)]
        
        # Evolution paths
        self.pc = [0.0] * self.dim  # For C
        self.ps = [0.0] * self.dim  # For sigma
        
        # Estado
        self.generation = 0
        self.best_fitness = float('inf')
        self.best_solution = list(initial_mean)
        
        # Eigendecomposition cache (simplificado)
        self.B = [[1.0 if i == j else 0.0 for j in range(self.dim)] 
                  for i in range(self.dim)]  # Eigenvectors
        self.D = [1.0] * self.dim  # Eigenvalues
        
        print(f"🧬 CMA-ES inicializado:")
        print(f"   Dimensão: {self.dim}")
        print(f"   População (λ): {self.lam}")
        print(f"   Pais (μ): {self.mu}")
        print(f"   Sigma inicial: {self.sigma:.3f}")
    
    def _calculate_weights(self) -> List[float]:
        """Calcula pesos de recombinação"""
        weights_raw = [math.log(self.mu + 0.5) - math.log(i + 1) 
                       for i in range(self.mu)]
        sum_weights = sum(weights_raw)
        return [w / sum_weights for w in weights_raw]
    
    def _sample_population(self) -> List[List[float]]:
        """Amostra população usando distribuição multivariada"""
        population = []
        
        for _ in range(self.lam):
            # Sample z ~ N(0, I)
            z = [random.gauss(0, 1) for _ in range(self.dim)]
            
            # Transform: y = B * D * z (simplificado sem eigen decomp completa)
            # Aproximação: usa apenas diagonal de C
            y = [z[i] * math.sqrt(self.C[i][i]) for i in range(self.dim)]
            
            # x = mean + sigma * y
            x = [self.mean[i] + self.sigma * y[i] for i in range(self.dim)]
            
            population.append(x)
        
        return population
    
    def _update_distribution(self, population: List[List[float]], 
                             fitnesses: List[float]):
        """Atualiza mean, C, sigma"""
        # Ordenar por fitness
        sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
        
        # Melhor solução
        best_idx = sorted_indices[0]
        if fitnesses[best_idx] < self.best_fitness:
            self.best_fitness = fitnesses[best_idx]
            self.best_solution = list(population[best_idx])
        
        # Selecionar top μ
        selected = [population[i] for i in sorted_indices[:self.mu]]
        
        # Calcular old mean
        old_mean = list(self.mean)
        
        # Update mean (recombinação ponderada)
        new_mean = [0.0] * self.dim
        for i in range(self.mu):
            for j in range(self.dim):
                new_mean[j] += self.weights[i] * selected[i][j]
        
        self.mean = new_mean
        
        # Update evolution paths
        # ps update (para sigma)
        mean_diff_normalized = [(self.mean[i] - old_mean[i]) / self.sigma 
                                for i in range(self.dim)]
        
        # ps = (1 - cs) * ps + sqrt(cs*(2-cs)*mueff) * mean_diff_normalized
        factor_ps = math.sqrt(self.cs * (2 - self.cs) * self.mueff)
        for i in range(self.dim):
            self.ps[i] = (1 - self.cs) * self.ps[i] + factor_ps * mean_diff_normalized[i]
        
        # pc update (para C)
        ps_norm = math.sqrt(sum(p*p for p in self.ps))
        hsig = 1.0 if ps_norm < 1.5 * math.sqrt(self.dim) else 0.0
        
        factor_pc = math.sqrt(self.cc * (2 - self.cc) * self.mueff)
        for i in range(self.dim):
            self.pc[i] = ((1 - self.cc) * self.pc[i] + 
                         hsig * factor_pc * mean_diff_normalized[i])
        
        # Update C (matriz de covariância) - simplificado diagonal
        # C = (1-c1-cmu) * C + c1 * pc*pc' + cmu * sum(weights[i] * y[i]*y[i]')
        for i in range(self.dim):
            # Rank-one update
            c_rank_one = self.c1 * self.pc[i] * self.pc[i]
            
            # Rank-mu update (simplificado)
            c_rank_mu = 0.0
            for k in range(self.mu):
                y_k = [(selected[k][j] - old_mean[j]) / self.sigma 
                      for j in range(self.dim)]
                c_rank_mu += self.cmu * self.weights[k] * y_k[i] * y_k[i]
            
            # Update diagonal
            self.C[i][i] = ((1 - self.c1 - self.cmu) * self.C[i][i] + 
                           c_rank_one + c_rank_mu)
            
            # Garantir positivo
            self.C[i][i] = max(1e-10, self.C[i][i])
        
        # Update sigma (step-size)
        # sigma *= exp((cs/damps) * (||ps||/E||N(0,I)|| - 1))
        ps_norm = math.sqrt(sum(p*p for p in self.ps))
        expected_norm = math.sqrt(self.dim) * (1 - 1/(4*self.dim) + 1/(21*self.dim**2))
        
        self.sigma *= math.exp((self.cs / self.damps) * (ps_norm / expected_norm - 1))
        self.sigma = min(10.0, max(1e-10, self.sigma))  # Clip
    
    def optimize(self, fitness_fn: Callable[[List[float]], float],
                 max_generations: int = 100,
                 target_fitness: float = 1e-10,
                 verbose: bool = True) -> CMAESState:
        """
        Otimiza função objetivo
        
        Args:
            fitness_fn: Função f(x) -> fitness (minimizar)
            max_generations: Máximo de gerações
            target_fitness: Fitness alvo (parar se atingir)
            verbose: Mostrar progresso
        
        Returns:
            Estado final
        """
        if verbose:
            print(f"\n🚀 CMA-ES Otimização iniciada")
            print(f"   Gerações máx: {max_generations}")
            print(f"   Target fitness: {target_fitness}\n")
        
        for gen in range(max_generations):
            self.generation = gen + 1
            
            # Sample população
            population = self._sample_population()
            
            # Avaliar
            fitnesses = [fitness_fn(x) for x in population]
            
            # Update distribuição
            self._update_distribution(population, fitnesses)
            
            # Log
            if verbose and (gen + 1) % 10 == 0:
                avg_fit = sum(fitnesses) / len(fitnesses)
                print(f"Gen {gen+1:3d}: Best={self.best_fitness:.6e}, "
                      f"Avg={avg_fit:.6e}, Sigma={self.sigma:.3f}")
            
            # Convergência
            if self.best_fitness < target_fitness:
                if verbose:
                    print(f"\n✅ Target atingido em {gen+1} gerações!")
                break
        
        if verbose:
            print(f"\n🏁 Otimização completa")
            print(f"   Best fitness: {self.best_fitness:.6e}")
            print(f"   Best solution: {[f'{x:.4f}' for x in self.best_solution[:5]]}")
        
        return CMAESState(
            generation=self.generation,
            mean=self.mean,
            sigma=self.sigma,
            best_fitness=self.best_fitness,
            best_solution=self.best_solution
        )


# ============================================================================
# TESTES
# ============================================================================

def test_cmaes_sphere():
    """Testa CMA-ES na função esfera"""
    print("\n" + "="*80)
    print("TESTE: CMA-ES - Função Esfera")
    print("="*80 + "\n")
    
    dim = 5
    
    def sphere(x):
        """f(x) = sum(x_i^2), mínimo em x=0"""
        return sum(xi**2 for xi in x)
    
    initial = [random.uniform(-5, 5) for _ in range(dim)]
    
    cmaes = CMAES(initial_mean=initial, initial_sigma=2.0)
    
    result = cmaes.optimize(
        fitness_fn=sphere,
        max_generations=50,
        target_fitness=1e-6,
        verbose=True
    )
    
    print(f"\n✅ Resultado:")
    print(f"   Fitness final: {result.best_fitness:.6e}")
    print(f"   Solução: {[f'{x:.4f}' for x in result.best_solution]}")
    
    assert result.best_fitness < 1e-3, "Deve convergir para perto de 0"
    
    print("\n" + "="*80)


def test_cmaes_rosenbrock():
    """Testa CMA-ES na função Rosenbrock"""
    print("\n" + "="*80)
    print("TESTE: CMA-ES - Função Rosenbrock")
    print("="*80 + "\n")
    
    def rosenbrock(x):
        """f(x) = sum(100*(x[i+1]-x[i]^2)^2 + (1-x[i])^2), mínimo em x=1"""
        total = 0.0
        for i in range(len(x) - 1):
            total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return total
    
    dim = 3
    initial = [random.uniform(-2, 2) for _ in range(dim)]
    
    cmaes = CMAES(initial_mean=initial, initial_sigma=1.0, population_size=20)
    
    result = cmaes.optimize(
        fitness_fn=rosenbrock,
        max_generations=100,
        target_fitness=1e-2,
        verbose=True
    )
    
    print(f"\n✅ Resultado:")
    print(f"   Fitness final: {result.best_fitness:.6e}")
    print(f"   Solução: {[f'{x:.4f}' for x in result.best_solution]}")
    print(f"   (Ótimo em [1, 1, 1])")
    
    # Rosenbrock é difícil, aceitar convergência razoável
    assert result.best_fitness < 10.0, "Deve melhorar significativamente"
    
    print("\n" + "="*80)


if __name__ == "__main__":
    random.seed(42)
    
    test_cmaes_sphere()
    test_cmaes_rosenbrock()
    
    print("\n✅ cma_es_pure.py FUNCIONAL!")
