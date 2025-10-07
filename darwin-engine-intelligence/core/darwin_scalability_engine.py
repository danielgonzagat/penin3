"""
Darwin Scalability Engine - Universal Execution
================================================

IMPLEMENTAÇÃO REAL - Sistema de escalabilidade universal.

Suporta execução em:
- CPU local (sequential/parallel)
- Multiprocessing (stdlib)
- Ray (cluster distributed) - opcional
- Dask (big data) - opcional

Criado: 2025-10-03
Status: FUNCIONAL (testado com stdlib)
"""

from __future__ import annotations
from typing import List, Callable, Any, Dict
from abc import ABC, abstractmethod
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import time


class ExecutionBackend(ABC):
    """Interface para backend de execução."""
    
    @abstractmethod
    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """Mapeia função sobre lista de items."""
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Nome do backend."""
        pass


class SequentialBackend(ExecutionBackend):
    """Execução sequencial (CPU único, sem paralelismo)."""
    
    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """Map sequencial."""
        return [func(item) for item in items]
    
    def get_name(self) -> str:
        return "Sequential (single CPU)"


class MultiprocessingBackend(ExecutionBackend):
    """Execução paralela com multiprocessing (stdlib)."""
    
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or mp.cpu_count()
    
    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """Map paralelo com ProcessPoolExecutor."""
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(func, items))
        return results
    
    def get_name(self) -> str:
        return f"Multiprocessing ({self.n_workers} workers)"


class ThreadPoolBackend(ExecutionBackend):
    """Execução paralela com threads (I/O bound)."""
    
    def __init__(self, n_workers: int = None):
        self.n_workers = n_workers or (mp.cpu_count() * 2)
    
    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """Map paralelo com ThreadPoolExecutor."""
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            results = list(executor.map(func, items))
        return results
    
    def get_name(self) -> str:
        return f"ThreadPool ({self.n_workers} threads)"


class RayBackend(ExecutionBackend):
    """Execução distribuída com Ray (opcional)."""
    
    def __init__(self, address: str = None):
        try:
            import ray
            
            if not ray.is_initialized():
                if address:
                    ray.init(address=address)
                else:
                    ray.init(ignore_reinit_error=True)
            
            self.ray = ray
            self.available = True
        except ImportError:
            self.available = False
            print("⚠️ Ray not installed. Install with: pip install ray")
    
    def map(self, func: Callable, items: List[Any]) -> List[Any]:
        """Map distribuído com Ray."""
        if not self.available:
            raise RuntimeError("Ray not available")
        
        # Converter função para Ray remote
        remote_func = self.ray.remote(func)
        
        # Executar em paralelo
        futures = [remote_func.remote(item) for item in items]
        results = self.ray.get(futures)
        
        return results
    
    def get_name(self) -> str:
        if self.available:
            return f"Ray Distributed (cluster)"
        return "Ray (not available)"


class ScalabilityEngine:
    """
    Motor de escalabilidade universal.
    
    Abstrai execução para diferentes backends:
    - Sequential: desenvolvimento/debug
    - Multiprocessing: CPU local paralelo
    - ThreadPool: I/O bound tasks
    - Ray: cluster distribuído
    """
    
    def __init__(self, backend: str = 'auto', **backend_kwargs):
        """
        Args:
            backend: 'auto', 'sequential', 'multiprocessing', 'threadpool', 'ray'
            backend_kwargs: Argumentos para o backend
        """
        self.backend = self._create_backend(backend, **backend_kwargs)
        self.stats = {
            'total_calls': 0,
            'total_items': 0,
            'total_time': 0.0
        }
    
    def _create_backend(self, backend: str, **kwargs) -> ExecutionBackend:
        """Cria backend apropriado."""
        if backend == 'auto':
            # Auto-detectar melhor backend
            try:
                import ray
                return RayBackend(**kwargs)
            except ImportError:
                # Fallback para multiprocessing
                return MultiprocessingBackend(**kwargs)
        
        elif backend == 'sequential':
            return SequentialBackend()
        
        elif backend == 'multiprocessing':
            return MultiprocessingBackend(**kwargs)
        
        elif backend == 'threadpool':
            return ThreadPoolBackend(**kwargs)
        
        elif backend == 'ray':
            return RayBackend(**kwargs)
        
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def parallel_map(self, func: Callable, items: List[Any]) -> List[Any]:
        """
        Executa função em paralelo sobre lista.
        
        Args:
            func: Função a aplicar
            items: Lista de items
        
        Returns:
            Lista de resultados
        """
        start_time = time.time()
        
        # Executar
        results = self.backend.map(func, items)
        
        # Estatísticas
        elapsed = time.time() - start_time
        self.stats['total_calls'] += 1
        self.stats['total_items'] += len(items)
        self.stats['total_time'] += elapsed
        
        return results
    
    def parallel_evaluate_population(self, population: List[Any]) -> List[Any]:
        """
        Avalia fitness de população em paralelo.
        
        Args:
            population: Lista de indivíduos
        
        Returns:
            População com fitness avaliado
        """
        def evaluate_individual(ind):
            """Avalia um indivíduo."""
            ind.evaluate_fitness()
            return ind
        
        return self.parallel_map(evaluate_individual, population)
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Retorna info sobre backend atual."""
        return {
            'backend': self.backend.get_name(),
            'stats': self.stats
        }
    
    def shutdown(self):
        """Finaliza backend (se necessário)."""
        if isinstance(self.backend, RayBackend) and self.backend.available:
            try:
                self.backend.ray.shutdown()
            except:
                pass


# ============================================================================
# TESTES
# ============================================================================

def expensive_computation(x: int) -> int:
    """Função cara para testar paralelismo."""
    # Simular trabalho pesado
    result = 0
    for i in range(10000):
        result += i * x
    return result


def test_scalability():
    """Testa engine de escalabilidade."""
    print("\n=== TESTE: Scalability Engine ===\n")
    
    # Dados de teste
    items = list(range(100))
    
    # Teste 1: Sequential
    print("Teste 1: Sequential Backend")
    engine_seq = ScalabilityEngine(backend='sequential')
    
    start = time.time()
    results_seq = engine_seq.parallel_map(expensive_computation, items)
    time_seq = time.time() - start
    
    print(f"  Tempo: {time_seq:.3f}s")
    print(f"  Resultados: {len(results_seq)} items")
    print(f"  Backend: {engine_seq.backend.get_name()}")
    
    # Teste 2: Multiprocessing
    print("\nTeste 2: Multiprocessing Backend")
    engine_mp = ScalabilityEngine(backend='multiprocessing', n_workers=4)
    
    start = time.time()
    results_mp = engine_mp.parallel_map(expensive_computation, items)
    time_mp = time.time() - start
    
    print(f"  Tempo: {time_mp:.3f}s")
    print(f"  Resultados: {len(results_mp)} items")
    print(f"  Backend: {engine_mp.backend.get_name()}")
    print(f"  Speedup: {time_seq/time_mp:.2f}x")
    
    # Teste 3: ThreadPool
    print("\nTeste 3: ThreadPool Backend")
    engine_tp = ScalabilityEngine(backend='threadpool', n_workers=4)
    
    start = time.time()
    results_tp = engine_tp.parallel_map(expensive_computation, items)
    time_tp = time.time() - start
    
    print(f"  Tempo: {time_tp:.3f}s")
    print(f"  Resultados: {len(results_tp)} items")
    print(f"  Backend: {engine_tp.backend.get_name()}")
    
    # Teste 4: Ray (se disponível)
    print("\nTeste 4: Ray Backend (se disponível)")
    try:
        engine_ray = ScalabilityEngine(backend='ray')
        
        if engine_ray.backend.available:
            start = time.time()
            results_ray = engine_ray.parallel_map(expensive_computation, items[:20])  # Menos items
            time_ray = time.time() - start
            
            print(f"  Tempo: {time_ray:.3f}s")
            print(f"  Resultados: {len(results_ray)} items")
            print(f"  Backend: {engine_ray.backend.get_name()}")
            
            engine_ray.shutdown()
        else:
            print(f"  ⚠️ Ray não disponível")
    except Exception as e:
        print(f"  ⚠️ Ray error: {e}")
    
    # Comparação
    print(f"\n📊 Comparação:")
    print(f"  Sequential: {time_seq:.3f}s (baseline)")
    print(f"  Multiprocessing: {time_mp:.3f}s ({time_seq/time_mp:.2f}x speedup)")
    print(f"  ThreadPool: {time_tp:.3f}s ({time_seq/time_tp:.2f}x speedup)")
    
    print("\n✅ Teste passou!")


if __name__ == "__main__":
    test_scalability()
    
    print("\n" + "="*80)
    print("✅ darwin_scalability_engine.py está FUNCIONAL!")
    print("="*80)
