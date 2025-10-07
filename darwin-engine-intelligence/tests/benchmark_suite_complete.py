"""
Benchmark Suite Completo - Validação SOTA
==========================================

IMPLEMENTAÇÃO COMPLETA E TESTADA
Status: FUNCIONAL
Data: 2025-10-03

Suite completa de benchmarks para validar todos os componentes SOTA
implementados no Darwin Engine Intelligence.
"""

import sys
sys.path.insert(0, '/workspace')

import time
import random
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Resultado de um benchmark"""
    name: str
    status: str  # PASS, FAIL, SKIP
    time_seconds: float
    metrics: Dict[str, Any]
    error: str = ""


class BenchmarkSuite:
    """Suite completa de benchmarks"""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def run_all(self, verbose: bool = True):
        """Executa todos os benchmarks"""
        if verbose:
            print("\n" + "="*80)
            print("🧪 BENCHMARK SUITE COMPLETO - DARWIN ENGINE INTELLIGENCE")
            print("="*80 + "\n")
        
        # 1. NSGA-III
        self._benchmark_nsga3(verbose)
        
        # 2. POET-Lite
        self._benchmark_poet(verbose)
        
        # 3. PBT
        self._benchmark_pbt(verbose)
        
        # 4. Hypervolume
        self._benchmark_hypervolume(verbose)
        
        # 5. CMA-ES
        self._benchmark_cmaes(verbose)
        
        # 6. Island Model
        self._benchmark_islands(verbose)
        
        # 7. SOTA Integrator
        self._benchmark_integrator(verbose)
        
        # 8. Omega Extensions
        self._benchmark_omega(verbose)
        
        # Print summary
        self._print_summary(verbose)
    
    def _benchmark_nsga3(self, verbose: bool):
        """Benchmark NSGA-III"""
        start = time.time()
        try:
            from core.nsga3_pure_python import NSGA3
            
            nsga3 = NSGA3(n_objectives=3, n_partitions=4)
            
            # População de teste
            population = list(range(20))
            objectives = [
                {'f1': random.random(), 'f2': random.random(), 'f3': random.random()}
                for _ in range(20)
            ]
            maximize = {'f1': True, 'f2': True, 'f3': True}
            
            # Seleção
            survivors = nsga3.select(population, objectives, maximize, n_survivors=10)
            
            elapsed = time.time() - start
            
            metrics = {
                'ref_points': len(nsga3.ref_points),
                'survivors': len(survivors),
                'time_ms': elapsed * 1000
            }
            
            status = "PASS" if len(survivors) == 10 else "FAIL"
            
            self.results.append(BenchmarkResult(
                name="NSGA-III",
                status=status,
                time_seconds=elapsed,
                metrics=metrics
            ))
            
            if verbose:
                print(f"✅ NSGA-III: {status} ({elapsed*1000:.1f}ms)")
                print(f"   Ref points: {len(nsga3.ref_points)}, Survivors: {len(survivors)}")
        
        except Exception as e:
            elapsed = time.time() - start
            self.results.append(BenchmarkResult(
                name="NSGA-III",
                status="FAIL",
                time_seconds=elapsed,
                metrics={},
                error=str(e)
            ))
            if verbose:
                print(f"❌ NSGA-III: FAIL - {e}")
    
    def _benchmark_poet(self, verbose: bool):
        """Benchmark POET-Lite"""
        start = time.time()
        try:
            from core.poet_lite_pure import POETLite, Environment, Agent
            
            # Funções dummy
            def env_gen():
                return Environment(f"env_{random.randint(1000,9999)}", 
                                  {'noise': random.random()}, 0.1)
            
            def agent_factory():
                return Agent(f"agent_{random.randint(1000,9999)}",
                            {'value': random.random()})
            
            def eval_fn(agent, env):
                return random.random()
            
            def mutate_env(env):
                return env_gen()
            
            def mutate_agent(agent):
                return agent_factory()
            
            # Criar POET
            poet = POETLite(env_gen, agent_factory, eval_fn, 
                           mutate_env, mutate_agent, mc_threshold=0.3)
            
            poet.initialize(n_initial_pairs=3)
            poet.evolve(n_iterations=5, verbose=False)
            
            elapsed = time.time() - start
            
            metrics = {
                'environments': len(poet.environments),
                'agents': len(poet.agents),
                'evaluations': poet.total_evaluations,
                'transfers': poet.transfer_successes,
                'time_ms': elapsed * 1000
            }
            
            status = "PASS" if len(poet.environments) >= 3 else "FAIL"
            
            self.results.append(BenchmarkResult(
                name="POET-Lite",
                status=status,
                time_seconds=elapsed,
                metrics=metrics
            ))
            
            if verbose:
                print(f"✅ POET-Lite: {status} ({elapsed*1000:.1f}ms)")
                print(f"   Envs: {len(poet.environments)}, Evals: {poet.total_evaluations}")
        
        except Exception as e:
            elapsed = time.time() - start
            self.results.append(BenchmarkResult(
                name="POET-Lite",
                status="FAIL",
                time_seconds=elapsed,
                metrics={},
                error=str(e)
            ))
            if verbose:
                print(f"❌ POET-Lite: FAIL - {e}")
    
    def _benchmark_pbt(self, verbose: bool):
        """Benchmark PBT"""
        start = time.time()
        try:
            from core.pbt_scheduler_pure import PBTScheduler
            
            def eval_fn(hp):
                return -(hp['lr'] - 0.01)**2 - (hp['mom'] - 0.9)**2 + 1.0
            
            pbt = PBTScheduler(
                n_workers=5,
                hyperparam_space={'lr': (0.001, 0.1), 'mom': (0.5, 0.99)},
                eval_fn=eval_fn,
                exploit_threshold=0.2
            )
            
            pbt.initialize()
            pbt.train_async(total_steps=50)
            
            elapsed = time.time() - start
            
            metrics = {
                'workers': pbt.n_workers,
                'exploits': pbt.n_exploits,
                'explores': pbt.n_explores,
                'best_perf': pbt.best_performance,
                'time_ms': elapsed * 1000
            }
            
            status = "PASS" if pbt.n_exploits > 0 else "FAIL"
            
            self.results.append(BenchmarkResult(
                name="PBT",
                status=status,
                time_seconds=elapsed,
                metrics=metrics
            ))
            
            if verbose:
                print(f"✅ PBT: {status} ({elapsed*1000:.1f}ms)")
                print(f"   Exploits: {pbt.n_exploits}, Best: {pbt.best_performance:.3f}")
        
        except Exception as e:
            elapsed = time.time() - start
            self.results.append(BenchmarkResult(
                name="PBT",
                status="FAIL",
                time_seconds=elapsed,
                metrics={},
                error=str(e)
            ))
            if verbose:
                print(f"❌ PBT: FAIL - {e}")
    
    def _benchmark_hypervolume(self, verbose: bool):
        """Benchmark Hypervolume"""
        start = time.time()
        try:
            from core.hypervolume_pure import HypervolumeCalculator
            
            points = [[0.8, 0.2], [0.6, 0.6], [0.2, 0.9]]
            reference = [1.0, 1.0]
            
            calc = HypervolumeCalculator(reference)
            hv = calc.calculate(points, maximize=[True, True])
            
            elapsed = time.time() - start
            
            metrics = {
                'hypervolume': hv,
                'points': len(points),
                'time_ms': elapsed * 1000
            }
            
            status = "PASS" if abs(hv - 0.46) < 0.01 else "FAIL"
            
            self.results.append(BenchmarkResult(
                name="Hypervolume",
                status=status,
                time_seconds=elapsed,
                metrics=metrics
            ))
            
            if verbose:
                print(f"✅ Hypervolume: {status} ({elapsed*1000:.1f}ms)")
                print(f"   HV: {hv:.4f} (expected ~0.46)")
        
        except Exception as e:
            elapsed = time.time() - start
            self.results.append(BenchmarkResult(
                name="Hypervolume",
                status="FAIL",
                time_seconds=elapsed,
                metrics={},
                error=str(e)
            ))
            if verbose:
                print(f"❌ Hypervolume: FAIL - {e}")
    
    def _benchmark_cmaes(self, verbose: bool):
        """Benchmark CMA-ES"""
        start = time.time()
        try:
            from core.cma_es_pure import CMAES
            
            def sphere(x):
                return sum(xi**2 for xi in x)
            
            cmaes = CMAES(initial_mean=[1.0, 1.0, 1.0], initial_sigma=1.0)
            result = cmaes.optimize(sphere, max_generations=20, verbose=False)
            
            elapsed = time.time() - start
            
            metrics = {
                'final_fitness': result.best_fitness,
                'generations': result.generation,
                'time_ms': elapsed * 1000
            }
            
            status = "PASS" if result.best_fitness < 0.1 else "FAIL"
            
            self.results.append(BenchmarkResult(
                name="CMA-ES",
                status=status,
                time_seconds=elapsed,
                metrics=metrics
            ))
            
            if verbose:
                print(f"✅ CMA-ES: {status} ({elapsed*1000:.1f}ms)")
                print(f"   Final fitness: {result.best_fitness:.6e}")
        
        except Exception as e:
            elapsed = time.time() - start
            self.results.append(BenchmarkResult(
                name="CMA-ES",
                status="FAIL",
                time_seconds=elapsed,
                metrics={},
                error=str(e)
            ))
            if verbose:
                print(f"❌ CMA-ES: FAIL - {e}")
    
    def _benchmark_islands(self, verbose: bool):
        """Benchmark Island Model"""
        start = time.time()
        try:
            from core.island_model_pure import IslandModel, SimpleIndividual
            
            def sphere_fitness(ind):
                return sum(x**2 for x in ind.genome)
            
            def mutate(ind):
                new_genome = [x + random.gauss(0, 0.5) for x in ind.genome]
                return SimpleIndividual(new_genome)
            
            def crossover(ind1, ind2):
                child_genome = [(g1 + g2) / 2 for g1, g2 in 
                               zip(ind1.genome, ind2.genome)]
                return SimpleIndividual(child_genome)
            
            island_model = IslandModel(
                n_islands=3,
                population_size_per_island=10,
                individual_factory=lambda: SimpleIndividual(),
                fitness_fn=sphere_fitness,
                mutation_fn=mutate,
                crossover_fn=crossover,
                migration_interval=5
            )
            
            island_model.evolve(n_generations=20, verbose=False)
            
            elapsed = time.time() - start
            
            metrics = {
                'best_fitness': island_model.global_best_fitness,
                'islands': island_model.n_islands,
                'migrations': len(island_model.migration_history),
                'time_ms': elapsed * 1000
            }
            
            status = "PASS" if island_model.global_best_fitness < 1.0 else "FAIL"
            
            self.results.append(BenchmarkResult(
                name="Island Model",
                status=status,
                time_seconds=elapsed,
                metrics=metrics
            ))
            
            if verbose:
                print(f"✅ Island Model: {status} ({elapsed*1000:.1f}ms)")
                print(f"   Best: {island_model.global_best_fitness:.6e}, "
                      f"Migrations: {len(island_model.migration_history)}")
        
        except Exception as e:
            elapsed = time.time() - start
            self.results.append(BenchmarkResult(
                name="Island Model",
                status="FAIL",
                time_seconds=elapsed,
                metrics={},
                error=str(e)
            ))
            if verbose:
                print(f"❌ Island Model: FAIL - {e}")
    
    def _benchmark_integrator(self, verbose: bool):
        """Benchmark SOTA Integrator"""
        start = time.time()
        try:
            from core.darwin_sota_integrator_COMPLETE import DarwinSOTAIntegrator, SimpleIndividual
            
            def eval_fn(ind):
                x, y = ind.genome['x'], ind.genome['y']
                return {
                    'obj0': max(0.0, 1.0 - x**2),
                    'obj1': max(0.0, 1.0 - y**2),
                    'obj2': max(0.0, 1.0 - (x**2 + y**2))
                }
            
            integrator = DarwinSOTAIntegrator(
                n_objectives=3,
                use_nsga3=True,
                use_poet=False,
                use_pbt=False,
                use_omega=False  # Omega pode não estar disponível
            )
            
            best = integrator.evolve_integrated(
                individual_factory=lambda: SimpleIndividual(),
                eval_multi_obj_fn=eval_fn,
                population_size=10,
                n_iterations=5
            )
            
            elapsed = time.time() - start
            
            metrics = {
                'best_fitness': best.fitness,
                'iterations': integrator.iteration,
                'time_ms': elapsed * 1000
            }
            
            status = "PASS" if best.fitness > 0.0 else "FAIL"
            
            self.results.append(BenchmarkResult(
                name="SOTA Integrator",
                status=status,
                time_seconds=elapsed,
                metrics=metrics
            ))
            
            if verbose:
                print(f"✅ SOTA Integrator: {status} ({elapsed*1000:.1f}ms)")
                print(f"   Best fitness: {best.fitness:.4f}")
        
        except Exception as e:
            elapsed = time.time() - start
            self.results.append(BenchmarkResult(
                name="SOTA Integrator",
                status="FAIL",
                time_seconds=elapsed,
                metrics={},
                error=str(e)
            ))
            if verbose:
                print(f"❌ SOTA Integrator: FAIL - {e}")
    
    def _benchmark_omega(self, verbose: bool):
        """Benchmark Omega Extensions"""
        start = time.time()
        try:
            from omega_ext.tests import quick_test
            
            # Run test
            quick_test.test_quick()
            
            elapsed = time.time() - start
            
            metrics = {'time_ms': elapsed * 1000}
            
            self.results.append(BenchmarkResult(
                name="Omega Extensions",
                status="PASS",
                time_seconds=elapsed,
                metrics=metrics
            ))
            
            if verbose:
                print(f"✅ Omega Extensions: PASS ({elapsed*1000:.1f}ms)")
        
        except Exception as e:
            elapsed = time.time() - start
            self.results.append(BenchmarkResult(
                name="Omega Extensions",
                status="FAIL",
                time_seconds=elapsed,
                metrics={},
                error=str(e)
            ))
            if verbose:
                print(f"❌ Omega Extensions: FAIL - {e}")
    
    def _print_summary(self, verbose: bool):
        """Imprime sumário final"""
        if not verbose:
            return
        
        print("\n" + "="*80)
        print("📊 SUMÁRIO FINAL - BENCHMARK SUITE")
        print("="*80 + "\n")
        
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        total = len(self.results)
        
        print(f"  Total: {total}")
        print(f"  ✅ Passou: {passed} ({passed/total*100:.0f}%)")
        print(f"  ❌ Falhou: {failed} ({failed/total*100:.0f}%)")
        
        total_time = sum(r.time_seconds for r in self.results)
        print(f"\n  ⏱️ Tempo total: {total_time:.2f}s")
        
        print(f"\n  📋 Detalhes:")
        for r in self.results:
            symbol = "✅" if r.status == "PASS" else "❌"
            print(f"     {symbol} {r.name:20s} ({r.time_seconds*1000:6.1f}ms)")
        
        print("\n" + "="*80)
        
        if passed == total:
            print("🎉 TODOS OS BENCHMARKS PASSARAM!")
        else:
            print(f"⚠️ {failed} benchmark(s) falharam")
        
        print("="*80 + "\n")


if __name__ == "__main__":
    random.seed(42)
    
    suite = BenchmarkSuite()
    suite.run_all(verbose=True)
