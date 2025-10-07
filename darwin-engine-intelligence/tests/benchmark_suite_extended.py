"""
Extended Benchmark Suite - Validação SOTA Completa
===================================================

Suite ESTENDIDA com TODOS os componentes SOTA implementados (11 total).

Componentes testados:
1. NSGA-III (Pareto multi-objetivo)
2. POET-Lite (Open-ended evolution)
3. PBT Scheduler (Population-based training)
4. Hypervolume Calculator
5. CMA-ES (Covariance Matrix Adaptation)
6. Island Model (Distributed evolution)
7. SOTA Integrator (Master orchestrator)
8. Omega Extensions (Full suite)
9. CVT-MAP-Elites (Quality-Diversity)
10. Multi-Emitter QD (CMA-MEGA framework)
11. Observability Tracker
"""

import sys
sys.path.insert(0, '/workspace')

import time
import random
import math
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    """Resultado de um benchmark."""
    name: str
    status: str  # PASS, FAIL, SKIP
    time_seconds: float
    metrics: Dict[str, Any]
    error: str = ""


class ExtendedBenchmarkSuite:
    """Suite estendida de benchmarks com todos os componentes SOTA."""
    
    def __init__(self):
        self.results: List[BenchmarkResult] = []
    
    def run_all(self, verbose: bool = True):
        """Executar todos os 11 benchmarks."""
        self._print_header(verbose)
        
        # Core 8 components
        self._benchmark_nsga3(verbose)
        self._benchmark_poet(verbose)
        self._benchmark_pbt(verbose)
        self._benchmark_hypervolume(verbose)
        self._benchmark_cmaes(verbose)
        self._benchmark_islands(verbose)
        self._benchmark_integrator(verbose)
        self._benchmark_omega(verbose)
        
        # New 3 components
        self._benchmark_cvt_map_elites(verbose)
        self._benchmark_multi_emitter(verbose)
        self._benchmark_observability(verbose)
        
        self._print_summary(verbose)
    
    def _print_header(self, verbose: bool):
        """Print header."""
        if verbose:
            print("\n" + "="*80)
            print("🧪 EXTENDED BENCHMARK SUITE - DARWIN SOTA (11 componentes)")
            print("="*80 + "\n")
    
    def _benchmark_nsga3(self, verbose: bool):
        """Benchmark NSGA-III."""
        name = "NSGA-III"
        start_time = time.time()
        
        try:
            from core.nsga3_pure_python import NSGA3
            
            nsga3 = NSGA3(n_objectives=3, n_partitions=4)
            population = list(range(20))
            objectives = [
                {'f1': random.random(), 'f2': random.random(), 'f3': random.random()}
                for _ in range(20)
            ]
            maximize = {'f1': True, 'f2': True, 'f3': True}
            survivors = nsga3.select(population, objectives, maximize, n_survivors=10)
            
            elapsed = (time.time() - start_time) * 1000
            
            if len(survivors) == 10:
                self.results.append(BenchmarkResult(
                    name=name,
                    status="PASS",
                    time_seconds=elapsed / 1000,
                    metrics={'ref_points': len(nsga3.ref_points), 'survivors': len(survivors)}
                ))
                if verbose:
                    print(f"✅ {name:20s}: PASS ({elapsed:6.1f}ms)")
            else:
                raise ValueError(f"Expected 10 survivors, got {len(survivors)}")
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.results.append(BenchmarkResult(
                name=name,
                status="FAIL",
                time_seconds=elapsed / 1000,
                metrics={},
                error=str(e)
            ))
            if verbose:
                print(f"❌ {name:20s}: FAIL ({elapsed:6.1f}ms) - {e}")
    
    def _benchmark_poet(self, verbose: bool):
        """Benchmark POET-Lite."""
        name = "POET-Lite"
        start_time = time.time()
        
        try:
            from core.poet_lite_pure import POETLite
            
            # FIX: All functions need to accept rng parameter
            def env_gen(rng=None): 
                r = rng if rng else random.Random()
                return {'difficulty': r.uniform(0.1, 1.0)}
            def agent_factory(rng=None):
                r = rng if rng else random.Random()
                return {'skill': r.uniform(0, 1)}
            def eval_fn(agent, env, rng=None):
                r = rng if rng else random.Random()
                return max(0, agent['skill'] - env['difficulty'] + r.gauss(0, 0.1))
            def mutate_env(env, rng=None):
                r = rng if rng else random.Random()
                e = env.copy()
                e['difficulty'] += r.gauss(0, 0.1)
                return e
            def mutate_agent(agent, rng=None):
                r = rng if rng else random.Random()
                a = agent.copy()
                a['skill'] += r.gauss(0, 0.05)
                return a
            
            poet = POETLite(env_gen, agent_factory, eval_fn, mutate_env, mutate_agent, mc_threshold=0.1)
            poet.initialize(n_initial_pairs=3)
            poet.evolve(n_iterations=3, verbose=False)
            
            elapsed = (time.time() - start_time) * 1000
            
            if len(poet.pairs) > 2:
                self.results.append(BenchmarkResult(
                    name=name,
                    status="PASS",
                    time_seconds=elapsed / 1000,
                    metrics={'pairs': len(poet.pairs)}
                ))
                if verbose:
                    print(f"✅ {name:20s}: PASS ({elapsed:6.1f}ms)")
            else:
                raise ValueError(f"Too few pairs: {len(poet.pairs)}")
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.results.append(BenchmarkResult(name=name, status="FAIL", time_seconds=elapsed/1000, metrics={}, error=str(e)))
            if verbose:
                print(f"❌ {name:20s}: FAIL ({elapsed:6.1f}ms) - {e}")
    
    def _benchmark_pbt(self, verbose: bool):
        """Benchmark PBT."""
        name = "PBT"
        start_time = time.time()
        
        try:
            from core.pbt_scheduler_pure import PBTScheduler
            
            def eval_fn(hparams): return sum(hparams.values()) / len(hparams)
            
            pbt = PBTScheduler(
                n_workers=5,
                hyperparam_space={'lr': (0.001, 0.1), 'momentum': (0.1, 0.9)},
                eval_fn=eval_fn,
                exploit_threshold=0.2,
                explore_prob=0.8
            )
            pbt.initialize()
            pbt.train_async(total_steps=20, ready_fn=lambda i: True)
            
            elapsed = (time.time() - start_time) * 1000
            
            best = max(pbt.workers, key=lambda w: w.performance)
            if best.performance > 0:
                self.results.append(BenchmarkResult(name=name, status="PASS", time_seconds=elapsed/1000, metrics={'best_perf': best.performance}))
                if verbose:
                    print(f"✅ {name:20s}: PASS ({elapsed:6.1f}ms)")
            else:
                raise ValueError("PBT performance is 0")
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.results.append(BenchmarkResult(name=name, status="FAIL", time_seconds=elapsed/1000, metrics={}, error=str(e)))
            if verbose:
                print(f"❌ {name:20s}: FAIL ({elapsed:6.1f}ms) - {e}")
    
    def _benchmark_hypervolume(self, verbose: bool):
        """Benchmark Hypervolume."""
        name = "Hypervolume"
        start_time = time.time()
        
        try:
            from core.hypervolume_pure import HypervolumeCalculator
            
            hv_calc = HypervolumeCalculator(reference_point=[2.0, 2.0])
            points = [[1.0, 1.5], [1.5, 1.0], [0.5, 0.8]]
            hv = hv_calc.calculate(points, maximize=[True, True])
            
            elapsed = (time.time() - start_time) * 1000
            
            if hv > 0:
                self.results.append(BenchmarkResult(name=name, status="PASS", time_seconds=elapsed/1000, metrics={'hypervolume': hv}))
                if verbose:
                    print(f"✅ {name:20s}: PASS ({elapsed:6.1f}ms)")
            else:
                raise ValueError(f"HV must be positive, got {hv}")
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.results.append(BenchmarkResult(name=name, status="FAIL", time_seconds=elapsed/1000, metrics={}, error=str(e)))
            if verbose:
                print(f"❌ {name:20s}: FAIL ({elapsed:6.1f}ms) - {e}")
    
    def _benchmark_cmaes(self, verbose: bool):
        """Benchmark CMA-ES."""
        name = "CMA-ES"
        start_time = time.time()
        
        try:
            from core.cma_es_pure import CMAES
            
            def sphere(x): return sum(xi**2 for xi in x)
            
            cmaes = CMAES(initial_mean=[2.0, 2.0], initial_sigma=1.0, population_size=10)
            result = cmaes.optimize(sphere, max_generations=10, verbose=False)
            
            elapsed = (time.time() - start_time) * 1000
            
            if result.best_fitness < 0.1:
                self.results.append(BenchmarkResult(name=name, status="PASS", time_seconds=elapsed/1000, metrics={'best_fitness': result.best_fitness}))
                if verbose:
                    print(f"✅ {name:20s}: PASS ({elapsed:6.1f}ms)")
            else:
                raise ValueError(f"CMA-ES didn't converge: {result.best_fitness}")
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.results.append(BenchmarkResult(name=name, status="FAIL", time_seconds=elapsed/1000, metrics={}, error=str(e)))
            if verbose:
                print(f"❌ {name:20s}: FAIL ({elapsed:6.1f}ms) - {e}")
    
    def _benchmark_islands(self, verbose: bool):
        """Benchmark Island Model."""
        name = "Island Model"
        start_time = time.time()
        
        try:
            from core.island_model_pure import IslandModel, MigrationTopology
            
            class SimpleInd:
                def __init__(self, val=None):
                    self.value = val if val is not None else random.uniform(-5, 5)
                    self.fitness = 0.0
            
            def fitness_fn(ind): return -(ind.value ** 2)
            def mutate_fn(ind): i = SimpleInd(ind.value + random.gauss(0, 0.5)); i.fitness = fitness_fn(i); return i
            def crossover_fn(i1, i2): i = SimpleInd((i1.value + i2.value) / 2); i.fitness = fitness_fn(i); return i
            
            island_model = IslandModel(
                n_islands=3,
                population_size_per_island=10,
                individual_factory=SimpleInd,
                fitness_fn=fitness_fn,
                mutation_fn=mutate_fn,
                crossover_fn=crossover_fn,
                topology=MigrationTopology.RING,
                migration_rate=0.1,
                migration_interval=5
            )
            island_model.evolve(n_generations=10, verbose=False)
            
            elapsed = (time.time() - start_time) * 1000
            best = min(island.best_fitness for island in island_model.islands)
            
            if best < -0.1:
                self.results.append(BenchmarkResult(name=name, status="PASS", time_seconds=elapsed/1000, metrics={'best_fitness': best}))
                if verbose:
                    print(f"✅ {name:20s}: PASS ({elapsed:6.1f}ms)")
            else:
                raise ValueError(f"Island model didn't improve: {best}")
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.results.append(BenchmarkResult(name=name, status="FAIL", time_seconds=elapsed/1000, metrics={}, error=str(e)))
            if verbose:
                print(f"❌ {name:20s}: FAIL ({elapsed:6.1f}ms) - {e}")
    
    def _benchmark_integrator(self, verbose: bool):
        """Benchmark SOTA Integrator."""
        name = "SOTA Integrator"
        start_time = time.time()
        
        try:
            from core.darwin_sota_integrator_COMPLETE import DarwinSOTAIntegrator, SimpleIndividual
            
            def factory(): return SimpleIndividual(genome={'x': random.uniform(-2, 2)})
            # FIX: Return dict for objectives instead of list
            def eval_fn(ind):
                x = ind.genome['x']
                f1 = x**2
                f2 = (x - 2)**2
                return {'f1': f1, 'f2': f2}, [x]  # Return dict, not list
            
            integrator = DarwinSOTAIntegrator(n_objectives=2, use_nsga3=True, use_poet=False, use_pbt=False, use_omega=False)
            integrator.evolve_integrated(factory, eval_fn, population_size=10, n_iterations=5)
            
            elapsed = (time.time() - start_time) * 1000
            
            if integrator.best_individual and integrator.best_individual.fitness > 0:
                self.results.append(BenchmarkResult(name=name, status="PASS", time_seconds=elapsed/1000, metrics={'best_fitness': integrator.best_individual.fitness}))
                if verbose:
                    print(f"✅ {name:20s}: PASS ({elapsed:6.1f}ms)")
            else:
                raise ValueError("Integrator failed to find solution")
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.results.append(BenchmarkResult(name=name, status="FAIL", time_seconds=elapsed/1000, metrics={}, error=str(e)))
            if verbose:
                print(f"❌ {name:20s}: FAIL ({elapsed:6.1f}ms) - {e}")
    
    def _benchmark_omega(self, verbose: bool):
        """Benchmark Omega Extensions."""
        name = "Omega Extensions"
        start_time = time.time()
        
        try:
            from omega_ext.core.bridge import DarwinOmegaBridge
            from omega_ext.plugins.adapter_darwin import autodetect
            
            init_fn, eval_fn = autodetect()
            engine = DarwinOmegaBridge(init_fn, eval_fn, seed=123, max_cycles=3)
            champ = engine.run(max_cycles=3)
            
            elapsed = (time.time() - start_time) * 1000
            
            if champ and champ.score > 0.0:
                self.results.append(BenchmarkResult(name=name, status="PASS", time_seconds=elapsed/1000, metrics={'champion_score': float(champ.score)}))
                if verbose:
                    print(f"✅ {name:20s}: PASS ({elapsed:6.1f}ms)")
            else:
                raise ValueError("Omega champion invalid")
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.results.append(BenchmarkResult(name=name, status="FAIL", time_seconds=elapsed/1000, metrics={}, error=str(e)))
            if verbose:
                print(f"❌ {name:20s}: FAIL ({elapsed:6.1f}ms) - {e}")
    
    def _benchmark_cvt_map_elites(self, verbose: bool):
        """Benchmark CVT-MAP-Elites."""
        name = "CVT-MAP-Elites"
        start_time = time.time()
        
        try:
            from core.cvt_map_elites_pure import CVTMAPElites
            
            def init(rng): return {'x': rng.uniform(-2, 2), 'y': rng.uniform(-2, 2)}
            def eval_fn(genome):
                x, y = genome['x'], genome['y']
                f = 10 + x*math.sin(4*math.pi*x) + y*math.cos(4*math.pi*y)
                return f, [x, y]
            def mutate(g):
                g = g.copy()
                if random.random() < 0.5: g['x'] += random.gauss(0, 0.3); g['x'] = max(-2, min(2, g['x']))
                if random.random() < 0.5: g['y'] += random.gauss(0, 0.3); g['y'] = max(-2, min(2, g['y']))
                return g
            
            cvt = CVTMAPElites(30, 2, [(-2, 2), (-2, 2)], init, eval_fn, mutate, seed=42)
            cvt.initialize(50)
            cvt.evolve(10, 5, verbose=False)
            metrics = cvt.get_metrics()
            
            elapsed = (time.time() - start_time) * 1000
            
            if metrics['coverage'] > 0.3:
                self.results.append(BenchmarkResult(name=name, status="PASS", time_seconds=elapsed/1000, metrics=metrics))
                if verbose:
                    print(f"✅ {name:20s}: PASS ({elapsed:6.1f}ms)")
            else:
                raise ValueError(f"CVT coverage too low: {metrics['coverage']}")
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.results.append(BenchmarkResult(name=name, status="FAIL", time_seconds=elapsed/1000, metrics={}, error=str(e)))
            if verbose:
                print(f"❌ {name:20s}: FAIL ({elapsed:6.1f}ms) - {e}")
    
    def _benchmark_multi_emitter(self, verbose: bool):
        """Benchmark Multi-Emitter QD."""
        name = "Multi-Emitter QD"
        start_time = time.time()
        
        try:
            from core.multi_emitter_qd import MultiEmitterQD
            
            def init(rng): return {'x': rng.uniform(-2, 2), 'y': rng.uniform(-2, 2)}
            def eval_fn(g):
                x, y = g['x'], g['y']
                f = 12 + x*math.sin(3*math.pi*x) + y*math.cos(3*math.pi*y)
                return f, [x, y]
            
            me = MultiEmitterQD(25, 2, [(-2, 2), (-2, 2)], init, eval_fn, seed=42)
            me.evolve(10, 3, verbose=False)
            metrics = me.get_metrics()
            
            elapsed = (time.time() - start_time) * 1000
            
            if metrics['coverage'] > 0.3:
                self.results.append(BenchmarkResult(name=name, status="PASS", time_seconds=elapsed/1000, metrics=metrics))
                if verbose:
                    print(f"✅ {name:20s}: PASS ({elapsed:6.1f}ms)")
            else:
                raise ValueError(f"Multi-emitter coverage too low: {metrics['coverage']}")
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.results.append(BenchmarkResult(name=name, status="FAIL", time_seconds=elapsed/1000, metrics={}, error=str(e)))
            if verbose:
                print(f"❌ {name:20s}: FAIL ({elapsed:6.1f}ms) - {e}")
    
    def _benchmark_observability(self, verbose: bool):
        """Benchmark Observability Tracker."""
        name = "Observability"
        start_time = time.time()
        
        try:
            from core.observability_tracker import ObservabilityTracker
            
            tracker = ObservabilityTracker(window_size=10)
            tracker.set_custom_metric('n_niches', 30)
            tracker.register_component('emitter_1', 'improvement')
            
            for i in range(15):
                archive = {j: type('Ind', (), {'fitness': 10 + i*0.5 + j*0.1})() for j in range(min(30, 5 + i*2))}
                tracker.record_snapshot(i, archive=archive, evaluations=50 + i*5)
                tracker.update_component('emitter_1', emissions=3, improvements=1)
            
            summary = tracker.get_summary()
            elapsed = (time.time() - start_time) * 1000
            
            if len(tracker.snapshots) == 15:
                self.results.append(BenchmarkResult(name=name, status="PASS", time_seconds=elapsed/1000, metrics={'snapshots': len(tracker.snapshots)}))
                if verbose:
                    print(f"✅ {name:20s}: PASS ({elapsed:6.1f}ms)")
            else:
                raise ValueError(f"Expected 15 snapshots, got {len(tracker.snapshots)}")
        
        except Exception as e:
            elapsed = (time.time() - start_time) * 1000
            self.results.append(BenchmarkResult(name=name, status="FAIL", time_seconds=elapsed/1000, metrics={}, error=str(e)))
            if verbose:
                print(f"❌ {name:20s}: FAIL ({elapsed:6.1f}ms) - {e}")
    
    def _print_summary(self, verbose: bool):
        """Print summary."""
        if not verbose:
            return
        
        print("\n" + "="*80)
        print("📊 SUMÁRIO FINAL - EXTENDED BENCHMARK SUITE")
        print("="*80 + "\n")
        
        passed = sum(1 for r in self.results if r.status == "PASS")
        failed = sum(1 for r in self.results if r.status == "FAIL")
        total = len(self.results)
        
        print(f"  Total: {total}")
        print(f"  ✅ Passou: {passed} ({100*passed//total if total else 0}%)")
        print(f"  ❌ Falhou: {failed} ({100*failed//total if total else 0}%)")
        print(f"\n  ⏱️ Tempo total: {sum(r.time_seconds for r in self.results):.2f}s")
        print("\n  📋 Detalhes:")
        for r in self.results:
            status_icon = "✅" if r.status == "PASS" else "❌"
            print(f"     {status_icon} {r.name:20s} ({r.time_seconds*1000:7.1f}ms)")
        
        print("\n" + "="*80)
        if passed == total:
            print("🎉 TODOS OS BENCHMARKS PASSARAM!")
        else:
            print(f"⚠️ {failed} BENCHMARK(S) FALHARAM")
        print("="*80 + "\n")


if __name__ == "__main__":
    random.seed(42)
    suite = ExtendedBenchmarkSuite()
    suite.run_all(verbose=True)
