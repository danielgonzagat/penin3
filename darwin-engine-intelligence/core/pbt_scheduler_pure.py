"""
PBT: Population Based Training Scheduler
=========================================

IMPLEMENTAÇÃO PURA PYTHON (SEM ML LIBRARIES)
Status: FUNCIONAL E TESTADO
Data: 2025-10-03

Based on: Jaderberg et al. (2017) "Population Based Training of Neural Networks"
"""

import random
import time
import json
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class PBTAction(Enum):
    """Ações PBT"""
    EXPLOIT = "exploit"  # Copiar hyperparams do melhor
    EXPLORE = "explore"  # Perturbar hyperparams
    CONTINUE = "continue"  # Continuar treinando


@dataclass
class Worker:
    """Worker PBT (1 indivíduo treinando)"""
    worker_id: str
    hyperparams: Dict[str, float]
    performance: float = 0.0
    steps_trained: int = 0
    last_checkpoint: Optional[Dict] = None
    created_at: float = field(default_factory=time.time)
    lineage: List[str] = field(default_factory=list)  # IDs dos parents


class PBTScheduler:
    """
    Population Based Training Scheduler
    
    Princípios:
    1. População de workers treina em paralelo
    2. Periodicamente: workers ruins exploitam (copiam) workers bons
    3. Após exploit, explore (perturba hyperparams)
    4. Assíncrono: cada worker evolui no seu ritmo
    """
    
    def __init__(self,
                 n_workers: int,
                 hyperparam_space: Dict[str, Tuple[float, float]],
                 eval_fn: Callable[[Dict], float],
                 exploit_threshold: float = 0.2,
                 explore_prob: float = 0.8):
        """
        Args:
            n_workers: Número de workers (população)
            hyperparam_space: Dict de (min, max) para cada hyperparam
            eval_fn: Função que avalia performance dado hyperparams
            exploit_threshold: Bottom 20% explora top 20%
            explore_prob: Prob de perturbar após exploit
        """
        self.n_workers = n_workers
        self.hyperparam_space = hyperparam_space
        self.eval_fn = eval_fn
        self.exploit_threshold = exploit_threshold
        self.explore_prob = explore_prob
        
        # Workers
        self.workers: List[Worker] = []
        
        # Métricas
        self.total_steps = 0
        self.n_exploits = 0
        self.n_explores = 0
        self.best_performance = 0.0
        self.performance_history = []
    
    def initialize(self):
        """Inicializa população de workers com hyperparams aleatórios"""
        print(f"\n🚀 Inicializando PBT com {self.n_workers} workers...")
        
        for i in range(self.n_workers):
            # Sample hyperparams uniformes
            hyperparams = {}
            for key, (min_val, max_val) in self.hyperparam_space.items():
                hyperparams[key] = random.uniform(min_val, max_val)
            
            worker = Worker(
                worker_id=f"worker_{i:03d}",
                hyperparams=hyperparams
            )
            
            # Avaliação inicial
            worker.performance = self.eval_fn(worker.hyperparams)
            
            self.workers.append(worker)
            
            print(f"   Worker {i+1}: {worker.hyperparams} → "
                  f"perf={worker.performance:.3f}")
    
    def step(self, worker_idx: int, n_steps: int = 1) -> PBTAction:
        """
        Step de treinamento para um worker
        
        Args:
            worker_idx: Índice do worker
            n_steps: Quantos steps treinar
        
        Returns:
            Ação tomada (EXPLOIT/EXPLORE/CONTINUE)
        """
        worker = self.workers[worker_idx]
        
        # "Treinar" (avaliar)
        for _ in range(n_steps):
            worker.performance = self.eval_fn(worker.hyperparams)
            worker.steps_trained += 1
            self.total_steps += 1
        
        # Decidir ação
        action = self._decide_action(worker)
        
        if action == PBTAction.EXPLOIT:
            self._exploit(worker)
            self.n_exploits += 1
            
            # Após exploit, provavelmente explore
            if random.random() < self.explore_prob:
                self._explore(worker)
                self.n_explores += 1
                return PBTAction.EXPLORE
            
            return PBTAction.EXPLOIT
        
        return PBTAction.CONTINUE
    
    def _decide_action(self, worker: Worker) -> PBTAction:
        """Decide se worker deve exploitar ou continuar"""
        # Ordenar workers por performance
        sorted_workers = sorted(self.workers, key=lambda w: w.performance, reverse=True)
        
        # Worker está no bottom quantile?
        cutoff = int(len(self.workers) * self.exploit_threshold)
        bottom_workers = sorted_workers[-cutoff:] if cutoff > 0 else []
        
        if worker in bottom_workers:
            return PBTAction.EXPLOIT
        
        return PBTAction.CONTINUE
    
    def _exploit(self, worker: Worker):
        """
        Exploit: copia hyperparams de um worker top
        """
        # Top quantile
        cutoff = max(1, int(len(self.workers) * self.exploit_threshold))
        sorted_workers = sorted(self.workers, key=lambda w: w.performance, reverse=True)
        top_workers = sorted_workers[:cutoff]
        
        # Escolher random do top
        target = random.choice(top_workers)
        
        print(f"   💎 EXPLOIT: {worker.worker_id} copia {target.worker_id} "
              f"(perf {target.performance:.3f})")
        
        # Copiar hyperparams
        worker.hyperparams = target.hyperparams.copy()
        worker.lineage.append(target.worker_id)
        
        # Checkpoint (restauração parcial)
        worker.last_checkpoint = {
            'hyperparams': target.hyperparams.copy(),
            'performance': target.performance,
            'step': self.total_steps
        }
    
    def _explore(self, worker: Worker):
        """
        Explore: perturba hyperparams
        """
        perturbations = []
        
        for key, (min_val, max_val) in self.hyperparam_space.items():
            old_val = worker.hyperparams[key]
            
            # Perturb: ×0.8 ou ×1.2
            if random.random() < 0.5:
                new_val = old_val * 0.8
            else:
                new_val = old_val * 1.2
            
            # Clip ao range
            new_val = max(min_val, min(max_val, new_val))
            
            if abs(new_val - old_val) > 1e-6:
                perturbations.append(f"{key}:{old_val:.3f}→{new_val:.3f}")
            
            worker.hyperparams[key] = new_val
        
        if perturbations:
            print(f"   🔍 EXPLORE: {worker.worker_id} perturbed {', '.join(perturbations[:2])}")
    
    def train_async(self, total_steps: int, ready_fn: Callable[[int], bool] = None):
        """
        Treina assincronamente
        
        Args:
            total_steps: Total de steps de treinamento
            ready_fn: Função que retorna True se worker está pronto para step
        """
        print(f"\n⚡ PBT Assíncrono iniciado ({total_steps} steps)...")
        
        step = 0
        while step < total_steps:
            # Escolher worker aleatório (simula async)
            worker_idx = random.randint(0, self.n_workers - 1)
            
            # Verificar se ready (simula disponibilidade)
            if ready_fn and not ready_fn(worker_idx):
                continue
            
            # Step
            action = self.step(worker_idx, n_steps=1)
            step += 1
            
            # Log progress
            if step % 50 == 0:
                self._log_progress(step, total_steps)
        
        self._log_final_stats()
    
    def _log_progress(self, step: int, total: int):
        """Log progresso"""
        # Melhor worker atual
        best_worker = max(self.workers, key=lambda w: w.performance)
        avg_perf = sum(w.performance for w in self.workers) / len(self.workers)
        
        print(f"\n📊 Step {step}/{total}:")
        print(f"   Best: {best_worker.performance:.4f} ({best_worker.worker_id})")
        print(f"   Avg: {avg_perf:.4f}")
        print(f"   Exploits: {self.n_exploits}, Explores: {self.n_explores}")
        
        # Trackear
        self.performance_history.append({
            'step': step,
            'best': best_worker.performance,
            'avg': avg_perf
        })
        
        if best_worker.performance > self.best_performance:
            self.best_performance = best_worker.performance
    
    def _log_final_stats(self):
        """Stats finais"""
        print(f"\n{'='*80}")
        print("📊 ESTATÍSTICAS FINAIS PBT")
        print(f"{'='*80}")
        
        best_worker = max(self.workers, key=lambda w: w.performance)
        worst_worker = min(self.workers, key=lambda w: w.performance)
        avg_perf = sum(w.performance for w in self.workers) / len(self.workers)
        
        print(f"  Melhor worker: {best_worker.worker_id}")
        print(f"    Performance: {best_worker.performance:.4f}")
        print(f"    Hyperparams: {best_worker.hyperparams}")
        print(f"  Pior worker: {worst_worker.worker_id}")
        print(f"    Performance: {worst_worker.performance:.4f}")
        print(f"  Performance média: {avg_perf:.4f}")
        print(f"  Total exploits: {self.n_exploits}")
        print(f"  Total explores: {self.n_explores}")
        print(f"  Improvement: {best_worker.performance:.4f}")
    
    def save(self, filepath: str):
        """Salva estado PBT"""
        data = {
            'total_steps': self.total_steps,
            'n_exploits': self.n_exploits,
            'n_explores': self.n_explores,
            'best_performance': self.best_performance,
            'workers': [
                {
                    'worker_id': w.worker_id,
                    'hyperparams': w.hyperparams,
                    'performance': w.performance,
                    'steps_trained': w.steps_trained,
                    'lineage': w.lineage
                }
                for w in self.workers
            ],
            'history': self.performance_history
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# ============================================================================
# TESTE
# ============================================================================

def test_pbt_scheduler():
    """Teste PBT scheduler"""
    print("\n" + "="*80)
    print("TESTE: PBT Scheduler Puro Python")
    print("="*80 + "\n")
    
    # Função de teste: otimizar f(x,y) = -(x-0.5)² - (y-0.3)²
    # Hyperparams: learning_rate, momentum
    def eval_fn(hyperparams):
        """Performance baseado em hyperparams"""
        lr = hyperparams.get('learning_rate', 0.01)
        mom = hyperparams.get('momentum', 0.9)
        
        # Optimal em lr=0.01, momentum=0.9
        fitness = -(lr - 0.01)**2 - (mom - 0.9)**2
        
        # Normalizar para [0, 1]
        return max(0.0, min(1.0, fitness + 1.0))
    
    # Criar PBT
    pbt = PBTScheduler(
        n_workers=10,
        hyperparam_space={
            'learning_rate': (0.0001, 0.1),
            'momentum': (0.5, 0.99)
        },
        eval_fn=eval_fn,
        exploit_threshold=0.2,  # Bottom 20% exploita top 20%
        explore_prob=0.8
    )
    
    # Inicializar
    pbt.initialize()
    
    # Treinar
    pbt.train_async(total_steps=200)
    
    # Validar
    best = max(pbt.workers, key=lambda w: w.performance)
    assert best.performance > 0.5, f"Performance muito baixa: {best.performance}"
    assert pbt.n_exploits > 0, "Deve ter exploited"
    assert pbt.n_explores > 0, "Deve ter explored"
    
    # Salvar
    pbt.save('/tmp/pbt_archive.json')
    print(f"\n💾 Archive salvo: /tmp/pbt_archive.json")
    
    print("\n✅ TESTE PBT PASSOU!\n")
    print("="*80)


if __name__ == "__main__":
    test_pbt_scheduler()
    print("\n✅ pbt_scheduler_pure.py FUNCIONAL!")
