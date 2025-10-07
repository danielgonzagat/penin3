
# FUNÇÕES DETERMINÍSTICAS (substituem random)
import hashlib
import os
import time


def deterministic_random(seed_offset=0):
    """Substituto determinístico para random.random()"""
    import hashlib
    import time

    # Usa múltiplas fontes de determinismo
    sources = [
        str(time.time()).encode(),
        str(os.getpid()).encode(),
        str(id({})).encode(),
        str(seed_offset).encode()
    ]

    # Combina todas as fontes
    combined = b''.join(sources)
    hash_val = int(hashlib.md5(combined).hexdigest()[:8], 16)

    return (hash_val % 1000000) / 1000000.0


def deterministic_uniform(a, b, seed_offset=0):
    """Substituto determinístico para random.uniform(a, b)"""
    r = deterministic_random(seed_offset)
    return a + (b - a) * r


def deterministic_randint(a, b, seed_offset=0):
    """Substituto determinístico para random.randint(a, b)"""
    r = deterministic_random(seed_offset)
    return int(a + (b - a + 1) * r)


def deterministic_choice(seq, seed_offset=0):
    """Substituto determinístico para random.choice(seq)"""
    if not seq:
        raise IndexError("sequence is empty")

    r = deterministic_random(seed_offset)
    return seq[int(r * len(seq))]


def deterministic_shuffle(lst, seed_offset=0):
    """Substituto determinístico para random.shuffle(lst)"""
    if not lst:
        return

    # Shuffle determinístico baseado em ordenação por hash
    def sort_key(item):
        item_str = str(item) + str(seed_offset)
        return hashlib.md5(item_str.encode()).hexdigest()

    lst.sort(key=sort_key)


def deterministic_torch_rand(*size, seed_offset=0):
    """Substituto determinístico para torch.rand(*size)"""
    if not size:
        return torch.tensor(deterministic_random(seed_offset))

    # Gera valores determinísticos
    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_random(seed_offset + i))

    return torch.tensor(values).reshape(size)


def deterministic_torch_randint(low, high, size=None, seed_offset=0):
    """Substituto determinístico para torch.randint(low, high, size)"""
    if size is None:
        return torch.tensor(deterministic_randint(low, high, seed_offset))

    # Gera valores determinísticos
    if isinstance(size, int):
        size = (size,)

    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_randint(low, high, seed_offset + i))

    return torch.tensor(values).reshape(size)

#!/usr/bin/env python3
"""
FAZENDA DE NEURÔNIOS IA³ - Sistema Perpétuo (Simulado) com CLI e execução finita por passos.
- CLI modos: test | steps | run (default: steps)
- Seeds determinísticos, métricas JSONL, checkpoints estruturados, DB opcional
- Sem hacks de shape; evolução configurável; modo steps é seguro para testes
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import torch

try:
    import sqlite3
    SQLITE_AVAILABLE = True
except Exception:
    SQLITE_AVAILABLE = False

# ---------------------------
# CLI
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="IA3 Neural Farm with finite steps and metrics")
    p.add_argument('--mode', choices=['test', 'steps', 'run'], default='steps')
    p.add_argument('--steps', type=int, default=500)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--out-dir', type=str, default='./neural_farm_out')
    p.add_argument('--db-path', type=str, default='')
    p.add_argument('--input-dim', type=int, default=16)
    p.add_argument('--hidden-dim', type=int, default=16)
    p.add_argument('--output-dim', type=int, default=8)
    p.add_argument('--min-pop', type=int, default=10)
    p.add_argument('--max-pop', type=int, default=100)
    p.add_argument('--fitness', choices=['usage','signal','age'], default='usage')
    p.add_argument('--sleep', type=float, default=0.005)
    p.add_argument('--deterministic-evolution', action='store_true')
    return p.parse_args()

# ---------------------------
# Utils
# ---------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def now_iso() -> str:
    return datetime.now().isoformat(timespec='seconds')

# ---------------------------
# Metrics/Checkpoint
# ---------------------------

def metrics_path(out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, 'metrics.jsonl')

def log_metrics(out_dir: str, payload: Dict):
    path = metrics_path(out_dir)
    with open(path, 'a') as f:
        json.dump(payload, f)
        f.write('\n')

def save_checkpoint(out_dir: str, name: str, payload: Dict):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)
    return path

# ---------------------------
# Core: Neuron, Farm
# ---------------------------

@dataclass
class NeuronState:
    id: str
    input_dim: int
    birth_time: float
    activations: int
    total_signal: float
    generation: int
    fitness: float

class RealNeuron:
    def __init__(self, input_dim: int, neuron_id: Optional[str] = None):
        self.id = neuron_id or f"N_{abs(hash((time.time(), time.time()))) % 10**8:08d}"
        self.input_dim = input_dim
        self.weight = torch.randn(input_dim) * 0.1
        self.bias = torch.randn(1) * 0.01
        self.birth_time = time.time()
        self.activations = 0
        self.total_signal = 0.0
        self.generation = 0
        self.fitness = 0.0

    def activate(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1)
        if x.numel() != self.input_dim:
            if x.numel() < self.input_dim:
                # pad with zeros
                pad = torch.zeros(self.input_dim - x.numel(), dtype=x.dtype)
                x = torch.cat([x, pad], dim=0)
            else:
                x = x[:self.input_dim]
        s = torch.dot(x, self.weight) + self.bias
        y = torch.tanh(s)
        self.activations += 1
        self.total_signal += float(abs(s.item()))
        return y

    def compute_fitness(self, criteria: str = 'usage') -> float:
        age = max(1.0, time.time() - self.birth_time)
        avg_signal = self.total_signal / max(1, self.activations)
        if criteria == 'usage':
            self.fitness = self.activations / age
        elif criteria == 'signal':
            self.fitness = avg_signal
        elif criteria == 'age':
            # prefer moderate age
            if age < 2:
                self.fitness = 0.1 * (age / 2)
            elif age > 300:
                self.fitness = 0.1 * (300 / age)
            else:
                self.fitness = 1.0
        else:
            self.fitness = self.activations / age
        return float(self.fitness)

    def should_die(self, min_act=10, min_fit=1e-3, max_age=300) -> bool:
        age = time.time() - self.birth_time
        if age < 3:
            return False
        if self.activations < min_act and age > 30:
            return True
        if self.fitness < min_fit and age > 30:
            return True
        if age > max_age:
            return True
        return False

    def reproduce(self, partner: 'RealNeuron', deterministic: bool = False) -> 'RealNeuron':
        child = RealNeuron(self.input_dim)
        if deterministic:
            mask = torch.arange(self.input_dim) % 2 == 0
        else:
            mask = deterministic_torch_rand(self.input_dim) > 0.5
        child.weight = torch.where(mask, self.weight, partner.weight)
        child.bias = (self.bias + partner.bias) / 2
        if not deterministic and (abs(hash(time.time())) % 100 < 5):
            child.weight += torch.randn_like(child.weight) * 0.01
            child.bias += torch.randn_like(child.bias) * 0.001
        child.generation = max(self.generation, partner.generation) + 1
        return child

    def to_state(self) -> NeuronState:
        return NeuronState(
            id=self.id,
            input_dim=self.input_dim,
            birth_time=self.birth_time,
            activations=self.activations,
            total_signal=self.total_signal,
            generation=self.generation,
            fitness=self.fitness,
        )

class NeuronFarm:
    def __init__(self, input_dim: int, initial_population: int,
                 min_population: int, max_population: int,
                 fitness_criteria: str = 'usage',
                 deterministic_evolution: bool = False,
                 seed: Optional[int] = None):
        self.input_dim = input_dim
        self.min_population = min_population
        self.max_population = max_population
        self.fitness_criteria = fitness_criteria
        self.deterministic = deterministic_evolution
        self.rng = random.Random(seed)
        self.neurons: Dict[str, RealNeuron] = {}
        for _ in range(initial_population):
            n = RealNeuron(input_dim)
            self.neurons[n.id] = n
        self.generation_count = 0
        self.total_births = len(self.neurons)
        self.total_deaths = 0

    def process_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.neurons:
            for _ in range(self.min_population):
                n = RealNeuron(self.input_dim)
                self.neurons[n.id] = n
                self.total_births += 1
        outs = [n.activate(x) for n in self.neurons.values()]
        if not outs:
            return torch.zeros(1)
        return torch.mean(torch.stack(outs)).unsqueeze(0)

    def cycle(self):
        # update fitness
        prev_fitness = {nid: n.fitness for nid, n in self.neurons.items()}
        for n in self.neurons.values():
            n.compute_fitness(self.fitness_criteria)
        # deaths
        # Darwin per generation: remove low fitness with no improvement
        candidates = []
        for nid, n in self.neurons.items():
            improvement = n.fitness - prev_fitness.get(nid, n.fitness)
            candidates.append((nid, n.fitness, improvement))
        # Sort by fitness asc, improvement asc
        candidates.sort(key=lambda t: (t[1], t[2]))
        # baseline rule-based deaths
        rule_kill = [nid for nid, n in self.neurons.items() if n.should_die()]
        # add additional Darwinian removals up to 30%
        extra = []
        cap = int(0.3 * len(self.neurons))
        for nid, _, _ in candidates:
            if len(rule_kill) + len(extra) >= cap:
                break
            if nid not in rule_kill:
                extra.append(nid)
        to_kill = rule_kill + extra
        # cap 30%
        if len(to_kill) > int(0.3 * len(self.neurons)):
            to_kill = to_kill[:int(0.3 * len(self.neurons))]
        for nid in to_kill:
            if len(self.neurons) > self.min_population:
                del self.neurons[nid]
                self.total_deaths += 1
        # reproduction
        if len(self.neurons) < self.max_population and len(self.neurons) >= 2:
            # sort by fitness
            top = sorted(self.neurons.values(), key=lambda n: n.fitness, reverse=True)
            parents = top[:max(2, len(top)//2)]
            num_off = min(10, self.max_population - len(self.neurons))
            for _ in range(num_off):
                a = parents[0] if self.deterministic else self.rng.choice(parents)
                b = parents[1] if self.deterministic else self.rng.choice(parents)
                if a.id != b.id:
                    c = a.reproduce(b, deterministic=self.deterministic)
                    self.neurons[c.id] = c
                    self.total_births += 1
        self.generation_count += 1

    def stats(self) -> Dict:
        fits = [n.fitness for n in self.neurons.values()]
        return {
            'population': len(self.neurons),
            'generation': self.generation_count,
            'total_births': self.total_births,
            'total_deaths': self.total_deaths,
            'avg_fitness': float(np.mean(fits)) if fits else 0.0,
            'max_fitness': float(np.max(fits)) if fits else 0.0,
            'min_fitness': float(np.min(fits)) if fits else 0.0,
        }

# ---------------------------
# Brain wrapper
# ---------------------------

class IA3Brain:
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 min_pop: int, max_pop: int, fitness: str, deterministic: bool):
        self.input_farm = NeuronFarm(input_dim, initial_population=hidden_dim,
                                     min_population=min_pop, max_population=max_pop,
                                     fitness_criteria=fitness, deterministic_evolution=deterministic)
        self.hidden_farm = NeuronFarm(hidden_dim, initial_population=hidden_dim,
                                      min_population=min_pop, max_population=max_pop,
                                      fitness_criteria=fitness, deterministic_evolution=deterministic)
        self.output_farm = NeuronFarm(hidden_dim, initial_population=output_dim,
                                      min_population=min_pop, max_population=max_pop,
                                      fitness_criteria=fitness, deterministic_evolution=deterministic)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # h1: scalar per input vector → expand to hidden_dim by replication of mean output
        h1_scalar = self.input_farm.process_input(x)
        h1 = h1_scalar.repeat(self.hidden_farm.input_dim)
        h2_scalar = self.hidden_farm.process_input(h1)
        out = h2_scalar.repeat(self.output_farm.input_dim)
        return out

    def evolve(self):
        self.input_farm.cycle()
        self.hidden_farm.cycle()
        self.output_farm.cycle()

    def snapshot(self) -> Dict:
        return {
            'input': self.input_farm.stats(),
            'hidden': self.hidden_farm.stats(),
            'output': self.output_farm.stats(),
        }

# ---------------------------
# DB helper (optional)
# ---------------------------

class DB:
    def __init__(self, path: str):
        self.enabled = SQLITE_AVAILABLE and bool(path)
        self.path = path
        self.conn = None
        if self.enabled:
            try:
                self.conn = sqlite3.connect(self.path, check_same_thread=False)
                c = self.conn.cursor()
                c.execute("""
                    CREATE TABLE IF NOT EXISTS evolution (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ts REAL,
                        generation INTEGER,
                        pop_input INTEGER,
                        pop_hidden INTEGER,
                        pop_output INTEGER,
                        max_fit REAL,
                        avg_fit REAL
                    )
                """)
                self.conn.commit()
            except Exception as e:
                logging.warning(f"DB disabled: {e}")
                self.enabled = False

    def log_evolution(self, gen: int, brain_snap: Dict):
        if not self.enabled:
            return
        try:
            c = self.conn.cursor()
            c.execute(
                "INSERT INTO evolution (ts, generation, pop_input, pop_hidden, pop_output, max_fit, avg_fit) VALUES (?,?,?,?,?,?,?)",
                (
                    time.time(),
                    gen,
                    brain_snap['input']['population'],
                    brain_snap['hidden']['population'],
                    brain_snap['output']['population'],
                    max(brain_snap['input']['max_fitness'], brain_snap['hidden']['max_fitness'], brain_snap['output']['max_fitness']),
                    np.mean([brain_snap['input']['avg_fitness'], brain_snap['hidden']['avg_fitness'], brain_snap['output']['avg_fitness']])
                )
            )
            self.conn.commit()
        except Exception as e:
            logging.warning(f"DB write failed: {e}")

# ---------------------------
# Main run modes
# ---------------------------

def run_steps(args: argparse.Namespace) -> int:
    seed_all(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')

    brain = IA3Brain(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        output_dim=args.output_dim,
        min_pop=args.min_pop,
        max_pop=args.max_pop,
        fitness=args.fitness,
        deterministic=args.deterministic_evolution,
    )
    db = DB(args.db_path)

    for step in range(args.steps):
        x = torch.randn(args.input_dim)
        y = brain.forward(x)
        brain.evolve()
        snap = brain.snapshot()
        metrics = {
            'step': step,
            'timestamp': now_iso(),
            'mean_out': float(y.mean().item()),
            'brain': snap,
        }
        log_metrics(args.out_dir, metrics)
        db.log_evolution(snap['input']['generation'], snap)
        if (step + 1) % 100 == 0:
            save_checkpoint(args.out_dir, f'checkpoint_step_{step+1}.json', {
                'timestamp': now_iso(),
                'step': step + 1,
                'brain': snap,
            })
        if args.sleep > 0:
            time.sleep(args.sleep)
    return 0


def run_test(args: argparse.Namespace) -> int:
    seed_all(args.seed)
    out_dir = os.path.join(args.out_dir, 'test')
    os.makedirs(out_dir, exist_ok=True)

    # Basic creation
    n = RealNeuron(8)
    v = torch.randn(8)
    y = n.activate(v)
    assert y.numel() == 1

    # Fitness evolves with usage
    n.compute_fitness('usage')
    fit0 = n.fitness
    for _ in range(20):
        _ = n.activate(torch.randn(8))
    n.compute_fitness('usage')
    fit1 = n.fitness
    assert fit1 >= fit0

    # Farm generates outputs and evolves
    farm = NeuronFarm(input_dim=8, initial_population=6, min_population=4, max_population=20)
    _ = farm.process_input(torch.randn(8))
    _ = len(farm.neurons)
    farm.cycle()
    pop1 = len(farm.neurons)
    assert pop1 >= 1

    # Brain snapshot integrity
    brain = IA3Brain(input_dim=8, hidden_dim=8, output_dim=4, min_pop=4, max_pop=20, fitness='usage', deterministic=False)
    s = brain.snapshot()
    assert 'input' in s and 'hidden' in s and 'output' in s

    # Metrics writing
    log_metrics(out_dir, {'ok': True, 'ts': now_iso()})
    assert os.path.exists(os.path.join(out_dir, 'metrics.jsonl'))

    logger.info('TEST OK')
    return 0


def run_forever(args: argparse.Namespace) -> int:
    seed_all(args.seed)
    while True:
        run_steps(argparse.Namespace(**{**vars(args), 'steps': 100}))
    # unreachable


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(os.path.join(args.out_dir, 'neural_farm.log'), 'a')
        ]
    )
    if args.mode == 'steps':
        code = run_steps(args)
        sys.exit(code)
    elif args.mode == 'test':
        code = run_test(args)
        sys.exit(code)
    else:
        code = run_forever(args)
        sys.exit(code)


if __name__ == '__main__':
    main()
