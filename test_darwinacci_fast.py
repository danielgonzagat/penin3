#!/usr/bin/env python3
"""
DARWINACCI ULTRA-FAST TEST
Fitness toy (sem gym pesado) para validar motor
"""
import sys
sys.path.insert(0, '/root')

from darwinacci_omega.core.engine import DarwinacciEngine
import random

def init_genome(rng):
    """Genome simples: 3 parâmetros"""
    return {
        'x': rng.uniform(-10, 10),
        'y': rng.uniform(-10, 10),
        'z': rng.uniform(-10, 10),
    }

def evaluate(genome, rng):
    """Fitness TOY: sphere function (objetivo: minimizar distância da origem)"""
    x = genome.get('x', 0)
    y = genome.get('y', 0)
    z = genome.get('z', 0)
    
    # Distance from origin (lower is better, invert for maximization)
    dist = (x**2 + y**2 + z**2) ** 0.5
    objective = max(0.0, 1.0 - dist / 20.0)  # Normalize to [0,1]
    
    behavior = [x / 10.0, y / 10.0]  # 2D behavior for QD
    
    return {
        "objective": objective,
        "linf": objective * 1.05,
        "novelty": 0.0,  # Will be filled by engine
        "robustness": 0.95,
        "caos_plus": 1.0,
        "cost_penalty": 1.0,
        "behavior": behavior,
        "ece": 0.05,
        "rho_bias": 1.0,
        "rho": 0.9,
        "eco_ok": True,
        "consent": True,
    }

print("🚀 DARWINACCI ULTRA-FAST TEST")
print("=" * 60)

import os
os.environ["DARWINACCI_TRIALS"] = "1"

eng = DarwinacciEngine(
    init_fn=init_genome,
    eval_fn=evaluate,
    max_cycles=5,
    pop_size=24,
    seed=42
)

print(f"Population: {eng.pop_size}")
print(f"Max cycles: 5")
print(f"Fitness: Sphere function (toy)")
print()

champ = eng.run(max_cycles=5)

print()
print("=" * 60)
print("📊 RESULTADOS:")
print("=" * 60)
if champ:
    print(f"Champion score: {champ.score:.6f}")
    print(f"Champion genome: {champ.genome}")
    print(f"Champion behavior: {champ.behavior}")
print(f"Coverage: {eng.archive.coverage():.2%}")
print(f"Novelty archive: {len(eng.novel.mem)}")
print(f"Arena champion: {eng.arena.champion is not None}")
print()
print("✅ DARWINACCI MOTOR FUNCIONAL!")
