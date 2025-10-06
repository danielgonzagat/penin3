import math, random
from typing import Dict, Any

def init_genome(rng: random.Random)->Dict[str,float]:
    return {"x": rng.uniform(-6,6), "y": rng.uniform(-6,6)}

def evaluate(genome: Dict[str,float], rng: random.Random)->Dict[str,Any]:
    x=float(genome.get("x", 0.0)); y=float(genome.get("y", 0.0))
    obj = math.sin(3*x)+0.6*math.cos(5*y)+0.1*(x-y)  # multi-ótimos
    cost = max(0.5, 1.0 - 0.02*(abs(x)+abs(y)))
    return {
        "objective": obj,
        "linf": 0.92 + 0.05*math.sin(x+y),
        "caos_plus": 1.0,
        "robustness": 1.0 - 0.1*abs(math.sin(x*y)),
        "cost_penalty": cost,
        "behavior": [x, y],
        "ece": 0.05,
        "rho_bias": 1.0,
        "rho": 0.9,
        "eco_ok": True,
        "consent": True
    }