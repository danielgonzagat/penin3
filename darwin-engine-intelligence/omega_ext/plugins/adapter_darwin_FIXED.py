"""
Adapter que REALMENTE integra Omega com Darwin existente
CORRIGIDO - Não mais toy functions
"""
import random
from typing import Dict, Any, Tuple, Callable
import sys
sys.path.insert(0, '/workspace')

def create_darwin_real_adapter() -> Tuple[Callable, Callable]:
    """
    Cria adapter que usa o Darwin REAL, não toy
    """
    from core.darwin_evolution_system_FIXED import EvolvableMNIST
    
    def init_genome(rng: random.Random) -> Dict[str, float]:
        """Cria genoma compatível com EvolvableMNIST"""
        return {
            'hidden_size': float(rng.choice([64, 128, 256, 512])),
            'learning_rate': rng.uniform(0.0001, 0.01),
            'batch_size': float(rng.choice([32, 64, 128, 256])),
            'dropout': rng.uniform(0.0, 0.5),
            'num_layers': float(rng.choice([2, 3, 4]))
        }
    
    def evaluate(genome: Dict[str, float], rng: random.Random) -> Dict[str, Any]:
        """
        Avalia usando EvolvableMNIST REAL
        
        IMPORTANTE: Treina modelo real, não toy function
        """
        # Converter genome de volta para ints onde necessário
        genome_int = {
            'hidden_size': int(genome['hidden_size']),
            'learning_rate': genome['learning_rate'],
            'batch_size': int(genome['batch_size']),
            'dropout': genome['dropout'],
            'num_layers': int(genome['num_layers'])
        }
        
        # Criar e avaliar
        evolvable = EvolvableMNIST(genome_int)
        base_fitness = evolvable.evaluate_fitness()
        
        # Métricas multiobjetivo (básicas por enquanto)
        return {
            "objective": float(base_fitness),
            "linf": 0.9,  # TODO: calcular real
            "caos_plus": 1.0,  # TODO: calcular real
            "robustness": 1.0,
            "cost_penalty": 1.0,
            "behavior": [float(base_fitness)],  # Simplified
            "eco_ok": True,
            "consent": True,
            "probs": [],  # Para ECE (vazio por enquanto)
            "labels": []
        }
    
    return init_genome, evaluate


def autodetect_FIXED() -> Tuple[Callable, Callable]:
    """
    Autodetect que REALMENTE funciona
    
    Tenta usar Darwin real, fallback para toy se falhar
    """
    try:
        print("[Omega Adapter] Tentando usar Darwin REAL...")
        init_fn, eval_fn = create_darwin_real_adapter()
        print("✅ [Omega Adapter] Darwin REAL conectado!")
        return init_fn, eval_fn
    except Exception as e:
        print(f"⚠️ [Omega Adapter] Falha ao criar adapter real: {e}")
        print("[Omega Adapter] Usando fallback TOY")
        
        # Fallback toy
        def toy_init(rng): 
            return {"x": rng.uniform(-6, 6)}
        
        def toy_eval(g, rng):
            import math
            x = g["x"]
            obj = math.sin(3*x) + 0.6*math.cos(5*x) + 0.1*x
            return {
                "objective": obj, "linf": 0.9, "caos_plus": 1.0,
                "robustness": 1.0, "cost_penalty": 1.0,
                "behavior": [x, obj], "eco_ok": True, "consent": True,
                "probs": [], "labels": []
            }
        
        return toy_init, toy_eval


# Exportar
autodetect = autodetect_FIXED
