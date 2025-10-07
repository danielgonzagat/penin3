import random, importlib
from typing import Dict, Any

# fallback simples
def _toy_init(rng: random.Random)->Dict[str,float]: return {"x": rng.uniform(-6,6)}
def _toy_eval(g:Dict[str,float], rng:random.Random)->Dict[str,Any]:
    import math
    x=float(g["x"]); obj=math.sin(3*x)+0.6*math.cos(5*x)+0.1*x
    return {"objective":obj, "linf":0.9, "caos_plus":1.0, "robustness":1.0,
            "cost_penalty":max(0.5,1.0-0.02*abs(x)), "behavior":[x,obj], "eco_ok":True,"consent":True}

def autodetect():
    # tenta padrões comuns
    cand = [
        ("core.darwin_engine_real","evaluate_individual","init_genome"),
        ("core.darwin_evolution_system_FIXED","evaluate_individual","init_genome"),
        ("darwin_engine.engine","evaluate","init_genome"),
        ("darwin_engine_real","evaluate","init_genome"),
    ]
    for mod, ev, init in cand:
        try:
            m=importlib.import_module(mod)
            eval_fn=getattr(m, ev, None); init_fn=getattr(m, init, None)
            if callable(eval_fn) and callable(init_fn):
                return init_fn, eval_fn
        except Exception:
            pass
    return _toy_init, _toy_eval
