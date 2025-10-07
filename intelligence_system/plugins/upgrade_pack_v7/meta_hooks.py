
import os, time, math, json, random, pathlib, pickle
from collections import defaultdict, deque

RUNTIME = pathlib.Path("/root/intelligence_system/runtime")
RUNTIME.mkdir(parents=True, exist_ok=True)

# ---------------- StrategyManager (UCB1) ----------------
class StrategyManager:
    def __init__(self, name="meta_bandit"):
        self.name = name
        self.state_path = RUNTIME / f"{name}.json"
        self.arms = [
            {"id":"tune_entropy_up",   "delta":+0.01},
            {"id":"tune_entropy_down", "delta":-0.01},
            {"id":"tune_clip_up",      "delta":+0.02},
            {"id":"tune_clip_down",    "delta":-0.02},
            {"id":"tune_lr_up",        "delta":+0.25},
            {"id":"tune_lr_down",      "delta":-0.20},
            {"id":"swap_opt_rmsprop",  "opt":"rmsprop"},
            {"id":"swap_opt_adam",     "opt":"adam"},
        ]
        self.N = [1]*len(self.arms)     # pulls
        self.Q = [0.0]*len(self.arms)   # value
        self.t = 1
        self._load()

    def _save(self):
        self.state_path.write_text(json.dumps({"N":self.N,"Q":self.Q,"t":self.t}))
    def _load(self):
        if self.state_path.exists():
            try:
                st = json.loads(self.state_path.read_text())
                self.N, self.Q, self.t = st["N"], st["Q"], st["t"]
            except Exception:
                pass

    def select(self):
        self.t += 1
        ucb = []
        for i,(q,n) in enumerate(zip(self.Q, self.N)):
            bonus = math.sqrt(2*math.log(max(self.t,2))/max(1,n))
            ucb.append(q+bonus)
        idx = max(range(len(self.arms)), key=lambda i: ucb[i])
        return self.arms[idx], idx

    def update(self, idx, reward):
        self.N[idx] += 1
        # moving average
        self.Q[idx] += (reward - self.Q[idx]) / self.N[idx]
        self._save()

STRATEGY = StrategyManager()

# ---------------- Episodic Memory (kNN) ----------------
class EpisodicMemory:
    def __init__(self, name="episodic_knn", k=5, maxlen=20000):
        self.path = RUNTIME / f"{name}.pkl"
        self.k=k; self.maxlen=maxlen
        self.buf = deque(maxlen=maxlen)
        if self.path.exists():
            try: self.buf = pickle.load(open(self.path,"rb"))
            except Exception: pass

    def add(self, state, action, value=0.0):
        self.buf.append((tuple(map(float,state)), int(action), float(value)))

    def suggest(self, state):
        if not self.buf: return None
        s = tuple(map(float,state))
        dists = []
        for i,(st,ac,val) in enumerate(self.buf):
            # L2 no estado (CartPole 4D)
            d = sum((a-b)**2 for a,b in zip(s,st))
            dists.append((d, ac, val))
        dists.sort(key=lambda x:x[0])
        top = dists[:min(self.k, len(dists))]
        # votação ponderada por 1/(d+1e-6)
        score = defaultdict(float)
        for d,ac,val in top:
            score[ac] += 1.0/(d+1e-6)
        best = max(score.items(), key=lambda kv: kv[1])[0]
        return int(best)

EPISODIC = EpisodicMemory()

# ------------- Domain Randomization (CartPole) ----------
def domain_randomize_cartpole(env):
    # Usa atributos padrão se existirem
    try:
        import numpy as np
        # perturbações leves
        env.gravity = getattr(env, "gravity", 9.8) * float(np.clip(np.random.normal(1.0, 0.02), 0.95, 1.05))
        if hasattr(env, "masscart"): env.masscart *= float(np.clip(np.random.normal(1.0, 0.03), 0.9, 1.1))
        if hasattr(env, "masspole"): env.masspole *= float(np.clip(np.random.normal(1.0, 0.03), 0.9, 1.1))
    except Exception:
        pass
    return env

# ------------- PBT-lite (exploit/explore) ----------------
class PBTLite:
    def __init__(self, root="/root/intelligence_system/models", interval=10):
        self.root = pathlib.Path(root)
        self.meta = RUNTIME / "pbt_meta.json"
        self.interval = interval
        self.state = {"last":0}
        if self.meta.exists():
            try: self.state=json.loads(self.meta.read_text())
            except: pass

    def step(self, cycle:int, avg100:float):
        # a cada N ciclos, copiar "melhor" e fazer mutações leves de hparams
        if cycle - int(self.state.get("last",0)) < self.interval:
            return None
        self.state["last"]=cycle
        self.meta.write_text(json.dumps(self.state))
        # slots (usa checkpoints do PPO gravados pelo teu agente)
        ppo_dir = self.root / "ppo_population"
        ppo_dir.mkdir(parents=True, exist_ok=True)
        stamp = f"c{cycle}_a{int(avg100)}"
        (ppo_dir / f"slot_{stamp}.touch").touch()
        # retorna sugestões de mutação de hiperparâmetro
        jitter = random.choice([0.8, 0.9, 1.1, 1.2])
        return {"lr_scale": jitter}

PBT = PBTLite()

# ------------- Meta-score (estabilidade/eficiência/OOD) ------------
def meta_score(delta_avg, steps, var_norm, ood=1.0):
    eff = (delta_avg / max(1.0, steps))
    stab = 1.0/(1.0+max(0.0,var_norm))
    return max(0.0, eff*stab*max(0.1, ood))

# ------------- API pública (usada pelos patches) -------------------
def on_meta_step(stats:dict):
    """
    stats esperados:
      {"cycle":int,"avg100":float,"var":float,"last":float,"reward":float,"loss":float,"steps":int}
    Retorna instruções de estratégia/op para o motor de auto-modificação.
    """
    # bandit escolhe um braço
    arm, idx = STRATEGY.select()

    # meta-recompensa: melhora de avg100 penalizada por var
    r = meta_score(stats.get("reward",0.0), max(1,stats.get("steps",1)), stats.get("var",0.0))
    STRATEGY.update(idx, r)

    plan = {"ops":[]}
    if arm["id"].startswith("tune_entropy"):
        plan["ops"].append({"op":"tune_entropy", "delta": arm["delta"]})
    elif arm["id"].startswith("tune_clip"):
        plan["ops"].append({"op":"tune_cliprange", "delta": arm["delta"]})
    elif arm["id"].startswith("tune_lr"):
        plan["ops"].append({"op":"tune_lr", "scale": 1.0+arm["delta"]})
    elif arm["id"].startswith("swap_opt"):
        plan["ops"].append({"op":"swap_optimizer", "to": arm["opt"]})

    # PBT sugestão periódica
    mut = PBT.step(stats.get("cycle",0), stats.get("avg100",0.0))
    if mut and "lr_scale" in mut:
        plan["ops"].append({"op":"tune_lr", "scale":float(mut["lr_scale"])})

    return plan

def episodic_add(state, action, value=0.0):
    EPISODIC.add(state, action, value)
    # salva eventual a cada 1k
    if len(EPISODIC.buf)%1000==0:
        try: pickle.dump(EPISODIC.buf, open(EPISODIC.path,"wb"))
        except: pass

def episodic_suggest(state):
    return EPISODIC.suggest(state)

def apply_domain_randomization(env):
    return domain_randomize_cartpole(env)
