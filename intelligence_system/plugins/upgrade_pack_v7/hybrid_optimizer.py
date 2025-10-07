
import math, pathlib, torch, logging
from torch import nn

logger = logging.getLogger("upgrade_pack_v7.hybrid")

RUNTIME = pathlib.Path("/root/intelligence_system/runtime"); RUNTIME.mkdir(parents=True, exist_ok=True)

# ---- bloco leve: Adam-like "shadow" ------------
class AdamLike:
    def __init__(self, params, lr=3e-4, betas=(0.9, 0.999), eps=1e-8):
        self.lr, self.betas, self.eps = float(lr), betas, eps
        self.state = {}
        self.params = list(params)
    
    @torch.no_grad()
    def step(self, grads):
        deltas = []
        b1, b2 = self.betas
        for p, g in zip(self.params, grads):
            if g is None: 
                deltas.append(None)
                continue
            sid = id(p)
            st = self.state.setdefault(sid, {"m": torch.zeros_like(p), "v": torch.zeros_like(p), "t": 0})
            st["t"] += 1
            m, v, t = st["m"], st["v"], st["t"]
            m.mul_(b1).add_(g, alpha=1-b1)
            v.mul_(b2).addcmul_(g, g, value=1-b2)
            mhat = m / (1 - b1**t)
            vhat = v / (1 - b2**t)
            delta = - self.lr * mhat / (torch.sqrt(vhat) + self.eps)
            deltas.append(delta)
        return deltas

# ---- MetaSGD coordenado ----
class MetaSGDCoord:
    def __init__(self, params, base_lr=3e-4, lr_bounds=(1e-6, 3e-3), name="hybrid_metasgd"):
        self.lr_min, self.lr_max = lr_bounds
        self.state_path = RUNTIME/f"{name}_state.pt"
        self.params = list(params)
        self.lrs = { id(p): torch.full_like(p, float(base_lr)) for p in self.params if p.requires_grad }
        self.mcorr = {}
        self._load()
    
    def _load(self):
        if self.state_path.exists():
            try:
                st = torch.load(self.state_path, map_location="cpu")
                for p in self.params:
                    if id(p) in st.get("lrs", {}):
                        self.lrs[id(p)] = st["lrs"][id(p)].to(p.device)
                self.mcorr.update(st.get("mcorr", {}))
            except Exception: pass
    
    def _save(self):
        try:
            torch.save({"lrs": {k: v.detach().cpu() for k,v in self.lrs.items()},
                        "mcorr": self.mcorr}, self.state_path)
        except Exception: pass
    
    @torch.no_grad()
    def step(self, grads):
        deltas = []
        for p, g in zip(self.params, grads):
            if g is None: 
                deltas.append(None)
                continue
            lrt = self.lrs[id(p)]
            deltas.append(- lrt * g)
        return deltas
    
    @torch.no_grad()
    def hyper_step(self, params_prev, grads, corr_beta=0.98, lr_eta=1e-3):
        for p, prev, g in zip(self.params, params_prev, grads):
            if (g is None) or (prev is None): 
                continue
            delta = p.detach() - prev
            if delta.numel()==0: 
                continue
            c = torch.mean(torch.sign(delta)*torch.sign(g)).item()
            k = str(id(p))
            self.mcorr[k] = corr_beta*self.mcorr.get(k,0.0) + (1-corr_beta)*c
            lrt = self.lrs[id(p)]
            lrt.mul_(1.0 + lr_eta*self.mcorr[k])
            lrt.clamp_(min=self.lr_min, max=self.lr_max)
        if torch.rand(1).item() < 0.05:
            self._save()

# ---- Hybrid optimizer ----
class HybridGatedOptimizer(torch.optim.Optimizer):
    """Híbrido: delta = alpha * MetaSGD + (1-alpha) * AdamLike"""
    def __init__(self, params, lr_base=3e-4, mode="layer", beta=0.98, name="hybrid_opt"):
        params = [p for p in params if p.requires_grad]
        assert params, "Hybrid: sem params"
        self.name = name
        self.beta = beta
        defaults = dict()
        super().__init__(params, defaults)
        self.meta = MetaSGDCoord(params, base_lr=lr_base, name=name+"_metasgd")
        self.adam = AdamLike(params, lr=lr_base)
        # Gate simple (layer-level alpha)
        self.alphas = {id(p): 0.5 for g in self.param_groups for p in g['params']}
        self.score = {id(p): 0.0 for g in self.param_groups for p in g['params']}
        self.prev_params = {id(p): p.detach().clone() for g in self.param_groups for p in g['params']}
        logger.info(f"HybridGatedOptimizer initialized: mode={mode}, lr={lr_base}, beta={beta}")

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        grads, params = [], []
        for group in self.param_groups:
            for p in group['params']:
                params.append(p)
                grads.append(None if (p.grad is None) else p.grad.detach())
        delta_meta = self.meta.step(grads)
        delta_adam = self.adam.step(grads)
        for p, dm, da in zip(params, delta_meta, delta_adam):
            if (dm is None) and (da is None): 
                continue
            if dm is None: dm = torch.zeros_like(p)
            if da is None: da = torch.zeros_like(p)
            a = self.alphas[id(p)]
            p.add_( a*dm + (1-a)*da )
        return loss

    @torch.no_grad()
    def meta_update(self, grads):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None: continue
                sid = id(p)
                prev = self.prev_params.get(sid, None)
                if prev is None:
                    self.prev_params[sid] = p.detach().clone()
                    continue
                delta_real = p.detach() - prev
                self.prev_params[sid] = p.detach().clone()
                if delta_real.numel()==0: 
                    continue
                g = p.grad.detach()
                v = torch.mean(torch.sign(-g)*torch.sign(delta_real)).item()
                self.score[sid] = self.beta*self.score.get(sid,0.0) + (1-self.beta)*v
                # Update alpha
                self.alphas[sid] = max(0.0, min(1.0, self.alphas[sid] + 0.01 * self.score[sid]))
        params_prev = [self.prev_params.get(id(p)) for g in self.param_groups for p in g['params']]
        grads_list = [p.grad.detach() if p.grad is not None else None for g in self.param_groups for p in g['params']]
        self.meta.hyper_step(params_prev, grads_list)

def build_hybrid_optimizer(agent, mode="layer", lr_base=None, beta=0.98):
    logger.info(f"Building HybridGatedOptimizer (mode={mode}, lr_base={lr_base}, beta={beta}) for agent={getattr(agent,'__class__',type(agent)).__name__}")
    
    params = []
    lr = float(getattr(agent, "learning_rate", 3e-4)) if lr_base is None else float(lr_base)
    if hasattr(agent, "policy") and hasattr(agent.policy, "parameters"):
        params = list(agent.policy.parameters())
    elif hasattr(agent, "parameters"):
        params = list(agent.parameters())
    else:
        for v in agent.__dict__.values():
            if hasattr(v, "parameters"): params += list(v.parameters())
    
    if not params:
        logger.error("hybrid_optimizer: no parameters found")
        raise RuntimeError("hybrid_optimizer: não encontrei parâmetros.")
    
    opt = HybridGatedOptimizer(params, lr_base=lr, mode=mode, beta=beta, name="hybrid_opt")
    setattr(agent, "optimizer", opt)
    logger.info(f"HybridGatedOptimizer attached to agent.optimizer")
    return f"hybrid_optimizer::{mode}"

def maybe_hybrid_update(agent):
    opt = getattr(agent, "optimizer", None)
    if isinstance(opt, HybridGatedOptimizer):
        grads = []
        for g in opt.param_groups:
            for p in g['params']: grads.append(None if p.grad is None else p.grad.detach())
        opt.meta_update(grads)
        return True
    return False
