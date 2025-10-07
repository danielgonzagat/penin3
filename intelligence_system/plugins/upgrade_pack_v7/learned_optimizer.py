
import math, json, pathlib, torch, logging
from torch import nn
from collections import defaultdict

logger = logging.getLogger("upgrade_pack_v7.learned_optimizer")

RUNTIME = pathlib.Path("/root/intelligence_system/runtime"); RUNTIME.mkdir(parents=True, exist_ok=True)

# ----------------------- META-SGD ---------------------------------
class MetaSGD(torch.optim.Optimizer):
    """
    Otimizador com taxa por-parâmetro aprendida (learnable lrs).
    """
    def __init__(self, params, base_lr=3e-4, lr_bounds=(1e-6, 3e-3), name="metasgd"):
        params = list(params)
        assert len(params)>0, "MetaSGD: sem parâmetros"
        self.name = name
        self.state_path = RUNTIME/f"{name}_state.pt"
        defaults = dict()
        super().__init__(params, defaults)
        # lrs por tensor
        self._lrs = { id(p): torch.full_like(p, float(base_lr)) for group in self.param_groups for p in group['params'] if p.requires_grad }
        # estatísticas para hyper_step
        self._m_corr = defaultdict(float)
        self.lr_min, self.lr_max = lr_bounds
        self._load()
        logger.info(f"MetaSGD initialized: {len(self._lrs)} parameter groups")

    def _save(self):
        try:
            torch.save({"lrs": {k: v.detach().cpu() for k,v in self._lrs.items()},
                        "m_corr": dict(self._m_corr)}, self.state_path)
        except Exception: pass

    def _load(self):
        if self.state_path.exists():
            try:
                st = torch.load(self.state_path, map_location="cpu")
                for group in self.param_groups:
                    for p in group['params']:
                        if id(p) in st["lrs"]:
                            self._lrs[id(p)] = st["lrs"][id(p)].to(p.device)
                self._m_corr.update(st.get("m_corr", {}))
                logger.info(f"MetaSGD loaded state from {self.state_path}")
            except Exception:
                pass

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None: loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad: continue
                if p.grad is None: continue
                g = p.grad
                lr_t = self._lrs[id(p)]
                p.add_( -lr_t * g )
        return loss

    @torch.no_grad()
    def hyper_step(self, corr_beta=0.98, lr_eta=1e-3):
        """Atualiza lrs com base na correlação de sinal"""
        for group in self.param_groups:
            for p in group['params']:
                if not p.requires_grad: continue
                if p.grad is None: continue
                g = p.grad
                st = self.state[p]
                prev = st.get("prev_param", None)
                if prev is None:
                    st["prev_param"] = p.detach().clone()
                    continue
                delta = p.detach() - prev
                st["prev_param"] = p.detach().clone()
                # correlação de sinal
                num = torch.sum(torch.sign(delta) * torch.sign(g)).item()
                den = delta.numel()
                c = num / max(1, den)
                k = str(id(p))
                self._m_corr[k] = corr_beta*self._m_corr.get(k, 0.0) + (1-corr_beta)*c
                adj = 1.0 + lr_eta * self._m_corr[k]
                lrt = self._lrs[id(p)]
                lrt.mul_(adj)
                lrt.clamp_(min=self.lr_min, max=self.lr_max)
        self._save()

# ------------------- helpers públicos -----------------------------
def build_learned_optimizer(agent, kind="metasgd"):
    """Substitui agent.optimizer por otim. aprendido"""
    logger.info(f"Building learned optimizer: kind={kind}, agent={type(agent).__name__}")
    
    lr = float(getattr(agent, "learning_rate", 3e-4))
    params = []
    if hasattr(agent, "policy") and hasattr(agent.policy, "parameters"):
        params = list(agent.policy.parameters())
    elif hasattr(agent, "parameters"):
        params = list(agent.parameters())
    else:
        for v in agent.__dict__.values():
            if hasattr(v, "parameters"):
                params += list(v.parameters())
    
    if not params:
        logger.error("learned_optimizer: no parameters found")
        raise RuntimeError("learned_optimizer: não encontrei parâmetros do agente.")

    if kind == "lstm":
        logger.warning("LSTM optimizer not implemented yet, falling back to metasgd")
        kind = "metasgd"
    
    opt = MetaSGD(params, base_lr=lr, name="metasgd")
    setattr(agent, "optimizer", opt)
    logger.info(f"Learned optimizer attached: {kind}")
    return f"learned_optimizer::{kind}"

def maybe_meta_update(agent):
    """Atualiza otimizador aprendido se presente"""
    opt = getattr(agent, "optimizer", None)
    if isinstance(opt, MetaSGD):
        opt.hyper_step()
        return True
    return False
