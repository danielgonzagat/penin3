# Operações para self_modification_engine (registradas via hook)
import logging
logger = logging.getLogger("upgrade_pack_v7.ops")

def op_tune_entropy(ctx, delta: float):
    # espera ctx["agent"].entropy_coef
    agent = ctx.get("agent")
    if not agent: return {"ok":False,"msg":"agent missing"}
    val = float(getattr(agent, "entropy_coef", 0.0)) + float(delta)
    val = max(0.0, min(0.5, val))
    setattr(agent, "entropy_coef", val)
    return {"ok":True,"msg":f"entropy_coef->{val:.4f}"}

def op_tune_cliprange(ctx, delta: float):
    agent = ctx.get("agent")
    if not agent: return {"ok":False,"msg":"agent missing"}
    cr = float(getattr(agent, "clip_range", 0.2)) + float(delta)
    cr = max(0.05, min(0.4, cr))
    setattr(agent, "clip_range", cr)
    return {"ok":True,"msg":f"clip_range->{cr:.3f}"}

def op_tune_lr(ctx, scale: float):
    agent = ctx.get("agent")
    if not agent: return {"ok":False,"msg":"agent missing"}
    base = float(getattr(agent, "learning_rate", 3e-4))
    new = max(1e-6, min(3e-3, base*float(scale)))
    setattr(agent, "learning_rate", new)
    # se tiver otimizador, atualiza
    opt = getattr(agent, "optimizer", None)
    if opt and hasattr(opt, "param_groups"):
        for g in opt.param_groups: g["lr"]=new
    return {"ok":True,"msg":f"lr->{new:.6f}"}

def op_swap_optimizer(ctx, to: str="adam", kind: str="metasgd", gating: str="layer", beta: float=0.98, **kwargs):
    agent = ctx.get("agent")
    if not agent: 
        logger.warning("swap_optimizer: agent missing in context")
        return {"ok":False,"msg":"agent missing"}
    
    logger.info(f"swap_optimizer: requested to={to}, kind={kind}, gating={gating}, beta={beta}")
    
    try:
        import torch
        
        # Coleta parâmetros
        base_params = []
        if hasattr(agent, "policy") and hasattr(agent.policy, "parameters"):
            base_params = list(agent.policy.parameters())
        elif hasattr(agent, "parameters"):
            base_params = list(agent.parameters())
        else:
            # fallback
            for v in agent.__dict__.values():
                if hasattr(v, "parameters"):
                    base_params += list(v.parameters())
        
        if not base_params:
            logger.error("swap_optimizer: no parameters found in agent")
            return {"ok":False,"msg":"no_parameters"}
        
        lr = float(getattr(agent, "learning_rate", 3e-4))
        
        # HYBRID
        if to.lower() == "hybrid":
            logger.info(f"swap_optimizer: building hybrid (gating={gating}, beta={beta})")
            try:
                from plugins.upgrade_pack_v7 import hybrid_optimizer as _hy
                msg = _hy.build_hybrid_optimizer(agent, mode=gating, lr_base=kwargs.get("lr_base", lr), beta=beta)
                logger.info(f"swap_optimizer: hybrid built successfully: {msg}")
                return {"ok": True, "msg": msg}
            except Exception as e:
                logger.exception(f"swap_optimizer: hybrid build failed: {e}")
                return {"ok": False, "msg": f"hybrid_failed: {e}"}
        
        # LEARNED
        elif to.lower() == "learned":
            logger.info(f"swap_optimizer: building learned optimizer (kind={kind})")
            try:
                from plugins.upgrade_pack_v7 import learned_optimizer as _lo
                msg = _lo.build_learned_optimizer(agent, kind=kind)
                logger.info(f"swap_optimizer: learned built successfully: {msg}")
                return {"ok": True, "msg": msg}
            except Exception as e:
                logger.exception(f"swap_optimizer: learned build failed: {e}")
                return {"ok": False, "msg": f"learned_failed: {e}"}
        
        # RMSPROP
        elif to.lower() == "rmsprop":
            optim = torch.optim.RMSprop(base_params, lr=lr, alpha=0.99)
            setattr(agent, "optimizer", optim)
            logger.info(f"swap_optimizer: switched to RMSprop (lr={lr})")
            return {"ok":True,"msg":f"optimizer->rmsprop"}
        
        # ADAM (default)
        else:
            optim = torch.optim.Adam(base_params, lr=lr)
            setattr(agent, "optimizer", optim)
            logger.info(f"swap_optimizer: switched to Adam (lr={lr})")
            return {"ok":True,"msg":f"optimizer->adam"}
            
    except Exception as e:
        logger.exception(f"swap_optimizer: failed with exception: {e}")
        return {"ok":False,"msg":str(e)}
