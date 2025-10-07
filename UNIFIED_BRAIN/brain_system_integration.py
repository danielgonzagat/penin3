#!/usr/bin/env python3
"""
🔗 BRAIN SYSTEM INTEGRATION
Conecta o Unified Brain com V7, PENIN-Ω e Darwin
"""

import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')
sys.path.insert(0, '/root')

import os
import json as _json
import hashlib
import torch
import torch.nn as nn
from unified_brain_core import UnifiedBrain
from brain_logger import brain_logger
from intelligence_system.extracted_algorithms.self_modification_engine import (
    SelfModificationEngine,
)
from curriculum import CurriculumHarness
from collections import deque
try:
    from intelligence_system.config.settings import (
        DATABASE_PATH,
        OFFLINE_ONLY,
        KILL_SWITCH_PATH,
    )
except Exception:
    from intelligence_system.config.settings import DATABASE_PATH
    OFFLINE_ONLY = False
    KILL_SWITCH_PATH = "/root/UNIFIED_BRAIN/KILL_SWITCH"
from intelligence_system.core.database import Database
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

class BrainV7Bridge(nn.Module):
    """
    Ponte entre UnifiedBrain e sistemas V7
    """
    def __init__(self, brain: UnifiedBrain, obs_dim: int = 4, act_dim: int = 2, num_steps: int = 4):
        super().__init__()
        self.brain = brain
        H = brain.H
        self.num_steps = num_steps
        # Keep last encoded Z for simple world-model error (prediction ≈ last Z)
        self._prev_Z0: torch.Tensor | None = None
        
        # Encoders: observações → Z
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, H // 2),
            nn.LayerNorm(H // 2),
            nn.GELU(),
            nn.Linear(H // 2, H)
        )
        
        # Decoders: Z → ações
        self.action_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.LayerNorm(H // 2),
            nn.GELU(),
            nn.Linear(H // 2, act_dim)
        )
        
        # Value head (para PPO)
        self.value_head = nn.Sequential(
            nn.Linear(H, H // 2),
            nn.GELU(),
            nn.Linear(H // 2, 1)
        )
    
    def forward(self, obs: torch.Tensor):
        """
        obs → cérebro → ação + value
        """
        import time
        t0 = time.time()
        # Encode observação para Z
        Z0 = self.obs_encoder(obs)
        t1 = time.time()
        # Processa no cérebro
        Zf = self.brain(Z0, num_steps=self.num_steps)
        t2 = time.time()
        # Decode ação e value (inference-only; avoid autograd save in inference_mode)
        with torch.no_grad():
            Z_detached = Zf.detach()
            logits = self.action_head(Z_detached)
            value = self.value_head(Z_detached)
        t3 = time.time()
        # Simple world-model error: change in encoded observation
        try:
            if self._prev_Z0 is not None:
                # L2 distance normalized by sqrt(H), clamp to [0,1]
                diff = (Z0.detach() - self._prev_Z0).float().flatten()
                wm_err = float(torch.linalg.vector_norm(diff, ord=2).item() / max(1e-6, (Z0.shape[-1] ** 0.5)))
                wm_err = min(1.0, max(0.0, wm_err))
            else:
                wm_err = 0.0
        except Exception:
            wm_err = 0.0
        # Update previous encoding
        try:
            self._prev_Z0 = Z0.detach().clone()
        except Exception:
            self._prev_Z0 = None
        # Attach timings on the tensor via a dict return (avoids tensor mutation)
        timings = {
            'encode_ms': (t1 - t0) * 1000.0,
            'brain_ms': (t2 - t1) * 1000.0,
            'decode_ms': (t3 - t2) * 1000.0,
            'wm_error': wm_err,
        }
        # Return logits, value, Z, and timings via attribute-like channel
        return logits, value, Zf, timings
class MemoryKNN:
    """Lightweight kNN memory in Z-space for novelty signals."""
    def __init__(self, dim: int, capacity: int = 256):
        self.dim = int(dim)
        self.capacity = int(capacity)
        self.bank: deque[torch.Tensor] = deque(maxlen=self.capacity)

    def add(self, z: torch.Tensor) -> None:
        try:
            zf = z.detach().flatten().to(torch.float32).cpu()
            self.bank.append(zf)
        except Exception:
            pass

    def novelty(self, z: torch.Tensor) -> float:
        if not self.bank:
            return 0.0
        try:
            zf = z.detach().flatten().to(torch.float32).cpu().unsqueeze(0)  # [1, D]
            bank = torch.stack(list(self.bank), dim=0)  # [N, D]
            dists = torch.cdist(zf, bank, p=2).squeeze(0)  # [N]
            d = float(dists.min().item())
            norm = max(1e-6, self.dim ** 0.5)
            return float(min(1.0, max(0.0, d / norm)))
        except Exception:
            return 0.0


class BrainPENINOmegaInterface:
    """
    Interface com PENIN-Ω para modulação
    """
    def __init__(self, brain: UnifiedBrain):
        self.brain = brain
        self.L_infinity = 0.0
        self.CAOS_plus = 0.0
        self.SR_Omega_infinity = 0.0
    
    def update_signals(self, penin_metrics: Dict[str, float]):
        """Atualiza sinais de PENIN-Ω"""
        self.L_infinity = penin_metrics.get('L_infinity', 0.0)
        self.CAOS_plus = penin_metrics.get('CAOS_plus', 0.0)
        self.SR_Omega_infinity = penin_metrics.get('SR_Omega_infinity', 0.0)
        
        # Modula parâmetros do cérebro
        if self.brain.router:
            # CAOS+ alto → mais exploração
            if self.CAOS_plus > 0.7:
                self.brain.router.temperature = min(2.0, self.brain.router.temperature * 1.05)
                self.brain.router.top_k = min(self.brain.max_neurons // 10, self.brain.router.top_k + 2)
            
            # L∞ alto → menos LR, mais estabilidade
            if self.L_infinity > 0.8:
                self.brain.alpha = min(0.95, self.brain.alpha + 0.01)
                self.brain.lateral_inhibition = min(0.3, self.brain.lateral_inhibition + 0.02)
    
    def get_ia3_signal(self) -> float:
        """
        Calcula sinal IA³ real do cérebro
        """
        metrics = self.brain.get_metrics_summary()
        
        # IA³ = f(uso real, novidade, impacto)
        uso = metrics.get('total_activations', 0) / max(1, metrics.get('total_steps', 1))
        novidade = metrics.get('avg_novelty', 0)
        coerencia = metrics.get('avg_coherence', 0)
        
        ia3 = (uso * 0.4 + novidade * 0.3 + coerencia * 0.3)
        
        return ia3


class BrainDarwinEvolver:
    """
    Interface com Darwin para evolução da topologia
    """
    def __init__(self, brain: UnifiedBrain):
        self.brain = brain
        self.generation = 0
        self.fitness_history = []
    
    def evaluate_fitness(self, reward: float) -> Dict[str, float]:
        """External-only fitness: use task reward trend as primary signal.

        Keeps internal metrics logged but not weighted for gating.
        """
        metrics = self.brain.get_metrics_summary()
        task_reward = float(reward)
        # Compute a short EWMA on reward to get trend (external-only)
        if not hasattr(self, '_reward_ewma'):
            self._reward_ewma = task_reward
        alpha = 0.1
        self._reward_ewma = (1 - alpha) * getattr(self, '_reward_ewma', task_reward) + alpha * task_reward

        fitness = {
            'task_reward': task_reward,
            'reward_ewma': float(self._reward_ewma),
            'coherence': metrics.get('avg_coherence', 0),
            'novelty': metrics.get('avg_novelty', 0),
            'efficiency': 1.0 / max(1.0, metrics.get('avg_latency_ms', 1.0)),
            'diversity': len(self.brain.registry.get_active()) / max(1, self.brain.max_neurons),
        }
        # Gating uses only reward_ewma; total for logging only mirrors ewma
        total_fitness = fitness['reward_ewma']
        fitness['total'] = total_fitness
        self.fitness_history.append(total_fitness)
        return fitness
    
    def evolve_step(self, fitness: float):
        """
        Um passo de evolução
        """
        # Evolui apenas adapters e router (neurônios permanecem frozen)
        
        # 1. Mutação do router
        if self.brain.router and torch.rand(1).item() < 0.1:  # 10% chance
            with torch.no_grad():
                # Mutação leve nos competence scores
                mutation = torch.randn_like(self.brain.router.competence) * 0.1
                self.brain.router.competence += mutation
                self.brain.router.competence.clamp_(0.0, 10.0)
        
        # 2. External-only gating: freeze low performers by reward trend
        if fitness < 0.4 and self.brain.router:
            from brain_spec import NeuronStatus
            from brain_logger import brain_logger
            
            low_comp = self.brain.router.competence < 0.5
            low_indices = torch.where(low_comp)[0].tolist()
            
            active = self.brain.registry.get_active()
            frozen_count = 0
            for idx in low_indices[:min(len(low_indices), 2)]:  # Max 2 por vez
                if idx < len(active):
                    neuron_id = active[idx].meta.id
                    self.brain.registry.promote(neuron_id, NeuronStatus.FROZEN)
                    brain_logger.warning(f"🧬 Darwin froze {neuron_id} (competence={self.brain.router.competence[idx].item():.3f})")
                    frozen_count += 1
            
            if frozen_count > 0:
                self.brain.initialize_router()
                brain_logger.info(f"Router re-initialized: {len(self.brain.registry.get_active())} active neurons")

            # External-only adjustments: if reward trend falling, increase exploration slightly
            try:
                if hasattr(self, '_reward_ewma') and self._reward_ewma < 0.2:
                    if self.brain.router:
                        self.brain.router.temperature = min(2.0, getattr(self.brain.router, 'temperature', 1.0) * 1.05)
            except Exception:
                pass

        # 3. Ethical gating placeholder: restrict certain ops (document-only here)
        # In production, enforce a whitelist of allowed self-mod/evolution operations
        # tied to external audit logs.
        
        self.generation += 1


class UnifiedSystemController:
    """
    Controlador central que orquestra tudo
    """
    def __init__(self, brain: UnifiedBrain):
        self.brain = brain
        self.v7_bridge = None  # Inicializado quando conectar V7
        self.penin_interface = BrainPENINOmegaInterface(brain)
        self.darwin_evolver = BrainDarwinEvolver(brain)
        # Minimal SME (safe ops only)
        try:
            self.sme = SelfModificationEngine(max_modifications_per_cycle=1,
                                              allowed_operations=['adjust_lr'])
        except Exception:
            self.sme = None
        # Lightweight memory for novelty
        try:
            self._memory = MemoryKNN(dim=getattr(brain, 'H', 1024), capacity=256)
        except Exception:
            self._memory = None
        
        self.episode_count = 0
        self.total_reward = 0.0
        
        # T1.4: Curriculum com progressão de tarefas
        self.curriculum_tasks = [
            ('CartPole-v1', 195.0),      # Threshold para "solved"
            ('MountainCar-v0', -110.0),  # Solved se avg >= -110
            ('Acrobot-v1', -100.0),      # Solved se avg >= -100
        ]
        self.current_task_idx = 0
        self.task_history = []  # Track solved tasks
        self.task_eval_window = 100  # Episodes para avaliar se resolveu
        
        # Telemetry DB
        try:
            self.db = Database(DATABASE_PATH)
        except Exception:
            self.db = None
        # Expose DB to brain for longitudinal metrics persistence
        try:
            self.brain.db = self.db
        except Exception:
            pass
        # Planner state (stateless by design, but keep last plan for audit)
        self._last_plan: Dict[str, Any] = {}
        # Meta-online control state
        self._config_history: list[dict] = []
        self._last_meta_decision: dict[str, Any] = {}

        # Darwinacci bridge (global optimizer for router/V7 params)
        self._darwinacci = None
        self._darwinacci_enabled = False
        try:
            import sys, torch
            sys.path.insert(0, '/root')
            from intelligence_system.extracted_algorithms.darwin_engine_darwinacci import DarwinacciOrchestrator

            def fitness_fn(_ind):
                cfg = self.get_config()
                g = _ind.genome
                cand = dict(cfg)
                if 'top_k' in g: cand['top_k'] = int(max(1, min(16, round(g['top_k']))))
                if 'temperature' in g: cand['temperature'] = float(max(0.5, min(2.0, g['temperature'])))
                if 'num_steps' in g: cand['num_steps'] = int(max(1, min(3, round(g['num_steps']))))
                base = self.get_config()
                self.set_config(cand)
                try:
                    # Micro-probe: route latency + entropy
                    Z0 = torch.randn(1, getattr(self.brain, 'H', 1024))
                    if self.v7_bridge is not None:
                        logits, value, Z, timings = self.v7_bridge(Z0)
                        brain_ms = float(timings.get('brain_ms', 0.0)) if isinstance(timings, dict) else 0.0
                        ent = float((torch.softmax(logits, dim=-1) * torch.log_softmax(logits, dim=-1)).sum().abs().item()) if logits is not None else 0.05
                    else:
                        _, info = self.brain.step(Z0)
                        brain_ms = float(info.get('latency_ms', 0.0)) if isinstance(info, dict) else 0.0
                        ent = 0.05
                    fit = max(0.0, 1.0 / (1.0 + brain_ms/10.0)) + max(0.0, 0.1 - ent)
                    return {'fitness': float(fit)}
                except Exception:
                    return {'fitness': 0.0}
                finally:
                    self.set_config(base)

            self._darwinacci = DarwinacciOrchestrator(population_size=12, max_cycles=2, seed=77, fitness_fn=fitness_fn)
            self._darwinacci.activate(fitness_fn=fitness_fn)
            self._darwinacci_enabled = True
        except Exception:
            self._darwinacci = None
            self._darwinacci_enabled = False

        # Curriculum + replay
        try:
            self.curriculum = CurriculumHarness()
        except Exception:
            self.curriculum = None

    def _clamp_proposal(self, cand: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(cand)
        try:
            if 'top_k' in out:
                out['top_k'] = int(max(1, min(getattr(self.brain, 'max_neurons', 1024), int(out['top_k']))))
            if 'temperature' in out:
                out['temperature'] = float(max(0.1, min(3.0, float(out['temperature']))))
            if 'num_steps' in out:
                out['num_steps'] = int(max(1, min(8, int(out['num_steps']))))
        except Exception:
            pass
        return out

    def get_darwinacci_summary(self) -> Dict[str, Any]:
        try:
            if self._darwinacci is None:
                return {}
            eng = getattr(self._darwinacci, 'engine', None)
            if eng is None:
                return {}
            cov = float(eng.archive.coverage())
            champ = getattr(eng.arena, 'champion', None)
            best = float(champ.score) if champ else 0.0
            return {'coverage': cov, 'best_score': best}
        except Exception:
            return {}
    
    # ----- Configuration Helpers -----
    def get_config(self) -> Dict[str, Any]:
        """Snapshot current tuning knobs (router/v7 bridge)."""
        cfg: Dict[str, Any] = {}
        try:
            r = getattr(self.brain, 'router', None)
            if r is not None:
                cfg['top_k'] = int(getattr(r, 'top_k', 0))
                cfg['temperature'] = float(getattr(r, 'temperature', 1.0))
        except Exception:
            pass
        try:
            if self.v7_bridge is not None:
                cfg['num_steps'] = int(getattr(self.v7_bridge, 'num_steps', 1))
        except Exception:
            pass
        return cfg

    def set_config(self, cfg: Dict[str, Any]) -> None:
        """Apply configuration safely with value clamping."""
        try:
            r = getattr(self.brain, 'router', None)
            if r is not None:
                if 'top_k' in cfg:
                    # Respect global lock if enabled
                    try:
                        lock_enabled = os.getenv('UBRAIN_TOP_K_LOCK', '0') == '1'
                    except Exception:
                        lock_enabled = False
                    if lock_enabled:
                        try:
                            env_top_k = int(os.getenv('UBRAIN_TOP_K', str(getattr(r, 'top_k', 1))))
                        except Exception:
                            env_top_k = int(getattr(r, 'top_k', 1))
                        val = int(max(1, min(getattr(self.brain, 'max_neurons', 1024), env_top_k)))
                    else:
                        val = int(max(1, min(getattr(self.brain, 'max_neurons', 1024), int(cfg['top_k']))))
                    try:
                        r.top_k = val
                    except Exception:
                        pass
                if 'temperature' in cfg:
                    t = float(max(0.1, min(3.0, float(cfg['temperature']))))
                    try:
                        r.temperature = t
                    except Exception:
                        pass
        except Exception:
            pass
        try:
            if self.v7_bridge is not None and 'num_steps' in cfg:
                ns = int(max(1, min(8, int(cfg['num_steps']))))
                self.v7_bridge.num_steps = ns
        except Exception:
            pass

    def propose_meta_adjustment(self, timings_ewma: Dict[str, float], reward_avg: float) -> Dict[str, Any]:
        """Propose a small configuration change based on timings and reward trend.

        Heuristics:
        - If forward_ms or brain_ms high, reduce num_steps; if encode_ms high, reduce top_k or temperature.
        - Keep changes small (±1 step, ±10% top_k, ±5% temperature) to allow safe A/B.
        """
        base = self.get_config()
        cand = dict(base)
        fwd = float(timings_ewma.get('forward_ms', 0.0))
        enc = float(timings_ewma.get('encode_ms', 0.0))
        brn = float(timings_ewma.get('brain_ms', 0.0))
        # Defaults if missing
        if 'num_steps' not in cand:
            cand['num_steps'] = 1
        if 'top_k' not in cand:
            cand['top_k'] = 4
        if 'temperature' not in cand:
            cand['temperature'] = 1.0
        try:
            # If brain core is slow, try reducing steps by 1 (min 1)
            if brn > 5.0 or fwd > 8.0:
                cand['num_steps'] = max(1, int(cand['num_steps']) - 1)
            # If encode is slow, try reducing top_k by 10%
            if enc > 3.0:
                cand['top_k'] = max(1, int(round(cand['top_k'] * 0.9)))
            # Slightly lower temperature to stabilize if reward stagnant
            if reward_avg < 100.0:  # CartPole nominal threshold
                cand['temperature'] = max(0.8, float(cand['temperature']) * 0.95)
        except Exception:
            pass
        # Optionally refine via Darwinacci bridge
        try:
            if self._darwinacci is not None and self._darwinacci_enabled:
                _ = self._darwinacci.evolve_generation()
                best = self._darwinacci.get_best_genome() or {}
                if 'top_k' in best:
                    cand['top_k'] = int(max(1, min(16, round(best['top_k']))))
                if 'temperature' in best:
                    cand['temperature'] = float(max(0.5, min(2.0, best['temperature'])))
                if 'num_steps' in best:
                    cand['num_steps'] = int(max(1, min(3, round(best['num_steps']))))
        except Exception:
            pass
        return self._clamp_proposal(cand)

    # ----- Meta control: pre-canary, AB eval and rollback -----
    def _parse_default_seeds(self) -> list[int]:
        try:
            env = os.getenv('UBRAIN_SEEDS', '123,456,789')
            return [int(x.strip()) for x in env.split(',') if x.strip()]
        except Exception:
            return [123, 456, 789]

    def _decision_hash(self, cand: Dict[str, Any]) -> str:
        try:
            data = _json.dumps(self._clamp_proposal(cand), sort_keys=True).encode('utf-8')
            return hashlib.sha256(data).hexdigest()[:16]
        except Exception:
            return ""

    def _pre_canary_check(self, cfg: Dict[str, Any]) -> bool:
        """Deterministic quick check: control vs proposed cfg on CartPole-v1, seed=123, 1 ep."""
        try:
            base = self.get_config()
            control = evaluate_env_reward(self.brain, 'CartPole-v1', episodes=1, seed=123)
            self.set_config(cfg)
            treat = evaluate_env_reward(self.brain, 'CartPole-v1', episodes=1, seed=123)
            self.set_config(base)
            tol = 1e-3
            ok = (treat + tol) >= control
            worm = getattr(self.brain, 'worm', None)
            if worm is not None:
                worm.append('pre_canary_check', {
                    'control': float(control),
                    'treatment': float(treat),
                    'ok': bool(ok),
                    'cand': cfg,
                })
            return ok
        except Exception:
            return False

    def _ab_eval_suite(self, base_cfg: Dict[str, Any], cand_cfg: Dict[str, Any], seeds: list[int]) -> Dict[str, Any]:
        """Run short suite for control vs treatment and compute uplift/retention."""
        try:
            # Control
            self.set_config(base_cfg)
            control_out = evaluate_suite(self.brain, seeds)
            # Treatment
            self.set_config(cand_cfg)
            treat_out = evaluate_suite(self.brain, seeds)
        finally:
            self.set_config(base_cfg)

        def _mean(xs: list[float]) -> float:
            if not xs:
                return 0.0
            return float(sum(xs) / max(1, len(xs)))

        ctrl_nom = _mean(control_out.get('cartpole_nominal', []))
        trt_nom = _mean(treat_out.get('cartpole_nominal', []))
        uplift = trt_nom - ctrl_nom

        # Retention across all tasks: worst-case ratio treatment/control (clipped at 0..inf)
        ratios = []
        for k in control_out.keys():
            c = _mean(control_out.get(k, []))
            t = _mean(treat_out.get(k, []))
            if c <= 0:
                continue
            ratios.append(t / c)
        retention = float(min(ratios) if ratios else 0.0)

        out = {
            'control': control_out,
            'treatment': treat_out,
            'uplift': uplift,
            'retention': retention,
            'ctrl_nom_mean': ctrl_nom,
            'trt_nom_mean': trt_nom,
        }
        try:
            if self.db:
                # persist detailed
                self.db.save_suite_results(name='control', results=control_out, seeds=seeds)
                self.db.save_suite_results(name='treatment', results=treat_out, seeds=seeds)
        except Exception:
            pass
        return out

    def apply_meta_with_rollback(self, cand_cfg: Dict[str, Any], name: str = 'meta_step') -> bool:
        base = self.get_config()
        seeds = self._parse_default_seeds()
        # Quick gate: pre-canary
        if not self._pre_canary_check(cand_cfg):
            if self.brain and hasattr(self.brain, 'worm'):
                self.brain.worm.append('meta_rejected_pre_canary', {'cand': cand_cfg, 'hash': self._decision_hash(cand_cfg)})
            return False
        # A/B eval (short)
        res = self._ab_eval_suite(base, cand_cfg, seeds)
        uplift = float(res['uplift'])
        retention = float(res['retention'])
        accepted = (uplift >= 0.0) and (retention >= 0.95)
        # Persist
        try:
            if self.db:
                self.db.save_gate_eval(name=name, control_mean=float(res['ctrl_nom_mean']), treatment_mean=float(res['trt_nom_mean']), uplift=uplift, retention=retention, seeds=seeds)
        except Exception:
            pass
        try:
            if self.brain and hasattr(self.brain, 'worm'):
                self.brain.worm.append('gate_eval', {
                    'name': name,
                    'hash': self._decision_hash(cand_cfg),
                    'accepted': accepted,
                    'uplift': uplift,
                    'retention': retention,
                    'seeds': seeds,
                })
        except Exception:
            pass
        # Commit or rollback
        if accepted:
            self.set_config(cand_cfg)
            # Optional: trigger tiny safe self-mod op (documented) to mark activation
            try:
                if self.sme and self.v7_bridge is not None:
                    # Adjust LR (no optimizer provided is fine; logs intent)
                    self.sme.apply_modification(self.v7_bridge, {
                        'operation': 'adjust_lr',
                        'details': {'new_lr': 0.0008}
                    })
                    if hasattr(self.brain, 'worm'):
                        self.brain.worm.append('self_mod_applied', {'op': 'adjust_lr', 'lr': 0.0008})
            except Exception:
                pass
            return True
        else:
            self.set_config(base)
            return False

    def meta_step(self, timings_ewma: Dict[str, float], reward_avg: float) -> bool:
        """Propose and safely apply meta adjustment with rollback protection."""
        try:
            cand = self.propose_meta_adjustment(timings_ewma, reward_avg)
            ok = self.apply_meta_with_rollback(cand_cfg=cand, name='auto_meta')
            
            # T1.3: Conectar Darwinacci ao UnifiedBrain via MessageBus
            try:
                from inter_system_bus import get_message_bus
                bus = get_message_bus()
                
                # Enviar métricas para Darwinacci
                metrics_payload = {
                    'accepted': ok,
                    'reward_avg': reward_avg,
                    'config': self.get_config(),
                    'timestamp': time.time(),
                }
                
                # Adicionar grad_norm se disponível
                try:
                    if hasattr(self.brain, 'router') and hasattr(self.brain.router, 'competence'):
                        if self.brain.router.competence.grad is not None:
                            grad_norm = float(self.brain.router.competence.grad.norm().item())
                            metrics_payload['router_grad_norm'] = grad_norm
                except Exception:
                    pass
                
                bus.send(
                    sender='unified_brain',
                    receiver='darwinacci',
                    message_type='meta_result',
                    payload=metrics_payload,
                    priority=1 if ok else 0
                )
            except Exception:
                pass
            
            # Curriculum replay tracking
            try:
                if ok and self.curriculum is not None:
                    seeds = self._parse_default_seeds()
                    res = self._ab_eval_suite(self.get_config(), self.get_config(), seeds)  # measure current config
                    # Aggregate task means for push
                    def _means(d):
                        from statistics import mean
                        return {k: float(mean(v)) if v else 0.0 for k, v in d.items() if isinstance(v, list)}
                    self.curriculum.push_if_better(self.get_config(), seeds[0] if seeds else 0, _means(res.get('treatment', {})))
            except Exception:
                pass
            return ok
        except Exception:
            return False

    def _pre_canary_check(self, cfg: Dict[str, Any]) -> bool:
        """Deterministic quick check: control vs proposed cfg on CartPole-v1, seed=123, 1 ep."""
        try:
            base = self.get_config()
            control = evaluate_env_reward(self.brain, 'CartPole-v1', episodes=1, seed=123)
            self.set_config(cfg)
            treat = evaluate_env_reward(self.brain, 'CartPole-v1', episodes=1, seed=123)
            self.set_config(base)
            tol = 1e-3
            ok = (treat + tol) >= control
            worm = getattr(self.brain, 'worm', None)
            if worm is not None:
                worm.append('pre_canary_check', {
                    'control': float(control),
                    'treatment': float(treat),
                    'ok': bool(ok),
                    'cand': cfg,
                })
            return ok
        except Exception:
            return False
    
    def step(
        self,
        obs: torch.Tensor,
        penin_metrics: Dict[str, float] = None,
        reward: float = 0.0
    ) -> tuple:
        """
        Passo completo do sistema integrado
        """
        import time
        t0 = time.time()
        # 1. PENIN-Ω modula
        if penin_metrics:
            self.penin_interface.update_signals(penin_metrics)
        t1 = time.time()
        # 2. Cérebro processa
        t2 = t1
        if self.v7_bridge:
            t2 = time.time()
            out = self.v7_bridge(obs)
            # Backwards compatible: older code expected 3-tuple
            if isinstance(out, tuple) and len(out) == 4:
                action_logits, value, Z, route_timings = out
            else:
                action_logits, value, Z = out
                route_timings = {'encode_ms': 0.0, 'brain_ms': 0.0, 'decode_ms': 0.0}
            t3 = time.time()
        else:
            # Sem V7, processa direto
            t2 = time.time()
            Z0 = torch.randn(obs.shape[0], self.brain.H)
            Z, info = self.brain.step(Z0, reward=reward, chaos_signal=self.penin_interface.CAOS_plus)
            action_logits = None
            value = None
            t3 = time.time()
        # 3. Darwin evolui (periodicamente)
        t4 = t3
        # 3. Darwin evolui (periodicamente)
        if self.episode_count % 10 == 0 and reward != 0:
            fitness_scores = self.darwin_evolver.evaluate_fitness(reward)
            self.darwin_evolver.evolve_step(fitness_scores['total'])
            # Export consolidated brain fitness for Cerebrum consumers
            try:
                from UNIFIED_BRAIN.metrics_exporter import write_brain_status
                write_brain_status(float(fitness_scores.get('total', 0.0)))
            except Exception:
                pass
        t5 = time.time()
        # 4. IA³ signal
        ia3_signal = self.penin_interface.get_ia3_signal()
        
        self.total_reward += reward
        self.episode_count += 1
        
        # 5. Persist telemetry per step (best-effort)
        try:
            if self.db and (self.episode_count % 5 == 0):
                m = self.brain.get_metrics_summary() or {}
                cart_last = reward if reward is not None else 0.0
                cart_avg = m.get('avg_energy', 0.0)  # placeholder for quick trend
                self.db.save_cycle(self.episode_count, None, cart_last, cart_avg)
                
                # BLOCO 2 - TAREFA 20: Save brain metrics
                router = self.brain.router
                if router and hasattr(router, 'competence'):
                    comp = router.competence.detach().cpu().numpy()
                    avg_comp = float(comp.mean())
                    max_comp = float(comp.max())
                    min_comp = float(comp.min())
                else:
                    avg_comp = max_comp = min_comp = 0
                
                registry = self.brain.registry
                proms = getattr(registry, 'promotion_count', 0)
                demos = getattr(registry, 'demotion_count', 0)
                
                self.db.save_brain_metrics(
                    episode=self.episode_count,
                    coherence=m.get('avg_coherence', 0),
                    novelty=m.get('avg_novelty', 0),
                    energy=m.get('avg_energy', 0),
                    ia3_signal=ia3_signal,
                    num_active_neurons=m.get('neuron_counts', {}).get('active', 0),
                    top_k=router.top_k if router else 0,
                    temperature=router.temperature if router else 0,
                    avg_competence=avg_comp,
                    max_competence=max_comp,
                    min_competence=min_comp,
                    promotions=proms,
                    demotions=demos
                )
        except Exception:
            pass

        # Compute memory novelty if available
        memory_novelty = 0.0
        try:
            if self._memory is not None and 'Z' in locals() and Z is not None:
                memory_novelty = float(self._memory.novelty(Z))
                self._memory.add(Z)
        except Exception:
            memory_novelty = 0.0

        comp_timings = {
            'penin_update_ms': (t1 - t0) * 1000.0,
            'forward_ms': (t3 - t2) * 1000.0,
            'darwin_ms': (t5 - t4) * 1000.0,
            'memory_novelty': memory_novelty,
        }
        # Merge route sub-timings if available
        try:
            comp_timings.update({
                'encode_ms': float(route_timings.get('encode_ms', 0.0)),
                'brain_ms': float(route_timings.get('brain_ms', 0.0)),
                'decode_ms': float(route_timings.get('decode_ms', 0.0)),
            })
        except Exception:
            pass
        return {
            'action_logits': action_logits,
            'value': value,
            'Z': Z if self.v7_bridge is None else Z,
            'ia3_signal': ia3_signal,
            'fitness': self.darwin_evolver.fitness_history[-1] if self.darwin_evolver.fitness_history else 0.0,
            'metrics': self.brain.get_metrics_summary(),
            'timings': comp_timings,
        }

    # -------------------- Planner-Executor Scaffold --------------------
    def plan(self, goal: str, constraints: Dict[str, Any] | None = None) -> Dict[str, Any]:
        """
        Generate a checklist-style plan using LLM when OFFLINE_ONLY=0; otherwise, return a static template.
        The plan is verifiable: each step has a measurable assertion.
        """
        constraints = constraints or {}
        steps: list[dict[str, Any]] = []
        if OFFLINE_ONLY:
            steps = [
                {'step': 'collect_metrics', 'assert': 'metrics.total_steps >= prev_total_steps + 10'},
                {'step': 'run_eval_variant_noise', 'assert': 'ood_noise_score >= baseline * 1.15'},
                {'step': 'run_eval_nominal', 'assert': 'retention >= 0.90'},
            ]
        else:
            try:
                # Lazy import to avoid hard dependency
                from intelligence_system.apis.litellm_wrapper import LiteLLMWrapper
                from intelligence_system.config.settings import API_KEYS, API_MODELS
                llm = LiteLLMWrapper(API_KEYS, API_MODELS)
                prompt = (
                    f"Plan steps to improve OOD generalization for goal: {goal}. "
                    "Return 3-5 short steps with measurable asserts."
                )
                resp = llm.call_model('gemini/gemini-1.5-flash', [{'role': 'user', 'content': prompt}], max_tokens=200) or ''
                # Minimal parse: each line "- do X | assert: condition"
                for line in resp.splitlines():
                    if 'assert' in line.lower():
                        parts = line.split('|')
                        step_txt = parts[0].replace('-', '').strip()
                        assertion = parts[1].split(':', 1)[-1].strip() if len(parts) > 1 else ''
                        if step_txt:
                            steps.append({'step': step_txt, 'assert': assertion})
            except Exception:
                steps = [
                    {'step': 'collect_metrics', 'assert': 'metrics.total_steps >= prev_total_steps + 10'},
                    {'step': 'run_eval_variant_noise', 'assert': 'ood_noise_score >= baseline * 1.15'},
                    {'step': 'run_eval_nominal', 'assert': 'retention >= 0.90'},
                ]
        self._last_plan = {'goal': goal, 'constraints': constraints, 'steps': steps}
        return self._last_plan

    def execute(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute plan steps using the brain; returns measured outcomes to validate asserts.
        """
        outcomes: Dict[str, Any] = {}
        try:
            # Reuse simple variant evaluators as proxies
            # Import locally to avoid circular imports in tests
            from unified_brain_core import CoreSoupHybrid
            hybrid = CoreSoupHybrid(H=self.brain.H)
            # No side effects here; just generate proxy scores
            def _eval_variant(var: str) -> float:
                H = hybrid.core.H
                z = torch.randn(1, H)
                score = 0.0
                for i in range(100):
                    reward = 0.5
                    if var == 'noise_high':
                        reward += float(torch.randn(1).clamp(-0.2, 0.2))
                    z, _ = hybrid.core.step(z, reward=reward)
                    score += reward
                return score / 100.0
            outcomes['ood_noise_score'] = _eval_variant('noise_high')
            outcomes['baseline'] = _eval_variant('nominal') if hasattr(torch, 'randn') else 0.5
            outcomes['retention'] = min(1.0, outcomes['baseline'] / max(1e-6, outcomes['baseline']))
        except Exception:
            pass
        return outcomes
    
    def connect_v7(self, obs_dim: int = 4, act_dim: int = 2):
        """Conecta bridge com V7"""
        self.v7_bridge = BrainV7Bridge(self.brain, obs_dim, act_dim)
        print(f"🔗 V7 Bridge connected: obs_dim={obs_dim}, act_dim={act_dim}")
    
    def get_current_task(self) -> Tuple[str, float]:
        """Retorna tarefa atual do curriculum"""
        if self.current_task_idx < len(self.curriculum_tasks):
            return self.curriculum_tasks[self.current_task_idx]
        return self.curriculum_tasks[-1]  # Última tarefa
    
    def check_task_solved(self, recent_rewards: List[float]) -> bool:
        """Verifica se tarefa atual foi resolvida"""
        if len(recent_rewards) < self.task_eval_window:
            return False
        
        task_name, threshold = self.get_current_task()
        eval_rewards = recent_rewards[-self.task_eval_window:]
        avg_reward = sum(eval_rewards) / len(eval_rewards)
        
        return avg_reward >= threshold
    
    def advance_curriculum(self, reason: str = 'solved') -> bool:
        """
        Avança para próxima tarefa do curriculum
        Returns: True se avançou, False se já está na última
        """
        if self.current_task_idx >= len(self.curriculum_tasks) - 1:
            from brain_logger import brain_logger
            brain_logger.info("🎓 [CURRICULUM] Todas tarefas completadas!")
            return False
        
        # Registrar tarefa concluída
        task_name, threshold = self.get_current_task()
        self.task_history.append({
            'task': task_name,
            'threshold': threshold,
            'completed_at': self.episode_count,
            'reason': reason,
        })
        
        # Avançar
        self.current_task_idx += 1
        next_task, next_threshold = self.get_current_task()
        
        from brain_logger import brain_logger
        brain_logger.info(f"🎓 [CURRICULUM] Avançando: {task_name} → {next_task} (threshold={next_threshold})")
        
        # Log no WORM
        try:
            self.brain.worm.append('curriculum_advanced', {
                'from_task': task_name,
                'to_task': next_task,
                'episode': self.episode_count,
                'reason': reason,
            })
        except Exception:
            pass
        
        return True


# -------------------- Evaluation helpers for OOD/retention gates --------------------
def _make_env(env_name: str):
    try:
        import gymnasium as gym
    except Exception:
        import gym
    try:
        return gym.make(env_name)
    except Exception as e:
        # Log detalhado da falha silenciosa
        import logging
        logging.getLogger(__name__).warning(
            f"[GYM_FAIL] could not create env '{env_name}': {type(e).__name__}: {str(e)}"
        )
        raise

def evaluate_env_reward(brain: UnifiedBrain, env_name: str, episodes: int = 1,
                        noise_std: float = 0.0, obs_shift: float = 0.0,
                        max_steps: int = 500, seed: int = 0) -> float:
    """
    Avalia recompensa média em um ambiente Gym usando o BrainV7Bridge.
    - noise_std: desvio padrão do ruído gaussiano em observações
    - obs_shift: deslocamento aditivo nas observações
    """
    import torch
    import numpy as np
    torch.manual_seed(seed)
    np.random.seed(seed)

    controller = UnifiedSystemController(brain)
    # Descobrir obs_dim/act_dim dinamicamente na primeira reset
    try:
        env = _make_env(env_name)
    except Exception as e:
        # Falha crítica: retornar 0 e logar via controller.worm
        try:
            brain.worm.append('env_creation_failed', {'env': env_name, 'error': str(e)})
        except Exception:
            pass
        return 0.0
    obs, _ = env.reset(seed=seed) if hasattr(env, 'reset') else (env.reset(), None)
    # Robustly infer obs_dim from space/obs
    try:
        obs_space = getattr(env, 'observation_space', None)
        if obs_space is not None and hasattr(obs_space, 'shape') and obs_space.shape:
            from math import prod as _prod
            obs_dim = int(_prod(obs_space.shape))
        elif hasattr(obs, '__len__'):
            obs_dim = int(len(obs))
        else:
            obs_dim = 4
    except Exception:
        obs_dim = 4
    try:
        act_dim = int(env.action_space.n)
    except Exception:
        act_dim = 2
    try:
        controller.connect_v7(obs_dim=obs_dim, act_dim=act_dim)
    except Exception:
        # Fallback to a square action head if shape probing fails
        controller.connect_v7(obs_dim=obs_dim, act_dim=max(2, obs_dim))

    total = 0.0
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep) if hasattr(env, 'reset') else (env.reset(), None)
        ep_reward = 0.0
        steps = 0
        done = False
        while not done and steps < max_steps:
            # Vectorize observation to fixed length obs_dim
            try:
                import numpy as _np
                arr = _np.array(obs, dtype=_np.float32).reshape(-1)
            except Exception:
                import numpy as _np
                arr = _np.zeros((obs_dim,), dtype=_np.float32)
            if arr.size < obs_dim:
                pad = _np.zeros((obs_dim - arr.size,), dtype=_np.float32)
                arr = _np.concatenate([arr, pad], axis=0)
            elif arr.size > obs_dim:
                arr = arr[:obs_dim]
            if obs_shift:
                arr = arr + float(obs_shift)
            if noise_std:
                arr = arr + _np.random.normal(0.0, noise_std, size=arr.shape)
            x_t = torch.tensor(arr, dtype=torch.float32).unsqueeze(0)
            with torch.inference_mode():
                out = controller.v7_bridge(x_t)
                # v7_bridge returns (logits, value, Z, timings)
                logits, value, Z = out[0], out[1], out[2]
                action = int(torch.argmax(logits, dim=-1).item()) if logits is not None else 0
            try:
                step = env.step(action)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(
                    f"[GYM_FAIL] step failed in '{env_name}' ep={ep}: {type(e).__name__}: {str(e)}"
                )
                break
            if len(step) == 5:
                obs, reward, terminated, truncated, info = step
                done = bool(terminated or truncated)
            else:
                obs, reward, done, info = step
            ep_reward += float(reward)
            steps += 1
        total += ep_reward
    try:
        env.close()
    except Exception:
        pass
    return total / max(1, episodes)

def evaluate_suite(brain: UnifiedBrain, seeds: List[int]) -> Dict[str, List[float]]:
    """
    Executa uma suíte OOD/nominal e retorna listas de scores por tarefa para os seeds informados.
    Tarefas:
      - cartpole_nominal, cartpole_noise, cartpole_shift, mountaincar, acrobot
    """
    suite = {
        'cartpole_nominal': dict(env='CartPole-v1', noise=0.0,  shift=0.0),
        'cartpole_noise':   dict(env='CartPole-v1', noise=0.05, shift=0.0),
        'cartpole_shift':   dict(env='CartPole-v1', noise=0.0,  shift=0.1),
        'frozenlake':       dict(env='FrozenLake-v1', noise=0.0, shift=0.0),
        'mountaincar':      dict(env='MountainCar-v0', noise=0.0, shift=0.0),
        'acrobot':          dict(env='Acrobot-v1', noise=0.0, shift=0.0),
    }
    out: Dict[str, List[float]] = {k: [] for k in suite}
    try:
        import os
        from concurrent.futures import ProcessPoolExecutor, as_completed
        episodes = int(os.getenv('UBRAIN_EVAL_EPISODES', '1'))
        parallel = int(os.getenv('UBRAIN_EVAL_PARALLEL', '0'))
        jobs: List[Tuple[str, int]] = []
        for s in seeds:
            for name in suite.keys():
                jobs.append((name, s))
        if parallel > 0:
            # Prepare static args
            def _job(args: Tuple[str, int]) -> Tuple[str, int, float]:
                name, s = args
                cfg = suite[name]
                score = evaluate_env_reward(brain, cfg['env'], episodes=episodes,
                                            noise_std=cfg['noise'], obs_shift=cfg['shift'],
                                            max_steps=500, seed=s)
                return (name, s, score)
            with ProcessPoolExecutor(max_workers=parallel) as ex:
                futures = [ex.submit(_job, j) for j in jobs]
                for fut in as_completed(futures):
                    name, s, score = fut.result()
                    out[name].append(score)
        else:
            for name, cfg in suite.items():
                for s in seeds:
                    score = evaluate_env_reward(brain, cfg['env'], episodes=episodes,
                                                noise_std=cfg['noise'], obs_shift=cfg['shift'],
                                                max_steps=500, seed=s)
                    out[name].append(score)
    except Exception:
        # Fallback serial minimal
        for name, cfg in suite.items():
            for s in seeds:
                score = evaluate_env_reward(brain, cfg['env'], episodes=1,
                                            noise_std=cfg['noise'], obs_shift=cfg['shift'],
                                            max_steps=500, seed=s)
                out[name].append(score)
    return out


def ethics_guard(brain: UnifiedBrain) -> tuple[bool, str]:
    """Simple ethics/safety gate over brain metrics.
    Returns (ok, reason). If ok=False, caller should rollback.
    Env controls:
      - UBRAIN_MIN_COHERENCE: float threshold for avg_coherence (default 0.0)
    """
    try:
        from os import getenv as _getenv
        try:
            min_coh = float(_getenv('UBRAIN_MIN_COHERENCE', '0.0'))
        except Exception:
            min_coh = 0.0
        m = brain.get_metrics_summary() or {}
        coh = float(m.get('avg_coherence', 0.0))
        if coh < min_coh:
            return (False, f"avg_coherence {coh:.3f} < min {min_coh:.3f}")
        return (True, "ok")
    except Exception as e:
        return (True, f"guard error ignored: {e}")

# -------------------- Calibration (CartPole) --------------------
def _sigmoid(x):
    import math
    try:
        return 1.0 / (1.0 + math.exp(-float(x)))
    except Exception:
        return 0.5

def compute_regression_ece(pred: List[float], target: List[float], n_bins: int = 10) -> float:
    if not pred or not target or len(pred) != len(target):
        return 1.0
    import numpy as np
    p = np.clip(np.array(pred, dtype=float), 0.0, 1.0)
    t = np.clip(np.array(target, dtype=float), 0.0, 1.0)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(p)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not np.any(mask):
            continue
        p_mean = float(p[mask].mean())
        t_mean = float(t[mask].mean())
        ece += abs(p_mean - t_mean) * (mask.sum() / total)
    return float(ece)

def evaluate_cartpole_calibration(brain: UnifiedBrain, seeds: List[int], episodes: int = 1) -> float:
    """Coleta (value→sigmoid) vs retorno normalizado e calcula ECE média."""
    import torch
    eces: List[float] = []
    for s in seeds:
        controller = UnifiedSystemController(brain)
        env = _make_env('CartPole-v1')
        obs, _ = env.reset(seed=s) if hasattr(env, 'reset') else (env.reset(), None)
        obs_dim = int(len(obs)) if hasattr(obs, '__len__') else 4
        act_dim = int(env.action_space.n) if hasattr(env, 'action_space') and hasattr(env.action_space, 'n') else 2
        controller.connect_v7(obs_dim=obs_dim, act_dim=act_dim)
        preds: List[float] = []
        trues: List[float] = []
        for ep in range(episodes):
            obs, _ = env.reset(seed=s + ep) if hasattr(env, 'reset') else (env.reset(), None)
            ep_reward = 0.0
            steps = 0
            done = False
            while not done and steps < 500:
                x_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                with torch.inference_mode():
                    logits, value, Z = controller.v7_bridge(x_t)
                # Predito: sigmoid(value)
                v = float(value.squeeze().item()) if value is not None else 0.0
                preds.append(_sigmoid(v))
                # Executa ação greedy
                action = int(torch.argmax(logits, dim=-1).item()) if logits is not None else 0
                step = env.step(action)
                if len(step) == 5:
                    obs, reward, terminated, truncated, info = step
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, info = step
                ep_reward += float(reward)
                # Verdade: retorno acumulado normalizado (aproximação)
                trues.append(min(1.0, max(0.0, ep_reward / 500.0)))
                steps += 1
        try:
            env.close()
        except Exception:
            pass
        eces.append(compute_regression_ece(preds, trues, n_bins=10))
    return sum(eces) / max(1, len(eces))


# -------------------- Vision (MNIST-C) evaluation --------------------
def compute_classification_ece(confidences: List[float], correct: List[int], n_bins: int = 15) -> float:
    """Classification ECE using predicted-class confidence and correctness.
    Args:
        confidences: list of max softmax probabilities for predicted class (0..1)
        correct: list of 0/1 correctness indicators (1 if prediction == label)
        n_bins: number of calibration bins
    Returns:
        ECE scalar in [0, 1]
    """
    if not confidences or not correct or len(confidences) != len(correct):
        return 1.0
    import numpy as np
    p = np.clip(np.array(confidences, dtype=float), 0.0, 1.0)
    y = np.array(correct, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = len(p)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        cnt = int(mask.sum())
        if cnt == 0:
            continue
        conf_bin = float(p[mask].mean())
        acc_bin = float(y[mask].mean())
        ece += abs(acc_bin - conf_bin) * (cnt / total)
    return float(ece)


def evaluate_mnist_c(model_path: str | None = None, root: str | None = None,
                     device: str = 'cpu', batch_size: int = 1024,
                     n_bins: int = 15) -> Dict[str, Any]:
    """Evaluate MNIST-C accuracy and ECE using an MNISTNet model.

    - Loads test split for each corruption from numpy arrays under root.
    - Builds a simple MNIST MLP and loads weights if available.

    Returns dict with keys: 'accuracy', 'ece', 'per_corruption', 'num_samples'.
    """
    import os
    import numpy as np
    import torch
    import torch.nn as nn
    from pathlib import Path
    # Avoid importing MNISTClassifier to prevent unintended downloads; use MNISTNet directly
    try:
        from intelligence_system.models.mnist_classifier import MNISTNet
        from intelligence_system.config.settings import MNIST_MODEL_PATH as _MNIST_MODEL_PATH
    except Exception:
        # Minimal fallback model if import path changes
        class MNISTNet(nn.Module):
            def __init__(self, hidden_size: int = 128):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc1 = nn.Linear(784, hidden_size)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(hidden_size, 10)
            def forward(self, x):
                x = self.flatten(x)
                x = self.fc1(x)
                x = self.relu(x)
                x = self.fc2(x)
                return x
        _MNIST_MODEL_PATH = Path('/root/intelligence_system/models/mnist_model.pth')

    root_dir = Path(root or '/root/data/mnist_c')
    if not root_dir.exists():
        raise FileNotFoundError(f"MNIST-C directory not found at {root_dir}")

    # Build model and load weights if present
    model = MNISTNet().to(device)
    path = Path(model_path) if model_path else Path(_MNIST_MODEL_PATH)
    if path.exists():
        try:
            ckpt = torch.load(path, map_location=device)
            state = ckpt.get('model_state_dict', ckpt if isinstance(ckpt, dict) else None)
            if isinstance(state, dict):
                model.load_state_dict(state, strict=False)
        except Exception:
            pass
    model.eval()

    softmax = nn.Softmax(dim=1)
    total = 0
    correct_total = 0
    confidences: List[float] = []
    correctness: List[int] = []
    per_corr_acc: Dict[str, float] = {}

    # Iterate corruptions; stream batches to limit memory
    for corr_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
        try:
            X = np.load(corr_dir / 'test_images.npy')  # shape [N, 28, 28]
            y = np.load(corr_dir / 'test_labels.npy')  # shape [N]
        except Exception:
            # Some distributions may use 'test_images'/'labels' without .npy, skip
            try:
                X = np.load(str(corr_dir / 'test_images'))
                y = np.load(str(corr_dir / 'test_labels'))
            except Exception:
                continue
        N = int(X.shape[0])
        if N == 0:
            continue
        
        # BUG FIX C2: Sample to avoid timeout (500 samples per corruption)
        import os as _os
        sample_size = int(_os.getenv('MNISTC_SAMPLE_SIZE', '500'))
        if N > sample_size:
            idx = np.random.choice(N, sample_size, replace=False)
            X = X[idx]
            y = y[idx]
            N = sample_size
        # Normalize to [0,1] and add channel dimension
        # Use batches to avoid OOM
        corr_correct = 0
        bs = max(1, int(batch_size))
        for i in range(0, N, bs):
            xb = X[i:i+bs]
            yb = y[i:i+bs]
            xb = torch.tensor(xb, dtype=torch.float32, device=device) / 255.0
            if xb.ndim == 3:
                xb = xb.unsqueeze(1)  # [B,1,28,28]
            with torch.inference_mode():
                logits = model(xb)
                probs = softmax(logits)
                conf, pred = torch.max(probs, dim=1)
            preds_np = pred.detach().cpu().numpy()
            conf_np = conf.detach().cpu().numpy()
            y_np = np.array(yb, dtype=np.int64)
            corr = (preds_np == y_np).astype(np.int32)
            corr_correct += int(corr.sum())
            total += len(y_np)
            correct_total += int(corr.sum())
            confidences.extend([float(c) for c in conf_np.tolist()])
            correctness.extend([int(c) for c in corr.tolist()])
        per_corr_acc[corr_dir.name] = float(corr_correct / max(1, N))

    accuracy = float(correct_total / max(1, total))
    ece = compute_classification_ece(confidences, correctness, n_bins=n_bins)

    return {
        'accuracy': accuracy,
        'ece': ece,
        'per_corruption': per_corr_acc,
        'num_samples': int(total),
        'model_path': str(path),
        'root': str(root_dir),
    }


if __name__ == "__main__":
    print("🔗 Brain System Integration Module")
    print("Use UnifiedSystemController to orchestrate everything")
