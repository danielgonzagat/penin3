#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω — Código 2/8: Módulo Estratégico Ω-META (Versão Fusão Definitiva)
=========================================================================
Transforma intenções em Planos Ω-META viáveis, restritos por Ética→Risco→Performance.
Integra SR-Ω∞ não-compensatório, trust-region adaptativo e provas WORM completas.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Literal
import hashlib
import json
import math
import threading
import time
import random
from copy import deepcopy
from functools import lru_cache
from collections import deque
import warnings
import uuid

# =============================================================================
# Integração com Módulo 1/8 (com fallback completo)
# =============================================================================

try:
    from penin_omega_1_core import (
        OmegaState, WORMLedger, PENINMotores,
        SigmaGuard, IRIC, CAOSPlus, SROmegaInfinity,
        EquacaoDaMorte, PeninUpdate, League,
        GOVERNANCE as CORE_GOVERNANCE
    )
    CORE_INTEGRATION = True
except ImportError:
    CORE_INTEGRATION = False
    
    # Fallback definitions para operação standalone
    @dataclass
    class OmegaState:
        """Mock do estado Omega para testes standalone."""
        # Componentes SR
        E_ok: float = 1.0
        M: float = 0.5
        C: float = 0.5
        A: float = 0.5
        
        # Métricas éticas
        ece: float = 0.01
        rho_bias: float = 1.0
        fairness: float = 1.0
        consent: bool = True
        eco_ok: bool = True
        
        # Métricas de risco
        rho: float = 0.5
        uncertainty: float = 0.3
        volatility: float = 0.2
        
        # Performance
        delta_linf: float = 0.01
        mdl_gain: float = 0.02
        ppl_ood: float = 100.0
        efficiency: float = 0.7
        
        # CAOS⁺
        caos_pre: float = 1.0
        caos_post: float = 1.0
        caos_stable: bool = True
        
        # Autonomia
        self_improvement: float = 0.5
        exploration: float = 0.5
        adaptation: float = 0.5
        learning_rate: float = 0.001
        
        # Estado geral
        sr_score: float = 1.0
        trust_region_radius: float = 0.1
        cycle_count: int = 0
        timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
        version: str = "2.8.fusion"
        
        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)
    
    class WORMLedger:
        """Mock do WORM Ledger para testes."""
        def __init__(self, path: Optional[str] = None):
            self.events = []
            
        def record_event(self, event_type: str, data: Dict[str, Any]) -> str:
            event = {"type": event_type, "data": data, "ts": datetime.now(timezone.utc).isoformat()}
            self.events.append(event)
            return hashlib.sha256(json.dumps(event).encode()).hexdigest()
    
    CORE_GOVERNANCE = {}

# =============================================================================
# IMPORTS BASE SEM CICLOS
# =============================================================================

from penin_omega_utils import _ts, _hash_data, log, BaseConfig, BaseWORMLedger, LAZY_IMPORTER

# =============================================================================
# INTEGRAÇÃO COM CONFIGURAÇÃO UNIFICADA - SEM CICLOS
# =============================================================================

def get_unified_config():
    """Obtém configuração unificada via lazy import"""
    unified_interface = LAZY_IMPORTER.get_module('penin_omega_unified_interface')
    if unified_interface and hasattr(unified_interface, 'UNIFIED_CONFIG'):
        return unified_interface.UNIFIED_CONFIG
    return None

USE_UNIFIED_CONFIG = get_unified_config() is not None

# =============================================================================
# Configuração Master Fusionada
# =============================================================================

OMEGA_META_CONFIG: Dict[str, Any] = {
    "version": BaseConfig.VERSION,  # Usa versão do módulo base
    "compatibility": "1.8.core",
    
    "sr_omega": {
        "tau_SR": 0.80,
        "weights": {"E": 0.40, "M": 0.30, "C": 0.20, "A": 0.10},
        "min_components": 3,
        "gate_mode": "strict",
        "cache_ttl": 60,
        "harmonic_epsilon": 1e-6,
        "confidence_bands": True,
    },
    
    "trust_region": {
        "initial_radius": 0.10,
        "min_radius": 0.02,
        "max_radius": 0.50,
        "shrink_factor": 0.90,
        "grow_factor": 1.10,
        "adaptive": True,
        "min_improvement": 0.01,
        "momentum": 0.9,
        "history_size": 10,
        "volatility_threshold": 0.05,
    },
    
    "ethics": {
        "ece_max": 0.01,
        "rho_bias_max": 1.05,
        "fairness_min": 0.95,
        "consent_required": True,
        "eco_ok_required": True,
        "priority_weight": 10.0,
        "violation_tolerance": 0.0,
    },
    
    "risk": {
        "rho_max": 0.95,
        "uncertainty_max": 0.30,
        "volatility_max": 0.25,
        "contraction_factor": 0.98,
        "priority_weight": 5.0,
        "cbf_active": True,
        "safe_set_margin": 0.05,
    },
    
    "performance": {
        "delta_linf_min": 0.01,
        "improvement_target": 0.05,
        "ppl_ood_target": 90.0,
        "efficiency_min": 0.70,
        "priority_weight": 1.0,
        "optimization_rounds": 3,
    },
    
    "budgets": {
        "max_tokens": 100000,
        "max_cost": 10.0,
        "max_latency_ms": 5000,
        "max_llm_calls": 100,
        "max_memory_mb": 1024,
        "quota_local": 0.8,
        "reserve_ratio": 0.1,
        "emergency_reserve": 0.05,
    },
    
    "deliberation": {
        "max_candidates": 10,
        "min_viability": 0.60,
        "consensus_threshold": 0.75,
        "exploration_bonus": 0.10,
        "seed": 42,
        "timeout_ms": 400,
        "parallel_evaluation": False,
    },
    
    "u_signal": {
        "lambda_U": 0.5,
        "kappa": 1.5,
        "budget_cap_factor": 0.1,
        "sigmoid_steepness": 2.0,
        "caos_coupling": True,
    },
    
    "worm": {
        "enabled": True,
        "hash_algorithm": "sha256",
        "proof_depth": 3,
        "retention_cycles": 1000,
        "batch_size": 10,
        "compression": True,
    },
    
    "llm": {
        "enabled": True,
        "base_url": "http://localhost:8010",
        "model": "falcon-mamba-7b",
        "max_tokens": 100,
        "temperature": 0.3,
        "timeout_ms": 200,
        "fallback_deterministic": True,
    },
    
    "telemetry": {
        "enabled": True,
        "sample_rate": 1.0,
        "metrics_window": 100,
        "export_interval": 60,
    },
    
    "integration": {
        "core_module": "penin_omega_1_core",
        "scheduler_module": "penin_omega_7_scheduler",
        "sync_mode": "async",
        "heartbeat_ms": 1000,
    },
}

# =============================================================================
# Eventos e Tipos Definitivos
# =============================================================================

class StrategyEvent(str, Enum):
    """Eventos estratégicos para WORM."""
    STRATEGY_START = "STRATEGY_START"
    STRATEGY_DECISION = "STRATEGY_DECISION"
    STRATEGY_ABORT = "STRATEGY_ABORT"
    STRATEGY_GATE_FAIL = "STRATEGY_GATE_FAIL"
    CONSTRAINT_VIOLATION = "CONSTRAINT_VIOLATION"
    BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
    PLAN_CREATED = "PLAN_CREATED"
    PLAN_VALIDATED = "PLAN_VALIDATED"
    PLAN_PROMOTED = "PLAN_PROMOTED"
    STRATEGY_ROLLBACK = "STRATEGY_ROLLBACK"
    GOAL_CREATED = "GOAL_CREATED"
    GOAL_ACHIEVED = "GOAL_ACHIEVED"
    GOAL_FAILED = "GOAL_FAILED"
    CACHE_HIT = "CACHE_HIT"
    CACHE_MISS = "CACHE_MISS"
    PERFORMANCE_METRIC = "PERFORMANCE_METRIC"
    SYNC_WITH_CORE = "SYNC_WITH_CORE"
    SYNC_WITH_SCHEDULER = "SYNC_WITH_SCHEDULER"

@dataclass
class Goal:
    """Objetivo estratégico completo com rastreamento."""
    id: str = field(default_factory=lambda: f"goal_{uuid.uuid4().hex[:8]}")
    name: str = ""
    description: str = ""
    metric: str = ""
    target: float = 0.0
    tolerance: float = 0.05
    confidence_interval: Tuple[float, float] = (0.95, 1.05)
    deadline: int = 10
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    priority: float = 1.0
    lexicographic_level: int = 3
    status: Literal["pending", "active", "achieved", "failed", "cancelled"] = "pending"
    progress: float = 0.0
    owner: str = "2/8"
    dependencies: List[str] = field(default_factory=list)
    conflicts: List[str] = field(default_factory=list)
    viability_score: float = 1.0
    risk_assessment: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def is_achieved(self, current: float) -> bool:
        return abs(current - self.target) <= self.tolerance
    
    def is_expired(self, current_cycle: int, creation_cycle: int) -> bool:
        return (current_cycle - creation_cycle) > self.deadline
    
    def update_progress(self, initial: float, current: float) -> None:
        if self.target != initial:
            self.progress = abs(current - initial) / abs(self.target - initial)
        self.progress = max(0.0, min(1.0, self.progress))

@dataclass
class Constraints:
    """Restrições com validação e projeção segura."""
    ece_max: float = 0.01
    rho_bias_max: float = 1.05
    fairness_min: float = 0.95
    consent_required: bool = True
    eco_ok_required: bool = True
    rho_max: float = 0.95
    uncertainty_max: float = 0.30
    volatility_max: float = 0.25
    cbf_margin: float = 0.05
    delta_linf_min: float = 0.01
    ppl_ood_max: float = 100.0
    efficiency_min: float = 0.70
    trust_region_radius_proposed: float = 0.10
    trust_region_radius_current: float = 0.10
    validation_timestamp: Optional[str] = None
    projection_applied: bool = False
    
    def validate(self, state: Dict[str, Any]) -> Tuple[bool, List[str]]:
        violations = []
        
        # Nível 1: Ética
        if state.get("ece", 0) > self.ece_max:
            violations.append(f"ETHICS: ECE {state['ece']:.4f} > {self.ece_max}")
        if state.get("rho_bias", 1) > self.rho_bias_max:
            violations.append(f"ETHICS: ρ_bias {state['rho_bias']:.4f} > {self.rho_bias_max}")
        if state.get("fairness", 1) < self.fairness_min:
            violations.append(f"ETHICS: fairness {state['fairness']:.4f} < {self.fairness_min}")
        if self.consent_required and not state.get("consent", True):
            violations.append("ETHICS: consent required but not given")
        if self.eco_ok_required and not state.get("eco_ok", True):
            violations.append("ETHICS: eco_ok required but not satisfied")
        
        if violations:
            self.validation_timestamp = datetime.now(timezone.utc).isoformat()
            return False, violations
        
        # Nível 2: Risco
        if state.get("rho", 0) > self.rho_max:
            violations.append(f"RISK: ρ {state['rho']:.4f} > {self.rho_max}")
        if state.get("uncertainty", 0) > self.uncertainty_max:
            violations.append(f"RISK: uncertainty {state['uncertainty']:.4f} > {self.uncertainty_max}")
        if state.get("volatility", 0) > self.volatility_max:
            violations.append(f"RISK: volatility {state['volatility']:.4f} > {self.volatility_max}")
        
        # Nível 3: Performance
        if not violations:
            if state.get("delta_linf", 0) < self.delta_linf_min:
                violations.append(f"PERF: ΔL∞ {state['delta_linf']:.4f} < {self.delta_linf_min}")
            if state.get("ppl_ood", 100) > self.ppl_ood_max:
                violations.append(f"PERF: PPL_OOD {state['ppl_ood']:.1f} > {self.ppl_ood_max}")
            if state.get("efficiency", 1) < self.efficiency_min:
                violations.append(f"PERF: efficiency {state['efficiency']:.3f} < {self.efficiency_min}")
        
        self.validation_timestamp = datetime.now(timezone.utc).isoformat()
        return len(violations) == 0, violations
    
    def project_safe(self, state: Dict[str, Any]) -> Dict[str, Any]:
        projected = deepcopy(state)
        projected["ece"] = min(state.get("ece", 0), self.ece_max * 0.95)
        projected["rho_bias"] = min(state.get("rho_bias", 1), self.rho_bias_max * 0.99)
        projected["fairness"] = max(state.get("fairness", 1), self.fairness_min)
        projected["consent"] = True if self.consent_required else state.get("consent", True)
        projected["eco_ok"] = True if self.eco_ok_required else state.get("eco_ok", True)
        projected["rho"] = min(state.get("rho", 0), self.rho_max - self.cbf_margin)
        projected["uncertainty"] = min(state.get("uncertainty", 0), self.uncertainty_max * 0.95)
        projected["volatility"] = min(state.get("volatility", 0), self.volatility_max * 0.95)
        projected["delta_linf"] = max(state.get("delta_linf", 0), self.delta_linf_min)
        projected["ppl_ood"] = min(state.get("ppl_ood", 100), self.ppl_ood_max)
        projected["efficiency"] = max(state.get("efficiency", 0.7), self.efficiency_min)
        self.projection_applied = True
        return projected
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class Budgets:
    """Orçamentos com rastreamento detalhado."""
    max_tokens: int = 100000
    max_cost: float = 10.0
    max_latency_ms: int = 5000
    max_llm_calls: int = 100
    max_memory_mb: int = 1024
    quota_local: float = 0.8
    reserve_ratio: float = 0.1
    emergency_reserve: float = 0.05
    used_tokens: int = 0
    used_cost: float = 0.0
    used_llm_calls: int = 0
    used_memory_mb: int = 0
    used_time_ms: int = 0
    reserved_tokens: int = 0
    reserved_cost: float = 0.0
    reserved_memory_mb: int = 0
    allocation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        self.reserved_tokens = int(self.max_tokens * self.reserve_ratio)
        self.reserved_cost = self.max_cost * self.reserve_ratio
        self.reserved_memory_mb = int(self.max_memory_mb * self.reserve_ratio)
    
    def remaining(self, include_reserves: bool = False) -> Dict[str, float]:
        base_remaining = {
            "tokens": self.max_tokens - self.used_tokens,
            "cost": self.max_cost - self.used_cost,
            "llm_calls": self.max_llm_calls - self.used_llm_calls,
            "memory_mb": self.max_memory_mb - self.used_memory_mb,
            "time_ms": self.max_latency_ms - self.used_time_ms,
        }
        
        if not include_reserves:
            base_remaining["tokens"] -= self.reserved_tokens
            base_remaining["cost"] -= self.reserved_cost
            base_remaining["memory_mb"] -= self.reserved_memory_mb
        
        return {k: max(0, v) for k, v in base_remaining.items()}
    
    def can_afford(self, required: Dict[str, float], safety_margin: float = 1.1) -> bool:
        rem = self.remaining()
        return all(rem.get(k, 0) >= v * safety_margin for k, v in required.items())
    
    def allocate(self, amount: Dict[str, float], purpose: str = "") -> bool:
        if not self.can_afford(amount):
            return False
        
        self.used_tokens += int(amount.get("tokens", 0))
        self.used_cost += amount.get("cost", 0)
        self.used_llm_calls += int(amount.get("llm_calls", 0))
        self.used_memory_mb += int(amount.get("memory_mb", 0))
        self.used_time_ms += int(amount.get("time_ms", 0))
        
        self.allocation_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "purpose": purpose,
            "amount": amount,
            "remaining_after": self.remaining(),
        })
        
        return True
    
    def get_usage_ratio(self) -> float:
        ratios = [
            self.used_tokens / max(1, self.max_tokens),
            self.used_cost / max(0.01, self.max_cost),
            self.used_llm_calls / max(1, self.max_llm_calls),
        ]
        return sum(ratios) / len(ratios)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class PlanOmega:
    """Plano Ω-META completo e auditável."""
    id: str
    timestamp: str
    cycle: int
    goals: List[Goal] = field(default_factory=list)
    constraints: Constraints = field(default_factory=Constraints)
    budgets: Budgets = field(default_factory=Budgets)
    priority_map: Dict[str, float] = field(default_factory=dict)
    promotion_policy: Dict[str, Any] = field(default_factory=dict)
    rollback_policy: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    confidence: float = 0.0
    sr_score: float = 0.0
    u_signal: float = 0.0
    input_hash: str = ""
    plan_hash: str = ""
    signature: Optional[str] = None
    parent_plan_id: Optional[str] = None
    generation_time_ms: float = 0.0
    validation_results: Dict[str, Any] = field(default_factory=dict)
    status: Literal["draft", "validated", "active", "completed", "rolled_back"] = "draft"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "cycle": self.cycle,
            "goals": [g.to_dict() for g in self.goals],
            "constraints": self.constraints.to_dict(),
            "budgets": self.budgets.to_dict(),
            "priority_map": self.priority_map,
            "promotion_policy": self.promotion_policy,
            "rollback_policy": self.rollback_policy,
            "rationale": self.rationale,
            "confidence": self.confidence,
            "sr_score": self.sr_score,
            "u_signal": self.u_signal,
            "input_hash": self.input_hash,
            "plan_hash": self.plan_hash,
            "signature": self.signature,
            "parent_plan_id": self.parent_plan_id,
            "generation_time_ms": self.generation_time_ms,
            "validation_results": self.validation_results,
            "status": self.status,
        }
    
    def compute_hash(self) -> str:
        content = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def sign(self, key: Optional[str] = None) -> str:
        if key:
            signature_content = f"{self.plan_hash}:{key}:{self.timestamp}"
            self.signature = hashlib.sha512(signature_content.encode()).hexdigest()
        else:
            self.signature = self.plan_hash
        return self.signature

@dataclass
class SRReport:
    """Relatório SR-Ω∞ detalhado."""
    sr_score: float
    components: Dict[str, float]
    tau_SR: float
    valid: bool
    decision: Literal["ALLOW", "CEDE", "ABORT"]
    confidence_interval: Tuple[float, float] = (0.0, 1.0)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    computation_time_ms: float = 0.0
    cache_hit: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# =============================================================================
# Cache e Performance
# =============================================================================

class AdaptiveCache:
    """Cache adaptativo com TTL dinâmico."""
    
    def __init__(self, base_ttl: int = 60, max_size: int = 1000):
        self.base_ttl = base_ttl
        self.max_size = max_size
        self.cache: Dict[str, Tuple[Any, float, int]] = {}
        self.lock = threading.RLock()
        self.stats = {"hits": 0, "misses": 0, "evictions": 0}
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                value, timestamp, hits = self.cache[key]
                adaptive_ttl = self.base_ttl * (1 + math.log(1 + hits))
                
                if time.time() - timestamp < adaptive_ttl:
                    self.cache[key] = (value, timestamp, hits + 1)
                    self.stats["hits"] += 1
                    return deepcopy(value)
                else:
                    del self.cache[key]
                    self.stats["evictions"] += 1
            
            self.stats["misses"] += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        with self.lock:
            if len(self.cache) >= self.max_size:
                lru_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
                del self.cache[lru_key]
                self.stats["evictions"] += 1
            
            self.cache[key] = (deepcopy(value), time.time(), 0)
    
    def get_stats(self) -> Dict[str, Any]:
        with self.lock:
            total = self.stats["hits"] + self.stats["misses"]
            return {
                **self.stats,
                "hit_rate": self.stats["hits"] / max(1, total),
                "size": len(self.cache),
            }

# =============================================================================
# Motor SR-Ω∞ Fusionado
# =============================================================================

class StrategicSROmegaFusion:
    """SR-Ω∞ definitivo com integração completa."""
    
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg["sr_omega"]
        self.weights = self.cfg["weights"]
        self.tau_SR = self.cfg["tau_SR"]
        self.min_components = self.cfg["min_components"]
        self.gate_mode = self.cfg["gate_mode"]
        self.epsilon = self.cfg["harmonic_epsilon"]
        self.cache = AdaptiveCache(self.cfg["cache_ttl"])
        
        self.external_sr = None
        if CORE_INTEGRATION:
            try:
                self.external_sr = SROmegaInfinity(cfg)
            except:
                pass
    
    def compute(self, state: Union[Dict[str, Any], OmegaState]) -> Tuple[float, SRReport]:
        start_time = time.time()
        
        if isinstance(state, OmegaState):
            state_dict = state.to_dict()
        else:
            state_dict = state
        
        cache_key = self._compute_cache_key(state_dict)
        cached = self.cache.get(cache_key)
        if cached:
            cached[1].cache_hit = True
            return cached
        
        if self.external_sr and CORE_INTEGRATION:
            sr_score = self.external_sr.compute(state_dict)
            components = self._extract_components(state_dict)
        else:
            components = self._compute_components(state_dict)
            sr_score = self._harmonic_mean(components)
        
        confidence_interval = self._compute_confidence_interval(sr_score, components)
        
        report = SRReport(
            sr_score=sr_score,
            components=components,
            tau_SR=self.tau_SR,
            valid=sr_score >= self.tau_SR,
            decision=self._decide(sr_score),
            confidence_interval=confidence_interval,
            computation_time_ms=(time.time() - start_time) * 1000,
            cache_hit=False
        )
        
        result = (sr_score, report)
        self.cache.set(cache_key, result)
        return result
    
    def gate(self, state: Union[Dict[str, Any], OmegaState]) -> bool:
        if self.gate_mode == "bypass":
            return True
        
        sr_score, _ = self.compute(state)
        
        if self.gate_mode == "relaxed":
            return sr_score >= (self.tau_SR * 0.8)
        
        return sr_score >= self.tau_SR
    
    def _compute_components(self, state: Dict[str, Any]) -> Dict[str, float]:
        return {
            "E": self._ethics(state),
            "M": self._mastery(state),
            "C": self._calibration(state),
            "A": self._autonomy(state),
        }
    
    def _extract_components(self, state: Dict[str, Any]) -> Dict[str, float]:
        return {
            "E": max(self.epsilon, state.get("E_ok", state.get("E", 1.0))),
            "M": max(self.epsilon, state.get("M", 0.5)),
            "C": max(self.epsilon, state.get("C", 0.5)),
            "A": max(self.epsilon, state.get("A", 0.5)),
        }
    
    def _harmonic_mean(self, components: Dict[str, float]) -> float:
        valid_count = sum(1 for v in components.values() if v > self.epsilon)
        
        if valid_count < self.min_components:
            return 0.0
        
        weighted_sum = sum(
            self.weights[k] / max(self.epsilon, v)
            for k, v in components.items()
        )
        
        total_weight = sum(self.weights.values())
        
        if weighted_sum > 0:
            return total_weight / weighted_sum
        return 0.0
    
    def _ethics(self, state: Dict[str, Any]) -> float:
        score = 1.0
        score *= max(0.1, 1.0 - state.get("ece", 0) * 100)
        score *= max(0.1, 2.0 - state.get("rho_bias", 1))
        score *= max(0.1, state.get("fairness", 1.0))
        score *= 1.0 if state.get("consent", True) else 0.2
        score *= 1.0 if state.get("eco_ok", True) else 0.3
        return max(self.epsilon, score)
    
    def _mastery(self, state: Dict[str, Any]) -> float:
        score = 1.0
        score *= max(0.1, min(1.0, state.get("delta_linf", 0) * 100))
        score *= max(0.1, min(1.0, state.get("mdl_gain", 0) * 50))
        ppl_ood = state.get("ppl_ood", 100)
        score *= max(0.1, min(1.0, 100.0 / (ppl_ood + 1)))
        score *= max(0.1, state.get("efficiency", 0.7))
        return max(self.epsilon, score)
    
    def _calibration(self, state: Dict[str, Any]) -> float:
        score = 1.0
        score *= max(0.1, 1.0 - state.get("uncertainty", 0.5))
        score *= max(0.1, state.get("confidence", 0.5))
        score *= 1.0 if state.get("caos_stable", True) else 0.5
        score *= max(0.1, 1.0 - state.get("volatility", 0))
        return max(self.epsilon, score)
    
    def _autonomy(self, state: Dict[str, Any]) -> float:
        components = [
            state.get("self_improvement", 0),
            state.get("exploration", 0),
            state.get("adaptation", 0),
            state.get("learning_rate", 0) * 100,
        ]
        
        if not components:
            return self.epsilon
        
        return max(self.epsilon, sum(components) / len(components))
    
    def _decide(self, sr_score: float) -> Literal["ALLOW", "CEDE", "ABORT"]:
        if sr_score >= self.tau_SR:
            return "ALLOW"
        elif sr_score >= self.tau_SR * 0.5:
            return "CEDE"
        else:
            return "ABORT"
    
    def _compute_confidence_interval(self, sr_score: float, components: Dict[str, float]) -> Tuple[float, float]:
        if not self.cfg.get("confidence_bands", True):
            return (sr_score, sr_score)
        
        values = list(components.values())
        if len(values) > 1:
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std_dev = math.sqrt(variance)
            margin = 1.96 * std_dev / math.sqrt(len(values))
            lower = max(0.0, sr_score - margin)
            upper = min(1.0, sr_score + margin)
        else:
            lower = upper = sr_score
        
        return (lower, upper)
    
    def _compute_cache_key(self, state: Dict[str, Any]) -> str:
        relevant_fields = [
            "ece", "rho_bias", "fairness", "consent", "eco_ok",
            "delta_linf", "mdl_gain", "ppl_ood", "efficiency",
            "uncertainty", "confidence", "volatility",
            "self_improvement", "exploration", "adaptation"
        ]
        
        filtered = {k: state.get(k) for k in relevant_fields if k in state}
        return hashlib.md5(json.dumps(filtered, sort_keys=True).encode()).hexdigest()

# =============================================================================
# Módulo Estratégico Principal Fusionado
# =============================================================================

class StrategyModuleFusion:
    """Módulo 2/8 Fusão Definitiva: Estratégia Ω-META."""
    
    def __init__(self, cfg: Optional[Dict[str, Any]] = None, worm_ledger: Optional[WORMLedger] = None):
        self.cfg = self._merge_configs(cfg or {}, OMEGA_META_CONFIG)
        self.worm = worm_ledger or WORMLedger()
        
        # Componentes principais
        self.sr_omega = StrategicSROmegaFusion(self.cfg)
        
        # Cache e performance
        self.plan_cache = AdaptiveCache(60, 100)
        
        # Estado interno
        self.cycle_count = 0
        self.last_plan: Optional[PlanOmega] = None
        self.plan_history = deque(maxlen=self.cfg["worm"]["retention_cycles"])
        self.lock = threading.RLock()
        
        # RNG determinístico
        self.rng = random.Random(self.cfg["deliberation"]["seed"])
        
        # Telemetria
        self.telemetry = {
            "plans_created": 0,
            "gates_passed": 0,
            "gates_failed": 0,
            "avg_latency_ms": 0.0,
            "errors": 0,
        }
        
        # Registro de inicialização
        self._log_event(StrategyEvent.STRATEGY_START, {
            "version": self.cfg["version"],
            "core_integration": CORE_INTEGRATION,
            "timestamp": self._timestamp(),
        })
    
    async def generate_plan(self, xt: Any, **kwargs) -> PlanOmega:
        """
        Gera plano Ω-META completo a partir do estado atual.
        CORREÇÃO: Análise REAL do estado, não fallback.
        """
        start_time = time.perf_counter()
        
        # Registra início no WORM
        proof_id = self.worm.record_event("PLAN_GENERATION_START", {
            "timestamp": self._timestamp(),
            "state_hash": _hash_data(xt.to_dict() if hasattr(xt, 'to_dict') else str(xt))
        })
        
        try:
            # CORREÇÃO: Análise REAL do estado atual
            state_analysis = self._analyze_current_state(xt)
            
            # Gera objetivos baseados na análise REAL
            goals = self._generate_real_goals(state_analysis, xt)
            
            # Se análise não gerou objetivos, usa heurísticas baseadas no estado
            if not goals:
                goals = self._generate_heuristic_goals(xt)
            
            # Constraints baseadas no estado REAL
            constraints = self._generate_real_constraints(xt, state_analysis)
            
            # Budgets adaptativos baseados no contexto REAL
            budgets = self._generate_adaptive_budgets(xt, kwargs)
            
            # Calcula U_signal baseado na análise REAL
            u_signal = self._calculate_real_u_signal(state_analysis, xt)
            
            # Cria plano REAL (não fallback)
            plan = PlanOmega(
                id=f"real_{uuid.uuid4().hex[:8]}",
                timestamp=self._timestamp(),
                cycle=getattr(xt, 'cycle_count', 0) + 1,
                goals=goals,
                constraints=constraints,
                budgets=budgets,
                u_signal=u_signal,
                priority_map={goal.name: getattr(goal, 'priority', 0.5) for goal in goals},
                promotion_policy={"type": "lexicographic", "order": ["ethics", "risk", "performance"]},
                rollback_policy={"enabled": True, "threshold": 0.95},
                rationale=f"Plano REAL baseado em análise de estado com {len(goals)} objetivos",
                confidence=state_analysis.get('confidence', 0.8),
                sr_score=state_analysis.get('sr_score', getattr(xt, 'sr_score', 1.0)),
                input_hash=_hash_data(xt.to_dict() if hasattr(xt, 'to_dict') else str(xt))
            )
            
            # Calcula hash do plano
            plan.plan_hash = _hash_data(plan.to_dict())
            
            # CORREÇÃO: Atualiza estado após geração do plano
            self._update_state_after_planning(xt, plan, state_analysis)
            
            # Registra sucesso
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            self.worm.record_event("PLAN_GENERATION_SUCCESS", {
                "plan_id": plan.id,
                "plan_hash": plan.plan_hash,
                "n_goals": len(goals),
                "u_signal": u_signal,
                "elapsed_ms": elapsed_ms,
                "proof_id": proof_id,
                "real_analysis": True
            })
            
            return plan
            
        except Exception as e:
            # Registra falha
            self.worm.record_event("PLAN_GENERATION_FAILURE", {
                "error": str(e),
                "proof_id": proof_id
            })
            raise
    
    def _analyze_current_state(self, xt: Any) -> Dict[str, Any]:
        """Análise REAL do estado atual"""
        analysis = {
            "timestamp": self._timestamp(),
            "confidence": 0.0,
            "sr_score": getattr(xt, 'sr_score', 1.0),
            "issues": [],
            "opportunities": [],
            "metrics": {}
        }
        
        # Análise de métricas críticas
        ece = getattr(xt, 'ece', 0.0)
        rho = getattr(xt, 'rho', 0.5)
        novelty_sim = getattr(xt, 'novelty_sim', 1.0)
        rag_recall = getattr(xt, 'rag_recall', 1.0)
        
        analysis["metrics"] = {
            "ece": ece,
            "rho": rho, 
            "novelty_sim": novelty_sim,
            "rag_recall": rag_recall
        }
        
        # Identifica problemas REAIS
        if ece > 0.01:
            analysis["issues"].append({"type": "high_ece", "value": ece, "severity": "high"})
        
        if rho > 0.9:
            analysis["issues"].append({"type": "high_rho", "value": rho, "severity": "medium"})
        
        if novelty_sim > 0.8:
            analysis["issues"].append({"type": "high_novelty", "value": novelty_sim, "severity": "low"})
        
        if rag_recall > 0.8:
            analysis["issues"].append({"type": "high_recall", "value": rag_recall, "severity": "low"})
        
        # Identifica oportunidades REAIS
        if ece < 0.005:
            analysis["opportunities"].append({"type": "optimize_performance", "potential": 0.8})
        
        if rho < 0.3:
            analysis["opportunities"].append({"type": "increase_exploration", "potential": 0.6})
        
        # Calcula confiança baseada na qualidade dos dados
        confidence = 1.0
        if not hasattr(xt, 'cycle_count') or xt.cycle_count == 0:
            confidence *= 0.5  # Estado inicial tem menos confiança
        
        if len(analysis["issues"]) > 2:
            confidence *= 0.7  # Muitos problemas reduzem confiança
        
        analysis["confidence"] = max(0.1, confidence)
        
        return analysis
    
    def _generate_real_goals(self, analysis: Dict[str, Any], xt: Any) -> List[Goal]:
        """Gera objetivos REAIS baseados na análise"""
        goals = []
        
        # Objetivos baseados em problemas identificados
        for issue in analysis["issues"]:
            if issue["type"] == "high_ece":
                goals.append(Goal(
                    name="reduce_ece",
                    description="Reduzir Expected Calibration Error",
                    metric="ece",
                    target=0.005,
                    tolerance=0.002,
                    priority=0.9,
                    deadline=5
                ))
            
            elif issue["type"] == "high_rho":
                goals.append(Goal(
                    name="reduce_rho",
                    description="Reduzir viés rho",
                    metric="rho",
                    target=0.7,
                    tolerance=0.1,
                    priority=0.7,
                    deadline=8
                ))
            
            elif issue["type"] == "high_novelty":
                goals.append(Goal(
                    name="optimize_novelty",
                    description="Otimizar similaridade de novidade",
                    metric="novelty_sim",
                    target=0.6,
                    tolerance=0.1,
                    priority=0.5,
                    deadline=10
                ))
        
        # Objetivos baseados em oportunidades
        for opp in analysis["opportunities"]:
            if opp["type"] == "optimize_performance":
                goals.append(Goal(
                    name="enhance_performance",
                    description="Melhorar performance geral",
                    metric="sr_score",
                    target=min(1.0, getattr(xt, 'sr_score', 1.0) * 1.1),
                    tolerance=0.05,
                    priority=0.8,
                    deadline=7
                ))
        
        return goals
    
    def _generate_heuristic_goals(self, xt: Any) -> List[Goal]:
        """Gera objetivos heurísticos quando análise não produz objetivos"""
        return [Goal(
            name="maintain_stability",
            description="Manter estabilidade do sistema",
            metric="sr_score",
            target=max(0.8, getattr(xt, 'sr_score', 1.0)),
            tolerance=0.1,
            priority=0.6,
            deadline=10
        )]
    
    def _generate_real_constraints(self, xt: Any, analysis: Dict[str, Any]) -> Constraints:
        """Gera constraints baseadas no estado REAL"""
        return Constraints(
            ece_max=0.01,
            rho_bias_max=1.05,
            rho_max=0.95,
            delta_linf_min=0.01,
            trust_region_radius_proposed=max(0.05, getattr(xt, 'trust_region_radius', 0.1) * analysis["confidence"])
        )
    
    def _generate_adaptive_budgets(self, xt: Any, kwargs: Dict[str, Any]) -> Budgets:
        """Gera budgets adaptativos baseados no contexto"""
        # Adapta budgets baseado no ciclo atual
        cycle = getattr(xt, 'cycle_count', 0)
        cycle_factor = min(2.0, 1.0 + cycle * 0.1)  # Aumenta budget com experiência
        
        return Budgets(
            max_tokens=int(kwargs.get('max_tokens', 50000) * cycle_factor),
            max_cost=kwargs.get('max_cost', 5.0) * cycle_factor,
            max_latency_ms=kwargs.get('max_latency_ms', 30000),
            max_llm_calls=int(kwargs.get('max_llm_calls', 20) * cycle_factor),
            quota_local=0.8
        )
    
    def _calculate_real_u_signal(self, analysis: Dict[str, Any], xt: Any) -> float:
        """Calcula U_signal baseado na análise REAL"""
        base_signal = getattr(xt, 'sr_score', 1.0) - 0.5
        
        # Ajusta baseado na confiança da análise
        confidence_factor = analysis["confidence"]
        
        # Ajusta baseado no número de problemas
        issue_penalty = len(analysis["issues"]) * 0.1
        
        # Ajusta baseado nas oportunidades
        opportunity_bonus = len(analysis["opportunities"]) * 0.05
        
        u_signal = base_signal * confidence_factor - issue_penalty + opportunity_bonus
        
        return max(0.0, min(1.0, u_signal))
    
    def _update_state_after_planning(self, xt: Any, plan: PlanOmega, analysis: Dict[str, Any]) -> None:
        """CORREÇÃO: Atualiza estado após planejamento"""
        # Incrementa contador de ciclo
        if hasattr(xt, 'cycle_count'):
            xt.cycle_count += 1
        else:
            setattr(xt, 'cycle_count', 1)
        
        # Adiciona hash do plano aos hashes
        if hasattr(xt, 'hashes'):
            xt.hashes.append(plan.plan_hash)
        else:
            setattr(xt, 'hashes', [plan.plan_hash])
        
        # Adiciona proof_id
        if hasattr(xt, 'proof_ids'):
            xt.proof_ids.append(plan.id)
        else:
            setattr(xt, 'proof_ids', [plan.id])
        
        # Atualiza métricas baseadas na análise
        if analysis["confidence"] > 0.7:
            # Alta confiança: ajusta métricas ligeiramente
            if hasattr(xt, 'sr_score'):
                xt.sr_score = min(1.0, xt.sr_score + 0.01)
        
        log(f"Estado atualizado: cycle={getattr(xt, 'cycle_count', 0)}, hashes={len(getattr(xt, 'hashes', []))}", "INFO", "STRATEGY")
    
    def create_plan(self, state: Union[Dict[str, Any], OmegaState, Any], intent: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Cria plano estratégico Ω-META."""
        start_time = time.time()
        
        with self.lock:
            try:
                self.cycle_count += 1
                
                # Preparar estado
                state_dict = self._prepare_state(state)
                
                # Hash para cache e auditoria
                input_hash = self._compute_input_hash(state_dict, intent, context)
                
                # Check cache
                cache_key = f"plan_{input_hash[:16]}"
                cached = self.plan_cache.get(cache_key)
                if cached:
                    self._log_event(StrategyEvent.CACHE_HIT, {"key": cache_key})
                    return cached
                
                # 1. SR Gate Check
                sr_score, sr_report = self.sr_omega.compute(state_dict)
                
                if not self.sr_omega.gate(state_dict):
                    result = self._create_conservative_result(state_dict, sr_report, input_hash, "SR_GATE_FAILED")
                    self._log_gate_failure(sr_score, sr_report.decision)
                    return result
                
                # 2. Gerar objetivos simples
                goals = self._generate_simple_goals(state_dict, intent)
                
                if not goals:
                    result = self._create_minimal_result(state_dict, sr_report, input_hash)
                    return result
                
                # 3. Construir constraints
                constraints = self._build_constraints(state_dict, context)
                
                # 4. Validação
                valid, violations = constraints.validate(state_dict)
                if not valid and any("ETHICS" in v for v in violations):
                    result = self._create_conservative_result(state_dict, sr_report, input_hash, "ETHICS_VIOLATION")
                    return result
                
                # 5. Alocar budgets
                budgets = self._allocate_budgets(goals, context)
                
                # 6. Trust region simples
                trust_radius = max(0.02, min(0.5, state_dict.get("trust_region_radius", 0.1) * 1.1))
                constraints.trust_region_radius_proposed = trust_radius
                
                # 7. Criar políticas básicas
                promotion_policy = {"sr_threshold": self.cfg["sr_omega"]["tau_SR"]}
                rollback_policy = {"ethics_violation": True}
                
                # 8. Priority map simples
                priority_map = {"F3": 0.2, "F4": 0.2, "F5": 0.2, "F6": 0.2, "F7": 0.1, "F8": 0.1}
                
                # 9. U signal
                u_signal = min(1.0, sr_score * state_dict.get("delta_linf", 0.01) * 10)
                
                # 10. Rationale
                rationale = f"Intent: {intent[:50]}... | SR: {sr_score:.3f} | Goals: {len(goals)}"
                
                # 11. Montar plano
                plan = PlanOmega(
                    id=f"plan_v{self.cycle_count}_{input_hash[:8]}",
                    timestamp=self._timestamp(),
                    cycle=self.cycle_count,
                    goals=goals,
                    constraints=constraints,
                    budgets=budgets,
                    priority_map=priority_map,
                    promotion_policy=promotion_policy,
                    rollback_policy=rollback_policy,
                    rationale=rationale,
                    confidence=sr_score * 0.8,
                    sr_score=sr_score,
                    u_signal=u_signal,
                    input_hash=input_hash,
                    parent_plan_id=self.last_plan.id if self.last_plan else None,
                    generation_time_ms=(time.time() - start_time) * 1000,
                    status="validated",
                )
                
                # 12. Hash e assinatura
                plan.plan_hash = plan.compute_hash()
                plan.sign()
                
                # 13. Registrar no WORM
                self._log_event(StrategyEvent.PLAN_CREATED, {
                    "plan_id": plan.id,
                    "sr_score": sr_score,
                    "num_goals": len(goals),
                    "u_signal": u_signal,
                })
                
                # 14. Atualizar histórico
                self.last_plan = plan
                self.plan_history.append(plan)
                
                # 15. Preparar resultado
                result = {
                    "PlanΩ": plan.to_dict(),
                    "SR_report": sr_report.to_dict(),
                    "proof": {
                        "plan_hash": plan.plan_hash,
                        "input_hash": input_hash,
                        "signature": plan.signature,
                        "timestamp": plan.timestamp,
                    },
                    "U_signal": u_signal,
                }
                
                # 16. Cachear
                self.plan_cache.set(cache_key, result)
                
                # 17. Atualizar telemetria
                self._update_telemetry(plan.generation_time_ms, True)
                
                return result
                
            except Exception as e:
                self.telemetry["errors"] += 1
                self._log_event(StrategyEvent.STRATEGY_ABORT, {"error": str(e), "cycle": self.cycle_count})
                return self._create_emergency_result(state_dict, str(e))
    
    def get_telemetry(self) -> Dict[str, Any]:
        """Retorna telemetria completa do módulo."""
        with self.lock:
            return deepcopy(self.telemetry)
    
    # =========================================================================
    # Métodos Privados de Suporte
    # =========================================================================
    
    def _prepare_state(self, state: Any) -> Dict[str, Any]:
        if isinstance(state, dict):
            return state
        elif hasattr(state, "to_dict"):
            return state.to_dict()
        elif hasattr(state, "__dict__"):
            return state.__dict__
        else:
            try:
                return dict(state)
            except:
                return {"raw_state": str(state)}
    
    def _generate_simple_goals(self, state: Dict[str, Any], intent: str) -> List[Goal]:
        """Gera objetivos simples baseados na intenção."""
        goals = []
        
        # Parse básico da intenção
        intent_lower = intent.lower()
        
        if "robust" in intent_lower or "ood" in intent_lower:
            goals.append(Goal(
                name="improve_robustness",
                description="Improve OOD robustness",
                metric="ppl_ood",
                target=state.get("ppl_ood", 100) * 0.95,
                tolerance=2.0,
                deadline=10,
                priority=1.2,
                lexicographic_level=3,
            ))
        
        if "fair" in intent_lower or "bias" in intent_lower:
            goals.append(Goal(
                name="improve_fairness",
                description="Improve fairness metrics",
                metric="fairness",
                target=min(1.0, state.get("fairness", 0.95) + 0.02),
                tolerance=0.01,
                deadline=5,
                priority=1.5,
                lexicographic_level=1,
            ))
        
        if "risk" in intent_lower or "rho" in intent_lower:
            goals.append(Goal(
                name="reduce_risk",
                description="Reduce risk coefficient",
                metric="rho",
                target=state.get("rho", 0.5) * 0.9,
                tolerance=0.05,
                deadline=7,
                priority=1.3,
                lexicographic_level=2,
            ))
        
        # Goal padrão se nenhum específico
        if not goals:
            goals.append(Goal(
                name="maintain_performance",
                description="Maintain system performance",
                metric="sr_score",
                target=max(0.8, state.get("sr_score", 0.85)),
                tolerance=0.05,
                deadline=5,
                priority=1.0,
                lexicographic_level=3,
            ))
        
        return goals
    
    def _build_constraints(self, state: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Constraints:
        constraints = Constraints(
            ece_max=self.cfg["ethics"]["ece_max"],
            rho_bias_max=self.cfg["ethics"]["rho_bias_max"],
            fairness_min=self.cfg["ethics"]["fairness_min"],
            consent_required=self.cfg["ethics"]["consent_required"],
            eco_ok_required=self.cfg["ethics"]["eco_ok_required"],
            rho_max=self.cfg["risk"]["rho_max"],
            uncertainty_max=self.cfg["risk"]["uncertainty_max"],
            volatility_max=self.cfg["risk"]["volatility_max"],
            delta_linf_min=self.cfg["performance"]["delta_linf_min"],
            ppl_ood_max=self.cfg["performance"]["ppl_ood_target"],
            efficiency_min=self.cfg["performance"]["efficiency_min"],
        )
        
        # Override com governança do contexto
        if context and "governance" in context:
            gov = context["governance"]
            if "ethics" in gov:
                constraints.ece_max = gov["ethics"].get("ece_max", constraints.ece_max)
            if "risk" in gov:
                constraints.rho_max = gov["risk"].get("rho_max", constraints.rho_max)
        
        return constraints
    
    def _allocate_budgets(self, goals: List[Goal], context: Optional[Dict[str, Any]]) -> Budgets:
        budgets = Budgets(
            max_tokens=self.cfg["budgets"]["max_tokens"],
            max_cost=self.cfg["budgets"]["max_cost"],
            max_latency_ms=self.cfg["budgets"]["max_latency_ms"],
            max_llm_calls=self.cfg["budgets"]["max_llm_calls"],
            max_memory_mb=self.cfg["budgets"]["max_memory_mb"],
        )
        
        # Escalar por número de objetivos
        num_goals = len(goals)
        if num_goals > 3:
            budgets.max_tokens = int(budgets.max_tokens * 1.2)
            budgets.max_llm_calls = int(budgets.max_llm_calls * 1.1)
        
        budgets.__post_init__()
        return budgets
    
    def _create_conservative_result(self, state: Dict[str, Any], sr_report: SRReport, input_hash: str, reason: str) -> Dict[str, Any]:
        plan = PlanOmega(
            id=f"conservative_{self.cycle_count}_{input_hash[:8]}",
            timestamp=self._timestamp(),
            cycle=self.cycle_count,
            goals=[],
            constraints=Constraints(trust_region_radius_proposed=0.02),
            budgets=Budgets(max_tokens=1000, max_cost=0.1, max_llm_calls=1),
            rationale=f"Conservative mode: {reason}",
            confidence=0.1,
            sr_score=sr_report.sr_score,
            u_signal=0.0,
            input_hash=input_hash,
            status="draft",
        )
        
        plan.plan_hash = plan.compute_hash()
        
        return {
            "PlanΩ": plan.to_dict(),
            "SR_report": sr_report.to_dict(),
            "proof": {"plan_hash": plan.plan_hash, "input_hash": input_hash, "reason": reason},
            "U_signal": 0.0,
        }
    
    def _create_minimal_result(self, state: Dict[str, Any], sr_report: SRReport, input_hash: str) -> Dict[str, Any]:
        plan = PlanOmega(
            id=f"minimal_{self.cycle_count}_{input_hash[:8]}",
            timestamp=self._timestamp(),
            cycle=self.cycle_count,
            goals=[Goal(name="maintain_stability", description="Maintain system stability", metric="stability", target=1.0, tolerance=0.1, deadline=1, priority=0.5, lexicographic_level=2)],
            constraints=Constraints(),
            budgets=Budgets(),
            rationale="No viable goals found - maintaining stability only",
            confidence=0.5,
            sr_score=sr_report.sr_score,
            u_signal=0.1,
            input_hash=input_hash,
            status="validated",
        )
        
        plan.plan_hash = plan.compute_hash()
        
        return {
            "PlanΩ": plan.to_dict(),
            "SR_report": sr_report.to_dict(),
            "proof": {"plan_hash": plan.plan_hash, "input_hash": input_hash},
            "U_signal": 0.1,
        }
    
    def _create_emergency_result(self, state: Dict[str, Any], error: str) -> Dict[str, Any]:
        return {
            "PlanΩ": {
                "id": f"emergency_{self.cycle_count}",
                "goals": [],
                "constraints": {"trust_region_radius_proposed": 0.02},
                "budgets": {"max_tokens": 100},
                "rationale": f"Emergency mode: {error[:100]}",
                "status": "draft",
            },
            "SR_report": {"sr_score": 0.0, "valid": False, "decision": "ABORT"},
            "proof": {"error": error, "timestamp": self._timestamp()},
            "U_signal": 0.0,
        }
    
    def _compute_input_hash(self, *args) -> str:
        content = []
        for arg in args:
            if arg is None:
                content.append("null")
            elif isinstance(arg, (dict, list)):
                content.append(json.dumps(arg, sort_keys=True))
            else:
                content.append(str(arg))
        
        combined = "|".join(content)
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def _timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()
    
    def _merge_configs(self, custom: Dict[str, Any], base: Dict[str, Any]) -> Dict[str, Any]:
        result = deepcopy(base)
        
        for key, value in custom.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(value, result[key])
            else:
                result[key] = value
        
        return result
    
    def _log_event(self, event: StrategyEvent, data: Dict[str, Any]) -> None:
        if not self.cfg["worm"]["enabled"] or not self.worm:
            return
        
        try:
            payload = {
                **data,
                "event": event.value,
                "timestamp": self._timestamp(),
                "cycle": self.cycle_count,
                "module": "2/8",
            }
            
            self.worm.record_event(event.value, payload)
        except Exception as e:
            warnings.warn(f"WORM logging failed: {e}")
    
    def _log_gate_failure(self, sr_score: float, decision: str) -> None:
        self.telemetry["gates_failed"] += 1
        self._log_event(StrategyEvent.STRATEGY_GATE_FAIL, {
            "sr_score": sr_score,
            "threshold": self.cfg["sr_omega"]["tau_SR"],
            "decision": decision,
        })
    
    def _update_telemetry(self, latency_ms: float, success: bool) -> None:
        self.telemetry["plans_created"] += 1
        
        if success:
            self.telemetry["gates_passed"] += 1
        
        # Média móvel exponencial de latência
        alpha = 0.1
        self.telemetry["avg_latency_ms"] = (
            alpha * latency_ms +
            (1 - alpha) * self.telemetry["avg_latency_ms"]
        )

# =============================================================================
# Factory Functions e Interface Principal
# =============================================================================

def create_strategy_module(config: Optional[Dict[str, Any]] = None, worm: Optional[WORMLedger] = None) -> StrategyModuleFusion:
    """Factory para criar módulo estratégico 2/8 fusionado."""
    return StrategyModuleFusion(config, worm)

# Alias para compatibilidade
StrategyModule = StrategyModuleFusion

# =============================================================================
# Main - Teste e Demonstração
# =============================================================================

if __name__ == "__main__":
    logger.info("="*80)
    logger.info("PENIN-Ω Strategy Module 2/8 - Fusão Definitiva")
    logger.info("="*80)
    
    # Criar módulo
    strategy = create_strategy_module()
    
    # Estado de teste
    test_state = OmegaState(
        E_ok=0.95, M=0.85, C=0.80, A=0.75,
        ece=0.008, rho_bias=1.02, fairness=0.96,
        consent=True, eco_ok=True,
        rho=0.82, uncertainty=0.25, volatility=0.15,
        delta_linf=0.018, mdl_gain=0.035, ppl_ood=88.0,
        efficiency=0.75, caos_post=1.35, caos_stable=True,
        self_improvement=0.72, exploration=0.65, adaptation=0.80,
        sr_score=0.85, trust_region_radius=0.12, cycle_count=42
    )
    
    # Intenção
    test_intent = "Melhorar robustez OOD em 5% mantendo ρ<0.9 com foco em fairness"
    
    # Contexto
    test_context = {
        "governance": {
            "ethics": {"ece_max": 0.01, "fairness_min": 0.95},
            "risk": {"rho_max": 0.90},
        }
    }
    
    logger.info("\nGerando Plano Ω-META...")
    logger.info("-"*40)
    
    # Criar plano
    result = strategy.create_plan(test_state, test_intent, test_context)
    
    # Exibir resultado
    plan = result["PlanΩ"]
    sr_report = result["SR_report"]
    
    logger.info(f"\n📋 PLANO GERADO")
    logger.info(f"   ID: {plan['id']}")
    logger.info(f"   Status: {plan['status']}")
    logger.info(f"   Tempo: {plan['generation_time_ms']:.2f}ms")
    
    logger.info(f"\n📊 SR-Ω∞ REPORT")
    logger.info(f"   Score: {sr_report['sr_score']:.3f}")
    logger.info(f"   Válido: {sr_report['valid']}")
    logger.info(f"   Decisão: {sr_report['decision']}")
    
    logger.info(f"\n🎯 OBJETIVOS ({len(plan['goals'])})")
    for i, goal in enumerate(plan['goals'], 1):
        logger.info(f"   {i}. {goal['name']} → {goal['target']:.3f}")
    
    logger.info(f"\n📡 SINAIS")
    logger.info(f"   U_signal: {result['U_signal']:.3f}")
    logger.info(f"   Confiança: {plan['confidence']:.3f}")
    
    logger.info(f"\n📊 TELEMETRIA")
    telemetry = strategy.get_telemetry()
    logger.info(f"   Planos criados: {telemetry['plans_created']}")
    logger.info(f"   Gates passados: {telemetry['gates_passed']}")
    logger.info(f"   Latência média: {telemetry['avg_latency_ms']:.2f}ms")
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ Módulo 2/8 Fusão Definitiva Funcional!")
    logger.info(f"{'='*80}")

# =============================================================================
# FUNÇÕES STANDALONE PARA COMPATIBILIDADE
# =============================================================================

def create_plan(intent: str, objectives: List[str], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Função standalone para criar plano estratégico.
    Implementação simplificada mas funcional.
    """
    import time
    from datetime import datetime, timezone
    
    plan_id = f'plan_{hash(intent) % 10000}_{int(time.time() % 10000)}'
    
    # Criar objetivos estruturados
    structured_objectives = []
    for i, obj_desc in enumerate(objectives):
        structured_objectives.append({
            'id': f'obj_{i+1}',
            'description': obj_desc,
            'priority': 1.0,
            'status': 'pending'
        })
    
    # Plano básico mas completo
    plan = {
        'plan_id': plan_id,
        'intent': intent,
        'objectives': structured_objectives,
        'status': 'created',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'context': context or {},
        'metadata': {
            'created_by': 'create_plan_standalone',
            'version': '1.0',
            'objectives_count': len(objectives)
        }
    }
    
    return plan

# Exportar função para compatibilidade
__all__ = ['create_plan', 'StrategyModuleFusion', 'PlanOmega', 'Goal']
