#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Classes Unificadas (CORRIGIDAS)
==========================================
Definições unificadas com métodos to_dict() implementados.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Union
from datetime import datetime, timezone
from enum import Enum

# =============================================================================
# CLASSES UNIFICADAS - COMPATIBILIDADE TOTAL
# =============================================================================

@dataclass
class Candidate:
    """Classe Candidate unificada - compatível com todos os módulos."""
    cand_id: str = ""
    
    # Campos do módulo 4/8 (mutation)
    parent_ids: List[str] = field(default_factory=list)
    op_seq: List[Dict[str, Any]] = field(default_factory=list)
    distance_to_base: float = 0.0
    patches: Union[Dict[str, Any], List[Dict[str, Any]]] = field(default_factory=dict)
    build_steps: List[str] = field(default_factory=list)
    env_caps: Dict[str, Any] = field(default_factory=dict)
    pred_metrics: Dict[str, Any] = field(default_factory=dict)
    risk_estimate: float = 0.0
    cost_estimate: float = 0.0
    latency_estimate: float = 0.0
    score: float = 0.0
    explain: str = ""
    proof_id: str = ""
    
    # Campos do módulo 5/8 (crucible) - ADICIONADOS
    patch_file: Optional[str] = None
    tr_dist: float = 0.0
    surrogate_delta: float = 0.0
    notes: str = ""
    
    # Campos compatibilidade - ADICIONADOS
    meta: Dict[str, Any] = field(default_factory=dict)
    code: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    async def from_dict(cls, data: Dict[str, Any]) -> "Candidate":
        """Cria instância a partir de dicionário, ignorando campos inexistentes."""
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return await cls(**valid_fields)

    async def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return await asdict(self)

@dataclass
class PlanOmega:
    """Classe PlanOmega unificada - compatível com todos os módulos."""
    id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cycle: int = 0
    
    # Campos principais
    goals: List[Dict[str, Any]] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    budgets: Dict[str, Any] = field(default_factory=dict)
    priority_map: Dict[str, float] = field(default_factory=dict)
    promotion_policy: Dict[str, Any] = field(default_factory=dict)
    rollback_policy: Dict[str, Any] = field(default_factory=dict)
    
    # Campos compatibilidade - ADICIONADOS
    policies: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    
    @classmethod
    async def from_dict(cls, data: Dict[str, Any]) -> "PlanOmega":
        """Cria instância a partir de dicionário, ignorando campos inexistentes."""
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return await cls(**valid_fields)

    async def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return await asdict(self)

@dataclass
class UnifiedOmegaState:
    """Estado unificado que sincroniza todos os módulos."""
    # Métricas principais
    rho: float = 0.4
    sr_score: float = 0.85
    ece: float = 0.003
    rho_bias: float = 1.0
    ppl_ood: float = 100.0
    caos_post: float = 1.2
    delta_linf: float = 0.0
    mdl_gain: float = 0.0
    
    # Governança
    consent: bool = True
    eco_ok: bool = True
    uncertainty: float = 0.1
    
    # Timestamps
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_update: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Métricas de sistema
    system_health: float = 1.0
    pipeline_status: str = "idle"
    active_workers: List[str] = field(default_factory=list)
    
    async def update_metrics(self, **kwargs):
        """Atualiza métricas e timestamp."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.last_update = datetime.now(timezone.utc).isoformat()

    @classmethod
    async def from_dict(cls, data: Dict[str, Any]) -> "UnifiedOmegaState":
        """Cria instância a partir de dicionário."""
        valid_fields = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return await cls(**valid_fields)

    async def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return await asdict(self)

@dataclass
class MutationBundle:
    """Bundle de mutação unificado."""
    bundle_id: str = ""
    plan_hash: str = ""
    seed: int = 0
    topK: List[Candidate] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    async def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return await asdict(self)

@dataclass
class ExecutionBundle:
    """Bundle de execução unificado."""
    bundle_id: str = ""
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    async def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return await asdict(self)

@dataclass
class AcquisitionReport:
    """Relatório de aquisição unificado."""
    report_id: str = ""
    questions: List[str] = field(default_factory=list)
    knowledge_items: List[Dict[str, Any]] = field(default_factory=list)
    quality_score: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    async def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return await asdict(self)

# =============================================================================
# ENUMS UNIFICADOS
# =============================================================================

class Verdict(Enum):
    ALLOW = "ALLOW"
    CANARY = "CANARY"
    REJECT = "REJECT"

class PipelineStage(Enum):
    F3_ACQUISITION = "F3_ACQUISITION"
    F4_MUTATION = "F4_MUTATION"
    F5_CRUCIBLE = "F5_CRUCIBLE"
    F6_AUTOREWRITE = "F6_AUTOREWRITE"
    F7_NEXUS = "F7_NEXUS"
    F8_GOVERNANCE = "F8_GOVERNANCE"

# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================

async def create_candidate(cand_id: str = None, **kwargs) -> Candidate:
    """Factory function para criar Candidate com validação."""
    if cand_id is None:
        cand_id = f"cand_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return await Candidate(cand_id=cand_id, **kwargs)

async def create_plan_omega(plan_id: str = None, **kwargs) -> PlanOmega:
    """Factory function para criar PlanOmega com validação."""
    if plan_id is None:
        plan_id = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    return await PlanOmega(id=plan_id, **kwargs)

async def create_unified_state(**kwargs) -> UnifiedOmegaState:
    """Factory function para criar UnifiedOmegaState."""
    return await UnifiedOmegaState(**kwargs)

# =============================================================================
# TESTE DAS CLASSES
# =============================================================================

async def test_unified_classes():
    """Testa todas as classes unificadas."""
    logger.info("🧪 Testando classes unificadas...")
    
    # Teste Candidate
    candidate = create_candidate("test_cand", score=0.8, notes="test")
    candidate_dict = candidate.to_dict()
    logger.info(f"✅ Candidate: {candidate.cand_id}")
    
    # Teste PlanOmega
    plan = create_plan_omega("test_plan", policies={"test": True})
    plan_dict = plan.to_dict()
    logger.info(f"✅ PlanOmega: {plan.id}")
    
    # Teste UnifiedOmegaState
    state = create_unified_state(rho=0.5)
    state_dict = state.to_dict()
    logger.info(f"✅ UnifiedOmegaState: {state.rho}")
    
    # Teste MutationBundle
    bundle = MutationBundle(bundle_id="test_bundle", topK=[candidate])
    bundle_dict = bundle.to_dict()
    logger.info(f"✅ MutationBundle: {bundle.bundle_id}")
    
    logger.info("🎉 Todas as classes funcionam corretamente!")
    return await True

if __name__ == "__main__":
    test_unified_classes()
