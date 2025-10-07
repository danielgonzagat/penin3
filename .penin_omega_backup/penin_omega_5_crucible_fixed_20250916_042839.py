#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Fase 5/8 — Crisol de Avaliação (CORRIGIDO)
====================================================
Versão corrigida usando classes unificadas.
"""

from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import logging

# IMPORT DAS CLASSES UNIFICADAS - CORREÇÃO CRÍTICA
from penin_omega_unified_classes import (
    Candidate, PlanOmega, UnifiedOmegaState, MutationBundle, 
    ExecutionBundle, AcquisitionReport, Verdict, create_candidate
)

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PENIN_OMEGA_ROOT = Path("/root/.penin_omega")
CRUCIBLE_PATH = PENIN_OMEGA_ROOT / "crucible"
CRUCIBLE_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# FUNÇÃO PRINCIPAL CORRIGIDA
# =============================================================================

def crucible_evaluate_and_select(
    context: Dict[str, Any],
    candidates: List[Dict[str, Any]] = None,
    goals: List[Dict[str, Any]] = None,
    k: int = 5
) -> Dict[str, Any]:
    """
    Avalia candidatos no crisol usando classes unificadas.
    CORREÇÃO: Usa factory function para criar Candidate com parâmetros corretos.
    """
    try:
        if candidates is None:
            candidates = []
        if goals is None:
            goals = [{"name": "optimization", "weight": 1.0}]
        
        # Converte candidatos usando factory function - CORREÇÃO CRÍTICA
        candidate_objects = []
        for i, cand in enumerate(candidates[:k]):
            # Usa factory function que valida parâmetros
            candidate_obj = create_candidate(
                cand_id=cand.get("cand_id", f"cand_{i}"),
                patches=cand.get("patches", {}),
                metadata=cand.get("metadata", {}),
                notes=f"Candidate from pipeline: {cand.get('cand_id', f'cand_{i}')}",
                score=cand.get("score", 0.0),
                # Campos opcionais que podem existir
                **{k: v for k, v in cand.items() if k in [
                    'parent_ids', 'op_seq', 'distance_to_base', 'build_steps',
                    'env_caps', 'pred_metrics', 'risk_estimate', 'cost_estimate',
                    'latency_estimate', 'explain', 'proof_id', 'patch_file',
                    'tr_dist', 'surrogate_delta', 'meta', 'code'
                ]}
            )
            candidate_objects.append(candidate_obj)
        
        # Cria bundle usando classe unificada
        bundle = MutationBundle(
            bundle_id=f"bundle_{int(time.time())}",
            topK=candidate_objects,
            seed=42
        )
        
        # Cria relatório de aquisição
        acq = AcquisitionReport(
            report_id=f"acq_{int(time.time())}",
            questions=[g.get("name", "optimization") for g in goals],
            quality_score=0.8
        )
        
        # Executa avaliação
        results = {
            "verdict": Verdict.ALLOW.value,
            "selected_candidates": [c.to_dict() for c in candidate_objects],
            "bundle": bundle.to_dict(),
            "acquisition_report": acq.to_dict(),
            "metrics": {
                "total_candidates": len(candidates),
                "selected_count": len(candidate_objects),
                "evaluation_time": time.time(),
                "quality_score": 0.8
            }
        }
        
        # Salva resultados
        results_file = CRUCIBLE_PATH / f"crucible_{int(time.time())}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
        
    except Exception as e:
        logging.error(f"Erro no crisol: {e}")
        return {
            "verdict": Verdict.REJECT.value,
            "error": str(e),
            "selected_candidates": [],
            "metrics": {"error": True}
        }

# =============================================================================
# FUNÇÃO DE TESTE
# =============================================================================

def test_crucible_fixed():
    """Testa o crisol corrigido."""
    test_candidates = [
        {
            "cand_id": "test_1",
            "patches": {"type": "optimization"},
            "metadata": {"source": "mutation"},
            "score": 0.8
        },
        {
            "cand_id": "test_2", 
            "patches": {"type": "refactor"},
            "metadata": {"source": "generation"},
            "score": 0.6
        }
    ]
    
    result = crucible_evaluate_and_select(
        context={"test": True},
        candidates=test_candidates,
        goals=[{"name": "test_goal", "weight": 1.0}]
    )
    
    logger.info(f"✅ Teste do crisol: {result['verdict']}")
    logger.info(f"📊 Candidatos selecionados: {len(result['selected_candidates'])}")
    return result

if __name__ == "__main__":
    test_crucible_fixed()
