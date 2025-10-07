#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω v6.0 FUSION - Interface Unificada (Factory Pattern)
============================================================
Factory pattern para integração sem dependências circulares.
PRESERVA toda funcionalidade, apenas remove ciclos.
"""

from __future__ import annotations
import json
import threading
import time
import uuid
import os
from collections import deque
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from datetime import datetime, timezone

# CORREÇÃO: Usa módulo base sem ciclos
from penin_omega_utils import _ts, _hash_data, save_json, load_json, log, BaseConfig, LAZY_IMPORTER

# =============================================================================
# INTERFACE UNIFICADA DE ESTADO (SEM CICLOS)
# =============================================================================

class UnifiedOmegaStateInterface:
    """Interface que unifica UnifiedOmegaState com OmegaState esperado pelas outras peças"""
    
    async def __init__(self, base_state: Any):
        self._state = base_state
    
    # Propriedades básicas (sempre presentes)
    @property
    async def ece(self) -> float:
        return await getattr(self._state, 'ece', 0.0)
    
    @ece.setter
    async def ece(self, value: float):
        setattr(self._state, 'ece', value)
    
    @property
    async def rho(self) -> float:
        return await getattr(self._state, 'rho', 0.5)
    
    @rho.setter
    async def rho(self, value: float):
        setattr(self._state, 'rho', value)
    
    @property
    async def sr_score(self) -> float:
        return await getattr(self._state, 'sr_score', 1.0)
    
    @sr_score.setter
    async def sr_score(self, value: float):
        setattr(self._state, 'sr_score', value)
    
    @property
    async def caos_post(self) -> float:
        return await getattr(self._state, 'caos_post', 1.0)
    
    @caos_post.setter
    async def caos_post(self, value: float):
        setattr(self._state, 'caos_post', value)
    
    @property
    async def novelty_sim(self) -> float:
        return await getattr(self._state, 'novelty_sim', 1.0)
    
    @novelty_sim.setter
    async def novelty_sim(self, value: float):
        setattr(self._state, 'novelty_sim', value)
    
    @property
    async def rag_recall(self) -> float:
        return await getattr(self._state, 'rag_recall', 1.0)
    
    @rag_recall.setter
    async def rag_recall(self, value: float):
        setattr(self._state, 'rag_recall', value)
    
    @property
    async def hashes(self) -> List[str]:
        if not hasattr(self._state, 'hashes'):
            setattr(self._state, 'hashes', [])
        return await getattr(self._state, 'hashes')
    
    @property
    async def proof_ids(self) -> List[str]:
        if not hasattr(self._state, 'proof_ids'):
            setattr(self._state, 'proof_ids', [])
        return await getattr(self._state, 'proof_ids')
    
    @property
    async def cycle_count(self) -> int:
        return await getattr(self._state, 'cycle_count', 0)
    
    @cycle_count.setter
    async def cycle_count(self, value: int):
        setattr(self._state, 'cycle_count', value)
    
    # Propriedades específicas para peça 4/8
    @property
    async def trust_region_radius(self) -> float:
        return await getattr(self._state, 'trust_region_radius', getattr(self._state, 'trust_region', 0.1))
    
    @trust_region_radius.setter
    async def trust_region_radius(self, value: float):
        setattr(self._state, 'trust_region_radius', value)
        setattr(self._state, 'trust_region', value)  # Compatibilidade
    
    @property
    async def delta_linf_pred(self) -> float:
        return await getattr(self._state, 'delta_linf_pred', 0.0)
    
    @delta_linf_pred.setter
    async def delta_linf_pred(self, value: float):
        setattr(self._state, 'delta_linf_pred', value)
    
    @property
    async def mdl_gain_pred(self) -> float:
        return await getattr(self._state, 'mdl_gain_pred', 0.0)
    
    @mdl_gain_pred.setter
    async def mdl_gain_pred(self, value: float):
        setattr(self._state, 'mdl_gain_pred', value)
    
    @property
    async def ppl_ood_pred(self) -> float:
        return await getattr(self._state, 'ppl_ood_pred', 100.0)
    
    @ppl_ood_pred.setter
    async def ppl_ood_pred(self, value: float):
        setattr(self._state, 'ppl_ood_pred', value)
    
    @property
    async def adv_capabilities(self) -> Dict[str, bool]:
        if not hasattr(self._state, 'adv_capabilities'):
            setattr(self._state, 'adv_capabilities', {})
        return await getattr(self._state, 'adv_capabilities')
    
    @adv_capabilities.setter
    async def adv_capabilities(self, value: Dict[str, bool]):
        setattr(self._state, 'adv_capabilities', value)
    
    # Métodos unificados
    async def validate_gates(self) -> bool:
        """Validação de gates unificada"""
        if hasattr(self._state, 'validate_gates'):
            return await self._state.validate_gates()
        else:
            # Validação manual para compatibilidade
            return await self.ece <= 0.01 and self.rho <= 0.95
    
    async def to_dict(self) -> Dict[str, Any]:
        """Conversão para dict unificada"""
        if hasattr(self._state, 'to_dict'):
            return await self._state.to_dict()
        else:
            return await asdict(self._state) if hasattr(self._state, '__dataclass_fields__') else {}
    
    @classmethod
    async def from_dict(cls, data: Dict[str, Any]) -> 'UnifiedOmegaStateInterface':
        """Criação a partir de dict"""
        # Cria um objeto simples com os dados
        class SimpleState:
            pass
        
        state = SimpleState()
        for key, value in data.items():
            setattr(state, key, value)
        
        return await cls(state)
    
    # Acesso direto ao estado base para compatibilidade total
    async def __getattr__(self, name: str) -> Any:
        return await getattr(self._state, name)
    
    async def __setattr__(self, name: str, value: Any) -> None:
        if name == '_state':
            super().__setattr__(name, value)
        else:
            setattr(self._state, name, value)

# =============================================================================
# WRAPPER PARA PEÇA 1/8
# =============================================================================

async def wrap_core_state(core_instance) -> UnifiedOmegaStateInterface:
    """Envolve o estado da peça 1/8 com interface unificada"""
    return await UnifiedOmegaStateInterface(core_instance.state)

# =============================================================================
# CONFIGURAÇÃO CENTRALIZADA
# =============================================================================

@dataclass
class UnifiedConfig:
    """Configuração centralizada para todo o sistema"""
    
    # Versioning
    version: str = "6.0.0-FUSION"
    
    # CORREÇÃO: Modo lite para desenvolvimento
    lite_mode: bool = False  # Se True, não carrega modelos pesados
    
    # Peça 1/8 - Núcleo
    core: Dict[str, Any] = field(default_factory=lambda: {
        "model_path": "/root/.penin_omega/models/falcon-mamba-7b",
        "device": "cpu",
        "cache_l1": 1000,
        "cache_l2": 10000,
        "init_timeout_s": 30,
        "lazy_loading": True,  # CORREÇÃO: Lazy loading por padrão
        "skip_model_in_lite": True  # CORREÇÃO: Pula modelo em modo lite
    })
    
    # Peça 2/8 - Estratégia
    strategy: Dict[str, Any] = field(default_factory=lambda: {
        "max_goals": 10,
        "planning_timeout_s": 5,
        "trust_region_default": 0.1,
        "lexicographic_order": ["ethics", "risk", "performance"]
    })
    
    # Peça 3/8 - Aquisição
    acquisition: Dict[str, Any] = field(default_factory=lambda: {
        "chunk_size": 900,
        "overlap": 140,
        "max_docs_per_cycle": 50,
        "embedding_model": "all-MiniLM-L6-v2",
        "ucb_c": 0.3,
        "novelty_sample_k": 64
    })
    
    # Peça 4/8 - Mutação
    mutation: Dict[str, Any] = field(default_factory=lambda: {
        "n_candidates": 32,
        "top_k": 5,
        "seed": 42,
        "target_latency_ms": 30000,
        "sandbox_timeout_s": 5,
        "trust_region_penalty": 2.0
    })
    
    # WORM Unificado
    worm: Dict[str, Any] = field(default_factory=lambda: {
        "ledger_path": "/root/.penin_omega/worm/unified_ledger.jsonl",
        "merkle_chain": True,
        "compression": True,
        "max_size_mb": 100
    })
    
    # Performance
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "max_memory_mb": 2048,
        "max_cpu_percent": 80,
        "timeout_global_s": 300,
        "enable_profiling": False
    })

# Instância global da configuração
UNIFIED_CONFIG = UnifiedConfig()

# =============================================================================
# IMPORTS UNIFICADOS
# =============================================================================

async def get_unified_imports(force_real_integration: bool = True):
    """Retorna imports corretos priorizando integração real"""
    imports = {}
    
    # CORREÇÃO: Prioriza integração real, fallback só em caso de falha
    integration_failures = []
    
    # Peça 1/8
    try:
        from penin_omega_1_core_v6 import PeninOmegaFusion, WormLedger as CoreWormLedger
        imports['core'] = {
            'PeninOmegaFusion': PeninOmegaFusion,
            'WormLedger': CoreWormLedger,
            'available': True,
            'mode': 'real'
        }
    except ImportError as e:
        integration_failures.append(f"core: {e}")
        imports['core'] = {'available': False, 'mode': 'failed'}
    
    # Peça 2/8
    try:
        from penin_omega_2_strategy import PlanOmega, Goal, Constraints, Budgets, StrategyModuleFusion
        imports['strategy'] = {
            'PlanOmega': PlanOmega,
            'Goal': Goal,
            'Constraints': Constraints,
            'Budgets': Budgets,
            'StrategyModuleFusion': StrategyModuleFusion,
            'available': True,
            'mode': 'real'
        }
    except ImportError as e:
        integration_failures.append(f"strategy: {e}")
        imports['strategy'] = {'available': False, 'mode': 'failed'}
    
    # Peça 3/8
    try:
        from penin_omega_3_acquisition_v6 import AcquisitionReport, acquire_ucb
        imports['acquisition'] = {
            'AcquisitionReport': AcquisitionReport,
            'acquire_ucb': acquire_ucb,
            'available': True,
            'mode': 'real'
        }
    except ImportError as e:
        integration_failures.append(f"acquisition: {e}")
        imports['acquisition'] = {'available': False, 'mode': 'failed'}
    
    # Peça 4/8
    try:
        from penin_omega_4_mutation_v6 import MutationBundle, mutate_and_rank
        imports['mutation'] = {
            'MutationBundle': MutationBundle,
            'mutate_and_rank': mutate_and_rank,
            'available': True,
            'mode': 'real'
        }
    except ImportError as e:
        integration_failures.append(f"mutation: {e}")
        imports['mutation'] = {'available': False, 'mode': 'failed'}
    
    # Log falhas de integração
    if integration_failures and force_real_integration:
        log(f"Falhas de integração detectadas: {integration_failures}", "WARNING", "UNIFIED")
    
    return await imports

# =============================================================================
# UTILITÁRIOS UNIFICADOS
# =============================================================================

async def _ts() -> str:
    """Timestamp unificado"""
    return await datetime.now(timezone.utc).isoformat()

async def _hash_data(data: Any) -> str:
    """Hash unificado"""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, ensure_ascii=False)
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif not isinstance(data, bytes):
        data = str(data).encode("utf-8")
    return await hashlib.sha256(data).hexdigest()

async def save_json(path: Path, data: Any) -> None:
    """Save JSON unificado"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

async def load_json(path: Path, default: Any = None) -> Any:
    """Load JSON unificado"""
    try:
        with path.open("r", encoding="utf-8") as f:
            return await json.load(f)
    except Exception:
        return await default

async def log(msg: str, level: str = "INFO", component: str = "UNIFIED") -> None:
    """Log unificado"""
    logger.info(f"[{_ts()}][{component}][{level}] {msg}")

class UnifiedWORMLedger:
    """WORM Ledger unificado que agrega todos os ledgers das peças"""
    
    async def __init__(self, unified_path: Optional[Path] = None):
        self.unified_path = unified_path or Path(UNIFIED_CONFIG.worm["ledger_path"])
        self.unified_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Mantém referências aos ledgers individuais (PRESERVA funcionalidade)
        self.individual_ledgers = {}
        self.lock = threading.RLock()
        
        # Cache para performance
        self._event_cache = deque(maxlen=1000)
        
    async def register_component_ledger(self, component: str, ledger_path: Path):
        """Registra ledger de uma peça individual (PRESERVA ledgers existentes)"""
        self.individual_ledgers[component] = ledger_path
        log(f"Ledger {component} registrado: {ledger_path}", "INFO", "WORM")
    
    async def record_unified_event(self, component: str, event_type: str, data: Dict[str, Any]) -> str:
        """Registra evento no ledger unificado E no individual"""
        with self.lock:
            event_id = str(uuid.uuid4())
            timestamp = _ts()
            
            # Evento unificado
            unified_event = {
                "event_id": event_id,
                "component": component,
                "type": event_type,
                "data": data,
                "timestamp": timestamp,
                "unified": True
            }
            
            # Hash Merkle
            prev_hash = self._get_last_hash()
            unified_event["prev_hash"] = prev_hash
            unified_event["hash"] = _hash_data({k: v for k, v in unified_event.items() if k != "hash"})
            
            # Escreve no ledger unificado
            with self.unified_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(unified_event, ensure_ascii=False) + "\n")
            
            # PRESERVA: Também escreve no ledger individual se existir
            if component in self.individual_ledgers:
                individual_path = self.individual_ledgers[component]
                try:
                    individual_event = {
                        "event_id": event_id,
                        "type": event_type,
                        "data": data,
                        "timestamp": timestamp,
                        "unified_ref": True
                    }
                    individual_event["hash"] = _hash_data(individual_event)
                    
                    with individual_path.open("a", encoding="utf-8") as f:
                        f.write(json.dumps(individual_event, ensure_ascii=False) + "\n")
                except Exception as e:
                    log(f"Falha ao escrever ledger individual {component}: {e}", "WARNING", "WORM")
            
            # Update cache
            self._event_cache.append(unified_event["hash"])
            
            return await event_id
    
    async def _get_last_hash(self) -> str:
        """Obtém último hash da chain Merkle"""
        if self._event_cache:
            return await self._event_cache[-1]
        
        if self.unified_path.exists() and self.unified_path.stat().st_size > 0:
            try:
                with self.unified_path.open("rb") as f:
                    f.seek(-1000, os.SEEK_END)
                    lines = f.read().decode("utf-8", errors="ignore").splitlines()
                    for line in reversed(lines):
                        if line.strip():
                            event = json.loads(line)
                            return await event.get("hash", "genesis")
            except Exception:
                pass
        
        return await "genesis"
    
    async def get_events_by_component(self, component: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtém eventos de um componente específico - MÉTODO COMPLETADO"""
        events = []
        
        if not self.unified_path.exists():
            return await events
        
        try:
            with self.unified_path.open("r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        event = json.loads(line)
                        if event.get("component") == component:
                            events.append(event)
                            if len(events) >= limit:
                                break
        except Exception as e:
            log(f"Erro ao ler eventos do componente {component}: {e}", "ERROR", "WORM")
        
        return await events[-limit:]  # Retorna os mais recentes

# Instância global do WORM unificado
UNIFIED_WORM = UnifiedWORMLedger()

class LazyComponentLoader:
    """Carregador lazy para componentes pesados"""
    
    async def __init__(self):
        self._loaded_components = {}
        self._loading_locks = {}
    
    async def get_component(self, component_name: str, loader_func: Callable):
        """Carrega componente sob demanda com thread safety"""
        if component_name in self._loaded_components:
            return await self._loaded_components[component_name]
        
        # Thread safety
        if component_name not in self._loading_locks:
            self._loading_locks[component_name] = threading.Lock()
        
        with self._loading_locks[component_name]:
            # Double-check pattern
            if component_name in self._loaded_components:
                return await self._loaded_components[component_name]
            
            # Verifica modo lite
            if is_lite_mode() and component_name in ['llm_model', 'embedder', 'sentence_transformer']:
                log(f"Pulando carregamento de {component_name} (modo lite)", "INFO", "LAZY")
                return await None
            
            log(f"Carregando componente {component_name}...", "INFO", "LAZY")
            start_time = time.time()
            
            try:
                component = loader_func()
                self._loaded_components[component_name] = component
                
                elapsed = (time.time() - start_time) * 1000
                log(f"Componente {component_name} carregado em {elapsed:.1f}ms", "INFO", "LAZY")
                
                return await component
                
            except Exception as e:
                log(f"Falha ao carregar {component_name}: {e}", "ERROR", "LAZY")
                return await None
    
    async def preload_components(self, component_loaders: Dict[str, Callable]):
        """Pré-carrega componentes em background (opcional)"""
        if is_lite_mode():
            log("Pré-carregamento pulado (modo lite)", "INFO", "LAZY")
            return
        
        async def background_load():
            for name, loader in component_loaders.items():
                try:
                    self.get_component(name, loader)
                except Exception as e:
                    log(f"Falha no pré-carregamento de {name}: {e}", "WARNING", "LAZY")
        
        thread = threading.Thread(target=background_load, daemon=True)
        thread.start()
        log(f"Pré-carregamento iniciado para {len(component_loaders)} componentes", "INFO", "LAZY")
    
    async def clear_cache(self):
        """Limpa cache de componentes carregados"""
        cleared = len(self._loaded_components)
        self._loaded_components.clear()
        log(f"Cache limpo: {cleared} componentes removidos", "INFO", "LAZY")

# Instância global do lazy loader
LAZY_LOADER = LazyComponentLoader()

async def lazy_load_component(name: str, loader: Callable):
    """Função helper para lazy loading"""
    return await LAZY_LOADER.get_component(name, loader)

# =============================================================================
# VALIDAÇÃO DE SISTEMA
# =============================================================================

async def enable_lite_mode():
    """Ativa modo lite para desenvolvimento (sem modelos pesados)"""
    global UNIFIED_CONFIG
    UNIFIED_CONFIG.lite_mode = True
    UNIFIED_CONFIG.core["skip_model_in_lite"] = True
    UNIFIED_CONFIG.acquisition["embedding_model"] = "lite"
    log("Modo lite ativado - modelos pesados desabilitados", "INFO", "UNIFIED")

async def disable_lite_mode():
    """Desativa modo lite (carrega todos os modelos)"""
    global UNIFIED_CONFIG
    UNIFIED_CONFIG.lite_mode = False
    UNIFIED_CONFIG.core["skip_model_in_lite"] = False
    UNIFIED_CONFIG.acquisition["embedding_model"] = "all-MiniLM-L6-v2"
    log("Modo lite desativado - todos os modelos habilitados", "INFO", "UNIFIED")

async def is_lite_mode() -> bool:
    """Verifica se está em modo lite"""
    return await UNIFIED_CONFIG.lite_mode
    """Valida integridade do sistema completo"""
    imports = get_unified_imports()
    
    report = {
        "timestamp": _ts(),
        "version": UNIFIED_CONFIG.version,
        "components": {},
        "overall_status": "unknown"
    }
    
    # Verifica cada peça
    for component, data in imports.items():
        report["components"][component] = {
            "available": data.get("available", False),
            "classes": list(data.keys()) if data.get("available") else [],
            "status": "OK" if data.get("available") else "MISSING"
        }
    
    # Status geral
    available_count = sum(1 for comp in report["components"].values() if comp["available"])
    total_count = len(report["components"])
    
    if available_count == total_count:
        report["overall_status"] = "FULL_INTEGRATION"
    elif available_count >= 2:
        report["overall_status"] = "PARTIAL_INTEGRATION"
    else:
        report["overall_status"] = "FALLBACK_MODE"
    
    return await report

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "UnifiedOmegaStateInterface",
    "wrap_core_state", 
    "UnifiedConfig",
    "UNIFIED_CONFIG",
    "get_unified_imports",
    "validate_system_integrity",
    # Modo lite
    "enable_lite_mode",
    "disable_lite_mode", 
    "is_lite_mode",
    # WORM unificado
    "UnifiedWORMLedger",
    "UNIFIED_WORM",
    # Lazy loading
    "LazyComponentLoader",
    "LAZY_LOADER",
    "lazy_load_component",
    # Utils
    "_ts", "_hash_data", "save_json", "load_json", "log"
]
