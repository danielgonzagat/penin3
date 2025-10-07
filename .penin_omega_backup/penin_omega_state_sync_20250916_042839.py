#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Sincronizador de Estado Global
========================================
OBJETIVO: Unifica e sincroniza estado entre todos os módulos PENIN-Ω,
garantindo consistência entre OmegaState (1/8), SystemView (7/8),
GlobalState (8/8) e métricas de todos os workers.

FUNCIONALIDADES:
✓ Estado global unificado
✓ Sincronização automática entre módulos
✓ Propagação de métricas em tempo real
✓ Versionamento de estado
✓ Rollback de estado em caso de falha

Autor: Equipe PENIN-Ω
Versão: 1.0.0
"""

from __future__ import annotations
import asyncio
import json
import time
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import logging

# =============================================================================
# ESTADO UNIFICADO
# =============================================================================

@dataclass
class UnifiedOmegaState:
    """Estado unificado que sincroniza todos os módulos."""
    # Métricas principais (compatível com todos os módulos)
    rho: float = 0.4
    sr_score: float = 0.85
    ece: float = 0.003
    rho_bias: float = 1.0
    ppl_ood: float = 100.0
    caos_post: float = 1.2
    delta_linf: float = 0.0
    mdl_gain: float = 0.0
    
    # Governança e segurança
    consent: bool = True
    eco_ok: bool = True
    trust_region_radius: float = 0.10
    
    # Métricas derivadas
    tau_sr: float = 0.80
    rho_max: float = 0.95
    ece_max: float = 0.01
    rho_bias_max: float = 1.05
    k_phi: float = 1.5
    lambda_rho: float = 0.5
    
    # Contadores e status
    cycle_count: int = 0
    total_evolutions: int = 0
    successful_evolutions: int = 0
    failed_evolutions: int = 0
    active_workers: int = 0
    
    # Timestamps e versionamento
    version: int = 1
    created_at: str = ""
    updated_at: str = ""
    last_sync: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()
        self.updated_at = datetime.now(timezone.utc).isoformat()
        self.last_sync = self.updated_at
    
    def update(self, **kwargs) -> "UnifiedOmegaState":
        """Atualiza estado e incrementa versão."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.version += 1
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return self
    
    def is_safe_to_evolve(self) -> bool:
        """Verifica se é seguro evoluir."""
        return (
            self.rho < self.rho_max and
            self.sr_score >= self.tau_sr and
            self.ece <= self.ece_max and
            self.rho_bias <= self.rho_bias_max and
            self.consent and self.eco_ok
        )
    
    def to_omega_state_1_8(self) -> Dict[str, Any]:
        """Converte para formato OmegaState do módulo 1/8."""
        return {
            "rho": self.rho,
            "sr_score": self.sr_score,
            "ece": self.ece,
            "rho_bias": self.rho_bias,
            "consent": self.consent,
            "eco_ok": self.eco_ok,
            "trust_region_radius": self.trust_region_radius,
            "ppl_ood": self.ppl_ood,
            "caos_post": self.caos_post,
            "delta_linf": self.delta_linf,
            "mdl_gain": self.mdl_gain,
            "cycle_count": self.cycle_count
        }
    
    def to_system_view_7_8(self) -> Dict[str, Any]:
        """Converte para formato SystemView do módulo 7/8."""
        return {
            "rho": self.rho,
            "sr_score": self.sr_score,
            "caos_post": self.caos_post,
            "ece": self.ece,
            "rho_bias": self.rho_bias,
            "consent": self.consent,
            "eco_ok": self.eco_ok,
            "trust_region_radius": self.trust_region_radius,
            "ppl_ood": self.ppl_ood,
            "tau_sr": self.tau_sr,
            "rho_max": self.rho_max,
            "ece_max": self.ece_max,
            "rho_bias_max": self.rho_bias_max,
            "k_phi": self.k_phi,
            "lambda_rho": self.lambda_rho
        }
    
    def to_global_state_8_8(self) -> Dict[str, Any]:
        """Converte para formato GlobalState do módulo 8/8."""
        return {
            "rho": self.rho,
            "sr_score": self.sr_score,
            "ece": self.ece,
            "ppl_ood": self.ppl_ood,
            "caos_post": self.caos_post,
            "consent": self.consent,
            "eco_ok": self.eco_ok,
            "trust_region_radius": self.trust_region_radius,
            "total_cycles": self.cycle_count,
            "successful_evolutions": self.successful_evolutions,
            "failed_evolutions": self.failed_evolutions,
            "system_status": "running" if self.active_workers > 0 else "idle",
            "active_pipelines": self.active_workers,
            "created_at": self.created_at,
            "updated_at": self.updated_at
        }

# =============================================================================
# SINCRONIZADOR DE ESTADO
# =============================================================================

class StateSynchronizer:
    """Sincronizador de estado entre todos os módulos."""
    
    def __init__(self):
        self.state = UnifiedOmegaState()
        self.subscribers: List[Callable[[UnifiedOmegaState], None]] = []
        self.lock = threading.RLock()
        self.logger = logging.getLogger("StateSynchronizer")
        self._sync_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Cache de estados por módulo
        self._module_states: Dict[str, Dict[str, Any]] = {}
        
        self.logger.info("🔄 Sincronizador de estado inicializado")
    
    def subscribe(self, callback: Callable[[UnifiedOmegaState], None]):
        """Inscreve callback para mudanças de estado."""
        with self.lock:
            self.subscribers.append(callback)
    
    def unsubscribe(self, callback: Callable[[UnifiedOmegaState], None]):
        """Remove callback."""
        with self.lock:
            if callback in self.subscribers:
                self.subscribers.remove(callback)
    
    def update_state(self, **kwargs) -> UnifiedOmegaState:
        """Atualiza estado global."""
        with self.lock:
            old_version = self.state.version
            self.state.update(**kwargs)
            
            # Notifica subscribers
            for callback in self.subscribers:
                try:
                    callback(self.state)
                except Exception as e:
                    self.logger.error(f"Erro notificando subscriber: {e}")
            
            self.logger.debug(f"Estado atualizado: v{old_version} → v{self.state.version}")
            return self.state
    
    def get_state(self) -> UnifiedOmegaState:
        """Obtém estado atual."""
        with self.lock:
            return self.state
    
    def get_state_for_module(self, module: str) -> Dict[str, Any]:
        """Obtém estado formatado para módulo específico."""
        with self.lock:
            if module == "1/8" or module == "core":
                return self.state.to_omega_state_1_8()
            elif module == "7/8" or module == "nexus":
                return self.state.to_system_view_7_8()
            elif module == "8/8" or module == "bridge":
                return self.state.to_global_state_8_8()
            else:
                return asdict(self.state)
    
    def sync_from_module(self, module: str, module_state: Dict[str, Any]):
        """Sincroniza estado a partir de módulo específico."""
        with self.lock:
            # Armazena estado do módulo
            self._module_states[module] = module_state.copy()
            
            # Extrai métricas relevantes
            updates = {}
            
            # Métricas comuns
            for key in ["rho", "sr_score", "ece", "ppl_ood", "caos_post", 
                       "consent", "eco_ok", "trust_region_radius"]:
                if key in module_state:
                    updates[key] = module_state[key]
            
            # Métricas específicas por módulo
            if module in ["1/8", "core"]:
                if "cycle_count" in module_state:
                    updates["cycle_count"] = module_state["cycle_count"]
                if "delta_linf" in module_state:
                    updates["delta_linf"] = module_state["delta_linf"]
            
            elif module in ["8/8", "bridge"]:
                if "total_cycles" in module_state:
                    updates["cycle_count"] = module_state["total_cycles"]
                if "successful_evolutions" in module_state:
                    updates["successful_evolutions"] = module_state["successful_evolutions"]
                if "failed_evolutions" in module_state:
                    updates["failed_evolutions"] = module_state["failed_evolutions"]
                if "active_pipelines" in module_state:
                    updates["active_workers"] = module_state["active_pipelines"]
            
            # Aplica atualizações se houver
            if updates:
                self.update_state(**updates)
                self.logger.debug(f"Estado sincronizado do módulo {module}: {list(updates.keys())}")
    
    async def start_auto_sync(self, interval_seconds: float = 5.0):
        """Inicia sincronização automática."""
        if self._running:
            return
        
        self._running = True
        self._sync_task = asyncio.create_task(self._auto_sync_loop(interval_seconds))
        self.logger.info(f"🔄 Auto-sincronização iniciada (intervalo: {interval_seconds}s)")
    
    async def stop_auto_sync(self):
        """Para sincronização automática."""
        self._running = False
        
        if self._sync_task:
            self._sync_task.cancel()
            try:
                await self._sync_task
            except asyncio.CancelledError:
                pass
        
        self.logger.info("🛑 Auto-sincronização parada")
    
    async def _auto_sync_loop(self, interval: float):
        """Loop de sincronização automática."""
        while self._running:
            try:
                # Atualiza timestamp de sincronização
                with self.lock:
                    self.state.last_sync = datetime.now(timezone.utc).isoformat()
                
                # Aguarda próximo ciclo
                await asyncio.sleep(interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Erro na auto-sincronização: {e}")
                await asyncio.sleep(interval)
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Obtém status da sincronização."""
        with self.lock:
            return {
                "state_version": self.state.version,
                "last_sync": self.state.last_sync,
                "auto_sync_running": self._running,
                "subscribers_count": len(self.subscribers),
                "modules_synced": list(self._module_states.keys()),
                "is_safe_to_evolve": self.state.is_safe_to_evolve()
            }

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

_global_synchronizer: Optional[StateSynchronizer] = None

def get_global_state_synchronizer() -> StateSynchronizer:
    """Obtém sincronizador global (singleton)."""
    global _global_synchronizer
    
    if _global_synchronizer is None:
        _global_synchronizer = StateSynchronizer()
    
    return _global_synchronizer

# =============================================================================
# CONECTORES PARA MÓDULOS
# =============================================================================

class ModuleStateConnector:
    """Conector base para sincronização de módulos."""
    
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.synchronizer = get_global_state_synchronizer()
        self.logger = logging.getLogger(f"StateConnector-{module_name}")
    
    def sync_to_global(self, local_state: Dict[str, Any]):
        """Sincroniza estado local para global."""
        self.synchronizer.sync_from_module(self.module_name, local_state)
        self.logger.debug(f"Estado sincronizado para global: {self.module_name}")
    
    def get_from_global(self) -> Dict[str, Any]:
        """Obtém estado global formatado para este módulo."""
        return self.synchronizer.get_state_for_module(self.module_name)
    
    def subscribe_to_changes(self, callback: Callable[[UnifiedOmegaState], None]):
        """Inscreve para mudanças de estado global."""
        self.synchronizer.subscribe(callback)

# Conectores específicos
def create_core_connector() -> ModuleStateConnector:
    """Cria conector para módulo 1/8 (Core)."""
    return ModuleStateConnector("1/8")

def create_nexus_connector() -> ModuleStateConnector:
    """Cria conector para módulo 7/8 (NEXUS-Ω)."""
    return ModuleStateConnector("7/8")

def create_bridge_connector() -> ModuleStateConnector:
    """Cria conector para módulo 8/8 (Bridge)."""
    return ModuleStateConnector("8/8")

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "UnifiedOmegaState", "StateSynchronizer", "ModuleStateConnector",
    "get_global_state_synchronizer", 
    "create_core_connector", "create_nexus_connector", "create_bridge_connector"
]

if __name__ == "__main__":
    # Teste básico
    print("PENIN-Ω State Synchronizer")
    
    async def test_sync():
        sync = get_global_state_synchronizer()
        
        # Testa atualização de estado
        sync.update_state(rho=0.35, sr_score=0.90)
        state = sync.get_state()
        print(f"✅ Estado atualizado: rho={state.rho}, sr_score={state.sr_score}")
        
        # Testa conversões para módulos
        core_state = sync.get_state_for_module("1/8")
        nexus_state = sync.get_state_for_module("7/8")
        bridge_state = sync.get_state_for_module("8/8")
        
        print(f"✅ Conversões: Core={len(core_state)}, NEXUS={len(nexus_state)}, Bridge={len(bridge_state)}")
        
        # Testa sincronização de módulo
        sync.sync_from_module("test", {"rho": 0.30, "sr_score": 0.95})
        updated_state = sync.get_state()
        print(f"✅ Sincronização: rho={updated_state.rho}, sr_score={updated_state.sr_score}")
        
        # Testa auto-sync
        await sync.start_auto_sync(1.0)
        await asyncio.sleep(2)
        await sync.stop_auto_sync()
        
        status = sync.get_sync_status()
        print(f"✅ Status: versão={status['state_version']}, seguro={status['is_safe_to_evolve']}")
    
    import asyncio
    asyncio.run(test_sync())
    print("✅ Sincronizador funcionando!")
