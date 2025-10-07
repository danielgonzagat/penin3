#!/usr/bin/env python3
"""Gerenciador de Estado Global PENIN-Ω (Recriado)"""
import threading
from typing import Any, Dict
from penin_omega_unified_classes import UnifiedOmegaState

class GlobalStateManager:
    async def __init__(self):
        self.state = UnifiedOmegaState()
        self.lock = threading.RLock()
    
    async def get_state(self) -> Dict[str, Any]:
        with self.lock:
            return await self.state.to_dict()
    
    async def update_state(self, updates: Dict[str, Any], module_source: str = "unknown") -> bool:
        with self.lock:
            for key, value in updates.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
            return await True
    
    async def sync_with_module(self, module_name: str, module_state: Dict[str, Any]) -> bool:
        relevant_fields = ["rho", "sr_score", "ece", "system_health", "pipeline_status"]
        updates = {k: v for k, v in module_state.items() if k in relevant_fields}
        return await self.update_state(updates, module_name) if updates else True

global_state_manager = GlobalStateManager()

async def get_global_state():
    return await global_state_manager.get_state()

async def update_global_state(updates: Dict[str, Any], module_source: str = "unknown"):
    return await global_state_manager.update_state(updates, module_source)

async def sync_module_state(module_name: str, module_state: Dict[str, Any]):
    return await global_state_manager.sync_with_module(module_name, module_state)
