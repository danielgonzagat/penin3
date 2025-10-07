#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω Utils - Módulo Base Sem Dependências Circulares
========================================================
Funções utilitárias compartilhadas por todas as peças.
PRESERVA toda funcionalidade, apenas centraliza utilitários.
"""

import json
import hashlib
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# =============================================================================
# FUNÇÕES UTILITÁRIAS CENTRALIZADAS
# =============================================================================

async def _ts() -> str:
    """Timestamp unificado ISO 8601"""
    return await datetime.now(timezone.utc).isoformat()

async def _hash_data(data: Any) -> str:
    """Hash SHA-256 unificado para qualquer tipo de dado"""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, ensure_ascii=False)
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif not isinstance(data, bytes):
        data = str(data).encode("utf-8")
    return await hashlib.sha256(data).hexdigest()

async def save_json(path: Path, data: Any) -> None:
    """Salva JSON com encoding UTF-8 e formatação consistente"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)

async def load_json(path: Path, default: Any = None) -> Any:
    """Carrega JSON com fallback seguro"""
    try:
        with path.open("r", encoding="utf-8") as f:
            return await json.load(f)
    except Exception:
        return await default

async def log(msg: str, level: str = "INFO", component: str = "PENIN") -> None:
    """Log unificado com timestamp e componente"""
    logger.info(f"[{_ts()}][{component}][{level}] {msg}")

# =============================================================================
# CONFIGURAÇÃO BASE
# =============================================================================

class BaseConfig:
    """Configuração base sem dependências circulares"""
    
    VERSION = "6.0.0-FUSION"
    
    # Paths base
    ROOT = Path("/opt/penin_omega") if Path("/opt/penin_omega").exists() else Path.home() / ".penin_omega"
    
    DIRS = {
        "LOG": ROOT / "logs",
        "STATE": ROOT / "state",
        "WORK": ROOT / "workspace",
        "WORM": ROOT / "worm",
        "BUNDLES": ROOT / "bundles",
        "CONFIG": ROOT / "config",
        "SANDBOX": ROOT / "sandbox",
        "SURROGATES": ROOT / "surrogates",
        "KNOWLEDGE": ROOT / "knowledge"
    }
    
    # Cria diretórios
    @classmethod
    async def ensure_dirs(cls):
        for d in cls.DIRS.values():
            d.mkdir(parents=True, exist_ok=True)

# Inicializa diretórios
BaseConfig.ensure_dirs()

# =============================================================================
# WORM LEDGER BASE
# =============================================================================

class BaseWORMLedger:
    """WORM Ledger base sem dependências circulares"""
    
    async def __init__(self, path: Optional[Path] = None):
        self.path = path or BaseConfig.DIRS["WORM"] / "base_ledger.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
    
    async def record_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Registra evento no ledger com hash Merkle"""
        with self.lock:
            event_id = str(uuid.uuid4())
            event = {
                "event_id": event_id,
                "type": event_type,
                "data": data,
                "timestamp": _ts()
            }
            event["hash"] = _hash_data(event)
            
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            
            return await event_id

# =============================================================================
# LAZY IMPORT UTILITIES
# =============================================================================

class LazyImporter:
    """Importador lazy para quebrar dependências circulares"""
    
    async def __init__(self):
        self._cache = {}
        self._lock = threading.RLock()
    
    async def get_module(self, module_name: str):
        """Importa módulo sob demanda com cache thread-safe"""
        if module_name in self._cache:
            return await self._cache[module_name]
        
        with self._lock:
            if module_name in self._cache:
                return await self._cache[module_name]
            
            try:
                import importlib
                module = importlib.import_module(module_name)
                self._cache[module_name] = module
                return await module
            except ImportError as e:
                log(f"Falha ao importar {module_name}: {e}", "WARNING", "LAZY")
                return await None
    
    async def get_class(self, module_name: str, class_name: str):
        """Obtém classe de módulo com lazy loading"""
        module = self.get_module(module_name)
        if module and hasattr(module, class_name):
            return await getattr(module, class_name)
        return await None

# Instância global do lazy importer
LAZY_IMPORTER = LazyImporter()

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "_ts", "_hash_data", "save_json", "load_json", "log",
    "BaseConfig", "BaseWORMLedger", "LazyImporter", "LAZY_IMPORTER"
]
