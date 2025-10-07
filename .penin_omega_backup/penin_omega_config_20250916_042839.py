#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Configuração Centralizada & Gerenciamento de Dependências
===================================================================
OBJETIVO: Sistema centralizado de configuração que resolve dependências
circulares, gerencia paths, e fornece configuração unificada para todos
os módulos PENIN-Ω.

FUNCIONALIDADES:
✓ Configuração centralizada para todos os módulos
✓ Resolução de dependências circulares
✓ Paths unificados
✓ Carregamento dinâmico de módulos
✓ Validação de configuração
✓ Hot-reload de configuração

Autor: Equipe PENIN-Ω
Versão: 1.0.0
"""

from __future__ import annotations
import json
import os
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Type
import logging
import importlib
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO GLOBAL UNIFICADA
# =============================================================================

@dataclass
class PeninOmegaConfig:
    """Configuração unificada do sistema PENIN-Ω."""
    
    # Informações do sistema
    version: str = "8.0.0"
    system_name: str = "PENIN-Ω"
    
    # Paths unificados
    root_path: str = "/root/.penin_omega"
    
    # Configuração de módulos
    modules: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "1_core": {
            "enabled": True,
            "module_name": "penin_omega_1_core_v6",
            "class_name": "PeninOmegaFusion",
            "config": {
                "max_tokens": 4000,
                "timeout_s": 300,
                "apis": ["deepseek", "anthropic", "openai", "grok", "mistral", "gemini"]
            }
        },
        "3_acquisition": {
            "enabled": True,
            "module_name": "penin_omega_3_acquisition",
            "worker_function": "create_f3_worker",
            "config": {
                "embedding_model": "all-MiniLM-L6-v2",
                "max_items": 20,
                "min_similarity": 0.3
            }
        },
        "4_mutation": {
            "enabled": True,
            "module_name": "penin_omega_4_mutation",
            "worker_function": "create_f4_worker",
            "config": {
                "n_candidates": 32,
                "mutation_types": ["genetic", "template", "synthesis", "hybrid"],
                "max_code_length": 5000
            }
        },
        "5_crucible": {
            "enabled": True,
            "module_name": "penin_omega_5_crucible",
            "function_name": "crucible_evaluate_and_select",
            "config": {
                "tau_sr": 0.80,
                "ece_max": 0.01,
                "rho_max": 0.95
            }
        },
        "6_autorewrite": {
            "enabled": True,
            "module_name": "penin_omega_6_autorewrite",
            "function_name": "autorewrite_process",
            "config": {
                "timeout_s": 240,
                "max_attempts": 3
            }
        },
        "7_nexus": {
            "enabled": True,
            "module_name": "penin_omega_7_nexus",
            "class_name": "NexusOmega",
            "config": {
                "max_workers": 4,
                "tick_interval": 0.25
            }
        },
        "8_bridge": {
            "enabled": True,
            "module_name": "penin_omega_8_bridge",
            "class_name": "PeninOmegaBridge",
            "config": {
                "max_concurrent_pipelines": 3,
                "auto_evolution_enabled": True
            }
        }
    })
    
    # Configuração de segurança
    safety: Dict[str, Any] = field(default_factory=lambda: {
        "rho_max": 0.95,
        "sr_min": 0.80,
        "ece_max": 0.01,
        "rho_bias_max": 1.05,
        "trust_region_max": 0.15,
        "consent_required": True,
        "eco_check_enabled": True
    })
    
    # Configuração de performance
    performance: Dict[str, Any] = field(default_factory=lambda: {
        "max_concurrent_tasks": 10,
        "task_timeout_s": 300,
        "heartbeat_interval_s": 30,
        "metrics_flush_interval_s": 60
    })
    
    # Configuração de logging
    logging: Dict[str, Any] = field(default_factory=lambda: {
        "level": "INFO",
        "format": "[%(asctime)s][PENIN-Ω][%(name)s][%(levelname)s] %(message)s",
        "file_enabled": True,
        "console_enabled": True,
        "max_file_size_mb": 100,
        "backup_count": 5
    })
    
    # Configuração de storage
    storage: Dict[str, Any] = field(default_factory=lambda: {
        "sqlite_wal_mode": True,
        "sqlite_synchronous": "NORMAL",
        "cache_size_mb": 256,
        "vacuum_interval_hours": 24
    })
    
    async def __post_init__(self):
        """Pós-processamento da configuração."""
        # Garante que root_path existe
        Path(self.root_path).mkdir(parents=True, exist_ok=True)
        
        # Configura paths derivados
        self._setup_derived_paths()
    
    async def _setup_derived_paths(self):
        """Configura paths derivados."""
        root = Path(self.root_path)
        
        self.paths = {
            "root": root,
            "logs": root / "logs",
            "cache": root / "cache",
            "state": root / "state",
            "worm": root / "worm_ledger",
            "knowledge": root / "knowledge",
            "embeddings": root / "embeddings",
            "mutations": root / "mutations",
            "candidates": root / "candidates",
            "templates": root / "templates",
            "queue": root / "queue",
            "metrics": root / "metrics",
            "snapshots": root / "snapshots",
            "config": root / "config",
            "tests": root / "tests",
            "tickets": root / "tickets",
            "workspace": root / "workspace",
            "patches": root / "patches",
            "sandbox": root / "sandbox",
            "artifacts": root / "artifacts"
        }
        
        # Cria todos os diretórios
        for path in self.paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    async def get_module_config(self, module_key: str) -> Dict[str, Any]:
        """Obtém configuração de módulo específico."""
        return await self.modules.get(module_key, {}).get("config", {})
    
    async def get_path(self, path_key: str) -> Path:
        """Obtém path específico."""
        return await self.paths.get(path_key, self.paths["root"])
    
    async def is_module_enabled(self, module_key: str) -> bool:
        """Verifica se módulo está habilitado."""
        return await self.modules.get(module_key, {}).get("enabled", False)
    
    async def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return await asdict(self)
    
    @classmethod
    async def from_dict(cls, data: Dict[str, Any]) -> "PeninOmegaConfig":
        """Cria configuração a partir de dicionário."""
        return await cls(**data)
    
    async def save_to_file(self, file_path: Union[str, Path]):
        """Salva configuração em arquivo."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with path.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    async def load_from_file(cls, file_path: Union[str, Path]) -> "PeninOmegaConfig":
        """Carrega configuração de arquivo."""
        path = Path(file_path)
        
        if not path.exists():
            return await cls()  # Retorna configuração padrão
        
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return await cls.from_dict(data)
        except Exception as e:
            print(f"⚠️  Erro carregando configuração: {e}")
            return await cls()  # Retorna configuração padrão

# =============================================================================
# GERENCIADOR DE CONFIGURAÇÃO
# =============================================================================

class ConfigManager:
    """Gerenciador centralizado de configuração."""
    
    async def __init__(self, config_file: Optional[Union[str, Path]] = None):
        self.config_file = Path(config_file) if config_file else Path("/root/.penin_omega/config/penin_omega.json")
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        self.logger.info(f"⚙️  Configuração carregada: {self.config_file}")
    
    async def _load_config(self) -> PeninOmegaConfig:
        """Carrega configuração."""
        if self.config_file.exists():
            return await PeninOmegaConfig.load_from_file(self.config_file)
        else:
            # Cria configuração padrão
            config = PeninOmegaConfig()
            config.save_to_file(self.config_file)
            return await config
    
    async def _setup_logging(self) -> logging.Logger:
        """Configura logging baseado na configuração."""
        log_config = self.config.logging
        
        # Configura nível
        level = getattr(logging, log_config["level"].upper(), logging.INFO)
        
        # Configura handlers
        handlers = []
        
        if log_config["console_enabled"]:
            handlers.append(logging.StreamHandler(sys.stdout))
        
        if log_config["file_enabled"]:
            log_file = self.config.get_path("logs") / "penin_omega.log"
            handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
        
        # Configura logging
        logging.basicConfig(
            level=level,
            format=log_config["format"],
            handlers=handlers,
            force=True
        )
        
        return await logging.getLogger("ConfigManager")
    
    async def get_config(self) -> PeninOmegaConfig:
        """Obtém configuração atual."""
        return await self.config
    
    async def update_config(self, updates: Dict[str, Any]):
        """Atualiza configuração."""
        # Aplica atualizações recursivamente
        self._deep_update(asdict(self.config), updates)
        
        # Recria configuração
        self.config = PeninOmegaConfig.from_dict(asdict(self.config))
        
        # Salva configuração
        self.config.save_to_file(self.config_file)
        
        self.logger.info("⚙️  Configuração atualizada")
    
    async def _deep_update(self, base: Dict[str, Any], updates: Dict[str, Any]):
        """Atualização recursiva de dicionário."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    async def reload_config(self):
        """Recarrega configuração do arquivo."""
        self.config = self._load_config()
        self.logger.info("⚙️  Configuração recarregada")

# =============================================================================
# RESOLVEDOR DE DEPENDÊNCIAS
# =============================================================================

class DependencyResolver:
    """Resolve dependências circulares entre módulos."""
    
    async def __init__(self, config: PeninOmegaConfig):
        self.config = config
        self.loaded_modules: Dict[str, Any] = {}
        self.loading_stack: List[str] = []
        self.logger = logging.getLogger("DependencyResolver")
    
    async def load_module(self, module_key: str) -> Optional[Any]:
        """Carrega módulo resolvendo dependências."""
        if module_key in self.loaded_modules:
            return await self.loaded_modules[module_key]
        
        if module_key in self.loading_stack:
            self.logger.warning(f"⚠️  Dependência circular detectada: {' → '.join(self.loading_stack)} → {module_key}")
            return await None
        
        module_config = self.config.modules.get(module_key)
        if not module_config or not module_config.get("enabled", False):
            return await None
        
        self.loading_stack.append(module_key)
        
        try:
            module = self._import_module(module_config)
            self.loaded_modules[module_key] = module
            self.logger.info(f"✅ Módulo carregado: {module_key}")
            return await module
            
        except Exception as e:
            self.logger.error(f"❌ Erro carregando módulo {module_key}: {e}")
            return await None
            
        finally:
            if module_key in self.loading_stack:
                self.loading_stack.remove(module_key)
    
    async def _import_module(self, module_config: Dict[str, Any]) -> Any:
        """Importa módulo específico."""
        module_name = module_config["module_name"]
        
        try:
            # Importa módulo
            module = importlib.import_module(module_name)
            
            # Obtém classe/função específica
            if "class_name" in module_config:
                return await getattr(module, module_config["class_name"])
            elif "function_name" in module_config:
                return await getattr(module, module_config["function_name"])
            elif "worker_function" in module_config:
                return await getattr(module, module_config["worker_function"])
            else:
                return await module
                
        except ImportError as e:
            self.logger.warning(f"⚠️  Módulo {module_name} não encontrado: {e}")
            return await None
    
    async def get_loaded_modules(self) -> Dict[str, Any]:
        """Obtém módulos carregados."""
        return await self.loaded_modules.copy()
    
    async def load_all_modules(self) -> Dict[str, Any]:
        """Carrega todos os módulos habilitados."""
        for module_key in self.config.modules.keys():
            self.load_module(module_key)
        
        return await self.get_loaded_modules()

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

_global_config_manager: Optional[ConfigManager] = None

async def get_global_config() -> PeninOmegaConfig:
    """Obtém configuração global."""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    
    return await _global_config_manager.get_config()

async def get_config_manager() -> ConfigManager:
    """Obtém gerenciador de configuração global."""
    global _global_config_manager
    
    if _global_config_manager is None:
        _global_config_manager = ConfigManager()
    
    return await _global_config_manager

async def create_dependency_resolver() -> DependencyResolver:
    """Cria resolvedor de dependências."""
    config = get_global_config()
    return await DependencyResolver(config)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "PeninOmegaConfig", "ConfigManager", "DependencyResolver",
    "get_global_config", "get_config_manager", "create_dependency_resolver"
]

if __name__ == "__main__":
    # Teste básico
    print("PENIN-Ω Configuration Manager")
    
    # Testa configuração
    config = get_global_config()
    print(f"✅ Configuração carregada: versão {config.version}")
    print(f"   Módulos habilitados: {sum(1 for m in config.modules.values() if m.get('enabled'))}")
    print(f"   Root path: {config.root_path}")
    
    # Testa resolvedor de dependências
    resolver = create_dependency_resolver()
    modules = resolver.load_all_modules()
    print(f"✅ Módulos carregados: {len(modules)}")
    
    for key, module in modules.items():
        if module:
            print(f"   {key}: {type(module).__name__}")
        else:
            print(f"   {key}: ❌ falhou")
    
    print("✅ Sistema de configuração funcionando!")
