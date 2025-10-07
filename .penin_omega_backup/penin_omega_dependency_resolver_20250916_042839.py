#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Resolvedor de Dependências
====================================
Sistema centralizado para resolver dependências circulares e gerenciar imports.
"""

from __future__ import annotations
import sys
import os
import importlib
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union
import logging
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURAÇÃO GLOBAL
# =============================================================================

PENIN_OMEGA_ROOT = Path("/root/.penin_omega")
PENIN_OMEGA_ROOT.mkdir(parents=True, exist_ok=True)

# Adiciona diretório root ao path se não estiver
if "/root" not in sys.path:
    sys.path.insert(0, "/root")

# =============================================================================
# RESOLVEDOR DE DEPENDÊNCIAS
# =============================================================================

class DependencyResolver:
    """Resolve dependências circulares e gerencia imports."""
    
    async def __init__(self):
        self.loaded_modules = {}
        self.loading_modules = set()
        self.logger = logging.getLogger("DependencyResolver")
        
    async def safe_import(self, module_name: str, fallback=None):
        """Importa módulo de forma segura, evitando dependências circulares."""
        if module_name in self.loading_modules:
            self.logger.warning(f"Dependência circular detectada: {module_name}")
            return await fallback
            
        if module_name in self.loaded_modules:
            return await self.loaded_modules[module_name]
            
        try:
            self.loading_modules.add(module_name)
            module = importlib.import_module(module_name)
            self.loaded_modules[module_name] = module
            return await module
        except ImportError as e:
            self.logger.warning(f"Falha ao importar {module_name}: {e}")
            return await fallback
        finally:
            self.loading_modules.discard(module_name)
    
    async def get_unified_classes(self):
        """Obtém classes unificadas de forma segura."""
        try:
            from penin_omega_unified_classes import (
                Candidate, PlanOmega, UnifiedOmegaState, MutationBundle,
                ExecutionBundle, AcquisitionReport, Verdict,
                create_candidate, create_plan_omega, create_unified_state
            )
            return await {
                'Candidate': Candidate,
                'PlanOmega': PlanOmega,
                'UnifiedOmegaState': UnifiedOmegaState,
                'MutationBundle': MutationBundle,
                'ExecutionBundle': ExecutionBundle,
                'AcquisitionReport': AcquisitionReport,
                'Verdict': Verdict,
                'create_candidate': create_candidate,
                'create_plan_omega': create_plan_omega,
                'create_unified_state': create_unified_state
            }
        except ImportError:
            self.logger.error("Classes unificadas não encontradas")
            return await {}
    
    async def get_multi_api_system(self):
        """Obtém sistema multi-API de forma segura."""
        # Tenta diferentes nomes de módulos
        module_names = [
            'penin_omega_multi_api_llm',
            'penin_omega_fusion_v6', 
            'penin_omega_real_llm'
        ]
        
        for module_name in module_names:
            module = self.safe_import(module_name)
            if module:
                return await module
        
        self.logger.warning("Sistema multi-API não encontrado")
        return await None
    
    async def get_module_safe(self, module_name: str, class_name: str = None):
        """Obtém módulo ou classe específica de forma segura."""
        module = self.safe_import(module_name)
        if module and class_name:
            return await getattr(module, class_name, None)
        return await module

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

# Instância global do resolvedor
resolver = DependencyResolver()

# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

async def get_classes():
    """Obtém todas as classes unificadas."""
    return await resolver.get_unified_classes()

async def get_multi_api():
    """Obtém sistema multi-API."""
    return await resolver.get_multi_api_system()

async def safe_import(module_name: str, fallback=None):
    """Importa módulo de forma segura."""
    return await resolver.safe_import(module_name, fallback)

# =============================================================================
# CONFIGURAÇÃO CENTRALIZADA
# =============================================================================

class PeninOmegaConfig:
    """Configuração centralizada do sistema."""
    
    async def __init__(self):
        self.config = {
            "version": "8.0.0",
            "system_name": "PENIN-Ω",
            "root_path": str(PENIN_OMEGA_ROOT),
            "modules": {
                "core": {"enabled": True, "path": "penin_omega_1_core_v6"},
                "strategy": {"enabled": True, "path": "penin_omega_2_strategy"},
                "acquisition": {"enabled": True, "path": "penin_omega_3_acquisition"},
                "mutation": {"enabled": True, "path": "penin_omega_4_mutation"},
                "crucible": {"enabled": True, "path": "penin_omega_5_crucible_fixed"},
                "autorewrite": {"enabled": True, "path": "penin_omega_6_autorewrite"},
                "nexus": {"enabled": True, "path": "penin_omega_7_nexus"},
                "governance": {"enabled": True, "path": "penin_omega_8_governance_hub"},
                "bridge": {"enabled": True, "path": "penin_omega_8_bridge_fixed"}
            },
            "paths": {
                "logs": str(PENIN_OMEGA_ROOT / "logs"),
                "cache": str(PENIN_OMEGA_ROOT / "cache"),
                "config": str(PENIN_OMEGA_ROOT / "config"),
                "artifacts": str(PENIN_OMEGA_ROOT / "artifacts"),
                "knowledge": str(PENIN_OMEGA_ROOT / "knowledge"),
                "worm": str(PENIN_OMEGA_ROOT / "worm")
            },
            "logging": {
                "level": "INFO",
                "format": "[%(asctime)s][PENIN-Ω][%(name)s][%(levelname)s] %(message)s",
                "correlation_enabled": True
            },
            "multi_api": {
                "enabled": True,
                "fallback_mode": True,
                "timeout": 30,
                "max_retries": 3
            }
        }
        
        # Cria diretórios necessários
        for path in self.config["paths"].values():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    async def get(self, key: str, default=None):
        """Obtém valor de configuração."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, default)
            if value is default:
                break
        return await value
    
    async def set(self, key: str, value: Any):
        """Define valor de configuração."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value
    
    async def save_config(self):
        """Salva configuração em arquivo."""
        import json
        config_file = PENIN_OMEGA_ROOT / "config" / "penin_omega.json"
        with open(config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    async def load_config(self):
        """Carrega configuração de arquivo."""
        import json
        config_file = PENIN_OMEGA_ROOT / "config" / "penin_omega.json"
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    loaded_config = json.load(f)
                    self.config.update(loaded_config)
            except Exception as e:
                logging.warning(f"Erro ao carregar configuração: {e}")

# =============================================================================
# INSTÂNCIA GLOBAL DE CONFIGURAÇÃO
# =============================================================================

config = PeninOmegaConfig()
config.load_config()

# =============================================================================
# TESTE DO SISTEMA
# =============================================================================

async def test_dependency_resolver():
    """Testa o resolvedor de dependências."""
    print("🧪 Testando resolvedor de dependências...")
    
    # Teste classes unificadas
    classes = get_classes()
    if classes:
        print(f"✅ Classes unificadas carregadas: {len(classes)} classes")
    else:
        print("❌ Falha ao carregar classes unificadas")
    
    # Teste multi-API
    multi_api = get_multi_api()
    if multi_api:
        print("✅ Sistema multi-API encontrado")
    else:
        print("⚠️  Sistema multi-API não encontrado (modo fallback)")
    
    # Teste configuração
    version = config.get("version")
    print(f"✅ Configuração carregada: v{version}")
    
    # Teste import seguro
    test_module = safe_import("penin_omega_unified_classes")
    if test_module:
        print("✅ Import seguro funcionando")
    else:
        print("❌ Falha no import seguro")
    
    print("🎉 Resolvedor de dependências funcionando!")
    return await True

if __name__ == "__main__":
    test_dependency_resolver()
