#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Configuration Manager - Configuração Centralizada
==========================================================
Sistema centralizado e rigoroso de configuração.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime, timezone

class ConfigurationManager:
    """Gerenciador centralizado de configuração."""
    
    async def __init__(self):
        self.config_dir = Path("/root/.penin_omega/config")
        self.config_file = self.config_dir / "penin_omega_master_config.json"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self._config: Dict[str, Any] = {}
        self._defaults: Dict[str, Any] = {}
        self._load_defaults()
        self._load_config()
    
    async def _load_defaults(self):
        """Carrega configurações padrão."""
        self._defaults = {
            # Sistema
            "system": {
                "name": "PENIN-Ω",
                "version": "1.0.0",
                "environment": "development",
                "debug": False,
                "max_threads": 4,
                "timeout_seconds": 30
            },
            
            # Paths
            "paths": {
                "root": "/root/.penin_omega",
                "logs": "/root/.penin_omega/logs",
                "cache": "/root/.penin_omega/cache",
                "worm": "/root/.penin_omega/worm",
                "state": "/root/.penin_omega/state",
                "modules": "/root/.penin_omega/modules",
                "config": "/root/.penin_omega/config"
            },
            
            # Logging
            "logging": {
                "level": "INFO",
                "max_file_size_mb": 10,
                "backup_count": 5,
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "enable_structured": True,
                "reduce_verbosity": True
            },
            
            # Performance
            "performance": {
                "enable_optimization": True,
                "cache_size_mb": 100,
                "gc_threshold": [700, 10, 10],
                "max_memory_mb": 512,
                "enable_profiling": False
            },
            
            # Security
            "security": {
                "enable_dlp": True,
                "enable_worm": True,
                "sigma_guard_threshold": 0.8,
                "ir_ic_threshold": 1.0,
                "sr_omega_threshold": 0.8,
                "max_violations": 10
            },
            
            # Database
            "database": {
                "worm_db_path": "/root/.penin_omega/worm/worm_ledger.db",
                "state_db_path": "/root/.penin_omega/state/global_state.db",
                "dlp_db_path": "/root/.penin_omega/security/dlp_violations.db",
                "backup_interval_hours": 24,
                "vacuum_interval_hours": 168
            },
            
            # Modules
            "modules": {
                "enable_all": True,
                "core_module": "penin_omega_1_core_v6",
                "strategy_module": "penin_omega_2_strategy",
                "acquisition_module": "penin_omega_3_acquisition",
                "mutation_module": "penin_omega_4_mutation",
                "crucible_module": "penin_omega_5_crucible",
                "autorewrite_module": "penin_omega_6_autorewrite",
                "nexus_module": "penin_omega_7_nexus",
                "governance_module": "penin_omega_8_governance_hub"
            },
            
            # Pipeline
            "pipeline": {
                "enable_real_data": True,
                "max_candidates": 10,
                "timeout_per_stage": 30,
                "enable_parallel": False,
                "retry_attempts": 3
            },
            
            # Autonomous
            "autonomous": {
                "enable_evolution": True,
                "evolution_interval_seconds": 30,
                "enable_creativity": True,
                "creativity_interval_seconds": 60,
                "enable_administration": True,
                "admin_interval_seconds": 120
            },
            
            # Testing
            "testing": {
                "enable_automated": True,
                "test_timeout": 10,
                "parallel_tests": False,
                "generate_reports": True
            },
            
            # API
            "api": {
                "enable_multi_api": True,
                "timeout_seconds": 15,
                "max_retries": 3,
                "enable_fallback": True
            }
        }
    
    async def _load_config(self):
        """Carrega configuração do arquivo."""
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    file_config = json.load(f)
                
                # Merge com defaults
                self._config = self._deep_merge(self._defaults.copy(), file_config)
            else:
                # Usa apenas defaults
                self._config = self._defaults.copy()
                self._save_config()
        
        except Exception as e:
            logger.info(f"Erro ao carregar configuração: {e}")
            self._config = self._defaults.copy()
    
    async def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge profundo de dicionários."""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return await result
    
    async def _save_config(self):
        """Salva configuração no arquivo."""
        try:
            config_to_save = self._config.copy()
            config_to_save["_metadata"] = {
                "last_updated": datetime.now(timezone.utc).isoformat(),
                "version": self._config["system"]["version"]
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_to_save, f, indent=2)
        
        except Exception as e:
            logger.info(f"Erro ao salvar configuração: {e}")
    
    async def get(self, key_path: str, default: Any = None) -> Any:
        """Obtém valor de configuração usando path com pontos."""
        try:
            keys = key_path.split('.')
            value = self._config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return await default
            
            return await value
        
        except Exception:
            return await default
    
    async def set(self, key_path: str, value: Any, save: bool = True):
        """Define valor de configuração usando path com pontos."""
        try:
            keys = key_path.split('.')
            config = self._config
            
            # Navega até o penúltimo nível
            for key in keys[:-1]:
                if key not in config:
                    config[key] = {}
                config = config[key]
            
            # Define o valor final
            config[keys[-1]] = value
            
            if save:
                self._save_config()
        
        except Exception as e:
            logger.info(f"Erro ao definir configuração {key_path}: {e}")
    
    async def get_section(self, section: str) -> Dict[str, Any]:
        """Obtém seção completa de configuração."""
        return await self._config.get(section, {})
    
    async def update_section(self, section: str, values: Dict[str, Any], save: bool = True):
        """Atualiza seção completa de configuração."""
        if section not in self._config:
            self._config[section] = {}
        
        self._config[section].update(values)
        
        if save:
            self._save_config()
    
    async def reset_to_defaults(self, section: Optional[str] = None):
        """Reseta configuração para padrões."""
        if section:
            if section in self._defaults:
                self._config[section] = self._defaults[section].copy()
        else:
            self._config = self._defaults.copy()
        
        self._save_config()
    
    async def validate_config(self) -> Dict[str, Any]:
        """Valida configuração atual."""
        issues = []
        
        # Valida paths
        for path_key, path_value in self.get_section("paths").items():
            path_obj = Path(path_value)
            if not path_obj.exists():
                try:
                    path_obj.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create path {path_key}: {path_value} - {e}")
        
        # Valida valores numéricos
        numeric_validations = [
            ("system.max_threads", 1, 32),
            ("system.timeout_seconds", 1, 300),
            ("performance.cache_size_mb", 1, 1024),
            ("performance.max_memory_mb", 64, 2048),
            ("security.max_violations", 1, 100)
        ]
        
        for key_path, min_val, max_val in numeric_validations:
            value = self.get(key_path)
            if not isinstance(value, (int, float)) or not (min_val <= value <= max_val):
                issues.append(f"Invalid {key_path}: {value} (should be {min_val}-{max_val})")
        
        return await {
            "valid": len(issues) == 0,
            "issues": issues,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    async def get_environment_config(self) -> Dict[str, Any]:
        """Obtém configuração baseada no ambiente."""
        env = self.get("system.environment", "development")
        
        env_configs = {
            "development": {
                "logging.level": "DEBUG",
                "system.debug": True,
                "performance.enable_profiling": True,
                "testing.enable_automated": True
            },
            "production": {
                "logging.level": "WARNING",
                "system.debug": False,
                "performance.enable_profiling": False,
                "testing.enable_automated": False
            },
            "testing": {
                "logging.level": "CRITICAL",
                "system.debug": False,
                "performance.enable_optimization": False,
                "autonomous.enable_evolution": False
            }
        }
        
        return await env_configs.get(env, {})
    
    async def apply_environment_config(self):
        """Aplica configuração do ambiente atual."""
        env_config = self.get_environment_config()
        
        for key_path, value in env_config.items():
            self.set(key_path, value, save=False)
        
        self._save_config()
    
    async def export_config(self, file_path: Union[str, Path]) -> bool:
        """Exporta configuração para arquivo."""
        try:
            export_data = {
                "config": self._config,
                "exported_at": datetime.now(timezone.utc).isoformat(),
                "version": self.get("system.version")
            }
            
            with open(file_path, 'w') as f:
                json.dump(export_data, f, indent=2)
            
            return await True
        
        except Exception as e:
            logger.info(f"Erro ao exportar configuração: {e}")
            return await False
    
    async def import_config(self, file_path: Union[str, Path]) -> bool:
        """Importa configuração de arquivo."""
        try:
            with open(file_path, 'r') as f:
                import_data = json.load(f)
            
            if "config" in import_data:
                self._config = self._deep_merge(self._defaults.copy(), import_data["config"])
                self._save_config()
                return await True
            
            return await False
        
        except Exception as e:
            logger.info(f"Erro ao importar configuração: {e}")
            return await False

# Instância global
config_manager = ConfigurationManager()

# Funções de conveniência
async def get_config(key_path: str, default: Any = None) -> Any:
    """Função de conveniência para obter configuração."""
    return await config_manager.get(key_path, default)

async def set_config(key_path: str, value: Any, save: bool = True):
    """Função de conveniência para definir configuração."""
    config_manager.set(key_path, value, save)

async def get_section(section: str) -> Dict[str, Any]:
    """Função de conveniência para obter seção."""
    return await config_manager.get_section(section)

async def validate_config() -> Dict[str, Any]:
    """Função de conveniência para validar configuração."""
    return await config_manager.validate_config()

# Inicialização automática
config_manager.apply_environment_config()

if __name__ == "__main__":
    # Teste do gerenciador de configuração
    logger.info("Testando gerenciador de configuração...")
    
    # Testa get/set
    logger.info(f"System name: {get_config('system.name')}")
    logger.info(f"Log level: {get_config('logging.level')}")
    
    # Testa validação
    validation = validate_config()
    logger.info(f"Config valid: {validation['valid']}")
    if validation['issues']:
        logger.info(f"Issues: {validation['issues']}")
    
    # Testa seção
    paths = get_section("paths")
    logger.info(f"Paths configured: {len(paths)}")
    
    logger.info("Configuração centralizada funcionando!")
