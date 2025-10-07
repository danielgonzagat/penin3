#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Log Controller - Controle Rigoroso de Logs
===================================================
Sistema para controlar e reduzir logs excessivos.
"""

import logging
import sys
from typing import Dict, Set
from pathlib import Path

class LogController:
    """Controlador de logs para reduzir verbosidade excessiva."""
    
    def __init__(self):
        self.suppressed_loggers: Set[str] = set()
        self.log_levels: Dict[str, int] = {}
        self.original_levels: Dict[str, int] = {}
        
    def suppress_logger(self, logger_name: str):
        """Suprime completamente um logger."""
        logger = logging.getLogger(logger_name)
        if logger_name not in self.original_levels:
            self.original_levels[logger_name] = logger.level
        
        logger.setLevel(logging.CRITICAL + 1)  # Acima de CRITICAL
        self.suppressed_loggers.add(logger_name)
    
    def set_log_level(self, logger_name: str, level: int):
        """Define nível específico para um logger."""
        logger = logging.getLogger(logger_name)
        if logger_name not in self.original_levels:
            self.original_levels[logger_name] = logger.level
        
        logger.setLevel(level)
        self.log_levels[logger_name] = level
    
    def reduce_verbose_logs(self):
        """Reduz logs verbosos do sistema."""
        
        # Suprime logs muito verbosos
        verbose_loggers = [
            "urllib3.connectionpool",
            "requests.packages.urllib3",
            "asyncio",
            "concurrent.futures",
            "threading"
        ]
        
        for logger_name in verbose_loggers:
            self.suppress_logger(logger_name)
        
        # Reduz nível de logs do sistema
        system_loggers = {
            "AutonomousCore": logging.WARNING,
            "CreativityEngine": logging.ERROR,
            "TotalAdmin": logging.ERROR,
            "FusionEngine": logging.WARNING,
            "PerformanceOptimizer": logging.WARNING,
            "UCBOptimizer": logging.ERROR,
            "NexusScheduler": logging.WARNING,
            "Watchdog": logging.ERROR,
            "ComplianceMonitor": logging.WARNING
        }
        
        for logger_name, level in system_loggers.items():
            self.set_log_level(logger_name, level)
    
    def set_production_mode(self):
        """Configura logs para modo produção (mínimo)."""
        
        # Nível geral mais alto
        logging.getLogger().setLevel(logging.WARNING)
        
        # Loggers críticos apenas
        critical_only = [
            "PENIN_OMEGA_MASTER",
            "PENIN_OMEGA_AUTONOMOUS", 
            "PENIN_OMEGA_FUSION",
            "PENIN_OMEGA_CREATIVITY",
            "PENIN_OMEGA_ADMIN"
        ]
        
        for logger_name in critical_only:
            self.set_log_level(logger_name, logging.ERROR)
        
        # Suprime logs de desenvolvimento
        dev_loggers = [
            "MasterSystem",
            "ArchitecturalAuditor",
            "ModuleAuditor", 
            "AdvancedAuditor",
            "TestFramework",
            "RealDataPipeline"
        ]
        
        for logger_name in dev_loggers:
            self.set_log_level(logger_name, logging.CRITICAL)
    
    def restore_original_levels(self):
        """Restaura níveis originais dos loggers."""
        for logger_name, original_level in self.original_levels.items():
            logger = logging.getLogger(logger_name)
            logger.setLevel(original_level)
        
        self.suppressed_loggers.clear()
        self.log_levels.clear()
        self.original_levels.clear()
    
    def get_log_summary(self) -> Dict[str, str]:
        """Retorna resumo do estado dos logs."""
        return {
            "suppressed_count": len(self.suppressed_loggers),
            "modified_count": len(self.log_levels),
            "suppressed_loggers": list(self.suppressed_loggers),
            "current_root_level": logging.getLevelName(logging.getLogger().level)
        }

class QuietFilter(logging.Filter):
    """Filtro para reduzir logs repetitivos."""
    
    def __init__(self):
        super().__init__()
        self.seen_messages: Dict[str, int] = {}
        self.max_repeats = 3
    
    def filter(self, record):
        """Filtra mensagens repetitivas."""
        message_key = f"{record.name}:{record.levelname}:{record.getMessage()}"
        
        if message_key in self.seen_messages:
            self.seen_messages[message_key] += 1
            if self.seen_messages[message_key] > self.max_repeats:
                return False  # Suprime mensagem repetitiva
        else:
            self.seen_messages[message_key] = 1
        
        return True

# Instância global
log_controller = LogController()

def setup_production_logging():
    """Configura logging para produção."""
    log_controller.set_production_mode()
    
    # Adiciona filtro anti-repetição
    quiet_filter = QuietFilter()
    logging.getLogger().addFilter(quiet_filter)

def setup_development_logging():
    """Configura logging para desenvolvimento."""
    log_controller.reduce_verbose_logs()

def setup_minimal_logging():
    """Configura logging mínimo."""
    # Apenas erros críticos
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Remove handlers desnecessários
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler) and handler.stream == sys.stdout:
            root_logger.removeHandler(handler)

def restore_logging():
    """Restaura configuração original de logging."""
    log_controller.restore_original_levels()

# Configuração automática baseada no contexto
def auto_configure_logging():
    """Configura logging automaticamente baseado no ambiente."""
    try:
        # Verifica se está em modo de teste
        if 'pytest' in sys.modules or 'unittest' in sys.modules:
            setup_minimal_logging()
        # Verifica se está em produção (simplificado)
        elif Path("/root/.penin_omega/production").exists():
            setup_production_logging()
        else:
            setup_development_logging()
    except Exception:
        # Fallback seguro
        setup_minimal_logging()

if __name__ == "__main__":
    # Teste do controlador
    print("Testando controle de logs...")
    
    # Configura logs verbosos
    logger1 = logging.getLogger("TestLogger1")
    logger2 = logging.getLogger("TestLogger2")
    
    logger1.setLevel(logging.DEBUG)
    logger2.setLevel(logging.DEBUG)
    
    print(f"Antes - Logger1: {logging.getLevelName(logger1.level)}")
    print(f"Antes - Logger2: {logging.getLevelName(logger2.level)}")
    
    # Aplica controle
    log_controller.set_log_level("TestLogger1", logging.ERROR)
    log_controller.suppress_logger("TestLogger2")
    
    print(f"Depois - Logger1: {logging.getLevelName(logger1.level)}")
    print(f"Depois - Logger2: {logging.getLevelName(logger2.level)}")
    
    # Resumo
    summary = log_controller.get_log_summary()
    print(f"Resumo: {summary}")
    
    # Restaura
    log_controller.restore_original_levels()
    print(f"Restaurado - Logger1: {logging.getLevelName(logger1.level)}")
    print(f"Restaurado - Logger2: {logging.getLevelName(logger2.level)}")
