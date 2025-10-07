#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Sistema de Logging Estruturado
=========================================
Sistema unificado de logging com correlação de execuções e rastreamento completo.
"""

from __future__ import annotations
import json
import logging
import logging.handlers
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import traceback
import sys

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PENIN_OMEGA_ROOT = Path("/root/.penin_omega")
LOGS_PATH = PENIN_OMEGA_ROOT / "logs"
LOGS_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CLASSES DE LOGGING
# =============================================================================

@dataclass
class LogContext:
    """Contexto de logging para correlação."""
    correlation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    module_name: str = "unknown"
    operation: str = "unknown"
    pipeline_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    parent_context: Optional[str] = None
    
    async def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return await {k: v for k, v in asdict(self).items() if v is not None}

@dataclass
class StructuredLogRecord:
    """Registro de log estruturado."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    level: str = "INFO"
    message: str = ""
    module: str = "unknown"
    function: str = "unknown"
    line_number: int = 0
    context: LogContext = field(default_factory=LogContext)
    extra_data: Dict[str, Any] = field(default_factory=dict)
    exception_info: Optional[Dict[str, Any]] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    
    async def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        data = asdict(self)
        data['context'] = self.context.to_dict()
        return await data

# =============================================================================
# FORMATADOR ESTRUTURADO
# =============================================================================

class StructuredFormatter(logging.Formatter):
    """Formatador para logs estruturados."""
    
    async def format(self, record: logging.LogRecord) -> str:
        """Formata registro de log."""
        # Obtém contexto atual
        context = getattr(threading.current_thread(), 'log_context', LogContext())
        
        # Cria registro estruturado
        structured_record = StructuredLogRecord(
            level=record.levelname,
            message=record.getMessage(),
            module=record.module if hasattr(record, 'module') else record.name,
            function=record.funcName,
            line_number=record.lineno,
            context=context,
            extra_data=getattr(record, 'extra_data', {})
        )
        
        # Adiciona informações de exceção se disponível
        if record.exc_info:
            structured_record.exception_info = {
                "type": record.exc_info[0].__name__ if record.exc_info[0] else None,
                "message": str(record.exc_info[1]) if record.exc_info[1] else None,
                "traceback": traceback.format_exception(*record.exc_info)
            }
        
        # Adiciona métricas de performance se disponível
        if hasattr(record, 'performance_metrics'):
            structured_record.performance_metrics = record.performance_metrics
        
        return await json.dumps(structured_record.to_dict(), ensure_ascii=False)

# =============================================================================
# GERENCIADOR DE LOGGING
# =============================================================================

class StructuredLoggingManager:
    """Gerencia sistema de logging estruturado."""
    
    async def __init__(self):
        self.loggers = {}
        self.handlers = {}
        self.context_stack = threading.local()
        self.correlation_map = {}
        
        # Configuração padrão
        self.config = {
            "level": logging.INFO,
            "format": "structured",
            "max_file_size": 50 * 1024 * 1024,  # 50MB
            "backup_count": 5,
            "correlation_enabled": True,
            "performance_tracking": True
        }
        
        self._setup_root_logger()
        self._setup_file_handlers()
    
    async def _setup_root_logger(self):
        """Configura logger raiz."""
        root_logger = logging.getLogger("PENIN-Ω")
        root_logger.setLevel(self.config["level"])
        
        # Remove handlers existentes
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Handler para console
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatador simples para console
        console_formatter = logging.Formatter(
            '[%(asctime)s][PENIN-Ω][%(name)s][%(levelname)s] %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        root_logger.addHandler(console_handler)
        
        self.loggers["root"] = root_logger
    
    async def _setup_file_handlers(self):
        """Configura handlers de arquivo."""
        # Handler principal estruturado
        main_handler = logging.handlers.RotatingFileHandler(
            LOGS_PATH / "penin_omega_structured.log",
            maxBytes=self.config["max_file_size"],
            backupCount=self.config["backup_count"]
        )
        main_handler.setLevel(logging.DEBUG)
        main_handler.setFormatter(StructuredFormatter())
        
        # Handler para erros
        error_handler = logging.handlers.RotatingFileHandler(
            LOGS_PATH / "penin_omega_errors.log",
            maxBytes=self.config["max_file_size"],
            backupCount=self.config["backup_count"]
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        
        # Handler para auditoria
        audit_handler = logging.handlers.RotatingFileHandler(
            LOGS_PATH / "penin_omega_audit.log",
            maxBytes=self.config["max_file_size"],
            backupCount=self.config["backup_count"]
        )
        audit_handler.setLevel(logging.INFO)
        audit_handler.setFormatter(StructuredFormatter())
        
        self.handlers.update({
            "main": main_handler,
            "error": error_handler,
            "audit": audit_handler
        })
        
        # Adiciona handlers ao logger raiz
        root_logger = self.loggers["root"]
        for handler in self.handlers.values():
            root_logger.addHandler(handler)
    
    async def get_logger(self, name: str) -> logging.Logger:
        """Obtém logger com nome específico."""
        if name not in self.loggers:
            logger = logging.getLogger(f"PENIN-Ω.{name}")
            logger.setLevel(self.config["level"])
            self.loggers[name] = logger
        
        return await self.loggers[name]
    
    @contextmanager
    async def log_context(self, **context_data):
        """Context manager para logging com contexto."""
        # Cria novo contexto
        context = LogContext(**context_data)
        
        # Salva contexto anterior
        thread = threading.current_thread()
        old_context = getattr(thread, 'log_context', None)
        
        # Define novo contexto
        thread.log_context = context
        
        try:
            yield context
        finally:
            # Restaura contexto anterior
            if old_context:
                thread.log_context = old_context
            else:
                delattr(thread, 'log_context')
    
    async def log_operation_start(self, operation: str, module: str, **extra_data):
        """Registra início de operação."""
        logger = self.get_logger(module)
        
        # Cria contexto se não existir
        thread = threading.current_thread()
        if not hasattr(thread, 'log_context'):
            thread.log_context = LogContext(module_name=module, operation=operation)
        
        # Adiciona métricas de performance
        start_time = time.time()
        extra_data['operation_start_time'] = start_time
        extra_data['operation'] = operation
        
        logger.info(f"🚀 Iniciando operação: {operation}", extra={'extra_data': extra_data})
        return await start_time
    
    async def log_operation_end(self, operation: str, module: str, start_time: float, success: bool = True, **extra_data):
        """Registra fim de operação."""
        logger = self.get_logger(module)
        
        # Calcula métricas
        end_time = time.time()
        duration = end_time - start_time
        
        performance_metrics = {
            "duration_seconds": duration,
            "start_time": start_time,
            "end_time": end_time,
            "success": success
        }
        
        status_icon = "✅" if success else "❌"
        status_text = "concluída" if success else "falhou"
        
        logger.info(
            f"{status_icon} Operação {status_text}: {operation} ({duration:.3f}s)",
            extra={
                'extra_data': extra_data,
                'performance_metrics': performance_metrics
            }
        )
    
    async def log_pipeline_stage(self, stage: str, pipeline_id: str, module: str, **extra_data):
        """Registra estágio de pipeline."""
        logger = self.get_logger(module)
        
        with self.log_context(pipeline_id=pipeline_id, operation=f"pipeline_stage_{stage}"):
            logger.info(f"📊 Pipeline {pipeline_id} - Estágio {stage}", extra={'extra_data': extra_data})
    
    async def log_error_with_context(self, error: Exception, module: str, operation: str = None, **extra_data):
        """Registra erro com contexto completo."""
        logger = self.get_logger(module)
        
        error_data = {
            "error_type": type(error).__name__,
            "error_message": str(error),
            "operation": operation,
            **extra_data
        }
        
        logger.error(
            f"❌ Erro em {module}: {error}",
            exc_info=True,
            extra={'extra_data': error_data}
        )
    
    async def log_audit_event(self, event: str, module: str, **extra_data):
        """Registra evento de auditoria."""
        audit_logger = logging.getLogger("PENIN-Ω.audit")
        
        audit_data = {
            "event_type": event,
            "module": module,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **extra_data
        }
        
        audit_logger.info(f"🔍 Auditoria: {event}", extra={'extra_data': audit_data})
    
    async def search_logs(self, query: str, level: str = None, module: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Busca logs estruturados."""
        results = []
        
        try:
            log_file = LOGS_PATH / "penin_omega_structured.log"
            if not log_file.exists():
                return await results
            
            with open(log_file, 'r') as f:
                for line in f:
                    try:
                        log_entry = json.loads(line.strip())
                        
                        # Filtros
                        if level and log_entry.get('level') != level:
                            continue
                        
                        if module and log_entry.get('module') != module:
                            continue
                        
                        # Busca por query
                        if query.lower() in log_entry.get('message', '').lower():
                            results.append(log_entry)
                        
                        if len(results) >= limit:
                            break
                            
                    except json.JSONDecodeError:
                        continue
        
        except Exception as e:
            self.log_error_with_context(e, "logging_manager", "search_logs")
        
        return await results[-limit:]  # Retorna os mais recentes
    
    async def get_correlation_trace(self, correlation_id: str) -> List[Dict[str, Any]]:
        """Obtém trace completo de uma correlação."""
        return await self.search_logs(correlation_id, limit=1000)
    
    async def cleanup_old_logs(self, days: int = 30):
        """Limpa logs antigos."""
        try:
            cutoff_time = time.time() - (days * 24 * 3600)
            
            for log_file in LOGS_PATH.glob("*.log*"):
                if log_file.stat().st_mtime < cutoff_time:
                    log_file.unlink()
                    self.get_logger("logging_manager").info(f"🧹 Log antigo removido: {log_file.name}")
        
        except Exception as e:
            self.log_error_with_context(e, "logging_manager", "cleanup_old_logs")

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

# Instância global do gerenciador
structured_logging = StructuredLoggingManager()
structured_logger = structured_logging  # Alias para compatibilidade

# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

async def get_structured_logger(name: str) -> logging.Logger:
    """Obtém logger estruturado."""
    return await structured_logging.get_logger(name)

async def log_context(**context_data):
    """Context manager para logging."""
    return await structured_logging.log_context(**context_data)

async def log_operation(operation: str, module: str):
    """Decorator para logging de operações."""
    async def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = structured_logging.log_operation_start(operation, module)
            try:
                result = func(*args, **kwargs)
                structured_logging.log_operation_end(operation, module, start_time, True)
                return await result
            except Exception as e:
                structured_logging.log_operation_end(operation, module, start_time, False)
                structured_logging.log_error_with_context(e, module, operation)
                raise
        return await wrapper
    return await decorator

# =============================================================================
# TESTE DO SISTEMA
# =============================================================================

async def test_structured_logging():
    """Testa o sistema de logging estruturado."""
    print("🧪 Testando sistema de logging estruturado...")
    
    # Obtém logger
    logger = get_structured_logger("test_module")
    
    # Teste básico
    logger.info("Teste de log básico")
    print("✅ Log básico funcionando")
    
    # Teste com contexto
    with log_context(operation="test_operation", pipeline_id="test_pipeline"):
        logger.info("Log com contexto")
        logger.warning("Aviso com contexto")
    print("✅ Log com contexto funcionando")
    
    # Teste de operação
    @log_operation("test_function", "test_module")
    async def test_function():
        time.sleep(0.1)
        return await "success"
    
    result = test_function()
    print("✅ Decorator de operação funcionando")
    
    # Teste de erro
    try:
        raise ValueError("Erro de teste")
    except Exception as e:
        structured_logging.log_error_with_context(e, "test_module", "test_error")
    print("✅ Log de erro funcionando")
    
    # Teste de auditoria
    structured_logging.log_audit_event("test_audit", "test_module", user="test_user")
    print("✅ Log de auditoria funcionando")
    
    # Teste de busca
    results = structured_logging.search_logs("test", limit=5)
    print(f"✅ Busca de logs: {len(results)} resultados encontrados")
    
    print("🎉 Sistema de logging estruturado funcionando!")
    return await True

if __name__ == "__main__":
    test_structured_logging()
