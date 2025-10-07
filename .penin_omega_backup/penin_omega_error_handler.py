#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Error Handler - Sistema de Tratamento de Erro Rigoroso
===============================================================
Tratamento consistente e robusto de erros em todo o sistema.
"""

import logging
import traceback
import functools
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional, Type, Union
from pathlib import Path

class PeninOmegaError(Exception):
    """Exceção base do sistema PENIN-Ω."""
    
    def __init__(self, message: str, error_code: str = None, context: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "PENIN_UNKNOWN"
        self.context = context or {}
        self.timestamp = datetime.now(timezone.utc)

class ValidationError(PeninOmegaError):
    """Erro de validação de entrada."""
    
    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(message, "PENIN_VALIDATION", {"field": field, "value": str(value)})

class ConfigurationError(PeninOmegaError):
    """Erro de configuração."""
    
    def __init__(self, message: str, config_key: str = None):
        super().__init__(message, "PENIN_CONFIG", {"config_key": config_key})

class SystemError(PeninOmegaError):
    """Erro de sistema."""
    
    def __init__(self, message: str, component: str = None):
        super().__init__(message, "PENIN_SYSTEM", {"component": component})

class ErrorHandler:
    """Manipulador centralizado de erros."""
    
    def __init__(self):
        self.logger = logging.getLogger("PeninOmegaErrorHandler")
        self.error_log_path = Path("/root/.penin_omega/logs/errors.log")
        self.error_log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Configura logger de erros
        self._setup_error_logger()
    
    def _setup_error_logger(self):
        """Configura logger específico para erros."""
        error_logger = logging.getLogger("PeninOmegaErrors")
        error_logger.setLevel(logging.ERROR)
        
        # Handler para arquivo
        file_handler = logging.FileHandler(self.error_log_path)
        file_handler.setLevel(logging.ERROR)
        
        # Formato detalhado para erros
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s\n'
            'Context: %(context)s\n'
            'Traceback: %(traceback)s\n'
            '---'
        )
        file_handler.setFormatter(formatter)
        error_logger.addHandler(file_handler)
        
        self.error_logger = error_logger
    
    def handle_error(self, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Manipula erro de forma consistente."""
        try:
            # Extrai informações do erro
            error_info = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "context": context or {},
                "traceback": traceback.format_exc()
            }
            
            # Adiciona informações específicas do PENIN-Ω
            if isinstance(error, PeninOmegaError):
                error_info["error_code"] = error.error_code
                error_info["penin_context"] = error.context
            
            # Log estruturado
            self.error_logger.error(
                f"{error_info['error_type']}: {error_info['error_message']}",
                extra={
                    "context": error_info["context"],
                    "traceback": error_info["traceback"]
                }
            )
            
            return error_info
            
        except Exception as handler_error:
            # Fallback se o próprio handler falhar
            fallback_info = {
                "error_type": "ErrorHandlerFailure",
                "error_message": f"Handler failed: {handler_error}",
                "original_error": str(error),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Log básico
            self.logger.critical(f"Error handler failed: {handler_error}")
            return fallback_info
    
    def validate_input(self, value: Any, expected_type: Type, field_name: str = "input") -> Any:
        """Valida entrada de forma rigorosa."""
        if value is None:
            raise ValidationError(f"{field_name} cannot be None", field_name, value)
        
        if not isinstance(value, expected_type):
            raise ValidationError(
                f"{field_name} must be {expected_type.__name__}, got {type(value).__name__}",
                field_name, value
            )
        
        return value
    
    def validate_string(self, value: str, field_name: str = "string", 
                       min_length: int = 1, max_length: int = None) -> str:
        """Valida string com critérios específicos."""
        self.validate_input(value, str, field_name)
        
        if len(value) < min_length:
            raise ValidationError(
                f"{field_name} must be at least {min_length} characters",
                field_name, value
            )
        
        if max_length and len(value) > max_length:
            raise ValidationError(
                f"{field_name} must be at most {max_length} characters",
                field_name, value
            )
        
        return value.strip()
    
    def validate_dict(self, value: Dict[str, Any], required_keys: list = None, 
                     field_name: str = "dict") -> Dict[str, Any]:
        """Valida dicionário com chaves obrigatórias."""
        self.validate_input(value, dict, field_name)
        
        if required_keys:
            missing_keys = [key for key in required_keys if key not in value]
            if missing_keys:
                raise ValidationError(
                    f"{field_name} missing required keys: {missing_keys}",
                    field_name, value
                )
        
        return value

def error_handler_decorator(component: str = None):
    """Decorator para tratamento automático de erros."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "component": component,
                    "args": str(args)[:200],  # Limita tamanho
                    "kwargs": str(kwargs)[:200]
                }
                
                error_info = error_handler.handle_error(e, context)
                
                # Re-raise como PeninOmegaError se não for uma
                if not isinstance(e, PeninOmegaError):
                    raise SystemError(
                        f"Error in {func.__name__}: {str(e)}",
                        component
                    ) from e
                else:
                    raise
        
        return wrapper
    return decorator

def async_error_handler_decorator(component: str = None):
    """Decorator para tratamento automático de erros em funções async."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                context = {
                    "function": func.__name__,
                    "component": component,
                    "args": str(args)[:200],
                    "kwargs": str(kwargs)[:200]
                }
                
                error_info = error_handler.handle_error(e, context)
                
                if not isinstance(e, PeninOmegaError):
                    raise SystemError(
                        f"Error in async {func.__name__}: {str(e)}",
                        component
                    ) from e
                else:
                    raise
        
        return wrapper
    return decorator

# Instância global
error_handler = ErrorHandler()

# Funções de conveniência
def validate_input(value: Any, expected_type: Type, field_name: str = "input") -> Any:
    """Função de conveniência para validação."""
    return error_handler.validate_input(value, expected_type, field_name)

def validate_string(value: str, field_name: str = "string", 
                   min_length: int = 1, max_length: int = None) -> str:
    """Função de conveniência para validação de string."""
    return error_handler.validate_string(value, field_name, min_length, max_length)

def validate_dict(value: Dict[str, Any], required_keys: list = None, 
                 field_name: str = "dict") -> Dict[str, Any]:
    """Função de conveniência para validação de dict."""
    return error_handler.validate_dict(value, required_keys, field_name)

def handle_error(error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Função de conveniência para tratamento de erro."""
    return error_handler.handle_error(error, context)

# Configuração global de exceções não capturadas
def global_exception_handler(exc_type, exc_value, exc_traceback):
    """Handler global para exceções não capturadas."""
    if issubclass(exc_type, KeyboardInterrupt):
        # Permite Ctrl+C
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    error_handler.handle_error(exc_value, {
        "source": "global_exception_handler",
        "exc_type": exc_type.__name__
    })

# Instala handler global
sys.excepthook = global_exception_handler

if __name__ == "__main__":
    # Teste do sistema de erro
    try:
        validate_string("", "test_field", min_length=5)
    except ValidationError as e:
        print(f"Validation error caught: {e.message}")
        print(f"Error code: {e.error_code}")
        print(f"Context: {e.context}")
