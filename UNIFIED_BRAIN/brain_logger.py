#!/usr/bin/env python3
"""
📝 BRAIN LOGGER - Bug #17 fix
Sistema de logging estruturado para todos os módulos
"""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime

class JSONFormatter(logging.Formatter):
    """
    Simple JSON log formatter for structured logging.
    """
    def format(self, record: logging.LogRecord) -> str:
        # Base fields
        log = {
            'timestamp': self.formatTime(record, self.datefmt),
            'logger': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
        }

        # Optional useful context
        if hasattr(record, 'module'):
            log['module'] = record.module
        if hasattr(record, 'funcName'):
            log['func'] = record.funcName
        if hasattr(record, 'lineno'):
            log['line'] = record.lineno

        try:
            import json as _json
            return _json.dumps(log, ensure_ascii=False)
        except Exception:
            # Fallback to default message if JSON fails for any reason
            return f"{log['timestamp']} - {log['logger']} - {log['level']} - {log['message']}"

def setup_logger(
    name: str,
    log_file: str = None,
    level: int = logging.INFO,
    structured: bool = True,
    rotate_max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    rotate_backups: int = 5,
):
    """
    Setup logger for brain modules
    
    Args:
        name: nome do logger (geralmente __name__)
        log_file: caminho do arquivo de log (opcional)
        level: nível de logging (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        logger configurado
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Evita duplicação de handlers
    if logger.handlers:
        return logger
    
    # Formato
    if structured:
        formatter = JSONFormatter(datefmt='%Y-%m-%d %H:%M:%S')
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console = logging.StreamHandler()
    console.setFormatter(formatter)
    logger.addHandler(console)
    
    # File handler
    if log_file is None:
        log_dir = Path("/root/UNIFIED_BRAIN/logs")
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"{name.replace('.', '_')}.log"
    
    # Rotating file handler for production hardening
    file_handler = RotatingFileHandler(
        str(log_file), maxBytes=rotate_max_bytes, backupCount=rotate_backups
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


# Logger global do brain
brain_logger = setup_logger('unified_brain')

if __name__ == "__main__":
    # Teste
    logger = setup_logger('test')
    logger.info("Logger test")
    logger.warning("Warning test")
    logger.error("Error test")
    print("✅ Logger OK")
