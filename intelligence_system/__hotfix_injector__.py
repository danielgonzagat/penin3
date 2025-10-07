"""
HOTFIX INJECTOR - Automaticamente aplicado via import
Este módulo força reload quando importado
"""
import sys
import importlib
import logging

logger = logging.getLogger(__name__)

try:
    logger.info("🔧 [HOTFIX INJECTOR] Forcing module reload...")
    
    # Força reload do self_modification_engine
    if 'extracted_algorithms.self_modification_engine' in sys.modules:
        importlib.reload(sys.modules['extracted_algorithms.self_modification_engine'])
        logger.info("   ✅ Reloaded: self_modification_engine")
    
    logger.info("   ✅ HOTFIX INJECTOR complete")
    
except Exception as e:
    logger.error(f"   ❌ HOTFIX INJECTOR failed: {e}")

# Auto-delete após execução para não interferir
import os
os.remove(__file__)
