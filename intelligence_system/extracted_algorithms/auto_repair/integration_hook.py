"""
INTEGRATION HOOK - Integra auto-repair ao sistema principal
Hook não-intrusivo que pode ser injetado no loop sem interromper
"""
import sys
import logging
from pathlib import Path
from typing import Optional, Dict

# Adiciona path do sistema
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from extracted_algorithms.auto_repair.repair_orchestrator import get_orchestrator

logger = logging.getLogger(__name__)

class AutoRepairHook:
    """
    Hook não-intrusivo para auto-repair
    
    Uso:
        hook = AutoRepairHook(dry_run=True)
        
        try:
            # código que pode falhar
            call_api()
        except Exception as e:
            hook.handle(e, context={'target_file': __file__})
    """
    
    def __init__(self, dry_run: bool = True, enabled: bool = True):
        self.enabled = enabled
        self.dry_run = dry_run
        self._orchestrator = None
        
        if enabled:
            try:
                self._orchestrator = get_orchestrator(dry_run=dry_run)
                logger.info(f"✅ AutoRepairHook initialized (dry_run={dry_run})")
            except Exception as e:
                logger.error(f"Failed to initialize AutoRepairHook: {e}")
                self.enabled = False
    
    def handle(self, exception: Exception, 
               context: Optional[Dict] = None,
               async_mode: bool = True) -> Optional[Dict]:
        """
        Manipula erro com auto-repair
        
        Args:
            exception: exceção capturada
            context: contexto do erro (arquivo, função, etc)
            async_mode: se True, não bloqueia (recomendado)
            
        Returns:
            Resultado ou None se async
        """
        if not self.enabled:
            return None
        
        if not self._orchestrator:
            return None
        
        # Contexto padrão se não fornecido
        if context is None:
            context = self._extract_context_from_exception(exception)
        
        try:
            return self._orchestrator.handle_error(
                exception,
                context,
                async_mode=async_mode
            )
        except Exception as e:
            logger.error(f"AutoRepairHook.handle failed: {e}")
            return None
    
    def _extract_context_from_exception(self, exception: Exception) -> Dict:
        """Extrai contexto da exceção"""
        import traceback
        
        tb = traceback.extract_tb(exception.__traceback__)
        
        if tb:
            frame = tb[-1]
            return {
                'target_file': frame.filename,
                'target_function': frame.name,
                'line_number': frame.lineno,
                'error_type': type(exception).__name__,
                'error_message': str(exception)
            }
        
        return {
            'error_type': type(exception).__name__,
            'error_message': str(exception)
        }
    
    def get_status(self) -> Dict:
        """Status do hook"""
        if not self.enabled or not self._orchestrator:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'dry_run': self.dry_run,
            **self._orchestrator.get_status()
        }


# Instância global para fácil uso
_global_hook = None

def initialize_global_hook(dry_run: bool = True) -> AutoRepairHook:
    """Inicializa hook global"""
    global _global_hook
    if _global_hook is None:
        _global_hook = AutoRepairHook(dry_run=dry_run, enabled=True)
    return _global_hook

def get_global_hook() -> Optional[AutoRepairHook]:
    """Retorna hook global se existir"""
    return _global_hook

def auto_repair_handle(exception: Exception, context: Optional[Dict] = None):
    """
    Função helper para uso rápido
    
    Uso:
        try:
            risky_operation()
        except Exception as e:
            auto_repair_handle(e, {'target_file': __file__})
            # continua execução normalmente
    """
    hook = get_global_hook()
    if hook:
        hook.handle(exception, context, async_mode=True)


# Wrapper decorator
def with_auto_repair(context: Optional[Dict] = None):
    """
    Decorator para adicionar auto-repair a funções
    
    Uso:
        @with_auto_repair(context={'target_file': __file__})
        def my_function():
            # código que pode falhar
            pass
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Tenta auto-repair
                auto_repair_handle(e, context)
                # Re-raise exception (não impede fluxo normal)
                raise
        return wrapper
    return decorator


if __name__ == "__main__":
    # Teste
    hook = initialize_global_hook(dry_run=True)
    
    # Simula erro
    try:
        raise ConnectionError("Test error: API timeout")
    except Exception as e:
        result = hook.handle(e, async_mode=False)
        print(f"Auto-repair triggered: {result is not None}")
        print(f"Status: {hook.get_status()}")
