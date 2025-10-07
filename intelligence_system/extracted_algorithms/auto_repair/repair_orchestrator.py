"""
REPAIR ORCHESTRATOR - Orquestrador principal de auto-repair
Conecta descoberta, teste, patch e emergência em um fluxo completo
"""
import threading
import time
import json
import logging
from typing import Dict, Optional, List
from pathlib import Path

from .code_discovery import CodeDiscoveryEngine, APICallDiscovery
from .snippet_tester import SnippetTester, ProgressiveTester
from .atomic_patcher import SmartPatcher
from .emergence_detector import EmergenceDetector

logger = logging.getLogger(__name__)

class RepairOrchestrator:
    """Orquestrador completo do sistema de auto-repair"""
    
    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        
        # Componentes
        self.code_discovery = CodeDiscoveryEngine()
        self.api_discovery = APICallDiscovery()
        self.snippet_tester = SnippetTester(timeout=10)
        self.progressive_tester = ProgressiveTester(self.snippet_tester)
        self.patcher = SmartPatcher()
        self.emergence_detector = EmergenceDetector()
        
        # Estado
        self.active_repairs = {}
        self.repair_lock = threading.Lock()
        self.error_count = {}
        
        # Comunicação
        self.comm_log = Path("/root/intelligence_system/auto_repair_agent_comm.log")
        
        logger.info(f"🔧 RepairOrchestrator initialized (dry_run={dry_run})")
        self._log_event({
            'event': 'start',
            'dry_run': dry_run,
            'timestamp': time.time()
        })
    
    def handle_error(self, exception: Exception,
                    error_context: Dict,
                    async_mode: bool = True) -> Optional[Dict]:
        """
        Ponto de entrada principal - lida com erro do sistema
        
        Args:
            exception: exceção capturada
            error_context: contexto do erro (arquivo, função, etc)
            async_mode: se True, executa em thread separada
            
        Returns:
            Dict com resultado ou None se async
        """
        error_key = self._generate_error_key(exception, error_context)
        
        # Throttling: não repara mesmo erro múltiplas vezes rapidamente
        if error_key in self.error_count:
            count, last_time = self.error_count[error_key]
            if time.time() - last_time < 300:  # 5 minutos
                logger.debug(f"Throttling repair for {error_key}")
                return None
        
        self.error_count[error_key] = (
            self.error_count.get(error_key, (0, 0))[0] + 1,
            time.time()
        )
        
        # Registra erro
        self.emergence_detector.record_error(error_context)
        
        if async_mode:
            # Executa em thread separada para não bloquear
            thread = threading.Thread(
                target=self._repair_worker,
                args=(exception, error_context, error_key),
                daemon=True
            )
            thread.start()
            return None
        else:
            return self._repair_worker(exception, error_context, error_key)
    
    def _repair_worker(self, exception: Exception, 
                      error_context: Dict,
                      error_key: str) -> Dict:
        """Worker que executa repair completo"""
        start_time = time.time()
        
        result = {
            'error_key': error_key,
            'success': False,
            'dry_run': self.dry_run,
            'phases': {}
        }
        
        try:
            with self.repair_lock:
                self.active_repairs[error_key] = {
                    'status': 'discovering',
                    'start_time': start_time
                }
            
            # FASE 1: Descoberta de código
            logger.info(f"🔍 Phase 1: Discovering solutions for {error_key}")
            self._log_event({
                'event': 'phase-start',
                'phase': 'discovery',
                'error_key': error_key
            })
            
            discovery_result = self._discover_solutions(exception, error_context)
            result['phases']['discovery'] = discovery_result
            
            if not discovery_result['candidates']:
                logger.warning(f"No candidates found for {error_key}")
                result['message'] = 'No solution candidates found'
                return result
            
            # FASE 2: Teste progressivo
            logger.info(f"🧪 Phase 2: Testing {len(discovery_result['candidates'])} candidates")
            self._log_event({
                'event': 'phase-start',
                'phase': 'testing',
                'candidate_count': len(discovery_result['candidates'])
            })
            
            with self.repair_lock:
                self.active_repairs[error_key]['status'] = 'testing'
            
            test_result = self._test_solutions(discovery_result['candidates'])
            result['phases']['testing'] = test_result
            
            if not test_result['winner']:
                logger.warning(f"No working solution found for {error_key}")
                result['message'] = 'All candidates failed testing'
                return result
            
            # FASE 3: Detecção de Emergência
            logger.info(f"🔥 Phase 3: Checking for emergence")
            emergence = self.emergence_detector.detect_emergence(
                error_context=error_context,
                winning_solution=test_result['winner'],
                attempt_count=test_result['total_attempts'],
                original_error_rate=self.error_count[error_key][0] / 10.0
            )
            
            result['phases']['emergence'] = {
                'detected': emergence is not None,
                'emergence_id': emergence['emergence_id'] if emergence else None
            }
            
            if emergence:
                self._log_event({
                    'event': 'emergence-detected',
                    'emergence_id': emergence['emergence_id'],
                    'classification': emergence['classification'],
                    'novelty': emergence['metrics']['novelty_score']
                })
            
            # FASE 4: Aplicação de Patch
            if not self.dry_run:
                logger.info(f"🔧 Phase 4: Applying patch")
                
                # Pede permissão se não dry-run
                patch_id = f"PATCH_{int(time.time())}"
                self._log_event({
                    'event': 'request-apply',
                    'patch_id': patch_id,
                    'target': error_context.get('target_file'),
                    'emergency': emergence is not None
                })
                
                # Aguarda aprovação (simulado - em produção seria input)
                # Por enquanto, aplica automaticamente se for emergência
                if emergence:
                    with self.repair_lock:
                        self.active_repairs[error_key]['status'] = 'patching'
                    
                    patch_result = self._apply_patch(
                        error_context,
                        test_result['winner'],
                        patch_id
                    )
                    result['phases']['patching'] = patch_result
                    
                    if patch_result['success']:
                        result['success'] = True
                        self._log_event({
                            'event': 'apply-result',
                            'patch_id': patch_id,
                            'applied': True,
                            'validated': patch_result.get('validated', False)
                        })
            else:
                logger.info(f"✓ Dry-run: Would apply patch")
                result['phases']['patching'] = {
                    'dry_run': True,
                    'would_apply': True
                }
                result['success'] = True
            
        except Exception as e:
            logger.error(f"Repair worker failed: {e}", exc_info=True)
            result['error'] = str(e)
        
        finally:
            with self.repair_lock:
                if error_key in self.active_repairs:
                    del self.active_repairs[error_key]
            
            result['duration'] = time.time() - start_time
            self._save_repair_result(result)
        
        return result
    
    def _discover_solutions(self, exception: Exception, error_context: Dict) -> Dict:
        """Fase de descoberta"""
        error_msg = str(exception)
        error_type = type(exception).__name__
        
        # Extrai keyword do erro
        keyword = self._extract_keyword(error_msg, error_context)
        
        # Busca implementações
        if 'api' in error_type.lower() or 'connection' in error_msg.lower():
            candidates = self.api_discovery.find_working_api_calls(keyword, error_msg)
        else:
            candidates = self.code_discovery.find_working_implementations(
                keyword,
                context=error_context.get('context')
            )
        
        # Filtra top N
        top_candidates = candidates[:20]  # Limita para não testar infinitos
        
        return {
            'keyword': keyword,
            'total_found': len(candidates),
            'candidates': top_candidates
        }
    
    def _test_solutions(self, candidates: List[Dict]) -> Dict:
        """Fase de teste"""
        result = self.progressive_tester.test_until_success(
            candidates,
            mutation_strategies=['', 'add_try_catch', 'add_timeout']
        )
        
        return {
            'total_attempts': self.progressive_tester.attempt_count,
            'winner': result,
            'success': result is not None
        }
    
    def _apply_patch(self, error_context: Dict, 
                     winning_solution: Dict,
                     patch_id: str) -> Dict:
        """Fase de aplicação"""
        target_file = error_context.get('target_file')
        
        if not target_file:
            # Cria helper file
            helper_path = "/root/intelligence_system/auto_repair_helpers/repair_helper.py"
            with open(helper_path, 'a', encoding='utf-8') as f:
                f.write(f"\n# Patch {patch_id}\n")
                f.write(winning_solution['snippet'])
                f.write("\n\n")
            
            return {
                'success': True,
                'target': helper_path,
                'type': 'helper_file'
            }
        
        # Patch incremental se possível
        if error_context.get('failing_function'):
            return self.patcher.apply_function_replacement(
                target_file,
                error_context['failing_function'],
                winning_solution['snippet'],
                patch_metadata={
                    'patch_id': patch_id,
                    'source': winning_solution.get('source'),
                    'emergence': True
                },
                dry_run=self.dry_run
            )
        else:
            # Patch completo
            with open(target_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Append solution
            new_content = content + f"\n# Auto-repair patch {patch_id}\n" + winning_solution['snippet']
            
            return self.patcher.apply_patch(
                target_file,
                new_content,
                patch_metadata={'patch_id': patch_id},
                dry_run=self.dry_run
            )
    
    def _extract_keyword(self, error_msg: str, error_context: Dict) -> str:
        """Extrai keyword relevante do erro"""
        # Patterns comuns
        keywords = ['openai', 'gpt', 'claude', 'anthropic', 'litellm', 
                   'mistral', 'gemini', 'deepseek', 'grok']
        
        for k in keywords:
            if k in error_msg.lower():
                return k
        
        # Fallback: primeira palavra do erro
        words = error_msg.split()
        return words[0] if words else 'error'
    
    def _generate_error_key(self, exception: Exception, error_context: Dict) -> str:
        """Gera chave única para tipo de erro"""
        import hashlib
        key_parts = [
            type(exception).__name__,
            str(exception)[:50],
            error_context.get('target_function', '')
        ]
        key_str = '|'.join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _log_event(self, event: Dict):
        """Log estruturado de eventos"""
        event['timestamp'] = time.time()
        
        # Log JSON
        with open(self.comm_log, 'a', encoding='utf-8') as f:
            f.write(json.dumps(event) + '\n')
        
        # Log legível
        event_type = event.get('event', 'unknown')
        logger.info(f"[AUTO-REPAIR] {event_type}: {json.dumps(event, indent=2)}")
    
    def _save_repair_result(self, result: Dict):
        """Salva resultado completo do repair"""
        results_dir = Path("/root/intelligence_system/auto_repair_dryrun_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        result_file = results_dir / f"repair_{timestamp}.json"
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dumps(result, f, indent=2)
    
    def get_status(self) -> Dict:
        """Retorna status do orquestrador"""
        return {
            'active_repairs': len(self.active_repairs),
            'error_types_seen': len(self.error_count),
            'emergences_detected': len(self.emergence_detector.emergences),
            'dry_run': self.dry_run
        }


# Singleton global para uso fácil
_orchestrator_instance = None

def get_orchestrator(dry_run: bool = True) -> RepairOrchestrator:
    """Retorna instância singleton do orquestrador"""
    global _orchestrator_instance
    if _orchestrator_instance is None:
        _orchestrator_instance = RepairOrchestrator(dry_run=dry_run)
    return _orchestrator_instance


if __name__ == "__main__":
    # Teste
    orchestrator = RepairOrchestrator(dry_run=True)
    
    # Simula erro
    try:
        raise ConnectionError("Connection timeout to gpt-5")
    except Exception as e:
        result = orchestrator.handle_error(
            e,
            error_context={
                'target_file': '/root/test.py',
                'target_function': 'call_api',
                'context': 'openai api call'
            },
            async_mode=False
        )
        
        print(f"Result: {json.dumps(result, indent=2)}")
