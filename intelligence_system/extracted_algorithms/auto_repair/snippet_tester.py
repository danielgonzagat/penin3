"""
SNIPPET TESTER - Testa código isoladamente
Executa snippets em ambiente isolado para validar funcionalidade
"""
import subprocess
import tempfile
import os
import sys
import json
import time
import signal
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class SnippetTester:
    """Testa snippets de código em ambiente isolado"""
    
    def __init__(self, timeout: int = 10, max_retries: int = 3):
        self.timeout = timeout
        self.max_retries = max_retries
        self.test_results = []
        
    def test_snippet(self, snippet: str, 
                     required_imports: List[str] = None,
                     test_assertion: Optional[str] = None) -> Dict:
        """
        Testa um snippet em processo isolado
        
        Args:
            snippet: código para testar
            required_imports: imports necessários
            test_assertion: asserção para validar sucesso
            
        Returns:
            Dict com resultado do teste
        """
        result = {
            'success': False,
            'stdout': '',
            'stderr': '',
            'exit_code': -1,
            'execution_time': 0,
            'error': None,
            'snippet_hash': hash(snippet)
        }
        
        # Prepara código de teste
        test_code = self._prepare_test_code(snippet, required_imports, test_assertion)
        
        # Executa em subprocesso
        start_time = time.time()
        
        try:
            exec_result = self._execute_isolated(test_code)
            result.update(exec_result)
            result['execution_time'] = time.time() - start_time
            
            # Valida sucesso
            if result['exit_code'] == 0:
                # Verifica output para markers de sucesso
                if '__TEST_SUCCESS__' in result['stdout']:
                    result['success'] = True
                elif test_assertion and '__ASSERTION_PASS__' in result['stdout']:
                    result['success'] = True
                elif result['stderr'] == '':
                    # Sem erro stderr = possível sucesso
                    result['success'] = True
                    
        except Exception as e:
            result['error'] = str(e)
            result['execution_time'] = time.time() - start_time
        
        self.test_results.append(result)
        return result
    
    def _prepare_test_code(self, snippet: str, 
                           required_imports: List[str] = None,
                           test_assertion: Optional[str] = None) -> str:
        """Prepara código completo para teste"""
        imports = required_imports or []
        
        # Template de teste
        template = '''
import sys
import json
import traceback

# Imports necessários
{imports}

# Setup
def __test_wrapper():
    try:
        # SNIPPET BEGIN
{snippet}
        # SNIPPET END
        
        # Asserção de teste
{assertion}
        
        # Sucesso
        print("__TEST_SUCCESS__")
        return True
        
    except Exception as e:
        traceback.print_exc()
        print(f"__TEST_FAIL__: {{str(e)}}", file=sys.stderr)
        return False

# Executa
if __name__ == "__main__":
    success = __test_wrapper()
    sys.exit(0 if success else 1)
'''
        
        # Formata snippet com indentação
        indented_snippet = '\n'.join('        ' + line for line in snippet.split('\n'))
        
        # Formata imports
        import_lines = '\n'.join(f'import {imp}' for imp in imports)
        
        # Formata asserção
        if test_assertion:
            assertion_code = f'''
        # Validação
        {test_assertion}
        print("__ASSERTION_PASS__")
'''
        else:
            assertion_code = '        pass  # No assertion'
        
        code = template.format(
            imports=import_lines,
            snippet=indented_snippet,
            assertion=assertion_code
        )
        
        return code
    
    def _execute_isolated(self, code: str) -> Dict:
        """Executa código em subprocesso isolado"""
        # Cria arquivo temporário
        with tempfile.NamedTemporaryFile(
            mode='w', 
            suffix='.py', 
            delete=False,
            encoding='utf-8'
        ) as f:
            f.write(code)
            temp_file = f.name
        
        try:
            # Prepara ambiente limpo
            env = os.environ.copy()
            env['PYTHONDONTWRITEBYTECODE'] = '1'
            
            # Executa com timeout
            proc = subprocess.Popen(
                [sys.executable, temp_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                preexec_fn=os.setsid  # Cria novo process group
            )
            
            try:
                stdout, stderr = proc.communicate(timeout=self.timeout)
                exit_code = proc.returncode
                
            except subprocess.TimeoutExpired:
                # Kill process group inteiro
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                stdout, stderr = proc.communicate()
                exit_code = -9
                stderr = b'TIMEOUT: Execution exceeded ' + str(self.timeout).encode() + b' seconds\n' + stderr
            
            return {
                'stdout': stdout.decode('utf-8', errors='ignore'),
                'stderr': stderr.decode('utf-8', errors='ignore'),
                'exit_code': exit_code
            }
            
        finally:
            # Limpa arquivo temporário
            try:
                os.unlink(temp_file)
            except:
                pass
    
    def test_multiple_variants(self, base_snippet: str, 
                               variants: List[Dict]) -> List[Dict]:
        """
        Testa múltiplas variações de um snippet
        
        Args:
            base_snippet: snippet base
            variants: lista de dicts com {'mutation': str, 'imports': [], ...}
            
        Returns:
            Lista de resultados ordenados por sucesso
        """
        results = []
        
        for v in variants:
            mutated = self._apply_mutation(base_snippet, v.get('mutation', ''))
            result = self.test_snippet(
                mutated,
                required_imports=v.get('imports', []),
                test_assertion=v.get('assertion')
            )
            result['variant_id'] = v.get('id', f"variant_{len(results)}")
            results.append(result)
        
        # Ordena por sucesso e tempo
        return sorted(results, key=lambda x: (x['success'], -x['execution_time']), reverse=True)
    
    def _apply_mutation(self, snippet: str, mutation: str) -> str:
        """Aplica mutação ao snippet"""
        if not mutation:
            return snippet
        
        # Mutações comuns
        if mutation == 'add_try_catch':
            return f'''
try:
{self._indent(snippet, 4)}
except Exception as e:
    print(f"Error: {{e}}")
    raise
'''
        elif mutation == 'add_timeout':
            return f'''
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Operation timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)
try:
{self._indent(snippet, 4)}
finally:
    signal.alarm(0)
'''
        else:
            # Mutação customizada
            return snippet + '\n' + mutation
    
    def _indent(self, text: str, spaces: int) -> str:
        """Indenta texto"""
        indent = ' ' * spaces
        return '\n'.join(indent + line for line in text.split('\n'))
    
    def get_statistics(self) -> Dict:
        """Retorna estatísticas dos testes"""
        if not self.test_results:
            return {'total': 0, 'success': 0, 'fail': 0, 'success_rate': 0.0}
        
        total = len(self.test_results)
        success = sum(1 for r in self.test_results if r['success'])
        
        return {
            'total': total,
            'success': success,
            'fail': total - success,
            'success_rate': success / total if total > 0 else 0.0,
            'avg_execution_time': sum(r['execution_time'] for r in self.test_results) / total
        }


class ProgressiveTester:
    """Testa progressivamente até encontrar solução"""
    
    def __init__(self, base_tester: SnippetTester = None):
        self.tester = base_tester or SnippetTester()
        self.attempt_count = 0
        self.max_attempts = 50
        
    def test_until_success(self, snippets: List[Dict], 
                           mutation_strategies: List[str] = None) -> Optional[Dict]:
        """
        Testa snippets com mutações progressivas até encontrar um que funcione
        
        Returns:
            Snippet vencedor com resultado, ou None se nenhum funcionar
        """
        mutation_strategies = mutation_strategies or [
            '', 'add_try_catch', 'add_timeout'
        ]
        
        for snippet_data in snippets:
            if self.attempt_count >= self.max_attempts:
                break
            
            snippet = snippet_data['snippet']
            
            # Testa snippet original
            result = self.tester.test_snippet(snippet)
            self.attempt_count += 1
            
            if result['success']:
                return {
                    'snippet': snippet,
                    'result': result,
                    'source': snippet_data.get('source_file'),
                    'attempts': self.attempt_count
                }
            
            # Tenta mutações
            for strategy in mutation_strategies:
                if self.attempt_count >= self.max_attempts:
                    break
                
                mutated = self.tester._apply_mutation(snippet, strategy)
                result = self.tester.test_snippet(mutated)
                self.attempt_count += 1
                
                if result['success']:
                    return {
                        'snippet': mutated,
                        'result': result,
                        'source': snippet_data.get('source_file'),
                        'mutation': strategy,
                        'attempts': self.attempt_count
                    }
        
        return None


if __name__ == "__main__":
    # Teste rápido
    tester = SnippetTester(timeout=5)
    
    test_snippet = """
def hello():
    return "Hello, World!"

result = hello()
assert result == "Hello, World!"
"""
    
    result = tester.test_snippet(test_snippet)
    print(f"Test result: {result['success']}")
    print(f"Stats: {tester.get_statistics()}")
