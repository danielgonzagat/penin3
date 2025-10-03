"""
CODE DISCOVERY ENGINE
Procura no filesystem por código funcional relacionado a erros
"""
import os
import re
import ast
import hashlib
import json
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict

class CodeDiscoveryEngine:
    """Descobre código funcional no filesystem"""
    
    def __init__(self, search_roots: List[str] = None):
        self.search_roots = search_roots or [
            "/root/intelligence_system",
            "/root/.venv/lib",
            "/root"
        ]
        self.cache = {}
        self.extensions = ['.py']
        
    def find_working_implementations(self, error_keyword: str, 
                                     context: Optional[str] = None) -> List[Dict]:
        """
        Busca implementações funcionais relacionadas ao erro
        
        Args:
            error_keyword: palavra-chave do erro (ex: 'openai', 'gpt-5')
            context: contexto adicional para refinar busca
            
        Returns:
            Lista de snippets candidatos com metadados
        """
        candidates = []
        cache_key = f"{error_keyword}:{context}"
        
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Busca em múltiplos roots
        for root in self.search_roots:
            if not os.path.exists(root):
                continue
                
            candidates.extend(self._scan_directory(root, error_keyword, context))
        
        # Deduplica por hash de conteúdo
        seen = set()
        unique = []
        for c in candidates:
            h = c['hash']
            if h not in seen:
                seen.add(h)
                unique.append(c)
        
        # Ordena por score de relevância
        scored = self._score_candidates(unique, error_keyword, context)
        
        self.cache[cache_key] = scored
        return scored
    
    def _scan_directory(self, root: str, keyword: str, context: Optional[str]) -> List[Dict]:
        """Escaneia diretório procurando código relevante"""
        results = []
        pattern = re.compile(re.escape(keyword), re.IGNORECASE)
        
        for dirpath, _, filenames in os.walk(root):
            # Skip diretorios que não interessam
            if any(skip in dirpath for skip in ['__pycache__', '.git', 'node_modules', 'venv']):
                continue
                
            for filename in filenames:
                if not any(filename.endswith(ext) for ext in self.extensions):
                    continue
                    
                filepath = os.path.join(dirpath, filename)
                
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                except:
                    continue
                
                if not pattern.search(content):
                    continue
                
                # Extrai snippets relevantes
                snippets = self._extract_snippets(content, keyword, context)
                
                for snippet in snippets:
                    results.append({
                        'source_file': filepath,
                        'snippet': snippet,
                        'hash': hashlib.md5(snippet.encode()).hexdigest(),
                        'size': len(snippet),
                        'keyword': keyword
                    })
        
        return results
    
    def _extract_snippets(self, content: str, keyword: str, context: Optional[str]) -> List[str]:
        """Extrai snippets relevantes usando AST"""
        snippets = []
        
        try:
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                # Funções
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    source = ast.get_source_segment(content, node)
                    if source and keyword.lower() in source.lower():
                        snippets.append(source)
                
                # Classes
                elif isinstance(node, ast.ClassDef):
                    source = ast.get_source_segment(content, node)
                    if source and keyword.lower() in source.lower():
                        # Extrai apenas métodos relevantes da classe
                        for item in node.body:
                            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                                method_source = ast.get_source_segment(content, item)
                                if method_source and keyword.lower() in method_source.lower():
                                    snippets.append(method_source)
                
        except SyntaxError:
            # Fallback: extração por linhas
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if keyword.lower() in line.lower():
                    # Pega contexto de ±5 linhas
                    start = max(0, i - 5)
                    end = min(len(lines), i + 6)
                    snippet = '\n'.join(lines[start:end])
                    snippets.append(snippet)
        
        return snippets
    
    def _score_candidates(self, candidates: List[Dict], keyword: str, context: Optional[str]) -> List[Dict]:
        """Score baseado em relevância"""
        for c in candidates:
            score = 0
            snippet = c['snippet'].lower()
            
            # Keyword presente
            score += snippet.count(keyword.lower()) * 10
            
            # Indicadores de sucesso
            success_indicators = ['return', 'success', 'ok', 'true', 'response', 'result']
            for ind in success_indicators:
                if ind in snippet:
                    score += 5
            
            # Indicadores de erro (penaliza)
            error_indicators = ['error', 'exception', 'fail', 'raise', 'assert false']
            for ind in error_indicators:
                if ind in snippet:
                    score -= 3
            
            # Contexto adicional
            if context and context.lower() in snippet:
                score += 15
            
            # Tamanho razoável (nem muito grande nem muito pequeno)
            if 100 < c['size'] < 500:
                score += 5
            
            c['score'] = max(0, score)
        
        # Ordena por score decrescente
        return sorted(candidates, key=lambda x: x['score'], reverse=True)


class APICallDiscovery(CodeDiscoveryEngine):
    """Especializado em encontrar chamadas de API funcionais"""
    
    def find_working_api_calls(self, api_name: str, error_msg: str) -> List[Dict]:
        """Encontra chamadas de API que funcionam"""
        # Extrai hints do erro
        hints = self._extract_api_hints(error_msg)
        
        # Busca implementações
        candidates = self.find_working_implementations(api_name, context=hints)
        
        # Filtra para focar em chamadas API
        api_candidates = []
        for c in candidates:
            snippet = c['snippet']
            
            # Verifica se tem chamada de API
            if any(pattern in snippet.lower() for pattern in [
                'client.', '.create(', 'api_key', 'headers', 'auth',
                'request', 'response', 'litellm', 'openai'
            ]):
                api_candidates.append(c)
        
        return api_candidates
    
    def _extract_api_hints(self, error_msg: str) -> str:
        """Extrai hints úteis da mensagem de erro"""
        hints = []
        
        # Patterns comuns
        if 'timeout' in error_msg.lower():
            hints.append('timeout')
        if 'auth' in error_msg.lower():
            hints.append('authentication')
        if 'key' in error_msg.lower():
            hints.append('api_key')
        if 'rate' in error_msg.lower():
            hints.append('rate_limit')
        
        return ' '.join(hints)


if __name__ == "__main__":
    # Teste rápido
    engine = CodeDiscoveryEngine()
    results = engine.find_working_implementations('openai', context='client')
    print(f"Encontrados {len(results)} candidatos")
    for r in results[:3]:
        print(f"  - {r['source_file']} (score: {r['score']})")
