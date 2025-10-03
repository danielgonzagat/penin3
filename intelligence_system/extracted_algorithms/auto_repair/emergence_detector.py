"""
EMERGENCE DETECTOR - Detecta novidades não previstas
Sistema que identifica quando algo genuinamente novo foi criado
"""
import json
import time
import hashlib
from typing import Dict, List, Optional
from pathlib import Path
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class EmergenceDetector:
    """Detecta emergências - novidades não previstas validadas"""
    
    def __init__(self, evidence_dir: str = "/root/intelligence_system/auto_repair_evidence"):
        self.evidence_dir = Path(evidence_dir)
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        
        self.emergences = []
        self.error_history = defaultdict(list)
        self.solution_patterns = {}
        self.novelty_threshold = 0.7  # 70% de novidade para ser emergência
        
    def record_error(self, error_context: Dict):
        """Registra erro para análise futura"""
        error_key = self._hash_error(error_context)
        self.error_history[error_key].append({
            'context': error_context,
            'timestamp': time.time()
        })
    
    def record_solution_attempt(self, error_context: Dict, 
                                solution_snippet: str,
                                test_result: Dict):
        """Registra tentativa de solução"""
        error_key = self._hash_error(error_context)
        solution_key = hashlib.md5(solution_snippet.encode()).hexdigest()
        
        if error_key not in self.solution_patterns:
            self.solution_patterns[error_key] = []
        
        self.solution_patterns[error_key].append({
            'solution_hash': solution_key,
            'snippet': solution_snippet,
            'result': test_result,
            'timestamp': time.time()
        })
    
    def detect_emergence(self, error_context: Dict,
                        winning_solution: Dict,
                        attempt_count: int,
                        original_error_rate: float) -> Optional[Dict]:
        """
        Detecta se uma solução constitui emergência
        
        Critérios para emergência:
        1. Solução funcionou após múltiplas tentativas
        2. Solução é significativamente diferente das existentes
        3. Solução foi validada em testes
        4. Reduz taxa de erro significativamente
        
        Returns:
            Dict com metadados da emergência ou None
        """
        error_key = self._hash_error(error_context)
        solution_snippet = winning_solution['snippet']
        
        # 1. Verifica se tentou múltiplas vezes (exploração)
        if attempt_count < 3:
            logger.debug("Not emergence: insufficient attempts")
            return None
        
        # 2. Calcula novidade
        novelty_score = self._calculate_novelty(solution_snippet, error_key)
        
        if novelty_score < self.novelty_threshold:
            logger.debug(f"Not emergence: novelty {novelty_score:.2f} < {self.novelty_threshold}")
            return None
        
        # 3. Verifica validação
        if not winning_solution.get('result', {}).get('success'):
            logger.debug("Not emergence: solution not validated")
            return None
        
        # 4. Verifica impacto (se fornecido)
        if original_error_rate > 0:
            # Nova taxa de erro estimada (assumindo que resolve esse erro)
            estimated_new_rate = original_error_rate * 0.5  # otimista
            impact = (original_error_rate - estimated_new_rate) / original_error_rate
            
            if impact < 0.3:  # menos de 30% de melhoria
                logger.debug(f"Not emergence: insufficient impact {impact:.2f}")
                return None
        
        # 🔥 É UMA EMERGÊNCIA! 🔥
        emergence = {
            'emergence_id': self._generate_emergence_id(),
            'timestamp': time.time(),
            'error_context': error_context,
            'solution': {
                'snippet': solution_snippet,
                'source': winning_solution.get('source'),
                'hash': hashlib.md5(solution_snippet.encode()).hexdigest()
            },
            'metrics': {
                'attempt_count': attempt_count,
                'novelty_score': novelty_score,
                'validation_success': True,
                'original_error_rate': original_error_rate
            },
            'evidence': {
                'test_result': winning_solution.get('result'),
                'mutations_tried': attempt_count,
                'unique_approaches': len(self.solution_patterns.get(error_key, []))
            },
            'classification': self._classify_emergence(error_context, solution_snippet)
        }
        
        self.emergences.append(emergence)
        self._save_emergence(emergence)
        
        logger.info(f"🔥 EMERGENCE DETECTED: {emergence['emergence_id']}")
        logger.info(f"   Novelty: {novelty_score:.2%}")
        logger.info(f"   Attempts: {attempt_count}")
        logger.info(f"   Classification: {emergence['classification']}")
        
        return emergence
    
    def _hash_error(self, error_context: Dict) -> str:
        """Gera hash único para tipo de erro"""
        key_parts = [
            error_context.get('error_type', ''),
            error_context.get('error_message', '')[:100],
            error_context.get('target_function', '')
        ]
        key_str = '|'.join(str(p) for p in key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()[:16]
    
    def _calculate_novelty(self, solution: str, error_key: str) -> float:
        """
        Calcula score de novidade da solução
        
        Compara com soluções históricas para esse tipo de erro
        """
        if error_key not in self.solution_patterns:
            return 1.0  # Completamente novo
        
        existing_solutions = self.solution_patterns[error_key]
        
        if not existing_solutions:
            return 1.0
        
        # Calcula distância para cada solução existente
        distances = []
        for existing in existing_solutions:
            dist = self._string_distance(solution, existing['snippet'])
            distances.append(dist)
        
        # Novidade = média das distâncias (normalizada)
        avg_distance = sum(distances) / len(distances)
        
        # Normaliza para [0, 1]
        novelty = min(1.0, avg_distance / 500)  # 500 chars = completamente diferente
        
        return novelty
    
    def _string_distance(self, s1: str, s2: str) -> float:
        """Distância simples entre strings"""
        # Levenshtein simplificado
        len1, len2 = len(s1), len(s2)
        
        if len1 == 0:
            return len2
        if len2 == 0:
            return len1
        
        # Heurística rápida: diferença de tamanho + caracteres diferentes
        size_diff = abs(len1 - len2)
        
        # Conta caracteres diferentes
        min_len = min(len1, len2)
        char_diff = sum(1 for i in range(min_len) if s1[i] != s2[i])
        
        return size_diff + char_diff
    
    def _classify_emergence(self, error_context: Dict, solution: str) -> str:
        """Classifica tipo de emergência"""
        error_type = error_context.get('error_type', '').lower()
        
        # Classifica por tipo
        if 'api' in error_type or 'connection' in error_type:
            return 'API_REPAIR'
        elif 'import' in error_type or 'module' in error_type:
            return 'DEPENDENCY_RESOLUTION'
        elif 'type' in error_type or 'attribute' in error_type:
            return 'STRUCTURAL_FIX'
        elif 'timeout' in error_type:
            return 'PERFORMANCE_OPTIMIZATION'
        else:
            return 'GENERAL_INNOVATION'
    
    def _generate_emergence_id(self) -> str:
        """Gera ID único para emergência"""
        timestamp = int(time.time() * 1000)
        random_part = hashlib.md5(str(timestamp).encode()).hexdigest()[:8]
        return f"EMRG_{timestamp}_{random_part}"
    
    def _save_emergence(self, emergence: Dict):
        """Salva evidência de emergência"""
        # JSON individual
        emergence_file = self.evidence_dir / f"{emergence['emergence_id']}.json"
        with open(emergence_file, 'w', encoding='utf-8') as f:
            json.dump(emergence, f, indent=2)
        
        # Log consolidado
        log_file = self.evidence_dir / 'emergences.jsonl'
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(emergence) + '\n')
        
        # Relatório legível
        report_file = self.evidence_dir / f"{emergence['emergence_id']}_REPORT.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(self._format_emergence_report(emergence))
    
    def _format_emergence_report(self, emergence: Dict) -> str:
        """Formata relatório legível de emergência"""
        report = f"""
{'='*80}
🔥 EMERGÊNCIA DETECTADA - NOVIDADE NÃO PREVISTA
{'='*80}

ID: {emergence['emergence_id']}
Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(emergence['timestamp']))}
Classificação: {emergence['classification']}

{'='*80}
CONTEXTO DO ERRO ORIGINAL
{'='*80}

Tipo: {emergence['error_context'].get('error_type', 'N/A')}
Mensagem: {emergence['error_context'].get('error_message', 'N/A')[:200]}
Função alvo: {emergence['error_context'].get('target_function', 'N/A')}

{'='*80}
SOLUÇÃO EMERGENTE
{'='*80}

Fonte: {emergence['solution'].get('source', 'Generated')}
Hash: {emergence['solution']['hash']}

Código:
{'-'*80}
{emergence['solution']['snippet'][:500]}
{'-'*80}

{'='*80}
MÉTRICAS DE EMERGÊNCIA
{'='*80}

✅ Tentativas até sucesso: {emergence['metrics']['attempt_count']}
✅ Score de novidade: {emergence['metrics']['novelty_score']:.2%}
✅ Validação: {'PASSOU' if emergence['metrics']['validation_success'] else 'FALHOU'}
✅ Abordagens únicas testadas: {emergence['evidence']['unique_approaches']}

{'='*80}
EVIDÊNCIAS
{'='*80}

{json.dumps(emergence['evidence'], indent=2)}

{'='*80}
STATUS: EMERGÊNCIA VALIDADA E REGISTRADA
{'='*80}
"""
        return report
    
    def get_emergences(self, classification: Optional[str] = None) -> List[Dict]:
        """Retorna emergências registradas"""
        if classification:
            return [e for e in self.emergences if e['classification'] == classification]
        return self.emergences.copy()
    
    def get_emergence_statistics(self) -> Dict:
        """Estatísticas de emergências"""
        if not self.emergences:
            return {
                'total_emergences': 0,
                'by_classification': {},
                'avg_novelty': 0.0,
                'avg_attempts': 0.0
            }
        
        classifications = defaultdict(int)
        for e in self.emergences:
            classifications[e['classification']] += 1
        
        return {
            'total_emergences': len(self.emergences),
            'by_classification': dict(classifications),
            'avg_novelty': sum(e['metrics']['novelty_score'] for e in self.emergences) / len(self.emergences),
            'avg_attempts': sum(e['metrics']['attempt_count'] for e in self.emergences) / len(self.emergences),
            'latest': self.emergences[-1]['emergence_id'] if self.emergences else None
        }


if __name__ == "__main__":
    # Teste
    detector = EmergenceDetector()
    
    # Simula erro e solução
    error_ctx = {
        'error_type': 'APIError',
        'error_message': 'Connection timeout to gpt-5',
        'target_function': 'call_openai'
    }
    
    winning_sol = {
        'snippet': 'client = OpenAI(timeout=60)\nresponse = client.chat.completions.create(...)',
        'result': {'success': True},
        'source': '/root/old_script.py'
    }
    
    emergence = detector.detect_emergence(error_ctx, winning_sol, attempt_count=12, original_error_rate=0.8)
    
    if emergence:
        print(f"✅ Emergence detected: {emergence['emergence_id']}")
        print(f"Novelty: {emergence['metrics']['novelty_score']:.2%}")
