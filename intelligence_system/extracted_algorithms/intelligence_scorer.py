"""
INTELLIGENCE SCORER - Extraído de supreme_intelligence_auditor.py
Detecta inteligência REAL vs FAKE com scoring científico
"""
import logging
import re
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)


class IntelligenceScorer:
    """
    Score intelligence in code
    Penaliza fake intelligence, recompensa real intelligence
    """
    
    def __init__(self):
        self.intelligence_markers = {
            'learning': ['gradient', 'backward', 'optimizer', 'loss.backward'],
            'adaptation': ['adapt', 'adjust', 'modify', 'evolve', 'improve'],
            'memory': ['memory', 'remember', 'recall', 'store', 'retrieve'],
            'reasoning': ['reason', 'think', 'infer', 'deduce', 'conclude'],
            'emergence': ['emergent', 'emerge', 'spontaneous', 'unexpected'],
            'consciousness': ['aware', 'conscious', 'self.analyze', 'introspect'],
            'creativity': ['create', 'generate', 'imagine', 'novel', 'invent'],
            'autonomy': ['autonomous', 'decide', 'choice', 'independent'],
            'recursion': ['recursive', 'self.improve', 'meta', 'self.modify'],
            'real_metrics': ['accuracy', 'f1_score', 'precision', 'recall', 'real']
        }
        
        logger.info("🧠 Intelligence Scorer initialized")
        logger.info(f"   Markers: {len(self.intelligence_markers)} categories")
    
    def score_code(self, code: str) -> Dict[str, float]:
        """
        Score intelligence in code
        Returns dict with score and breakdown
        """
        score = 0.0
        breakdown = {
            'penalties': [],
            'rewards': [],
            'total': 0.0
        }
        
        # PENALIZE FAKE INTELLIGENCE
        if 'random.random()' in code and 'fitness' in code:
            score -= 50
            breakdown['penalties'].append(('fake_fitness', -50))
        
        if 'while True:' in code and 'evolve' in code and 'break' not in code:
            score -= 30
            breakdown['penalties'].append(('infinite_loop', -30))
        
        if 'consciousness' in code and 'return await True' in code:
            score -= 40
            breakdown['penalties'].append(('fake_consciousness', -40))
        
        # REWARD REAL INTELLIGENCE
        
        # 1. Real gradient learning
        if 'loss.backward()' in code and 'optimizer.step()' in code:
            score += 30
            breakdown['rewards'].append(('real_learning', 30))
        
        # 2. Real metrics being calculated
        if 'accuracy' in code and '/' in code and 'correct' in code:
            score += 20
            breakdown['rewards'].append(('real_metrics', 20))
        
        # 3. Real neural architecture
        if 'nn.Module' in code or 'tf.keras.Model' in code:
            score += 15
            breakdown['rewards'].append(('neural_architecture', 15))
        
        # 4. Real persistent memory
        if ('pickle.dump' in code or 'torch.save' in code or 'json.dump' in code):
            if 'memory' in code.lower() or 'state' in code.lower():
                score += 25
                breakdown['rewards'].append(('persistent_memory', 25))
        
        # 5. Real adaptation based on feedback
        if 'reward' in code and 'if' in code and 'adjust' in code.lower():
            score += 20
            breakdown['rewards'].append(('real_adaptation', 20))
        
        # 6. Experience replay (real learning)
        if 'replay' in code.lower() and 'sample' in code and 'batch' in code:
            score += 25
            breakdown['rewards'].append(('experience_replay', 25))
        
        # 7. Multi-agent or distributed
        if 'multi' in code.lower() and 'agent' in code:
            score += 15
            breakdown['rewards'].append(('multi_agent', 15))
        
        # 8. Self-modification (real)
        if 'self.modify' in code or 'modify_architecture' in code:
            score += 20
            breakdown['rewards'].append(('self_modification', 20))
        
        breakdown['total'] = score
        
        return {
            'score': score,
            'breakdown': breakdown,
            'is_real': score > 0,
            'is_fake': score < 0
        }
    
    def score_markers(self, code: str) -> Dict[str, int]:
        """
        Count intelligence markers by category
        """
        marker_counts = {}
        
        for category, markers in self.intelligence_markers.items():
            count = 0
            for marker in markers:
                count += code.lower().count(marker.lower())
            marker_counts[category] = count
        
        return marker_counts
    
    def audit_file(self, file_path: str) -> Dict[str, Any]:
        """
        Audit a Python file for real intelligence
        """
        try:
            with open(file_path, 'r') as f:
                code = f.read()
            
            score_result = self.score_code(code)
            markers = self.score_markers(code)
            
            return {
                'file': file_path,
                'score': score_result['score'],
                'is_real': score_result['is_real'],
                'is_fake': score_result['is_fake'],
                'breakdown': score_result['breakdown'],
                'markers': markers,
                'total_markers': sum(markers.values())
            }
        
        except Exception as e:
            return {
                'file': file_path,
                'error': str(e),
                'score': 0
            }
