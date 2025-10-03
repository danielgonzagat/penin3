"""
SUPREME INTELLIGENCE AUDITOR - Extraído de IA3_REAL
Sistema de scoring para detectar inteligência REAL vs FAKE

Fonte: IA3_REAL/supreme_intelligence_auditor.py
"""
import logging
import ast
import re
from typing import Dict, List, Tuple, Any

logger = logging.getLogger(__name__)

class IntelligenceScorer:
    """
    Avalia código para detectar inteligência REAL
    Penaliza simulações e theater computacional
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
    
    def score_code(self, code: str) -> Dict[str, Any]:
        """
        Score código para inteligência real
        
        Returns:
            Dict com score, breakdown, is_real, is_fake
        """
        score = 0.0
        breakdown = {'penalties': [], 'rewards': [], 'total': 0.0}
        
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
        
        # 2. Real metrics
        if 'accuracy' in code and '/' in code and 'correct' in code:
            score += 20
            breakdown['rewards'].append(('real_metrics', 20))
        
        # 3. Real neural architecture
        if 'nn.Module' in code or 'tf.keras.Model' in code:
            score += 15
            breakdown['rewards'].append(('real_architecture', 15))
        
        # 4. Real persistence
        if ('pickle.dump' in code or 'torch.save' in code or 'json.dump' in code) and \
           ('memory' in code.lower() or 'state' in code.lower()):
            score += 25
            breakdown['rewards'].append(('real_persistence', 25))
        
        # 5. Real adaptation
        if 'reward' in code and 'if' in code and 'adjust' in code.lower():
            score += 20
            breakdown['rewards'].append(('real_adaptation', 20))
        
        # 6. Experience replay
        if 'replay' in code.lower() and 'sample' in code and 'batch' in code:
            score += 25
            breakdown['rewards'].append(('experience_replay', 25))
        
        # 7. Multi-agent
        if 'multi' in code.lower() and 'agent' in code:
            score += 15
            breakdown['rewards'].append(('multi_agent', 15))
        
        # 8. Self-modification
        if 'self.modify' in code or 'modify_architecture' in code:
            score += 20
            breakdown['rewards'].append(('self_modification', 20))
        
        # 9. Transfer learning
        if 'transfer' in code.lower() and 'learn' in code.lower():
            score += 30
            breakdown['rewards'].append(('transfer_learning', 30))
        
        # 10. Meta-learning
        if 'meta' in code.lower() and 'gradient' in code:
            score += 35
            breakdown['rewards'].append(('meta_learning', 35))
        
        breakdown['total'] = score
        
        return {
            'score': score,
            'breakdown': breakdown,
            'is_real': score > 0,
            'is_fake': score < 0
        }
    
    def score_system(self, system_path: str) -> Dict[str, Any]:
        """Score an entire system file"""
        try:
            with open(system_path, 'r') as f:
                code = f.read()
            
            result = self.score_code(code)
            result['path'] = system_path
            
            return result
            
        except Exception as e:
            return {
                'score': 0.0,
                'error': str(e),
                'is_real': False,
                'is_fake': False
            }
    
    def rank_systems(self, system_paths: List[str]) -> List[Dict[str, Any]]:
        """Rank multiple systems by intelligence score"""
        results = []
        
        for path in system_paths:
            result = self.score_system(path)
            results.append(result)
        
        # Sort by score (descending)
        results.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return results


if __name__ == "__main__":
    # Test the scorer
    scorer = IntelligenceScorer()
    
    test_code = """
    import torch
    import torch.nn as nn
    
    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(100):
        x = torch.randn(10)
        y = torch.randint(0, 2, (1,))
        
        output = model(x)
        loss = nn.functional.cross_entropy(output.unsqueeze(0), y)
        
        loss.backward()
        optimizer.step()
        
        correct = (output.argmax() == y).float().mean()
        accuracy = correct / 1.0
    """
    
    result = scorer.score_code(test_code)
    
    print("✅ Intelligence Scorer Test")
    print(f"   Score: {result['score']}")
    print(f"   Is Real: {result['is_real']}")
    print(f"   Breakdown: {result['breakdown']}")
