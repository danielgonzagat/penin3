"""
LangGraph Integration - FIXED VERSION
Simple orchestrator without complex async
"""
import logging
from typing import Dict, Any, List, Optional, Callable

logger = logging.getLogger(__name__)

# Don't try to use LangGraph if it causes async issues
LANGGRAPH_AVAILABLE = False

class AgentOrchestrator:
    """
    Simple agent orchestrator
    Coordinates training cycles efficiently
    """
    
    def __init__(self):
        self.langgraph_available = LANGGRAPH_AVAILABLE
        logger.info(f"🔄 Agent Orchestrator initialized (simple mode)")
    
    def orchestrate_cycle(self, cycle: int,
                         mnist_fn: Callable,
                         cartpole_fn: Callable,
                         meta_fn: Callable,
                         api_fn: Callable) -> Dict[str, Any]:
        """
        Orchestrate training cycle - SIMPLE and ROBUST
        
        Args:
            cycle: Current cycle number
            mnist_fn: MNIST training function
            cartpole_fn: CartPole training function
            meta_fn: Meta-learning function
            api_fn: API consultation function
        
        Returns:
            Complete results dictionary
        """
        results = {
            'cycle': cycle,
            'orchestration_type': 'simple_sequential'
        }
        
        try:
            # 1. MNIST training
            results['mnist'] = mnist_fn()
            
            # 2. CartPole training  
            results['cartpole'] = cartpole_fn()
            
            # 3. Meta-learning (if functions accept args)
            try:
                results['meta'] = meta_fn(results['mnist'], results['cartpole'])
            except TypeError:
                results['meta'] = meta_fn()
            
            # 4. API consultation (periodic)
            if cycle % 10 == 0:
                try:
                    results['api'] = api_fn(results['mnist'], results['cartpole'])
                except TypeError:
                    results['api'] = api_fn()
            else:
                results['api'] = None
            
        except Exception as e:
            logger.error(f"Orchestration error: {e}")
            raise
        
        return results
    
    def get_orchestration_stats(self) -> Dict[str, Any]:
        """Get orchestration statistics"""
        return {
            'langgraph_available': self.langgraph_available,
            'orchestration_type': 'simple_sequential',
            'status': 'operational'
        }

