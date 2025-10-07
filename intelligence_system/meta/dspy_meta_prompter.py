"""
DSPy Integration - Meta-Prompting and LM Programming
Merge do dspy para otimização automática de prompts
"""
import logging
from typing import Dict, Any, List, Optional
import json

logger = logging.getLogger(__name__)

try:
    import dspy
    DSPY_AVAILABLE = True
except ImportError:
    logger.warning("DSPy not available - using manual prompting")
    DSPY_AVAILABLE = False

class DSPyMetaPrompter:
    """
    DSPy-based meta-prompting system
    Automatically optimizes prompts for API consultations
    """
    
    def __init__(self, api_keys: Dict[str, str]):
        self.dspy_available = DSPY_AVAILABLE
        self.api_keys = api_keys
        
        if DSPY_AVAILABLE:
            self._init_dspy()
        else:
            logger.info("⚠️  Using manual prompt engineering (no DSPy)")
        
        # Prompt templates (manual fallback)
        self.prompt_templates = {
            'improvement': self._get_improvement_template(),
            'diagnosis': self._get_diagnosis_template(),
            'strategy': self._get_strategy_template()
        }
        
        logger.info(f"🔄 DSPy Meta-Prompter initialized (DSPy: {DSPY_AVAILABLE})")
    
    def _init_dspy(self):
        """Initialize DSPy with LM configuration"""
        try:
            # Configure DSPy with OpenAI (example)
            lm = dspy.OpenAI(
                model='gpt-4o-mini',
                api_key=self.api_keys.get('openai', ''),
                max_tokens=300
            )
            dspy.settings.configure(lm=lm)
            
            # Define signatures (future)
            self.improvement_signature = dspy.Signature(
                "metrics -> suggestions",
                instructions="Given ML metrics, suggest concrete improvements"
            )
            
            logger.info("✅ DSPy configured with OpenAI")
        except Exception as e:
            logger.warning(f"DSPy initialization failed: {e}")
            self.dspy_available = False
    
    def _get_improvement_template(self) -> str:
        """Manual improvement prompt template"""
        return """You are an expert ML engineer. Analyze these metrics and suggest 1-2 concrete improvements:

MNIST: Train {mnist_train}%, Test {mnist_test}%
CartPole: Last {cartpole_last}, Avg {cartpole_avg}
Cycle: {cycle}, Stagnation: {stagnation}

Respond with JSON:
{{
  "increase_lr": true/false,
  "decrease_lr": true/false,
  "increase_exploration": true/false,
  "decrease_exploration": true/false,
  "architecture_change": true/false,
  "reasoning": "brief explanation"
}}"""
    
    def _get_diagnosis_template(self) -> str:
        """Manual diagnosis prompt template"""
        return """Diagnose this ML system's performance:

Recent MNIST: {recent_mnist}
Recent CartPole: {recent_cartpole}
Best MNIST: {best_mnist}%, Best CartPole: {best_cartpole}

What's the main issue? (overfitting, underfitting, stagnation, exploration, etc)"""
    
    def _get_strategy_template(self) -> str:
        """Manual strategy prompt template"""
        return """Given this diagnosis: {diagnosis}

What's the best strategy? (tune_lr, add_regularization, change_architecture, increase_exploration, etc)"""
    
    def generate_improvement_prompt(self, metrics: Dict[str, Any]) -> str:
        """
        Generate optimized improvement prompt
        
        Args:
            metrics: Current system metrics
        
        Returns:
            Optimized prompt string
        """
        if self.dspy_available:
            # Future: Use DSPy optimizer
            return self._generate_dspy_prompt(metrics)
        else:
            return self._generate_manual_prompt(metrics)
    
    def _generate_manual_prompt(self, metrics: Dict[str, Any]) -> str:
        """Generate prompt manually (fallback)"""
        template = self.prompt_templates['improvement']
        return template.format(
            mnist_train=metrics.get('mnist_train', 0),
            mnist_test=metrics.get('mnist_test', 0),
            cartpole_last=metrics.get('cartpole_last', 0),
            cartpole_avg=metrics.get('cartpole_avg', 0),
            cycle=metrics.get('cycle', 0),
            stagnation=metrics.get('stagnation', 0)
        )
    
    def _generate_dspy_prompt(self, metrics: Dict[str, Any]) -> str:
        """Generate prompt with DSPy optimization (future)"""
        # For now, same as manual
        return self._generate_manual_prompt(metrics)
    
    def optimize_prompts(self, examples: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Optimize prompts based on examples (DSPy feature)
        
        Args:
            examples: List of (metrics, response) pairs
        
        Returns:
            Optimized prompt templates
        """
        if not self.dspy_available:
            logger.info("⚠️  DSPy optimization not available")
            return self.prompt_templates
        
        # Future: Use DSPy's MIPRO or BootstrapFewShot
        logger.info("🔄 Optimizing prompts with DSPy (future feature)")
        return self.prompt_templates
    
    def get_meta_stats(self) -> Dict[str, Any]:
        """Get meta-prompting statistics"""
        return {
            'dspy_available': self.dspy_available,
            'prompt_types': list(self.prompt_templates.keys()),
            'optimization_enabled': False  # Future
        }

