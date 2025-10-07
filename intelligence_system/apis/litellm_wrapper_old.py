"""
LiteLLM Integration - Universal API Gateway
Merge completo do litellm para unificar todas as 6 APIs
"""
import logging
from typing import Dict, Any, Optional, List
import os

logger = logging.getLogger(__name__)

# Try to import litellm, fallback to basic if not available
try:
    from litellm import completion, embedding
    LITELLM_AVAILABLE = True
    logger.info("✅ LiteLLM imported successfully")
except ImportError:
    LITELLM_AVAILABLE = False
    logger.warning("⚠️ LiteLLM not available, using fallback")

class LiteLLMWrapper:
    """Unified API wrapper using LiteLLM for all 6 frontier APIs"""
    
    def __init__(self, api_keys: Dict[str, str], api_models: Dict[str, str]):
        self.api_keys = api_keys
        self.api_models = api_models
        self.litellm_available = LITELLM_AVAILABLE
        
        # Set environment variables for litellm
        os.environ["OPENAI_API_KEY"] = api_keys.get("openai", "")
        os.environ["ANTHROPIC_API_KEY"] = api_keys.get("anthropic", "")
        os.environ["GEMINI_API_KEY"] = api_keys.get("gemini", "")
        os.environ["MISTRAL_API_KEY"] = api_keys.get("mistral", "")
        os.environ["DEEPSEEK_API_KEY"] = api_keys.get("deepseek", "")
        os.environ["XAI_API_KEY"] = api_keys.get("grok", "")
        
        logger.info(f"🚀 LiteLLM Wrapper initialized (available: {LITELLM_AVAILABLE})")
    
    def call_model(self, model: str, messages: List[Dict[str, str]], 
                   max_tokens: int = 200, temperature: float = 0.7) -> Optional[str]:
        """
        Universal model call using LiteLLM
        
        Args:
            model: Model name (e.g., 'gpt-4o-mini', 'claude-3-5-sonnet', etc)
            messages: List of message dicts
            max_tokens: Max tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Response text or None if failed
        """
        if not self.litellm_available:
            return self._fallback_call(model, messages, max_tokens)
        
        try:
            response = completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                timeout=15
            )
            
            content = response.choices[0].message.content
            logger.info(f"✅ LiteLLM call successful: {model}")
            return content
            
        except Exception as e:
            logger.error(f"❌ LiteLLM call failed for {model}: {e}")
            return None
    
    def call_all_models(self, prompt: str, max_tokens: int = 150) -> Dict[str, str]:
        """
        Call all 6 frontier APIs in parallel using LiteLLM
        
        Returns dict of {model_name: response}
        """
        messages = [{"role": "user", "content": prompt}]
        
        results = {}
        
        # OpenAI
        try:
            results['openai'] = self.call_model(
                self.api_models.get('openai', 'gpt-4o-mini'),
                messages, max_tokens
            )
        except Exception as e:
            logger.warning(f"OpenAI failed: {e}")
        
        # Anthropic
        try:
            results['anthropic'] = self.call_model(
                self.api_models.get('anthropic', 'claude-3-5-sonnet-20241022'),
                messages, max_tokens
            )
        except Exception as e:
            logger.warning(f"Anthropic failed: {e}")
        
        # Gemini
        try:
            model_name = f"gemini/{self.api_models.get('gemini', 'gemini-2.0-flash-exp')}"
            results['gemini'] = self.call_model(model_name, messages, max_tokens)
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")
        
        # Mistral
        try:
            results['mistral'] = self.call_model(
                self.api_models.get('mistral', 'mistral-large-latest'),
                messages, max_tokens
            )
        except Exception as e:
            logger.warning(f"Mistral failed: {e}")
        
        # DeepSeek
        try:
            model_name = f"deepseek/{self.api_models.get('deepseek', 'deepseek-chat')}"
            results['deepseek'] = self.call_model(model_name, messages, max_tokens)
        except Exception as e:
            logger.warning(f"DeepSeek failed: {e}")
        
        # Grok
        try:
            model_name = f"xai/{self.api_models.get('grok', 'grok-beta')}"
            results['grok'] = self.call_model(model_name, messages, max_tokens)
        except Exception as e:
            logger.warning(f"Grok failed: {e}")
        
        successful = sum(1 for v in results.values() if v is not None)
        logger.info(f"📊 Called {successful}/6 APIs successfully")
        
        return results
    
    def _fallback_call(self, model: str, messages: List, max_tokens: int) -> Optional[str]:
        """Fallback to direct API calls if litellm not available"""
        # Use original api_manager logic as fallback
        logger.warning(f"Using fallback for {model}")
        return None
    
    def consult_for_improvement(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consult all 6 APIs for improvement suggestions
        Uses litellm for unified calling
        """
        prompt = self._build_improvement_prompt(metrics)
        
        suggestions = {
            "increase_lr": False,
            "decrease_lr": False,
            "increase_exploration": False,
            "decrease_exploration": False,
            "add_regularization": False,
            "architecture_change": False,
            "reasoning": []
        }
        
        # Call all models
        responses = self.call_all_models(prompt, max_tokens=200)
        
        # Parse all responses
        for api_name, response in responses.items():
            if response is None:
                continue
            
            suggestions["reasoning"].append({
                "api": api_name,
                "response": response[:500]  # Truncate for storage
            })
            
            response_lower = response.lower()
            
            # Parse suggestions
            if "increase learning rate" in response_lower or "higher lr" in response_lower:
                suggestions["increase_lr"] = True
            if "decrease learning rate" in response_lower or "lower lr" in response_lower:
                suggestions["decrease_lr"] = True
            if "more exploration" in response_lower or "increase epsilon" in response_lower:
                suggestions["increase_exploration"] = True
            if "less exploration" in response_lower or "decrease epsilon" in response_lower:
                suggestions["decrease_exploration"] = True
            if "regularization" in response_lower or "dropout" in response_lower:
                suggestions["add_regularization"] = True
            if "architecture" in response_lower or "network" in response_lower:
                suggestions["architecture_change"] = True
        
        logger.info(f"✅ Got suggestions from {len(suggestions['reasoning'])} APIs")
        return suggestions
    
    def _build_improvement_prompt(self, metrics: Dict[str, Any]) -> str:
        """Build prompt for API consultation"""
        return f"""Analyze AI training performance and suggest ONE specific improvement (max 80 words):

Current Metrics:
- MNIST Test: {metrics.get('mnist_test', 0):.1f}%
- MNIST Train: {metrics.get('mnist_train', 0):.1f}%
- CartPole Avg: {metrics.get('cartpole_avg', 0):.1f}
- CartPole Last: {metrics.get('cartpole_last', 0):.1f}
- Epsilon: {metrics.get('epsilon', 1.0):.3f}
- Cycle: {metrics.get('cycle', 0)}
- Stagnation: {metrics.get('stagnation', 0):.2f}

Recent 5 MNIST: {metrics.get('recent_mnist', [])}
Recent 5 CartPole: {metrics.get('recent_cartpole', [])}

Give ONE actionable suggestion."""
