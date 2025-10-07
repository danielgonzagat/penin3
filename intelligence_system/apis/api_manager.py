"""
Professional API Manager
Uses responses productively, not just logging!
"""
import logging
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import requests

logger = logging.getLogger(__name__)


class APIManager:
    """Smart API manager that uses responses productively"""
    
    def __init__(self, api_keys: Dict[str, str], api_models: Dict[str, str]):
        self.api_keys = api_keys
        self.api_models = api_models
        self.response_cache = {}
    
    def consult_for_improvement(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consult APIs for actionable improvements
        Returns: dict with suggestions for system improvement
        """
        prompt = self._build_improvement_prompt(metrics)
        
        suggestions = {
            "increase_lr": False,
            "decrease_lr": False,
            "increase_exploration": False,
            "decrease_exploration": False,
            "add_regularization": False,
            "reasoning": []
        }
        
        # Consult multiple APIs and aggregate
        responses = []
        
        # Try DeepSeek (fast and good)
        try:
            response = self._call_deepseek(prompt)
            if response:
                responses.append(("deepseek", response))
                logger.info(f"✅ DeepSeek consulted")
        except Exception as e:
            logger.warning(f"DeepSeek failed: {e}")
        
        # Try Gemini (fast)
        try:
            response = self._call_gemini(prompt)
            if response:
                responses.append(("gemini", response))
                logger.info(f"✅ Gemini consulted")
        except Exception as e:
            logger.warning(f"Gemini failed: {e}")

        # Try OpenAI Responses API (GPT‑5 per user docs)
        try:
            import os, requests
            api_key = os.environ.get("OPENAI_API_KEY", "")
            model = self.api_models.get('openai', 'gpt-5')
            resp = requests.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "input": prompt},
                timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                content = data.get("output_text")
                if not content:
                    # Extract from output list
                    for out in data.get("output", []):
                        if out.get("type") == "message":
                            for c in out.get("content", []):
                                if c.get("type") in ("text", "output_text"):
                                    content = c.get("text"); break
                        if content:
                            break
                if content:
                    responses.append(("openai", content))
                    logger.info("✅ OpenAI (responses) consulted")
        except Exception as e:
            logger.warning(f"OpenAI responses failed: {e}")
        
        # Parse responses
        for api_name, response in responses:
            analysis = self._parse_improvement_response(response)
            suggestions["reasoning"].append({
                "api": api_name,
                "analysis": analysis
            })
            
            # Aggregate suggestions (majority vote)
            if "increase learning rate" in response.lower() or "higher lr" in response.lower():
                suggestions["increase_lr"] = True
            if "decrease learning rate" in response.lower() or "lower lr" in response.lower():
                suggestions["decrease_lr"] = True
            if "more exploration" in response.lower() or "increase epsilon" in response.lower():
                suggestions["increase_exploration"] = True
            if "less exploration" in response.lower() or "decrease epsilon" in response.lower():
                suggestions["decrease_exploration"] = True
            if "regularization" in response.lower() or "overfitting" in response.lower():
                suggestions["add_regularization"] = True
        
        return suggestions
    
    def _build_improvement_prompt(self, metrics: Dict[str, Any]) -> str:
        """Build concise prompt for API consultation"""
        return f"""Analyze this AI training performance and suggest ONE specific improvement:

Current Metrics:
- MNIST Test Accuracy: {metrics.get('mnist_test', 0):.1f}%
- MNIST Train Accuracy: {metrics.get('mnist_train', 0):.1f}%
- CartPole Avg Reward: {metrics.get('cartpole_avg', 0):.1f}
- CartPole Last Reward: {metrics.get('cartpole_last', 0):.1f}
- DQN Epsilon: {metrics.get('epsilon', 1.0):.3f}
- Cycles: {metrics.get('cycle', 0)}
- Stagnation Score: {metrics.get('stagnation', 0):.2f}

Previous 5 cycles MNIST: {metrics.get('recent_mnist', [])}
Previous 5 cycles CartPole: {metrics.get('recent_cartpole', [])}

Respond in under 100 words with ONE actionable suggestion."""
    
    def _parse_improvement_response(self, response: str) -> str:
        """Extract key insights from API response"""
        # Simple parsing for now
        lines = response.strip().split('\n')
        return ' '.join(lines[:3])  # First 3 lines usually have the key info
    
    def _call_deepseek(self, prompt: str, max_tokens: int = 150) -> Optional[str]:
        """Call DeepSeek API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_keys['deepseek']}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": self.api_models['deepseek'],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.7
            }
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                return result['choices'][0]['message']['content']
            else:
                logger.warning(f"DeepSeek API error: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"DeepSeek API call failed: {e}")
            return None
    
    def _call_gemini(self, prompt: str, max_tokens: int = 150) -> Optional[str]:
        """Call Gemini API"""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_keys['gemini'])
            
            model = genai.GenerativeModel(self.api_models['gemini'])
            response = model.generate_content(
                prompt,
                generation_config={"max_output_tokens": max_tokens}
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini API call failed: {e}")
            return None
    
    def get_meta_advice(self, cycle: int, history: List[Dict]) -> Optional[str]:
        """Get high-level meta-learning advice"""
        if cycle % 50 != 0:  # Only every 50 cycles
            return None
        
        prompt = f"""You are an AI research advisor. Review this learning history:

Last 10 cycles summary:
{json.dumps(history[-10:], indent=2)}

Provide ONE strategic insight about the overall learning process in 2-3 sentences."""
        
        try:
            response = self._call_deepseek(prompt, max_tokens=100)
            return response
        except Exception as e:
            logger.error(f"Meta advice failed: {e}")
            return None
