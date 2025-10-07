"""
MULTI-API CLIENT - UPDATED 2025-10-02
Suporta novos formatos de API (OpenAI responses, DeepSeek V3.1, etc)
"""
import logging
import os
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)

class MultiAPIClient:
    """Cliente unificado para múltiplas APIs com novos formatos"""
    
    def __init__(self, keys: Dict[str, str], models: Dict[str, str], configs: Dict[str, Dict]):
        self.keys = keys
        self.models = models
        self.configs = configs
        
        # Set environment variables
        for api, key in keys.items():
            os.environ[f"{api.upper()}_API_KEY"] = key
        
        logger.info(f"🌐 Multi-API Client initialized with {len(keys)} APIs")
    
    def call_openai(self, prompt: str, **kwargs) -> Optional[str]:
        """OpenAI com novo formato responses.create"""
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.keys.get("openai"))
            
            response = client.responses.create(
                model=self.models.get("gpt", "gpt-5"),
                input=prompt
            )
            
            return response.output_text
            
        except Exception as e:
            logger.warning(f"❌ OpenAI failed: {type(e).__name__}: {str(e)[:100]}")
            return None
    
    def call_deepseek(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """DeepSeek V3.1 (deepseek-chat ou deepseek-reasoner)"""
        try:
            from openai import OpenAI
            
            client = OpenAI(
                api_key=self.keys.get("deepseek"),
                base_url="https://api.deepseek.com"
            )
            
            response = client.chat.completions.create(
                model=self.models.get("deepseek", "deepseek-chat"),
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 200)
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"❌ DeepSeek failed: {type(e).__name__}: {str(e)[:100]}")
            return None
    
    def call_mistral(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Mistral (codestral-latest)"""
        try:
            from mistralai import Mistral
            
            client = Mistral(api_key=self.keys.get("mistral"))
            
            chat_response = client.chat.complete(
                model=self.models.get("mistral", "codestral-latest"),
                messages=messages
            )
            
            return chat_response.choices[0].message.content
            
        except Exception as e:
            logger.warning(f"❌ Mistral failed: {type(e).__name__}: {str(e)[:100]}")
            return None
    
    def call_gemini(self, prompt: str, **kwargs) -> Optional[str]:
        """Google Gemini (gemini-2.5-pro)"""
        try:
            from google import genai
            
            client = genai.Client(api_key=self.keys.get("gemini"))
            
            response = client.models.generate_content(
                model=self.models.get("gemini", "gemini-2.5-pro"),
                contents=prompt
            )
            
            return response.text
            
        except Exception as e:
            logger.warning(f"❌ Gemini failed: {type(e).__name__}: {str(e)[:100]}")
            return None
    
    def call_anthropic(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Anthropic Claude (claude-opus-4-1-20250805)"""
        try:
            import anthropic
            
            client = anthropic.Anthropic(api_key=self.keys.get("anthropic"))
            
            message = client.messages.create(
                model=self.models.get("claude", "claude-opus-4-1-20250805"),
                max_tokens=kwargs.get("max_tokens", 1024),
                messages=messages
            )
            
            return message.content[0].text
            
        except Exception as e:
            logger.warning(f"❌ Anthropic failed: {type(e).__name__}: {str(e)[:100]}")
            return None
    
    def call_grok(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """xAI Grok (grok-4)"""
        try:
            from xai_sdk import Client
            from xai_sdk.chat import user, system
            
            client = Client(
                api_key=self.keys.get("xai"),
                timeout=kwargs.get("timeout", 3600)
            )
            
            chat = client.chat.create(model=self.models.get("grok", "grok-4"))
            
            for msg in messages:
                if msg["role"] == "system":
                    chat.append(system(msg["content"]))
                elif msg["role"] == "user":
                    chat.append(user(msg["content"]))
            
            response = chat.sample()
            return response.content
            
        except Exception as e:
            logger.warning(f"❌ Grok failed: {type(e).__name__}: {str(e)[:100]}")
            return None
    
    def call_best_available(self, prompt: str, messages: Optional[List[Dict]] = None) -> Optional[Dict]:
        """
        Tenta todas as APIs na ordem de preferência
        Retorna: {"api": str, "response": str, "success": bool}
        """
        if messages is None:
            messages = [{"role": "user", "content": prompt}]
        
        # Ordem de preferência
        apis = [
            ("mistral", self.call_mistral, messages),
            ("gemini", self.call_gemini, prompt),
            ("deepseek", self.call_deepseek, messages),
            ("grok", self.call_grok, messages),
            ("openai", self.call_openai, prompt),
            ("anthropic", self.call_anthropic, messages),
        ]
        
        for api_name, api_func, api_input in apis:
            try:
                logger.debug(f"🔄 Trying {api_name}...")
                response = api_func(api_input)
                
                if response:
                    logger.info(f"✅ {api_name} succeeded")
                    return {
                        "api": api_name,
                        "response": response,
                        "success": True
                    }
            except Exception as e:
                logger.debug(f"❌ {api_name} error: {type(e).__name__}")
                continue
        
        logger.warning("❌ All APIs failed")
        return {"api": "none", "response": None, "success": False}


# Convenience function
def get_multi_api_client():
    """Get configured multi-API client"""
    try:
        from config.api_keys_updated import API_KEYS, API_MODELS, API_CONFIGS
        return MultiAPIClient(API_KEYS, API_MODELS, API_CONFIGS)
    except Exception as e:
        logger.error(f"Failed to initialize multi-API client: {e}")
        return None
