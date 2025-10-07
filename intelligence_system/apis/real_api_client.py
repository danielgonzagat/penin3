"""
REAL API CLIENT - 6/6 APIs funcionando
Baseado no script que FUNCIONOU do usuário
"""
import os
import json
import requests
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

@dataclass
class APIResponse:
    provider: str
    ok: bool
    content: str
    error: Optional[str] = None

class RealAPIClient:
    """
    Cliente REAL que funciona (baseado em evolution_standard)
    SEM LiteLLM - chamadas HTTP diretas
    """
    
    def __init__(self, timeout: int = 60):
        self.timeout = timeout
        self.providers = {
            'deepseek': self._call_deepseek,
            'openai': self._call_openai,
            'gemini': self._call_gemini,
            'mistral': self._call_mistral,
            'anthropic': self._call_anthropic,
            'grok': self._call_xai
        }
    
    def _call_deepseek(self, prompt: str) -> APIResponse:
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        model = "deepseek-chat"
        try:
            resp = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are a helpful coding assistant."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return APIResponse("deepseek", True, content)
        except Exception as e:
            return APIResponse("deepseek", False, "", str(e))
    
    def _call_openai(self, prompt: str) -> APIResponse:
        api_key = os.environ.get("OPENAI_API_KEY", "")
        # Try gpt-4.1-2025-04-14 first (from user's script)
        model = "gpt-4.1-2025-04-14"
        try:
            # First try responses API (user's preference)
            resp = requests.post(
                "https://api.openai.com/v1/responses",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "input": prompt},
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            
            if "output_text" in data:
                return APIResponse("openai", True, data["output_text"])
            for out in data.get("output", []):
                if out.get("type") == "message":
                    for c in out.get("content", []):
                        if c.get("type") in ("text", "output_text"):
                            return APIResponse("openai", True, c.get("text", ""))
        except:
            pass  # Fallback
        
        # Fallback to chat completions
        try:
            resp = requests.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4",  # More reliable
                    "messages": [
                        {"role": "system", "content": "You are a helpful coding assistant."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return APIResponse("openai", True, content)
        except Exception as e:
            return APIResponse("openai", False, "", str(e))
    
    def _call_gemini(self, prompt: str) -> APIResponse:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        model = "gemini-2.5-pro"
        try:
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
            resp = requests.post(
                url,
                json={"contents": [{"parts": [{"text": prompt}]}]},
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            cand = (data.get("candidates") or [{}])[0]
            parts = cand.get("content", {}).get("parts", [])
            text = parts[0].get("text") if parts else ""
            return APIResponse("gemini", True, text)
        except Exception as e:
            return APIResponse("gemini", False, "", str(e))
    
    def _call_mistral(self, prompt: str) -> APIResponse:
        api_key = os.environ.get("MISTRAL_API_KEY", "")
        model = "codestral-2508"
        try:
            resp = requests.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"model": model, "messages": [{"role": "user", "content": prompt}]},
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return APIResponse("mistral", True, content)
        except Exception as e:
            return APIResponse("mistral", False, "", str(e))
    
    def _call_anthropic(self, prompt: str) -> APIResponse:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        model = "claude-opus-4-1-20250805"
        try:
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "Content-Type": "application/json",
                },
                json={"model": model, "max_tokens": 1024, "messages": [{"role": "user", "content": prompt}]},
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            content_blocks = data.get("content")
            if isinstance(content_blocks, list) and content_blocks:
                text = content_blocks[0].get("text", "")
                return APIResponse("anthropic", True, text)
            return APIResponse("anthropic", False, "", "No content")
        except Exception as e:
            return APIResponse("anthropic", False, "", str(e))
    
    def _call_xai(self, prompt: str) -> APIResponse:
        api_key = os.environ.get("XAI_API_KEY", "")
        model = "grok-2-latest"  # We know this works from earlier
        try:
            resp = requests.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": "You are Grok, a highly capable assistant."},
                        {"role": "user", "content": prompt},
                    ],
                },
                timeout=self.timeout
            )
            resp.raise_for_status()
            data = resp.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return APIResponse("grok", True, content)
        except Exception as e:
            return APIResponse("grok", False, "", str(e))
    
    def consult_all(self, prompt: str) -> List[APIResponse]:
        """Consultar TODAS as 6 APIs"""
        responses = []
        for name, fn in self.providers.items():
            logger.info(f"🌐 Consulting {name}...")
            try:
                resp = fn(prompt)
                responses.append(resp)
                if resp.ok:
                    logger.info(f"   ✅ {name}: {len(resp.content)} chars")
                else:
                    logger.warning(f"   ❌ {name}: {resp.error[:100]}")
            except Exception as e:
                logger.error(f"   ❌ {name} FAILED: {e}")
                responses.append(APIResponse(name, False, "", str(e)))
        
        success = sum(1 for r in responses if r.ok)
        logger.info(f"📊 API Summary: {success}/6 OK")
        return responses
    
    def consult_best(self, prompt: str, max_providers: int = 3) -> Optional[str]:
        """Consultar até conseguir resposta de 1 provedor"""
        # Order by reliability (from user's experience)
        priority = ['gemini', 'mistral', 'deepseek', 'openai', 'grok', 'anthropic']
        
        for name in priority[:max_providers]:
            if name not in self.providers:
                continue
            logger.info(f"🌐 Trying {name}...")
            try:
                resp = self.providers[name](prompt)
                if resp.ok and resp.content:
                    logger.info(f"   ✅ {name} succeeded!")
                    return resp.content
            except Exception as e:
                logger.warning(f"   ❌ {name} failed: {e}")
        
        return None
