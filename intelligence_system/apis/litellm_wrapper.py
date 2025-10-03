"""
LiteLLM Wrapper - FIXED VERSION
Handles API errors gracefully, fallback to individual APIs
"""
import logging
from typing import Dict, Any, Optional, List
import os
import time

logger = logging.getLogger(__name__)

# ========== CIRCUIT BREAKER (V7 UPGRADE) ==========
_PROVIDER_STATUS = {}  # provider -> {"fail_count":int, "down_until":ts}

def provider_ok(provider):
    st = _PROVIDER_STATUS.get(provider, {})
    down_until = st.get("down_until", 0)
    is_ok = time.time() > down_until
    if not is_ok:
        remaining = int(down_until - time.time())
        logger.debug(f"provider {provider} in cooldown for {remaining}s more")
    return is_ok

def record_failure(provider, backoff_sec=60):
    st = _PROVIDER_STATUS.setdefault(provider, {"fail_count":0, "down_until":0})
    st["fail_count"] += 1
    if st["fail_count"] >= 3:
        st["down_until"] = time.time() + backoff_sec
        logger.warning(f"🔴 Provider {provider} marked DOWN for {backoff_sec}s after {st['fail_count']} failures")
    _PROVIDER_STATUS[provider] = st

def record_success(provider):
    if provider in _PROVIDER_STATUS:
        _PROVIDER_STATUS[provider] = {"fail_count": 0, "down_until": 0}

def get_provider_status_summary():
    return {k: {"fails": v.get("fail_count", 0), 
                "down": time.time() < v.get("down_until", 0)} 
            for k,v in _PROVIDER_STATUS.items()}
# ================================================

try:
    import litellm
    litellm.suppress_debug_info = True
    LITELLM_AVAILABLE = True
except ImportError:
    LITELLM_AVAILABLE = False

class LiteLLMWrapper:
    """
    LiteLLM wrapper with ROBUST error handling
    Falls back gracefully when APIs fail
    """
    
    def __init__(self, api_keys: Dict[str, str], api_models: Dict[str, str]):
        # Normalize and record available providers
        self.api_keys = {k: (v.strip() if isinstance(v, str) else v) for k, v in api_keys.items()}
        self.available_providers = {k: bool(v) for k, v in self.api_keys.items()}
        # Filter out models for providers without keys
        self.api_models = {p: m for p, m in api_models.items() if self.available_providers.get(p, False)}
        self.litellm_available = LITELLM_AVAILABLE
        
        # Set environment variables
        for key, value in self.api_keys.items():
            env_key = f"{key.upper()}_API_KEY"
            if isinstance(value, str):
                value = value.strip()
            os.environ[env_key] = value or ""
        # Aliases for providers expecting different ENV names
        if 'grok' in api_keys and api_keys['grok']:
            # x.ai expects XAI_API_KEY
            os.environ['XAI_API_KEY'] = api_keys['grok']
        if 'openai' in api_keys and api_keys['openai']:
            os.environ['OPENAI_API_KEY'] = api_keys['openai']
        if 'deepseek' in api_keys and api_keys['deepseek']:
            os.environ['DEEPSEEK_API_KEY'] = api_keys['deepseek']
        if 'gemini' in api_keys and api_keys['gemini']:
            # Some SDKs expect GOOGLE_API_KEY instead of GEMINI_API_KEY
            os.environ['GOOGLE_API_KEY'] = api_keys['gemini']
        
        # Track API health
        self.api_health = {api: True for api in self.api_models.keys()}
        self.last_check = {api: 0 for api in self.api_models.keys()}
        
        logger.info(f"🚀 LiteLLM Wrapper initialized (available: {LITELLM_AVAILABLE})")
        logger.info(f"   Providers configured: {len(self.api_models)}/{len(api_models)}")
    
    def _canonical_model(self, model: str) -> str:
        """Return a litellm-compatible model string when user-provided name is not supported."""
        m = model.strip()
        # Anthropic: map Opus 4.1 (docs) → Claude 3 Opus (litellm canonical)
        if m.lower().startswith("claude-opus-4-1") or "opus-4-1" in m:
            return "anthropic/claude-3-opus-20240229"
        if m.lower().startswith("anthropic/") and "opus-4-1" in m.lower():
            return "anthropic/claude-3-opus-20240229"
        # Gemini: keep 2.5-pro if requested; otherwise pass through
        if m == "gemini-2.5-pro":
            return "gemini/gemini-2.5-pro"
        # DeepSeek: keep chat alias
        if m == "deepseek-chat":
            return "deepseek/deepseek-chat"
        # OpenAI Responses API stays as is (handled specially)
        return m

    def call_model(self, model: str, messages: List[Dict[str, str]], 
                   max_tokens: int = 200, temperature: float = 0.7,
                   timeout: int = 120) -> Optional[str]:
        """
        Call model with ROBUST error handling + CIRCUIT BREAKER
        Uses native SDKs for OpenAI (responses API), DeepSeek, Gemini, etc.
        """
        if not self.litellm_available:
            return None
        
        # V7: Check circuit breaker
        canonical = self._canonical_model(model)
        provider = canonical.split('/')[0] if '/' in canonical else canonical.split('-')[0]
        # Normalize provider names
        if model.startswith('gpt-5'):
            provider = 'openai'
        if not provider_ok(provider):
            logger.debug(f"Skipping {provider} (circuit breaker active)")
            return None
        
        try:
            # SPECIAL: OpenAI GPT-5 uses Responses API (per user docs)
            if canonical == 'gpt-5':
                import requests
                api_key = os.environ.get("OPENAI_API_KEY", "")
                # Extract prompt text from messages
                prompt_text = "\n".join([m.get("content", "") for m in messages if m.get("role") == "user"])
                resp = requests.post(
                    "https://api.openai.com/v1/responses",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={"model": model, "input": prompt_text},
                    timeout=timeout
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
                                        content = c.get("text")
                                        break
                            if content:
                                break
                    record_success(provider)
                    return content if content else None
                else:
                    # Fallback once to GPT-4.1 per user docs when GPT-5 is unavailable
                    fallback_model = 'gpt-4.1-2025-04-14'
                    resp2 = requests.post(
                        "https://api.openai.com/v1/responses",
                        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                        json={"model": fallback_model, "input": prompt_text},
                        timeout=timeout
                    )
                    if resp2.status_code == 200:
                        data = resp2.json()
                        content = data.get("output_text")
                        if not content:
                            for out in data.get("output", []):
                                if out.get("type") == "message":
                                    for c in out.get("content", []):
                                        if c.get("type") in ("text", "output_text"):
                                            content = c.get("text")
                                            break
                                if content:
                                    break
                        record_success(provider)
                        return content if content else None
                    else:
                        raise Exception(f"OpenAI Responses API error: {resp.status_code} and fallback {resp2.status_code}")
            
            # SPECIAL: Gemini – use google-genai directly
            if provider == 'gemini':
                try:
                    from google import genai
                except Exception as import_err:
                    raise Exception(f"google-genai not installed: {import_err}")
                client = genai.Client()
                # Concatenate user messages into a single prompt
                prompt_text = "\n".join([m.get("content", "") for m in messages if m.get("role") == "user"]) or messages[-1].get('content','')
                try:
                    # Use canonical model name without provider prefix if present
                    genai_model = canonical.split('/', 1)[1] if '/' in canonical else canonical
                    r = client.models.generate_content(model=genai_model, contents=prompt_text)
                except Exception:
                    # Fallback to 1.5-pro
                    fallback_model = 'gemini-1.5-pro'
                    r = client.models.generate_content(model=fallback_model, contents=prompt_text)
                text = getattr(r, 'text', None)
                if not text:
                    # Try candidates content fallback
                    try:
                        cand = getattr(r, 'candidates', [])[0]
                        parts = getattr(getattr(cand, 'content', {}), 'parts', [])
                        text = "\n".join([getattr(p, 'text', '') for p in parts if getattr(p, 'text', '')])
                    except Exception:
                        text = None
                record_success(provider)
                return text

            # SPECIAL: DeepSeek – use OpenAI-compatible REST directly
            if provider == 'deepseek':
                import requests
                api_key = os.environ.get('DEEPSEEK_API_KEY', '').strip()
                if not api_key:
                    raise Exception('Missing DEEPSEEK_API_KEY')
                payload = {
                    "model": "deepseek-chat",
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                }
                resp = requests.post(
                    "https://api.deepseek.com/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=timeout,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    content = (
                        data.get('choices', [{}])[0]
                            .get('message', {})
                            .get('content')
                    )
                    record_success(provider)
                    return content
                else:
                    raise Exception(f"DeepSeek API error: {resp.status_code}")

            # SPECIAL: Anthropic – use SDK directly
            if provider == 'anthropic':
                try:
                    import anthropic
                except Exception as import_err:
                    raise Exception(f"anthropic SDK not installed: {import_err}")
                api_key = os.environ.get('ANTHROPIC_API_KEY', '').strip()
                if not api_key:
                    raise Exception('Missing ANTHROPIC_API_KEY')
                client = anthropic.Anthropic(api_key=api_key)
                # Build messages (simple: last user content)
                prompt_text = "\n".join([m.get("content", "") for m in messages if m.get("role") == "user"]) or messages[-1].get('content','')
                try:
                    # Try user-requested model first, then fallbacks
                    anthropic_model = canonical.split('/', 1)[1] if canonical.startswith('anthropic/') else canonical
                    try_models = [
                        anthropic_model,
                        'claude-opus-4-1-20250805',
                        'claude-3-5-sonnet-20240620',
                        'claude-3-haiku-20240307',
                    ]
                    msg = None
                    last_err = None
                    for am in try_models:
                        if not am:
                            continue
                        try:
                            msg = client.messages.create(
                                model=am,
                                max_tokens=512,
                                messages=[{"role": "user", "content": prompt_text}]
                            )
                            anthropic_model = am
                            break
                        except Exception as e:
                            last_err = e
                            continue
                    if msg is None:
                        raise last_err or Exception('No Anthropic model available')
                    # Extract text
                    content = "\n".join([
                        getattr(part, 'text', '') if hasattr(part, 'text') else (part.get('text','') if isinstance(part, dict) else '')
                        for part in getattr(msg, 'content', [])
                    ])
                    record_success(provider)
                    return content or None
                except Exception as e:
                    raise

            # All other models: use LiteLLM
            # FIX: DeepSeek needs base_url
            kwargs = {
                "model": canonical,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "timeout": timeout
            }
            
            if "deepseek" in model:
                kwargs["api_base"] = "https://api.deepseek.com"
            
            try:
                response = litellm.completion(**kwargs)
            except Exception as primary_error:
                # Retry once with a safer fallback per provider
                fallback = None
                if provider == 'gemini' and '1.5-pro' in canonical:
                    fallback = 'gemini/gemini-1.5-flash'
                elif provider == 'gemini' and '2.5-pro' in model:
                    fallback = 'gemini/gemini-1.5-pro'
                elif provider == 'anthropic' and ('opus-4-1' in model or 'opus-4-1' in canonical):
                    fallback = 'anthropic/claude-3-opus-20240229'
                elif provider == 'mistral' and canonical.endswith('codestral-2508'):
                    fallback = 'mistral/codestral-latest'
                elif provider == 'deepseek' and 'deepseek-chat' in canonical:
                    # No alternate; retry once without api_base
                    kwargs.pop('api_base', None)
                if fallback:
                    kwargs['model'] = fallback
                response = litellm.completion(**kwargs)
            
            # V7: Record success
            record_success(provider)
            
            return response.choices[0].message.content
        
        except Exception as e:
            # V7: Record failure
            record_failure(provider, backoff_sec=60)
            logger.warning(f"LiteLLM call failed for {model}: {str(e)[:100]}")
            return None
    
    def call_all_models_robust(self, prompt: str, max_tokens: int = 150) -> Dict[str, str]:
        """
        Call ALL models, skip failures gracefully
        """
        results = {}
        messages = [{"role": "user", "content": prompt}]
        
        for api_name, model_name in self.api_models.items():
            # Skip unhealthy APIs
            if not self.api_health.get(api_name, True):
                if time.time() - self.last_check[api_name] < 300:  # 5 min cooldown
                    continue
            
            try:
                response = self.call_model(model_name, messages, max_tokens)
                
                if response:
                    results[api_name] = response
                    self.api_health[api_name] = True
                else:
                    self.api_health[api_name] = False
                    self.last_check[api_name] = time.time()
            
            except Exception as e:
                logger.warning(f"API {api_name} failed: {str(e)[:50]}")
                self.api_health[api_name] = False
                self.last_check[api_name] = time.time()
        
        return results
    
    def consult_for_improvement(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Consult APIs with ROBUST handling + CIRCUIT BREAKER
        """
        # V7: Log provider status before consulting
        status_summary = get_provider_status_summary()
        logger.debug(f"Provider status: {status_summary}")
        
        prompt = self._build_prompt(metrics)
        
        responses = self.call_all_models_robust(prompt, max_tokens=150)
        
        # V7: Log final summary
        total_providers = len(self.api_models)
        successful = len(responses)
        failed = total_providers - successful
        logger.info(f"📊 API Summary: {successful}/{total_providers} OK, {failed} failed | Status: {status_summary}")
        
        # Parse responses
        suggestions = {
            "increase_lr": False,
            "decrease_lr": False,
            "increase_exploration": False,
            "decrease_exploration": False,
            "architecture_change": False,
            "reasoning": []
        }
        
        for api_name, response in responses.items():
            reasoning = self._parse_response(response)
            reasoning['api'] = api_name
            suggestions['reasoning'].append(reasoning)
            
            # Aggregate suggestions
            if 'increase' in response.lower() and 'lr' in response.lower():
                suggestions['increase_lr'] = True
            if 'decrease' in response.lower() and 'lr' in response.lower():
                suggestions['decrease_lr'] = True
        
        logger.info(f"✅ Consulted {len(responses)}/{len(self.api_models)} APIs successfully")
        
        return suggestions
    
    def _build_prompt(self, metrics: Dict[str, Any]) -> str:
        """Build consultation prompt"""
        return f"""You are an ML expert. Analyze these metrics briefly:

MNIST: Train {metrics.get('mnist_train', 0):.1f}%, Test {metrics.get('mnist_test', 0):.1f}%
CartPole: Last {metrics.get('cartpole_last', 0):.1f}, Avg {metrics.get('cartpole_avg', 0):.1f}
Cycle: {metrics.get('cycle', 0)}, Stagnation: {metrics.get('stagnation', 0):.2f}

Suggest ONE concrete improvement (increase/decrease LR, exploration, architecture, etc)."""
    
    def _parse_response(self, response: str) -> Dict[str, str]:
        """Parse API response"""
        return {
            'response': response[:200],
            'analysis': 'improvement_suggested'
        }
    
    def get_healthy_apis(self) -> List[str]:
        """Get list of currently healthy APIs"""
        return [api for api, health in self.api_health.items() if health]

