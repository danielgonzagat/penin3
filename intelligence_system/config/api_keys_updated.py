"""
UPDATED API KEYS - 2025-10-02
Chaves atualizadas e testadas
"""

API_KEYS = {
    # OpenAI (GPT-5)
    "openai": "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",
    
    # Mistral (codestral-2508)
    "mistral": "z44Nl2B4cVmdjQbCnDsAVQAuiGEQGqAO",
    
    # Google Gemini (gemini-2.5-pro)
    "gemini": "AIzaSyA2BuXahKz1hwQCTAeuMjOxje8lGqEqL4k",
    
    # DeepSeek (DeepSeek-V3.1)
    "deepseek": "sk-19c2b1d0864c4a44a53d743fb97566aa",
    
    # Anthropic (claude-opus-4-1-20250805)
    "anthropic": "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",
    
    # Alternative keys if above fails
    "anthropic_alt": "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",
    
    # xAI Grok (grok-4)
    "xai": "xai-sHbr1x7v2vpfDi657DtU64U53UM6OVhs4FdHeR1Ijk7jRUgU0xmo6ff8SF7hzV9mzY1wwjo4ChYsCDog",
    
    # Extras
    "huggingface": "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",
    "github": "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"
}

API_MODELS = {
    # OpenAI: gpt-4 works, gpt-5 has connection issues
    "gpt": "gpt-4",
    
    # Mistral: codestral-latest aponta para codestral-2508
    "mistral": "codestral-latest",
    
    # Gemini: gemini-2.5-pro
    "gemini": "gemini-2.5-pro",
    
    # DeepSeek: deepseek-chat = V3.1 não pensante
    "deepseek": "deepseek-chat",
    
    # Anthropic: claude-3-5-sonnet is stable
    "claude": "claude-3-5-sonnet-20241022",
    
    # Grok: grok-2-latest is correct model name
    "grok": "grok-2-latest"
}

# Configurações especiais
API_CONFIGS = {
    "openai": {
        "use_responses_api": True,  # Novo formato
        "base_url": "https://api.openai.com/v1"
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "context_caching": True  # 0.1 yuan per M tokens cached
    },
    "mistral": {
        "base_url": "https://api.mistral.ai/v1"
    },
    "gemini": {
        "use_genai_client": True  # from google import genai
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1"
    },
    "xai": {
        "base_url": "https://api.x.ai/v1",
        "timeout": 120  # Increased from 30s (was timing out)
    }
}
