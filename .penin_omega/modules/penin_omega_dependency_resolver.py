#!/usr/bin/env python3
"""
PENIN-Ω Dependency Resolver - Mínimo
===================================
"""

async def safe_import(module_name, fallback=None):
    """Import seguro com fallback."""
    try:
        return await __import__(module_name)
    except ImportError:
        return await fallback

# Configuração básica
config = {
    'deepseek_api_key': 'sk-19c2b1d0864c4a44a53d743fb97566aa',
    'anthropic_api_key': 'sk-ant-api03-your-key-here',
    'openai_api_key': 'sk-your-openai-key-here',
    'grok_api_key': 'xai-sHbr1x7v2vpfDi657DtUF7hzV9mzY1wwjo4ChYsCDog',
    'mistral_api_key': 'your-mistral-key-here',
    'gemini_api_key': 'your-gemini-key-here'
}
