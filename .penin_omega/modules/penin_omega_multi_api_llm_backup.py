#!/usr/bin/env python3
"""
PENIN-Ω Multi-API LLM Integration Module
========================================
Integrates all major AI APIs: DeepSeek, OpenAI GPT-5, Gemini, Mistral, Grok, Anthropic
"""

import os
import json
import time
from typing import Dict, List, Optional, Any

class MultiAPILLM:
    async def __init__(self):
        self.apis = {
            'deepseek': {
                'key': 'sk-19c2b1d0864c4a44a53d743fb97566aa',
                'model': 'deepseek-chat',
                'base_url': 'https://api.deepseek.com'
            },
            'openai': {
                'key': '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',
                'model': 'gpt-5'
            },
            'gemini': {
                'key': 'AIzaSyA2BuXahKz1hwQCTAeuMjOxje8lGqEqL4k',
                'model': 'gemini-2.5-pro'
            },
            'mistral': {
                'key': 'AMTeAQrzudpGvU2jkU9hVRvSsYr1hcni',
                'model': 'codestral-2508'
            },
            'grok': {
                'key': 'xai-sHbr1x7v2vpfDi657DtU64U53UM6OVhs4FdHeR1Ijk7jRUgU0xmo6ff8SF7hzV9mzY1wwjo4ChYsCDog',
                'model': 'grok-4'
            },
            'anthropic': {
                'key': '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',
                'model': 'claude-opus-4-1-20250805'
            }
        }
        self.active_api = 'deepseek'
        
    async def call_deepseek(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(
                api_key=self.apis['deepseek']['key'],
                base_url=self.apis['deepseek']['base_url']
            )
            response = client.chat.completions.create(
                model=self.apis['deepseek']['model'],
                messages=[{"role": "user", "content": prompt}]
            )
            return await response.choices[0].message.content
        except Exception as e:
            return await f"DeepSeek Error: {e}"
            
    async def call_openai(self, prompt: str) -> str:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=self.apis['openai']['key'])
            # Use chat completions API (GPT-5 not available yet)
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100
            )
            return await response.choices[0].message.content
        except Exception as e:
            return await f"OpenAI Error: {e}"
            
    async def call_gemini(self, prompt: str) -> str:
        try:
            from google import genai
            os.environ['GEMINI_API_KEY'] = self.apis['gemini']['key']
            client = genai.Client()
            response = client.models.generate_content(
                model=self.apis['gemini']['model'],
                contents=prompt
            )
            return await response.text
        except Exception as e:
            return await f"Gemini Error: {e}"
            
    async def call_mistral(self, prompt: str) -> str:
        try:
            from mistralai import Mistral
            client = Mistral(api_key=self.apis['mistral']['key'])
            response = client.chat.complete(
                model=self.apis['mistral']['model'],
                messages=[{"role": "user", "content": prompt}]
            )
            return await response.choices[0].message.content
        except Exception as e:
            return await f"Mistral Error: {e}"
            
    async def call_grok(self, prompt: str) -> str:
        try:
            from xai_sdk import Client
            from xai_sdk.chat import user
            client = Client(api_key=self.apis['grok']['key'])
            chat = client.chat.create(model=self.apis['grok']['model'])
            chat.append(user(prompt))
            response = chat.sample()
            return await response.content
        except Exception as e:
            return await f"Grok Error: {e}"
            
    async def call_anthropic(self, prompt: str) -> str:
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=self.apis['anthropic']['key'])
            message = client.messages.create(
                model=self.apis['anthropic']['model'],
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            return await message.content[0].text
        except Exception as e:
            return await f"Anthropic Error: {e}"
            
    async def query(self, prompt: str, api: str = None) -> str:
        """Main query method with fallback"""
        if api is None:
            api = self.active_api
            
        methods = {
            'deepseek': self.call_deepseek,
            'openai': self.call_openai,
            'gemini': self.call_gemini,
            'mistral': self.call_mistral,
            'grok': self.call_grok,
            'anthropic': self.call_anthropic
        }
        
        if api in methods:
            return await methods[api](prompt)
        else:
            return await "API not supported"
            
    async def query_all(self, prompt: str) -> Dict[str, str]:
        """Query all APIs and return await results"""
        results = {}
        for api_name in self.apis.keys():
            results[api_name] = self.query(prompt, api_name)
        return await results

# Global instance for PENIN modules
multi_api_llm = MultiAPILLM()
MULTI_API_LLM = multi_api_llm  # Alias for compatibility

async def get_llm_response(prompt: str, api: str = 'deepseek') -> str:
    """Simple function for PENIN modules to use"""
    return await multi_api_llm.query(prompt, api)

async def initialize_multi_api_llm():
    """Initialize function required by PENIN modules"""
    return await multi_api_llm

if __name__ == "__main__":
    # Test the multi-API system
    llm = MultiAPILLM()
    test_prompt = "Hello, respond with just 'API Working'"
    
    logger.info("🧪 Testing Multi-API LLM System")
    logger.info("=" * 40)
    
    for api_name in llm.apis.keys():
        logger.info(f"Testing {api_name}...")
        result = llm.query(test_prompt, api_name)
        if result:
            logger.info(f"✅ {api_name}: {result[:50]}...")
        else:
            logger.info(f"❌ {api_name}: No response")
        logger.info()
