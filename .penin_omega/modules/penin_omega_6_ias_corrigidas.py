#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import time
import logging
import os
from api_keys_config import configure_api_keys

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

class SixIAsCorrigidas:
    def __init__(self):
        configure_api_keys()
    
    async def chamar_openai(self, session, prompt):
        try:
            payload = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100
            }
            async with session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"ia": "OpenAI GPT-4", "resposta": data["choices"][0]["message"]["content"], "tokens": data["usage"]["total_tokens"], "status": "SUCCESS"}
                return {"ia": "OpenAI GPT-4", "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": "OpenAI GPT-4", "status": f"ERROR: {e}"}
    
    async def chamar_anthropic(self, session, prompt):
        try:
            payload = {
                "model": "claude-3-5-sonnet-20241022",
                "max_tokens": 100,
                "messages": [{"role": "user", "content": prompt}]
            }
            async with session.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": os.environ['ANTHROPIC_API_KEY'],
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"ia": "Anthropic Claude-3.5", "resposta": data["content"][0]["text"], "tokens": data["usage"]["input_tokens"] + data["usage"]["output_tokens"], "status": "SUCCESS"}
                return {"ia": "Anthropic Claude-3.5", "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": "Anthropic Claude-3.5", "status": f"ERROR: {e}"}
    
    async def chamar_deepseek(self, session, prompt):
        try:
            payload = {
                "model": "deepseek-reasoner",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100
            }
            async with session.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"ia": "DeepSeek Reasoner", "resposta": data["choices"][0]["message"]["content"], "tokens": data["usage"]["total_tokens"], "status": "SUCCESS"}
                return {"ia": "DeepSeek Reasoner", "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": "DeepSeek Reasoner", "status": f"ERROR: {e}"}
    
    async def chamar_mistral(self, session, prompt):
        try:
            payload = {
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100
            }
            async with session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"},
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"ia": "Mistral Large", "resposta": data["choices"][0]["message"]["content"], "tokens": data["usage"]["total_tokens"], "status": "SUCCESS"}
                return {"ia": "Mistral Large", "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": "Mistral Large", "status": f"ERROR: {e}"}
    
    async def chamar_xai(self, session, prompt):
        try:
            payload = {
                "messages": [{"role": "user", "content": prompt}],
                "model": "grok-beta",
                "stream": False,
                "temperature": 0
            }
            async with session.post(
                "https://api.x.ai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {os.environ['XAI_API_KEY']}",
                    "Content-Type": "application/json"
                },
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"ia": "xAI Grok", "resposta": data["choices"][0]["message"]["content"], "tokens": data["usage"]["total_tokens"], "status": "SUCCESS"}
                return {"ia": "xAI Grok", "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": "xAI Grok", "status": f"ERROR: {e}"}
    
    async def chamar_google(self, session, prompt):
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 100}
            }
            async with session.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={os.environ['GOOGLE_API_KEY']}",
                headers={"Content-Type": "application/json"},
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {"ia": "Google Gemini", "resposta": data["candidates"][0]["content"]["parts"][0]["text"], "tokens": 50, "status": "SUCCESS"}
                return {"ia": "Google Gemini", "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": "Google Gemini", "status": f"ERROR: {e}"}
    
    async def testar_todas_6_ias(self, prompt):
        logging.info(f"🚀 Testando 6 IAs corrigidas...")
        
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
            tasks = [
                self.chamar_openai(session, prompt),
                self.chamar_anthropic(session, prompt),
                self.chamar_deepseek(session, prompt),
                self.chamar_mistral(session, prompt),
                self.chamar_xai(session, prompt),
                self.chamar_google(session, prompt)
            ]
            
            resultados = await asyncio.gather(*tasks, return_exceptions=True)
        
        sucessos = sum(1 for r in resultados if isinstance(r, dict) and r.get('status') == 'SUCCESS')
        
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 TESTE 6 IAs CORRIGIDAS: {sucessos}/6")
        logger.info(f"{'='*60}")
        
        for resultado in resultados:
            if isinstance(resultado, dict):
                if resultado.get('status') == 'SUCCESS':
                    logger.info(f"✅ {resultado['ia']}: {resultado['tokens']} tokens")
                    logger.info(f"   {resultado['resposta'][:100]}...")
                else:
                    logger.info(f"❌ {resultado['ia']}: {resultado['status']}")
        
        return sucessos

async def main():
    sistema = SixIAsCorrigidas()
    sucessos = await sistema.testar_todas_6_ias("Explique IA em uma frase")
    logger.info(f"\n🎯 RESULTADO: {sucessos}/6 IAs funcionando")

if __name__ == "__main__":
    asyncio.run(main())
