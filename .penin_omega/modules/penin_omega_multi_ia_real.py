#!/usr/bin/env python3

import asyncio
import aiohttp
import json
import time
import logging
from api_keys_config import configure_api_keys

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s')

class MultiIAReal:
    def __init__(self):
        configure_api_keys()
        
        self.apis = {
            "openai": {
                "url": "https://api.openai.com/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                "model": "gpt-4"
            },
            "anthropic": {
                "url": "https://api.anthropic.com/v1/messages", 
                "headers": {"x-api-key": os.environ['ANTHROPIC_API_KEY'], "anthropic-version": "2023-06-01"},
                "model": "claude-3-sonnet-20240229"
            },
            "deepseek": {
                "url": "https://api.deepseek.com/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
                "model": "deepseek-reasoner"
            },
            "mistral": {
                "url": "https://api.mistral.ai/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"},
                "model": "mistral-large-latest"
            },
            "xai": {
                "url": "https://api.x.ai/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {os.environ['XAI_API_KEY']}"},
                "model": "grok-beta"
            },
            "google": {
                "url": f"https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key={os.environ['GOOGLE_API_KEY']}",
                "headers": {"Content-Type": "application/json"},
                "model": "gemini-pro"
            }
        }
    
    async def chamar_openai(self, session, prompt):
        """Chamada real OpenAI GPT-4"""
        try:
            payload = {
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150
            }
            
            async with session.post(
                self.apis["openai"]["url"],
                headers=self.apis["openai"]["headers"],
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "ia": "OpenAI GPT-4",
                        "resposta": data["choices"][0]["message"]["content"],
                        "tokens": data["usage"]["total_tokens"],
                        "status": "SUCCESS"
                    }
                else:
                    return {"ia": "OpenAI GPT-4", "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": "OpenAI GPT-4", "status": f"ERROR: {e}"}
    
    async def chamar_anthropic(self, session, prompt):
        """Chamada real Anthropic Claude"""
        try:
            payload = {
                "model": "claude-3-sonnet-20240229",
                "max_tokens": 150,
                "messages": [{"role": "user", "content": prompt}]
            }
            
            async with session.post(
                self.apis["anthropic"]["url"],
                headers=self.apis["anthropic"]["headers"],
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "ia": "Anthropic Claude-3",
                        "resposta": data["content"][0]["text"],
                        "tokens": data["usage"]["input_tokens"] + data["usage"]["output_tokens"],
                        "status": "SUCCESS"
                    }
                else:
                    return {"ia": "Anthropic Claude-3", "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": "Anthropic Claude-3", "status": f"ERROR: {e}"}
    
    async def chamar_deepseek(self, session, prompt):
        """Chamada real DeepSeek"""
        try:
            payload = {
                "model": "deepseek-reasoner",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150
            }
            
            async with session.post(
                self.apis["deepseek"]["url"],
                headers=self.apis["deepseek"]["headers"],
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "ia": "DeepSeek Reasoner",
                        "resposta": data["choices"][0]["message"]["content"],
                        "tokens": data["usage"]["total_tokens"],
                        "status": "SUCCESS"
                    }
                else:
                    return {"ia": "DeepSeek Reasoner", "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": "DeepSeek Reasoner", "status": f"ERROR: {e}"}
    
    async def chamar_mistral(self, session, prompt):
        """Chamada real Mistral"""
        try:
            payload = {
                "model": "mistral-large-latest",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150
            }
            
            async with session.post(
                self.apis["mistral"]["url"],
                headers=self.apis["mistral"]["headers"],
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "ia": "Mistral Large",
                        "resposta": data["choices"][0]["message"]["content"],
                        "tokens": data["usage"]["total_tokens"],
                        "status": "SUCCESS"
                    }
                else:
                    return {"ia": "Mistral Large", "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": "Mistral Large", "status": f"ERROR: {e}"}
    
    async def chamar_xai(self, session, prompt):
        """Chamada real xAI Grok"""
        try:
            payload = {
                "model": "grok-beta",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 150
            }
            
            async with session.post(
                self.apis["xai"]["url"],
                headers=self.apis["xai"]["headers"],
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "ia": "xAI Grok",
                        "resposta": data["choices"][0]["message"]["content"],
                        "tokens": data["usage"]["total_tokens"],
                        "status": "SUCCESS"
                    }
                else:
                    return {"ia": "xAI Grok", "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": "xAI Grok", "status": f"ERROR: {e}"}
    
    async def chamar_google(self, session, prompt):
        """Chamada real Google Gemini"""
        try:
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 150}
            }
            
            async with session.post(
                self.apis["google"]["url"],
                headers=self.apis["google"]["headers"],
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "ia": "Google Gemini Pro",
                        "resposta": data["candidates"][0]["content"]["parts"][0]["text"],
                        "tokens": len(prompt.split()) + len(data["candidates"][0]["content"]["parts"][0]["text"].split()),
                        "status": "SUCCESS"
                    }
                else:
                    return {"ia": "Google Gemini Pro", "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": "Google Gemini Pro", "status": f"ERROR: {e}"}
    
    async def consultar_todas_ias_real(self, prompt):
        """Consulta REAL e simultânea a todas as 6 IAs"""
        logging.info(f"🚀 Consultando 6 IAs REAIS simultaneamente...")
        logging.info(f"📝 Prompt: {prompt}")
        
        inicio = time.time()
        
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
        
        tempo_total = time.time() - inicio
        sucessos = sum(1 for r in resultados if isinstance(r, dict) and r.get('status') == 'SUCCESS')
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📊 RESPOSTAS REAIS DE {sucessos}/6 IAs:")
        logger.info(f"{'='*80}")
        
        for resultado in resultados:
            if isinstance(resultado, dict):
                if resultado.get('status') == 'SUCCESS':
                    logger.info(f"\n✅ {resultado['ia']} ({resultado['tokens']} tokens):")
                    logger.info(f"   {resultado['resposta'][:200]}...")
                else:
                    logger.info(f"\n❌ {resultado['ia']}: {resultado['status']}")
        
        logging.info(f"📊 RESULTADO: {sucessos}/6 IAs responderam em {tempo_total:.2f}s")
        return resultados

import os
async def main():
    sistema = MultiIAReal()
    
    prompt = "Explique em 2 frases o que é inteligência artificial"
    await sistema.consultar_todas_ias_real(prompt)

if __name__ == "__main__":
    asyncio.run(main())
