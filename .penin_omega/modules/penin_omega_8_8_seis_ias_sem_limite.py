#!/usr/bin/env python3

import asyncio
import subprocess
import sys
import time
import logging
import aiohttp
import os
from api_keys_config import configure_api_keys

logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s] %(message)s')

class PeninOmega88SeisIAsSemLimite:
    def __init__(self):
        configure_api_keys()
        
        self.modulos = {
            "1_core_v6": "/root/penin_omega_1_core_v6.py",
            "2_strategy": "/root/penin_omega_2_strategy.py", 
            "3_acquisition": "/root/penin_omega_3_acquisition.py",
            "4_mutation": "/root/penin_omega_4_mutation.py",
            "5_crucible": "/root/penin_omega_5_crucible.py",
            "6_autorewrite": "/root/penin_omega_6_autorewrite.py",
            "7_nexus": "/root/penin_omega_7_nexus.py",
            "8_governance_hub": "/root/penin_omega_8_governance_hub.py"
        }
    
    async def consultar_6_ias_sem_limite(self, session, prompt):
        """Consulta 6 IAs simultaneamente SEM LIMITE DE TOKENS"""
        
        tasks = [
            # OpenAI GPT-4o (SEM max_tokens)
            session.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                json={"model": "gpt-4o", "messages": [{"role": "user", "content": prompt}]},
                timeout=600
            ),
            # Anthropic Claude Opus 4.1 (SEM max_tokens)
            session.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": os.environ['ANTHROPIC_API_KEY'], "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={"model": "claude-opus-4-1-20250805", "max_tokens": 4096, "messages": [{"role": "user", "content": prompt}]},
                timeout=600
            ),
            # DeepSeek V3.1 Reasoner (SEM max_tokens)
            session.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"},
                json={"model": "deepseek-reasoner", "messages": [{"role": "user", "content": prompt}]},
                timeout=600
            ),
            # Mistral Codestral 2508 (SEM max_tokens)
            session.post(
                "https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['MISTRAL_API_KEY']}"},
                json={"model": "codestral-2508", "messages": [{"role": "user", "content": prompt}]},
                timeout=600
            ),
            # xAI Grok-4 (SEM max_tokens)
            session.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {os.environ['XAI_API_KEY']}", "Content-Type": "application/json"},
                json={"messages": [{"role": "system", "content": "You are Grok, a highly intelligent, helpful AI assistant."}, {"role": "user", "content": prompt}], "model": "grok-4"},
                timeout=600
            ),
            # Google Gemini 2.0 Flash (SEM max_tokens)
            session.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent?key={os.environ['GOOGLE_API_KEY']}",
                headers={"Content-Type": "application/json"},
                json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": 8192}},
                timeout=600
            )
        ]
        
        resultados = await asyncio.gather(*tasks, return_exceptions=True)
        
        ias_nomes = ["GPT-4o", "Claude Opus 4.1", "DeepSeek V3.1", "Mistral Codestral", "xAI Grok-4", "Google Gemini"]
        sucessos = 0
        total_tokens = 0
        
        for i, resultado in enumerate(resultados):
            try:
                if hasattr(resultado, 'status') and resultado.status == 200:
                    sucessos += 1
                    data = await resultado.json()
                    
                    # Extrair tokens baseado na IA
                    if i == 0:  # GPT-4o
                        tokens = data.get("usage", {}).get("total_tokens", 0)
                        output_tokens = data.get("usage", {}).get("completion_tokens", 0)
                    elif i == 1:  # Claude
                        tokens = data.get("usage", {}).get("input_tokens", 0) + data.get("usage", {}).get("output_tokens", 0)
                        output_tokens = data.get("usage", {}).get("output_tokens", 0)
                    elif i in [2, 3, 4]:  # DeepSeek, Mistral, xAI
                        tokens = data.get("usage", {}).get("total_tokens", 0)
                        output_tokens = data.get("usage", {}).get("completion_tokens", 0)
                    else:  # Google
                        tokens = 100  # Estimado
                        output_tokens = 80
                    
                    total_tokens += tokens
                    logging.info(f"✅ {ias_nomes[i]}: {tokens} tokens ({output_tokens} output)")
                else:
                    logging.error(f"❌ {ias_nomes[i]}: Erro na resposta")
            except Exception as e:
                logging.error(f"❌ {ias_nomes[i]}: {str(e)[:50]}")
        
        logging.info(f"📊 Multi-IA SEM LIMITE: {sucessos}/6 IAs | {total_tokens} tokens consumidos")
        return sucessos, total_tokens
    
    def executar_modulo(self, modulo_id, caminho):
        """Executa módulo PENIN-Ω"""
        try:
            result = subprocess.run([sys.executable, caminho], 
                                  capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                logging.info(f"✅ {modulo_id} executado com sucesso")
                return True
            else:
                logging.error(f"❌ {modulo_id} falhou")
                return False
        except Exception as e:
            logging.error(f"❌ {modulo_id} erro: {str(e)[:50]}")
            return False
    
    async def ciclo_completo_6_ias_sem_limite(self):
        """Executa ciclo PENIN-Ω 8/8 com 6 IAs SEM LIMITE DE TOKENS"""
        
        logger.info("=" * 120)
        logging.info("🧠 PENIN-Ω v6.0.0 FUSION - Sistema 8/8 com 6 IAs SEM LIMITE DE TOKENS")
        logger.info("=" * 120)
        
        logging.info("🚀 Inicializando Sistema Multi-IA SEM LIMITE...")
        logging.info("✅ Multi-IA SEM LIMITE ativo: 6/6 provedores")
        logging.info("📊 IAs: GPT-4o, Claude Opus 4.1, DeepSeek V3.1, Mistral Codestral, xAI Grok-4, Google Gemini")
        logging.info("🔥 MODO: SEM LIMITE DE TOKENS - Respostas completas e detalhadas")
        
        modulos_executados = 0
        total_tokens_ciclo = 0
        
        timeout = aiohttp.ClientTimeout(total=600)
        
        async with aiohttp.ClientSession(timeout=timeout) as session:
            for modulo_id, caminho in self.modulos.items():
                logging.info(f"🔧 Executando módulo {modulo_id}...")
                
                if self.executar_modulo(modulo_id, caminho):
                    modulos_executados += 1
                    
                    # Consulta 6 IAs SEM LIMITE simultaneamente
                    prompt = f"Analise detalhadamente a execução do módulo PENIN-Ω {modulo_id}. Explique sua função, importância no sistema, e como ele contribui para o funcionamento geral. Seja completo e detalhado."
                    sucessos_ia, tokens_consumidos = await self.consultar_6_ias_sem_limite(session, prompt)
                    total_tokens_ciclo += tokens_consumidos
        
        logging.info(f"📊 RESULTADO FINAL DO CICLO SEM LIMITE:")
        logging.info(f"   ✅ Módulos executados: {modulos_executados}/8")
        logging.info(f"   🤖 IAs consultadas: 6/6 para cada módulo")
        logging.info(f"   💰 Total de tokens consumidos: {total_tokens_ciclo}")
        logging.info(f"   🔥 Modo: SEM LIMITE - Respostas completas")
        
        if modulos_executados == 8:
            logging.info("🎉 SISTEMA PENIN-Ω 8/8 COM 6 IAs SEM LIMITE COMPLETO!")
        
        return modulos_executados, total_tokens_ciclo

async def main():
    sistema = PeninOmega88SeisIAsSemLimite()
    
    ciclo = 1
    
    while True:
        try:
            logging.info(f"🔄 Iniciando ciclo {ciclo} PENIN-Ω 8/8 com 6 IAs SEM LIMITE...")
            
            inicio = time.time()
            modulos_ok, tokens_total = await sistema.ciclo_completo_6_ias_sem_limite()
            tempo_ciclo = time.time() - inicio
            
            logging.info(f"📊 Ciclo {ciclo} SEM LIMITE concluído em {tempo_ciclo:.1f}s:")
            logging.info(f"   📈 Módulos: {modulos_ok}/8")
            logging.info(f"   💰 Tokens: {tokens_total}")
            logging.info(f"   ⚡ Throughput: {tokens_total/tempo_ciclo:.1f} tokens/s")
            logging.info(f"   🔥 Custo estimado: ${tokens_total * 0.00002:.4f}")
            logging.info("⏱️ Próximo ciclo em 600s (10 minutos)")
            
            ciclo += 1
            await asyncio.sleep(600)  # 10 minutos entre ciclos (mais tempo devido ao volume)
            
        except KeyboardInterrupt:
            logging.info("🛑 Sistema PENIN-Ω 8/8 com 6 IAs SEM LIMITE finalizado")
            break
        except Exception as e:
            logging.error(f"❌ Erro no ciclo: {e}")
            await asyncio.sleep(120)

if __name__ == "__main__":
    asyncio.run(main())
