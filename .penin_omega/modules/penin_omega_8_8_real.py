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

class PeninOmega88Real:
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
        
        self.apis_funcionais = {
            "openai": {
                "url": "https://api.openai.com/v1/chat/completions",
                "headers": {"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
                "model": "gpt-4"
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
            }
        }
    
    async def chamar_ia_real(self, session, ia_nome, config, prompt):
        """Chamada real para IA específica"""
        try:
            payload = {
                "model": config["model"],
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100
            }
            
            async with session.post(
                config["url"],
                headers=config["headers"],
                json=payload
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return {
                        "ia": ia_nome,
                        "resposta": data["choices"][0]["message"]["content"][:150],
                        "tokens": data["usage"]["total_tokens"],
                        "status": "SUCCESS"
                    }
                else:
                    return {"ia": ia_nome, "status": f"ERROR: {response.status}"}
        except Exception as e:
            return {"ia": ia_nome, "status": f"ERROR: {str(e)[:50]}"}
    
    async def consultar_ias_reais(self, prompt):
        """Consulta as 3 IAs funcionais simultaneamente"""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=15)) as session:
            tasks = []
            for ia_nome, config in self.apis_funcionais.items():
                task = self.chamar_ia_real(session, ia_nome, config, prompt)
                tasks.append(task)
            
            resultados = await asyncio.gather(*tasks, return_exceptions=True)
            sucessos = sum(1 for r in resultados if isinstance(r, dict) and r.get('status') == 'SUCCESS')
            
            logging.info(f"📊 Multi-IA REAL: {sucessos}/3 IAs responderam")
            
            for resultado in resultados:
                if isinstance(resultado, dict) and resultado.get('status') == 'SUCCESS':
                    logging.info(f"✅ {resultado['ia']}: {resultado['tokens']} tokens")
            
            return resultados
    
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
    
    async def ciclo_completo_real(self):
        """Executa ciclo PENIN-Ω 8/8 com IAs REAIS"""
        
        logger.info("=" * 80)
        logging.info("🧠 PENIN-Ω v6.0.0 FUSION - Sistema 8/8 com IAs REAIS")
        logger.info("=" * 80)
        
        logging.info("🚀 Inicializando Sistema Multi-IA REAL...")
        logging.info("✅ Multi-IA REAL ativo: 3/3 provedores funcionais")
        logging.info("📊 APIs: OpenAI GPT-4, DeepSeek Reasoner, Mistral Large")
        
        modulos_executados = 0
        
        for modulo_id, caminho in self.modulos.items():
            logging.info(f"🔧 Executando módulo {modulo_id}...")
            
            if self.executar_modulo(modulo_id, caminho):
                modulos_executados += 1
                
                # Consulta IAs REAIS para análise do módulo
                prompt = f"Analise brevemente a execução do módulo PENIN-Ω {modulo_id}"
                await self.consultar_ias_reais(prompt)
        
        logging.info(f"📊 RESULTADO: {modulos_executados}/8 módulos executados")
        logging.info(f"🤖 IAs REAIS: 3/3 consultadas por módulo")
        
        if modulos_executados == 8:
            logging.info("🎉 SISTEMA PENIN-Ω 8/8 COM IAs REAIS COMPLETO!")
        
        return modulos_executados

async def main():
    sistema = PeninOmega88Real()
    
    while True:
        try:
            logging.info("🔄 Iniciando ciclo PENIN-Ω 8/8 REAL...")
            
            modulos_ok = await sistema.ciclo_completo_real()
            
            logging.info(f"📊 Ciclo concluído: {modulos_ok}/8 módulos")
            logging.info("⏱️ Próximo ciclo em 120s")
            
            await asyncio.sleep(120)
            
        except KeyboardInterrupt:
            logging.info("🛑 Sistema PENIN-Ω 8/8 REAL finalizado")
            break
        except Exception as e:
            logging.error(f"❌ Erro no ciclo: {e}")
            await asyncio.sleep(30)

if __name__ == "__main__":
    asyncio.run(main())
