#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω Integration Layer - Conecta módulos 1/8 e 2/8
====================================================
Interface de integração entre Núcleo (1/8) e Estratégico (2/8)
com conexão HTTP ao Falcon Mamba 7B
"""

import asyncio
import aiohttp
import json
import time
from typing import Dict, Any, Optional, Union
from dataclasses import asdict

# Importar módulos
try:
    from penin_omega_2_strategy import (
        StrategyModuleFusion, 
        OmegaState, 
        create_strategy_module,
        CORE_INTEGRATION
    )
    STRATEGY_AVAILABLE = True
except ImportError:
    STRATEGY_AVAILABLE = False
    logger.info("⚠️ Módulo 2/8 não encontrado")

# =============================================================================
# HTTP LLM Provider para Falcon Mamba
# =============================================================================

class FalconMambaProvider:
    """Provider HTTP para Falcon Mamba 7B na porta 8010"""
    
    def __init__(self, base_url: str = "http://localhost:8010"):
        self.base_url = base_url
        self.session = None
        
    async def _get_session(self):
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def generate(self, prompt: str, max_tokens: int = 150, temperature: float = 0.7) -> str:
        """Gera resposta via HTTP"""
        try:
            session = await self._get_session()
            
            payload = {
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stream": False
            }
            
            async with session.post(f"{self.base_url}/generate", json=payload) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return result.get("response", "").strip()
                else:
                    return f"HTTP Error {resp.status}"
                    
        except Exception as e:
            return f"Connection error: {str(e)}"
    
    async def close(self):
        if self.session:
            await self.session.close()

# =============================================================================
# Integração Principal 1/8 + 2/8
# =============================================================================

class PeninOmegaIntegration:
    """Integração completa entre módulos 1/8 e 2/8"""
    
    def __init__(self):
        logger.info("🔗 Inicializando integração PENIN-Ω 1/8 + 2/8...")
        
        # Inicializar módulo 2/8
        if STRATEGY_AVAILABLE:
            self.strategy_module = create_strategy_module()
            logger.info("✅ Módulo 2/8 (Estratégico) carregado")
        else:
            self.strategy_module = None
            logger.info("❌ Módulo 2/8 não disponível")
        
        # Provider LLM
        self.llm_provider = FalconMambaProvider()
        
        # Estado atual simulado (normalmente viria do 1/8)
        self.current_state = OmegaState(
            E_ok=0.95, M=0.85, C=0.80, A=0.75,
            ece=0.008, rho_bias=1.02, fairness=0.96,
            consent=True, eco_ok=True,
            rho=0.75, uncertainty=0.25, volatility=0.15,
            delta_linf=0.018, mdl_gain=0.035, ppl_ood=88.0,
            efficiency=0.75, caos_post=1.35, caos_stable=True,
            self_improvement=0.72, exploration=0.65, adaptation=0.80,
            sr_score=0.85, trust_region_radius=0.12, cycle_count=1
        )
        
        logger.info("🧠 Estado inicial configurado")
        logger.info("🚀 Integração pronta!")
    
    async def process_intent(self, intent: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Processa intenção através do pipeline 1/8 → 2/8"""
        
        if not self.strategy_module:
            return {"error": "Módulo 2/8 não disponível"}
        
        logger.info(f"\n📝 Processando intenção: '{intent}'")
        
        # 1. Atualizar estado (simulado - normalmente viria do 1/8)
        self.current_state.cycle_count += 1
        self.current_state.timestamp = time.time()
        
        # 2. Criar plano estratégico via 2/8
        logger.info("🎯 Gerando plano estratégico...")
        start_time = time.time()
        
        plan_result = self.strategy_module.create_plan(
            state=self.current_state,
            intent=intent,
            context=context or {}
        )
        
        generation_time = (time.time() - start_time) * 1000
        logger.info(f"⚡ Plano gerado em {generation_time:.2f}ms")
        
        # 3. Enriquecer com LLM se necessário
        if "explain" in intent.lower() or "rationale" in intent.lower():
            logger.info("🤖 Enriquecendo com Falcon Mamba...")
            llm_prompt = f"Explique brevemente este plano estratégico: {plan_result['PlanΩ']['rationale']}"
            llm_response = await self.llm_provider.generate(llm_prompt, max_tokens=100)
            plan_result["llm_explanation"] = llm_response
        
        # 4. Preparar resposta integrada
        response = {
            "success": True,
            "intent": intent,
            "processing_time_ms": generation_time,
            "state_cycle": self.current_state.cycle_count,
            "plan": plan_result["PlanΩ"],
            "sr_report": plan_result["SR_report"],
            "u_signal": plan_result["U_signal"],
            "proof": plan_result["proof"],
        }
        
        if "llm_explanation" in plan_result:
            response["llm_explanation"] = plan_result["llm_explanation"]
        
        return response
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Status completo do sistema integrado"""
        
        status = {
            "timestamp": time.time(),
            "integration_active": True,
            "modules": {
                "1/8_core": "simulated",  # Seria real com 1/8
                "2/8_strategy": STRATEGY_AVAILABLE,
            },
            "llm_provider": {
                "type": "falcon_mamba_7b",
                "endpoint": self.llm_provider.base_url,
                "status": "unknown"  # Testaremos
            },
            "current_state": {
                "cycle": self.current_state.cycle_count,
                "sr_score": self.current_state.sr_score,
                "rho": self.current_state.rho,
                "trust_region": self.current_state.trust_region_radius,
            }
        }
        
        # Testar conexão com Falcon Mamba
        try:
            test_response = await self.llm_provider.generate("Test", max_tokens=10)
            if "error" not in test_response.lower():
                status["llm_provider"]["status"] = "connected"
            else:
                status["llm_provider"]["status"] = "error"
                status["llm_provider"]["error"] = test_response
        except Exception as e:
            status["llm_provider"]["status"] = "disconnected"
            status["llm_provider"]["error"] = str(e)
        
        # Telemetria do 2/8
        if self.strategy_module:
            status["strategy_telemetry"] = self.strategy_module.get_telemetry()
        
        return status
    
    async def simulate_evolution_cycle(self) -> Dict[str, Any]:
        """Simula um ciclo completo de evolução"""
        
        logger.info("\n🔄 Simulando ciclo de evolução...")
        
        # 1. Intenção automática baseada no estado
        if self.current_state.rho > 0.8:
            intent = "Reduzir risco ρ para níveis seguros mantendo performance"
        elif self.current_state.fairness < 0.97:
            intent = "Melhorar fairness sem comprometer eficiência"
        elif self.current_state.ppl_ood > 90:
            intent = "Melhorar robustez OOD em 10% com foco em calibração"
        else:
            intent = "Otimizar performance geral mantendo ética e segurança"
        
        # 2. Processar intenção
        result = await self.process_intent(intent)
        
        # 3. Simular execução (atualizar estado)
        if result.get("success"):
            plan = result["plan"]
            
            # Aplicar mudanças simuladas baseadas nos objetivos
            for goal in plan.get("goals", []):
                metric = goal.get("metric")
                target = goal.get("target", 0)
                
                if metric == "rho":
                    self.current_state.rho = min(0.95, target * 1.1)  # Progresso parcial
                elif metric == "fairness":
                    self.current_state.fairness = min(1.0, target * 0.98)
                elif metric == "ppl_ood":
                    self.current_state.ppl_ood = max(80, target * 1.05)
                elif metric == "sr_score":
                    self.current_state.sr_score = min(1.0, target * 0.95)
            
            # Atualizar métricas derivadas
            self.current_state.delta_linf = min(0.05, self.current_state.delta_linf * 1.1)
            self.current_state.uncertainty = max(0.1, self.current_state.uncertainty * 0.95)
            
            logger.info(f"✅ Ciclo completo - SR: {self.current_state.sr_score:.3f}, ρ: {self.current_state.rho:.3f}")
        
        return result
    
    async def close(self):
        """Limpa recursos"""
        await self.llm_provider.close()

# =============================================================================
# Interface CLI para Testes
# =============================================================================

async def main():
    """Função principal de teste da integração"""
    
    logger.info("="*80)
    logger.info("🧠 PENIN-Ω Integration Test - 1/8 + 2/8 + Falcon Mamba")
    logger.info("="*80)
    
    # Inicializar integração
    integration = PeninOmegaIntegration()
    
    try:
        # 1. Status do sistema
        logger.info("\n📊 Status do Sistema:")
        logger.info("-"*40)
        status = await integration.get_system_status()
        
        logger.info(f"Módulos ativos:")
        for module, active in status["modules"].items():
            status_icon = "✅" if active else "❌"
            logger.info(f"  {status_icon} {module}: {active}")
        
        logger.info(f"\nFalcon Mamba:")
        llm_status = status["llm_provider"]["status"]
        status_icon = "✅" if llm_status == "connected" else "⚠️" if llm_status == "error" else "❌"
        logger.info(f"  {status_icon} Status: {llm_status}")
        if "error" in status["llm_provider"]:
            logger.info(f"  Error: {status['llm_provider']['error']}")
        
        logger.info(f"\nEstado atual:")
        state = status["current_state"]
        logger.info(f"  Ciclo: {state['cycle']}")
        logger.info(f"  SR Score: {state['sr_score']:.3f}")
        logger.info(f"  Risco ρ: {state['rho']:.3f}")
        logger.info(f"  Trust Region: {state['trust_region']:.3f}")
        
        # 2. Teste de intenções
        logger.info(f"\n🎯 Testando Processamento de Intenções:")
        logger.info("-"*40)
        
        test_intents = [
            "Melhorar robustez OOD em 5% mantendo ρ<0.9",
            "Reduzir bias algorítmico com foco em fairness",
            "Otimizar eficiência sem comprometer ética",
        ]
        
        for i, intent in enumerate(test_intents, 1):
            logger.info(f"\n{i}. Intenção: '{intent}'")
            
            result = await integration.process_intent(intent)
            
            if result.get("success"):
                plan = result["plan"]
                sr_report = result["sr_report"]
                
                logger.info(f"   ✅ Plano gerado: {plan['id']}")
                logger.info(f"   📊 SR Score: {sr_report['sr_score']:.3f} ({sr_report['decision']})")
                logger.info(f"   🎯 Objetivos: {len(plan['goals'])}")
                logger.info(f"   ⚡ Tempo: {result['processing_time_ms']:.2f}ms")
                logger.info(f"   📡 U-Signal: {result['u_signal']:.3f}")
                
                if "llm_explanation" in result:
                    logger.info(f"   🤖 LLM: {result['llm_explanation'][:100]}...")
            else:
                logger.info(f"   ❌ Erro: {result.get('error', 'Unknown')}")
        
        # 3. Simulação de ciclos evolutivos
        logger.info(f"\n🔄 Simulando Ciclos Evolutivos:")
        logger.info("-"*40)
        
        for cycle in range(3):
            logger.info(f"\nCiclo {cycle + 1}:")
            result = await integration.simulate_evolution_cycle()
            
            if result.get("success"):
                logger.info(f"   Intent: {result['intent']}")
                logger.info(f"   Objetivos: {len(result['plan']['goals'])}")
                logger.info(f"   SR: {result['sr_report']['sr_score']:.3f}")
            
            await asyncio.sleep(0.5)  # Pausa entre ciclos
        
        # 4. Status final
        logger.info(f"\n📈 Status Final:")
        logger.info("-"*40)
        final_status = await integration.get_system_status()
        final_state = final_status["current_state"]
        
        logger.info(f"Ciclos executados: {final_state['cycle']}")
        logger.info(f"SR Score final: {final_state['sr_score']:.3f}")
        logger.info(f"Risco ρ final: {final_state['rho']:.3f}")
        
        if STRATEGY_AVAILABLE:
            telemetry = final_status["strategy_telemetry"]
            logger.info(f"Planos criados: {telemetry['plans_created']}")
            logger.info(f"Gates passados: {telemetry['gates_passed']}")
            logger.info(f"Latência média: {telemetry['avg_latency_ms']:.2f}ms")
        
    except Exception as e:
        logger.info(f"❌ Erro na integração: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await integration.close()
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ Teste de Integração Completo!")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(main())
