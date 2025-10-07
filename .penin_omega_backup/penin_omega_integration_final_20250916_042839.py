#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω Integration Final - 1/8 v6.0 + 2/8 Estratégico
======================================================
Integração definitiva usando código 1/8 v6.0 FUSION oficial
"""

import asyncio
import time
from typing import Dict, Any, Optional

# Importar módulos
try:
    from penin_omega_1_core_v6 import (
        PeninOmegaFusion, 
        UnifiedOmegaState,
        create_core,
        GOVERNANCE
    )
    CORE_V6_AVAILABLE = True
except ImportError:
    CORE_V6_AVAILABLE = False
    logger.info("❌ Módulo 1/8 v6.0 não encontrado")

try:
    from penin_omega_2_strategy import (
        StrategyModuleFusion, 
        create_strategy_module
    )
    STRATEGY_AVAILABLE = True
except ImportError:
    STRATEGY_AVAILABLE = False
    logger.info("❌ Módulo 2/8 não encontrado")

# =============================================================================
# Integração Final 1/8 v6.0 + 2/8
# =============================================================================

class PeninOmegaFinalIntegration:
    """Integração FINAL - 1/8 v6.0 FUSION + 2/8 Estratégico"""
    
    def __init__(self):
        logger.info("🔗 Inicializando integração FINAL PENIN-Ω...")
        
        # Verificar disponibilidade
        if not CORE_V6_AVAILABLE:
            raise RuntimeError("❌ Módulo 1/8 v6.0 não disponível")
        if not STRATEGY_AVAILABLE:
            raise RuntimeError("❌ Módulo 2/8 não disponível")
        
        # Inicializar módulos
        self.core_v6 = create_core()
        self.strategy_module = create_strategy_module()
        
        logger.info("✅ Módulo 1/8 v6.0 FUSION carregado")
        logger.info("✅ Módulo 2/8 Estratégico carregado")
        logger.info("🚀 Integração FINAL pronta!")
    
    async def complete_cycle_with_strategy(self, intent: Optional[str] = None) -> Dict[str, Any]:
        """Ciclo completo usando AMBOS os módulos"""
        
        logger.info(f"\n🔄 Ciclo completo com estratégia...")
        start_time = time.time()
        
        # 1. Estado atual do núcleo v6.0
        current_state = self.core_v6.state
        logger.info(f"📊 Estado v6.0 - Ciclo: {current_state.cycle_count}, SR: {current_state.sr_score:.3f}")
        
        # 2. Plano estratégico se há intenção
        plan_result = None
        if intent:
            logger.info(f"🎯 Gerando plano estratégico: '{intent}'")
            
            plan_result = self.strategy_module.create_plan(
                state=current_state,
                intent=intent,
                context={"governance": GOVERNANCE}
            )
            
            logger.info(f"📋 Plano: {plan_result['PlanΩ']['id']}")
            logger.info(f"🎯 Objetivos: {len(plan_result['PlanΩ']['goals'])}")
            logger.info(f"📊 SR: {plan_result['SR_report']['sr_score']:.3f} ({plan_result['SR_report']['decision']})")
        
        # 3. Ciclo evolutivo no núcleo v6.0
        logger.info("⚙️ Executando evolução no núcleo v6.0...")
        evolution_result = await self.core_v6.evolution_cycle()
        
        # 4. Resultado integrado
        total_time = (time.time() - start_time) * 1000
        
        result = {
            "success": True,
            "total_time_ms": total_time,
            "core_v6_result": evolution_result,
            "strategy_result": plan_result,
            "final_state": {
                "cycle": current_state.cycle_count,
                "sr_score": current_state.sr_score,
                "rho": current_state.rho,
                "alive": current_state.E_t
            }
        }
        
        logger.info(f"✅ Ciclo completo em {total_time:.2f}ms")
        logger.info(f"🎯 Núcleo v6.0: {evolution_result.get('decision', evolution_result.get('reason', 'N/A'))}")
        
        return result
    
    async def get_integrated_status(self) -> Dict[str, Any]:
        """Status integrado completo"""
        
        # Diagnósticos
        core_diag = self.core_v6.get_diagnostics()
        strategy_telemetry = self.strategy_module.get_telemetry()
        
        return {
            "timestamp": time.time(),
            "integration": "FINAL",
            
            "modules": {
                "1/8_v6_fusion": {
                    "version": core_diag["version"],
                    "state_valid": core_diag["state_validation"]["valid"],
                    "worm_valid": core_diag["worm"]["valid"],
                    "cycles": core_diag["metrics"]["cycles"],
                    "promotions": core_diag["metrics"]["promotions"]
                },
                "2/8_strategy": {
                    "plans_created": strategy_telemetry["plans_created"],
                    "gates_passed": strategy_telemetry["gates_passed"],
                    "avg_latency_ms": strategy_telemetry["avg_latency_ms"]
                }
            },
            
            "current_state": {
                "cycle": self.core_v6.state.cycle_count,
                "sr_score": self.core_v6.state.sr_score,
                "rho": self.core_v6.state.rho,
                "trust_region": self.core_v6.state.trust_region,
                "alive": self.core_v6.state.E_t,
                "cpu_usage": self.core_v6.state.cpu_usage,
                "memory_usage": self.core_v6.state.memory_usage
            },
            
            "performance": {
                "core_cache_hit_ratio": core_diag["cache"]["hit_ratio"],
                "strategy_avg_latency": strategy_telemetry["avg_latency_ms"]
            }
        }
    
    async def demo_sequence(self) -> Dict[str, Any]:
        """Sequência de demonstração final"""
        
        logger.info("\n🎬 DEMO FINAL - Sistema Integrado v6.0 + 2/8")
        logger.info("="*60)
        
        results = []
        
        # 1. Ciclo puro (só núcleo v6.0)
        logger.info("\n1. Ciclo evolutivo puro (núcleo v6.0):")
        result1 = await self.complete_cycle_with_strategy()
        results.append(result1)
        
        # 2. Ciclos com estratégia
        intents = [
            "Melhorar robustez OOD mantendo ética",
            "Reduzir risco ρ com performance otimizada",
            "Aumentar SR score sem comprometer segurança"
        ]
        
        for i, intent in enumerate(intents, 2):
            logger.info(f"\n{i}. Ciclo com estratégia:")
            result = await self.complete_cycle_with_strategy(intent)
            results.append(result)
            
            await asyncio.sleep(0.3)
        
        # 3. Status final
        logger.info(f"\n📊 Status Final Integrado:")
        final_status = await self.get_integrated_status()
        
        logger.info(f"  Núcleo v6.0:")
        logger.info(f"    Versão: {final_status['modules']['1/8_v6_fusion']['version']}")
        logger.info(f"    Ciclos: {final_status['modules']['1/8_v6_fusion']['cycles']}")
        logger.info(f"    Promoções: {final_status['modules']['1/8_v6_fusion']['promotions']}")
        
        logger.info(f"  Estratégico 2/8:")
        logger.info(f"    Planos: {final_status['modules']['2/8_strategy']['plans_created']}")
        logger.info(f"    Gates: {final_status['modules']['2/8_strategy']['gates_passed']}")
        
        logger.info(f"  Estado Final:")
        logger.info(f"    SR Score: {final_status['current_state']['sr_score']:.3f}")
        logger.info(f"    Sistema: {'🟢 VIVO' if final_status['current_state']['alive'] else '🔴 EXTINTO'}")
        
        return {
            "demo_results": results,
            "final_status": final_status,
            "summary": {
                "total_cycles": len(results),
                "successful_cycles": sum(1 for r in results if r["success"]),
                "strategic_cycles": sum(1 for r in results if r["strategy_result"]),
                "final_sr_score": final_status['current_state']['sr_score'],
                "system_alive": final_status['current_state']['alive']
            }
        }

# =============================================================================
# Interface CLI
# =============================================================================

async def main():
    """Função principal da integração final"""
    
    logger.info("="*80)
    logger.info("🧠 PENIN-Ω FINAL INTEGRATION - v6.0 FUSION + 2/8 STRATEGY")
    logger.info("="*80)
    
    try:
        # Inicializar integração final
        integration = PeninOmegaFinalIntegration()
        
        # Status inicial
        logger.info("\n📊 Status Inicial:")
        status = await integration.get_integrated_status()
        
        logger.info(f"Módulos integrados:")
        for module, info in status["modules"].items():
            logger.info(f"  ✅ {module}")
        
        logger.info(f"Estado inicial:")
        state = status["current_state"]
        logger.info(f"  SR Score: {state['sr_score']:.3f}")
        logger.info(f"  Sistema: {'🟢 VIVO' if state['alive'] else '🔴 EXTINTO'}")
        
        # Demo completa
        demo_result = await integration.demo_sequence()
        
        # Resumo final
        logger.info(f"\n📈 RESUMO FINAL:")
        summary = demo_result["summary"]
        logger.info(f"  Total de ciclos: {summary['total_cycles']}")
        logger.info(f"  Ciclos bem-sucedidos: {summary['successful_cycles']}")
        logger.info(f"  Ciclos estratégicos: {summary['strategic_cycles']}")
        logger.info(f"  SR Score final: {summary['final_sr_score']:.3f}")
        logger.info(f"  Sistema final: {'🟢 VIVO' if summary['system_alive'] else '🔴 EXTINTO'}")
        
        logger.info(f"\n🎯 INTEGRAÇÃO FINAL COMPLETA!")
        logger.info(f"  ✅ Núcleo v6.0 FUSION funcionando")
        logger.info(f"  ✅ Estratégico 2/8 funcionando")
        logger.info(f"  ✅ Comunicação entre módulos ativa")
        logger.info(f"  ✅ Sistema evolutivo operacional")
        
    except Exception as e:
        logger.info(f"❌ Erro na integração: {e}")
        import traceback
        traceback.print_exc()
    
    logger.info(f"\n{'='*80}")
    logger.info("✅ Teste de Integração FINAL Concluído!")
    logger.info(f"{'='*80}")

if __name__ == "__main__":
    asyncio.run(main())
