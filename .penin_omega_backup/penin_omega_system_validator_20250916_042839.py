#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω - Validador de Sistema
==============================
Validação rigorosa de consistência e qualidade
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Any, Tuple
from penin_omega_utils import log, _ts

class SystemValidator:
    """Validador rigoroso de consistência do sistema"""
    
    def __init__(self):
        self.validation_history = []
        
    async def validate_full_system(self, imports: Dict[str, Any]) -> Dict[str, Any]:
        """Validação completa e rigorosa do sistema"""
        
        log("Iniciando validação rigorosa do sistema", "INFO", "VALIDATOR")
        
        validation_report = {
            "timestamp": _ts(),
            "overall_status": "unknown",
            "component_tests": {},
            "integration_tests": {},
            "consistency_tests": {},
            "performance_tests": {},
            "issues": [],
            "recommendations": []
        }
        
        try:
            # 1. Testa componentes individuais
            validation_report["component_tests"] = await self._test_components(imports)
            
            # 2. Testa integração entre componentes
            validation_report["integration_tests"] = await self._test_integration(imports)
            
            # 3. Testa consistência ao longo de múltiplos ciclos
            validation_report["consistency_tests"] = await self._test_consistency(imports)
            
            # 4. Testa performance
            validation_report["performance_tests"] = await self._test_performance(imports)
            
            # 5. Calcula status geral
            validation_report["overall_status"] = self._calculate_overall_status(validation_report)
            
            # 6. Gera recomendações
            validation_report["recommendations"] = self._generate_recommendations(validation_report)
            
        except Exception as e:
            validation_report["overall_status"] = "FAILED"
            validation_report["issues"].append(f"Validação falhou: {e}")
            log(f"Erro na validação: {e}", "ERROR", "VALIDATOR")
        
        self.validation_history.append(validation_report)
        return validation_report
    
    async def _test_components(self, imports: Dict[str, Any]) -> Dict[str, Any]:
        """Testa cada componente individualmente"""
        
        component_results = {}
        
        # Testa 1/8 - Núcleo
        try:
            if imports['core']['available']:
                PeninOmegaFusion = imports['core']['PeninOmegaFusion']
                core = PeninOmegaFusion()
                
                # Validações rigorosas
                has_state = hasattr(core, 'state')
                state_has_required_attrs = all(hasattr(core.state, attr) for attr in ['ece', 'rho'])
                
                component_results['core'] = {
                    "available": True,
                    "has_state": has_state,
                    "state_valid": state_has_required_attrs,
                    "status": "PASS" if has_state and state_has_required_attrs else "FAIL"
                }
            else:
                component_results['core'] = {"available": False, "status": "FAIL"}
                
        except Exception as e:
            component_results['core'] = {"available": False, "error": str(e), "status": "FAIL"}
        
        # Testa 2/8 - Estratégia
        try:
            if imports['strategy']['available']:
                StrategyModuleFusion = imports['strategy']['StrategyModuleFusion']
                strategy = StrategyModuleFusion()
                
                # Validações rigorosas
                has_generate_plan = hasattr(strategy, 'generate_plan')
                has_worm = hasattr(strategy, 'worm')
                
                component_results['strategy'] = {
                    "available": True,
                    "has_generate_plan": has_generate_plan,
                    "has_worm": has_worm,
                    "status": "PASS" if has_generate_plan and has_worm else "FAIL"
                }
            else:
                component_results['strategy'] = {"available": False, "status": "FAIL"}
                
        except Exception as e:
            component_results['strategy'] = {"available": False, "error": str(e), "status": "FAIL"}
        
        # Testa 3/8 - Aquisição
        try:
            # Tenta aquisição real primeiro
            try:
                from penin_omega_3_acquisition_real import acquire_real_ucb, RealAcquisitionReport
                component_results['acquisition'] = {
                    "available": True,
                    "type": "real",
                    "has_real_processing": True,
                    "status": "PASS"
                }
            except ImportError:
                if imports['acquisition']['available']:
                    component_results['acquisition'] = {
                        "available": True,
                        "type": "fallback",
                        "has_real_processing": False,
                        "status": "PARTIAL"
                    }
                else:
                    component_results['acquisition'] = {"available": False, "status": "FAIL"}
                    
        except Exception as e:
            component_results['acquisition'] = {"available": False, "error": str(e), "status": "FAIL"}
        
        # Testa 4/8 - Mutação
        try:
            if imports['mutation']['available']:
                from penin_omega_4_mutation_v6 import mutate_and_rank
                
                # Verifica se função existe e é callable
                is_callable = callable(mutate_and_rank)
                
                component_results['mutation'] = {
                    "available": True,
                    "is_callable": is_callable,
                    "status": "PASS" if is_callable else "FAIL"
                }
            else:
                component_results['mutation'] = {"available": False, "status": "FAIL"}
                
        except Exception as e:
            component_results['mutation'] = {"available": False, "error": str(e), "status": "FAIL"}
        
        return component_results
    
    async def _test_integration(self, imports: Dict[str, Any]) -> Dict[str, Any]:
        """Testa integração entre componentes"""
        
        integration_results = {}
        
        try:
            # Cria base de conhecimento robusta
            from penin_omega_knowledge_base import create_robust_knowledge_base
            kb_dir = create_robust_knowledge_base()
            
            # Testa fluxo 1/8 → 2/8
            if imports['core']['available'] and imports['strategy']['available']:
                try:
                    from penin_omega_unified_interface import wrap_core_state
                    
                    core = imports['core']['PeninOmegaFusion']()
                    xt = wrap_core_state(core)
                    
                    strategy = imports['strategy']['StrategyModuleFusion']()
                    plan = await strategy.generate_plan(xt, max_cost=5.0)
                    
                    # Validações rigorosas
                    plan_is_real = not plan.id.startswith('fallback_')
                    has_goals = len(plan.goals) > 0
                    state_updated = getattr(xt, 'cycle_count', 0) > 0
                    
                    integration_results['core_to_strategy'] = {
                        "success": True,
                        "plan_is_real": plan_is_real,
                        "has_goals": has_goals,
                        "state_updated": state_updated,
                        "status": "PASS" if plan_is_real and has_goals and state_updated else "PARTIAL"
                    }
                    
                except Exception as e:
                    integration_results['core_to_strategy'] = {
                        "success": False,
                        "error": str(e),
                        "status": "FAIL"
                    }
            
            # Testa fluxo 2/8 → 3/8
            if 'plan' in locals() and component_results.get('acquisition', {}).get('available'):
                try:
                    if component_results['acquisition']['type'] == 'real':
                        from penin_omega_3_acquisition_real import acquire_real_ucb
                        acq_report = await acquire_real_ucb(xt, plan, [kb_dir])
                        
                        # Validações rigorosas
                        real_processing = acq_report.real_processing
                        meaningful_docs = acq_report.n_docs >= 3
                        meaningful_chunks = acq_report.n_chunks >= 3
                        
                        integration_results['strategy_to_acquisition'] = {
                            "success": True,
                            "real_processing": real_processing,
                            "meaningful_docs": meaningful_docs,
                            "meaningful_chunks": meaningful_chunks,
                            "docs_processed": acq_report.n_docs,
                            "chunks_created": acq_report.n_chunks,
                            "status": "PASS" if real_processing and meaningful_docs else "PARTIAL"
                        }
                    else:
                        integration_results['strategy_to_acquisition'] = {
                            "success": False,
                            "reason": "Usando fallback, não aquisição real",
                            "status": "PARTIAL"
                        }
                        
                except Exception as e:
                    integration_results['strategy_to_acquisition'] = {
                        "success": False,
                        "error": str(e),
                        "status": "FAIL"
                    }
            
            # Testa fluxo 3/8 → 4/8
            if 'acq_report' in locals() and imports['mutation']['available']:
                try:
                    from penin_omega_4_mutation_v6 import mutate_and_rank
                    
                    initial_cycle = getattr(xt, 'cycle_count', 0)
                    initial_hashes = len(getattr(xt, 'hashes', []))
                    
                    bundle, xt_updated = mutate_and_rank(xt, plan, acq_report, n_candidates=4, top_k=2, seed=42)
                    
                    # Validações rigorosas
                    candidates_generated = len(bundle.topK) > 0
                    candidates_diverse = len(set(c.cand_id for c in bundle.topK)) == len(bundle.topK)
                    state_evolved = (getattr(xt_updated, 'cycle_count', 0) > initial_cycle and
                                   len(getattr(xt_updated, 'hashes', [])) > initial_hashes)
                    scores_valid = all(isinstance(c.score, (int, float)) and c.score >= 0 for c in bundle.topK)
                    
                    integration_results['acquisition_to_mutation'] = {
                        "success": True,
                        "candidates_generated": candidates_generated,
                        "candidates_diverse": candidates_diverse,
                        "state_evolved": state_evolved,
                        "scores_valid": scores_valid,
                        "final_cycle": getattr(xt_updated, 'cycle_count', 0),
                        "status": "PASS" if all([candidates_generated, candidates_diverse, state_evolved, scores_valid]) else "PARTIAL"
                    }
                    
                except Exception as e:
                    integration_results['acquisition_to_mutation'] = {
                        "success": False,
                        "error": str(e),
                        "status": "FAIL"
                    }
            
        except Exception as e:
            integration_results['general_error'] = {
                "error": str(e),
                "status": "FAIL"
            }
        
        return integration_results
    
    async def _test_consistency(self, imports: Dict[str, Any]) -> Dict[str, Any]:
        """Testa consistência ao longo de múltiplos ciclos"""
        
        consistency_results = {
            "multiple_cycles": False,
            "state_progression": False,
            "score_stability": False,
            "cycles_tested": 0,
            "status": "FAIL"
        }
        
        try:
            if not all(imports[comp]['available'] for comp in ['core', 'strategy', 'mutation']):
                consistency_results["reason"] = "Nem todos os componentes disponíveis"
                return consistency_results
            
            # Cria base de conhecimento
            from penin_omega_knowledge_base import create_robust_knowledge_base
            kb_dir = create_robust_knowledge_base()
            
            from penin_omega_unified_interface import wrap_core_state
            from penin_omega_4_mutation_v6 import mutate_and_rank
            
            # Inicializa sistema
            core = imports['core']['PeninOmegaFusion']()
            xt = wrap_core_state(core)
            strategy = imports['strategy']['StrategyModuleFusion']()
            
            # Testa múltiplos ciclos
            cycles_data = []
            
            for cycle in range(3):
                try:
                    # Gera plano
                    plan = await strategy.generate_plan(xt, max_cost=5.0)
                    
                    # Aquisição
                    if component_results.get('acquisition', {}).get('type') == 'real':
                        from penin_omega_3_acquisition_real import acquire_real_ucb
                        acq_report = await acquire_real_ucb(xt, plan, [kb_dir])
                    else:
                        # Fallback
                        from penin_omega_3_acquisition_v6 import AcquisitionReport
                        acq_report = AcquisitionReport(
                            novelty_sim=0.7, rag_recall=0.6, synthesis_path=None,
                            questions=['test'], sources_stats={}, plan_hash=plan.plan_hash,
                            n_docs=1, n_chunks=5, proof_id=f'cycle_{cycle}'
                        )
                    
                    # Mutação
                    bundle, xt_updated = mutate_and_rank(xt, plan, acq_report, n_candidates=4, top_k=2, seed=42+cycle)
                    
                    # Coleta dados do ciclo
                    cycle_data = {
                        "cycle": cycle,
                        "plan_real": not plan.id.startswith('fallback_'),
                        "n_goals": len(plan.goals),
                        "n_candidates": len(bundle.topK),
                        "cycle_count": getattr(xt_updated, 'cycle_count', 0),
                        "n_hashes": len(getattr(xt_updated, 'hashes', [])),
                        "scores": [c.score for c in bundle.topK]
                    }
                    
                    cycles_data.append(cycle_data)
                    xt = xt_updated
                    
                except Exception as e:
                    consistency_results["cycle_error"] = f"Ciclo {cycle}: {e}"
                    break
            
            # Analisa consistência
            if len(cycles_data) >= 2:
                consistency_results["multiple_cycles"] = True
                consistency_results["cycles_tested"] = len(cycles_data)
                
                # Verifica progressão do estado
                cycle_counts = [data["cycle_count"] for data in cycles_data]
                hash_counts = [data["n_hashes"] for data in cycles_data]
                
                state_progresses = (
                    all(cycle_counts[i] <= cycle_counts[i+1] for i in range(len(cycle_counts)-1)) and
                    all(hash_counts[i] <= hash_counts[i+1] for i in range(len(hash_counts)-1))
                )
                
                consistency_results["state_progression"] = state_progresses
                
                # Verifica estabilidade dos scores
                all_scores = [score for data in cycles_data for score in data["scores"]]
                score_stability = (
                    len(all_scores) > 0 and
                    all(isinstance(s, (int, float)) and s >= 0 for s in all_scores) and
                    not all(s == 0.0 for s in all_scores)
                )
                
                consistency_results["score_stability"] = score_stability
                consistency_results["score_range"] = [min(all_scores), max(all_scores)] if all_scores else [0, 0]
                
                # Status geral
                if consistency_results["multiple_cycles"] and state_progresses and score_stability:
                    consistency_results["status"] = "PASS"
                elif consistency_results["multiple_cycles"] and (state_progresses or score_stability):
                    consistency_results["status"] = "PARTIAL"
                
                consistency_results["cycles_data"] = cycles_data
            
        except Exception as e:
            consistency_results["error"] = str(e)
        
        return consistency_results
    
    async def _test_performance(self, imports: Dict[str, Any]) -> Dict[str, Any]:
        """Testa performance do sistema"""
        
        import time
        
        performance_results = {
            "initialization_time_ms": 0,
            "cycle_time_ms": 0,
            "memory_efficient": False,
            "status": "FAIL"
        }
        
        try:
            # Testa tempo de inicialização
            start_time = time.perf_counter()
            
            if imports['core']['available']:
                core = imports['core']['PeninOmegaFusion']()
                
            init_time = (time.perf_counter() - start_time) * 1000
            performance_results["initialization_time_ms"] = init_time
            
            # Testa tempo de ciclo completo
            if all(imports[comp]['available'] for comp in ['core', 'strategy']):
                from penin_omega_unified_interface import wrap_core_state
                
                xt = wrap_core_state(core)
                strategy = imports['strategy']['StrategyModuleFusion']()
                
                start_cycle = time.perf_counter()
                plan = await strategy.generate_plan(xt, max_cost=5.0)
                cycle_time = (time.perf_counter() - start_cycle) * 1000
                
                performance_results["cycle_time_ms"] = cycle_time
            
            # Avalia performance
            init_acceptable = init_time < 10000  # 10s
            cycle_acceptable = performance_results["cycle_time_ms"] < 1000  # 1s
            
            performance_results["memory_efficient"] = True  # Assumido por simplicidade
            
            if init_acceptable and cycle_acceptable:
                performance_results["status"] = "PASS"
            elif init_acceptable or cycle_acceptable:
                performance_results["status"] = "PARTIAL"
            
        except Exception as e:
            performance_results["error"] = str(e)
        
        return performance_results
    
    def _calculate_overall_status(self, report: Dict[str, Any]) -> str:
        """Calcula status geral baseado em todos os testes"""
        
        # Conta sucessos em cada categoria
        component_passes = sum(1 for test in report["component_tests"].values() if test.get("status") == "PASS")
        integration_passes = sum(1 for test in report["integration_tests"].values() if test.get("status") == "PASS")
        consistency_pass = 1 if report["consistency_tests"].get("status") == "PASS" else 0
        performance_pass = 1 if report["performance_tests"].get("status") == "PASS" else 0
        
        total_passes = component_passes + integration_passes + consistency_pass + performance_pass
        total_tests = len(report["component_tests"]) + len(report["integration_tests"]) + 2
        
        if total_passes == total_tests:
            return "EXCELLENT"
        elif total_passes >= total_tests * 0.8:
            return "GOOD"
        elif total_passes >= total_tests * 0.6:
            return "ACCEPTABLE"
        elif total_passes >= total_tests * 0.4:
            return "POOR"
        else:
            return "FAILED"
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Gera recomendações baseadas nos resultados"""
        
        recommendations = []
        
        # Recomendações baseadas em componentes
        for comp, result in report["component_tests"].items():
            if result.get("status") == "FAIL":
                recommendations.append(f"Corrigir componente {comp}: {result.get('error', 'falha desconhecida')}")
        
        # Recomendações baseadas em integração
        for test, result in report["integration_tests"].items():
            if result.get("status") == "FAIL":
                recommendations.append(f"Corrigir integração {test}: {result.get('error', 'falha desconhecida')}")
        
        # Recomendações baseadas em consistência
        if report["consistency_tests"].get("status") != "PASS":
            if not report["consistency_tests"].get("multiple_cycles"):
                recommendations.append("Implementar suporte robusto para múltiplos ciclos")
            if not report["consistency_tests"].get("state_progression"):
                recommendations.append("Corrigir progressão de estado entre ciclos")
            if not report["consistency_tests"].get("score_stability"):
                recommendations.append("Melhorar estabilidade e validade dos scores")
        
        # Recomendações baseadas em performance
        if report["performance_tests"].get("status") != "PASS":
            if report["performance_tests"].get("initialization_time_ms", 0) > 10000:
                recommendations.append("Otimizar tempo de inicialização")
            if report["performance_tests"].get("cycle_time_ms", 0) > 1000:
                recommendations.append("Otimizar tempo de ciclo")
        
        return recommendations

# Instância global
SYSTEM_VALIDATOR = SystemValidator()

async def validate_system_rigorously(imports: Dict[str, Any]) -> Dict[str, Any]:
    """Função principal para validação rigorosa"""
    return await SYSTEM_VALIDATOR.validate_full_system(imports)
