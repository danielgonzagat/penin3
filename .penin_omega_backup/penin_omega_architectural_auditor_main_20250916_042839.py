#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Auditor Arquitetural - Executor Principal
==================================================
Executor principal que integra todos os componentes de auditoria.
"""

import asyncio
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

# Importa todos os auditores
from penin_omega_architectural_auditor import ArchitecturalAuditor
from penin_omega_architectural_auditor_modules import ModuleAuditor
from penin_omega_architectural_auditor_advanced import AdvancedAuditor

# =============================================================================
# CONFIGURAÇÃO DE LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/root/.penin_omega/audit/architectural_audit.log')
    ]
)

# =============================================================================
# EXECUTOR PRINCIPAL
# =============================================================================

class ArchitecturalAuditExecutor:
    """Executor principal da auditoria arquitetural completa."""
    
    def __init__(self):
        self.logger = logging.getLogger("ArchitecturalAuditExecutor")
        self.architectural_auditor = ArchitecturalAuditor()
        self.module_auditor = ModuleAuditor()
        self.advanced_auditor = AdvancedAuditor()
        
        # Cria diretório de auditoria
        self.audit_path = Path("/root/.penin_omega/audit")
        self.audit_path.mkdir(parents=True, exist_ok=True)
    
    async def execute_complete_audit(self) -> Dict[str, Any]:
        """Executa auditoria arquitetural completa."""
        self.logger.info("🚀 INICIANDO AUDITORIA ARQUITETURAL COMPLETA PENIN-Ω")
        self.logger.info("=" * 80)
        
        start_time = time.time()
        
        try:
            # 1. Auditoria Arquitetural Principal
            self.logger.info("📋 Fase 1: Auditoria Arquitetural Principal")
            architectural_report = await self.architectural_auditor.audit_complete_architecture()
            
            # 2. Auditoria de Módulos Individuais
            self.logger.info("📋 Fase 2: Auditoria de Módulos Individuais")
            module_results = await self.module_auditor.audit_individual_modules()
            
            # 3. Auditorias Avançadas
            self.logger.info("📋 Fase 3: Auditorias Avançadas")
            advanced_findings = []
            
            # Budget & Circuit Breaker
            budget_findings = await self.advanced_auditor.audit_budget_circuit_breaker()
            advanced_findings.extend(budget_findings)
            
            # Segurança e Robustez
            security_findings = await self.advanced_auditor.audit_security_robustness()
            advanced_findings.extend(security_findings)
            
            # Observabilidade
            observability_findings = await self.advanced_auditor.audit_observability()
            advanced_findings.extend(observability_findings)
            
            # Performance
            performance_findings = await self.advanced_auditor.audit_performance()
            advanced_findings.extend(performance_findings)
            
            # Validação Completa
            validation_findings = await self.advanced_auditor.audit_complete_validation()
            advanced_findings.extend(validation_findings)
            
            # 4. Consolida Relatório Final
            total_duration = time.time() - start_time
            final_report = self._consolidate_final_report(
                architectural_report,
                module_results,
                advanced_findings,
                total_duration
            )
            
            # 5. Salva Relatório
            report_file = self.audit_path / f"complete_architectural_audit_{int(time.time())}.json"
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            
            # 6. Gera Relatório Executivo
            executive_summary = self._generate_executive_summary(final_report)
            summary_file = self.audit_path / f"executive_summary_{int(time.time())}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(executive_summary, f, indent=2, ensure_ascii=False)
            
            # 7. Log Final
            self.logger.info("=" * 80)
            self.logger.info(f"✅ AUDITORIA COMPLETA FINALIZADA em {total_duration:.2f}s")
            self.logger.info(f"📄 Relatório completo: {report_file}")
            self.logger.info(f"📊 Sumário executivo: {summary_file}")
            self.logger.info("=" * 80)
            
            # 8. Exibe Sumário no Console
            self._display_console_summary(executive_summary)
            
            return final_report
        
        except Exception as e:
            self.logger.error(f"❌ FALHA NA AUDITORIA: {str(e)}")
            raise
    
    def _consolidate_final_report(
        self,
        architectural_report: Dict[str, Any],
        module_results: Dict[str, Any],
        advanced_findings: list,
        total_duration: float
    ) -> Dict[str, Any]:
        """Consolida relatório final."""
        
        # Combina todos os findings
        all_findings = []
        all_findings.extend(architectural_report.get("detailed_findings", []))
        
        # Adiciona findings dos módulos
        for module_id, result in module_results.items():
            all_findings.extend([finding.__dict__ if hasattr(finding, '__dict__') else finding for finding in result.findings])
        
        # Adiciona findings avançados
        all_findings.extend([finding.__dict__ if hasattr(finding, '__dict__') else finding for finding in advanced_findings])
        
        # Categoriza findings
        findings_by_category = {
            "CRITICAL": [f for f in all_findings if f.get("category") == "CRITICAL"],
            "HIGH": [f for f in all_findings if f.get("category") == "HIGH"],
            "MEDIUM": [f for f in all_findings if f.get("category") == "MEDIUM"],
            "LOW": [f for f in all_findings if f.get("category") == "LOW"],
            "INFO": [f for f in all_findings if f.get("category") == "INFO"]
        }
        
        # Calcula métricas consolidadas
        total_findings = len(all_findings)
        critical_count = len(findings_by_category["CRITICAL"])
        high_count = len(findings_by_category["HIGH"])
        compliance_count = len(findings_by_category["INFO"])
        
        # Score de conformidade consolidado
        compliance_score = compliance_count / max(1, total_findings)
        
        # Score de prontidão para produção
        production_readiness_score = max(0.0, 1.0 - (critical_count * 0.3) - (high_count * 0.1))
        
        # Determina status geral
        if critical_count == 0 and high_count <= 2 and compliance_score >= 0.6:
            overall_status = "READY_FOR_PRODUCTION"
            status_description = "Sistema pronto para integração Falcon-Mamba e produção"
        elif critical_count <= 1 and compliance_score >= 0.5:
            overall_status = "MOSTLY_READY"
            status_description = "Sistema quase pronto, resolver issues críticos"
        elif compliance_score >= 0.3:
            overall_status = "NEEDS_WORK"
            status_description = "Sistema precisa de correções significativas"
        else:
            overall_status = "NOT_READY"
            status_description = "Sistema não está pronto para produção"
        
        return {
            "audit_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_duration_seconds": total_duration,
                "auditor_version": "1.0.0",
                "audit_scope": "complete_penin_omega_architecture",
                "audit_phases": [
                    "architectural_core",
                    "individual_modules",
                    "advanced_features"
                ]
            },
            "executive_summary": {
                "overall_status": overall_status,
                "status_description": status_description,
                "production_readiness_score": production_readiness_score,
                "compliance_score": compliance_score,
                "total_findings": total_findings,
                "findings_by_category": {
                    "critical": critical_count,
                    "high": high_count,
                    "medium": len(findings_by_category["MEDIUM"]),
                    "low": len(findings_by_category["LOW"]),
                    "info": compliance_count
                },
                "key_metrics": {
                    "omega_state_integrity": architectural_report.get("executive_summary", {}).get("omega_state_integrity", False),
                    "worm_ledger_integrity": architectural_report.get("executive_summary", {}).get("worm_ledger_integrity", False),
                    "security_gates_compliance": len(architectural_report.get("executive_summary", {}).get("gates_compliance", {})),
                    "modules_functional": sum(1 for r in module_results.values() if r.functional),
                    "modules_total": len(module_results)
                }
            },
            "detailed_results": {
                "architectural_audit": architectural_report,
                "module_audit_results": {
                    module_id: {
                        "module_name": result.module_name,
                        "exists": result.exists,
                        "importable": result.importable,
                        "functional": result.functional,
                        "compliance_score": result.compliance_score,
                        "findings_count": len(result.findings),
                        "findings": [finding.__dict__ if hasattr(finding, '__dict__') else finding for finding in result.findings]
                    }
                    for module_id, result in module_results.items()
                },
                "advanced_findings": [finding.__dict__ if hasattr(finding, '__dict__') else finding for finding in advanced_findings]
            },
            "consolidated_findings": all_findings,
            "findings_by_category": findings_by_category,
            "recommendations": self._generate_consolidated_recommendations(findings_by_category),
            "next_steps": self._generate_next_steps(overall_status, findings_by_category)
        }
    
    def _generate_consolidated_recommendations(self, findings_by_category: Dict[str, list]) -> list:
        """Gera recomendações consolidadas."""
        recommendations = []
        
        critical_count = len(findings_by_category["CRITICAL"])
        high_count = len(findings_by_category["HIGH"])
        
        if critical_count > 0:
            recommendations.append({
                "priority": "CRITICAL",
                "action": f"Resolver {critical_count} finding(s) crítico(s) imediatamente",
                "impact": "Bloqueadores para produção",
                "timeline": "Imediato"
            })
        
        if high_count > 0:
            recommendations.append({
                "priority": "HIGH",
                "action": f"Resolver {high_count} finding(s) de alta prioridade",
                "impact": "Riscos significativos para produção",
                "timeline": "1-2 dias"
            })
        
        # Recomendações específicas baseadas nos findings
        critical_findings = findings_by_category["CRITICAL"]
        
        # Agrupa por componente
        components_with_critical = {}
        for finding in critical_findings:
            component = finding.get("component", "unknown")
            if component not in components_with_critical:
                components_with_critical[component] = []
            components_with_critical[component].append(finding)
        
        for component, findings in components_with_critical.items():
            recommendations.append({
                "priority": "CRITICAL",
                "action": f"Corrigir componente {component}",
                "impact": f"{len(findings)} issue(s) crítico(s) neste componente",
                "timeline": "Imediato",
                "details": [f.get("title", "Unknown issue") for f in findings]
            })
        
        return recommendations
    
    def _generate_next_steps(self, overall_status: str, findings_by_category: Dict[str, list]) -> list:
        """Gera próximos passos baseados no status."""
        
        if overall_status == "READY_FOR_PRODUCTION":
            return [
                "✅ Sistema aprovado para integração Falcon-Mamba",
                "✅ Prosseguir com deployment em produção",
                "📊 Implementar monitoramento contínuo",
                "🔄 Agendar auditorias regulares"
            ]
        
        elif overall_status == "MOSTLY_READY":
            return [
                "🔧 Resolver findings críticos restantes",
                "📋 Revisar findings de alta prioridade",
                "🧪 Executar testes de regressão",
                "📊 Re-executar auditoria após correções"
            ]
        
        elif overall_status == "NEEDS_WORK":
            return [
                "🚨 Resolver todos os findings críticos",
                "🔧 Corrigir findings de alta prioridade",
                "🏗️ Implementar componentes ausentes",
                "🧪 Executar testes completos",
                "📊 Re-executar auditoria completa"
            ]
        
        else:  # NOT_READY
            return [
                "🚨 PARAR: Sistema não está pronto",
                "🔧 Resolver TODOS os findings críticos",
                "🏗️ Implementar componentes essenciais ausentes",
                "🔄 Refatorar arquitetura se necessário",
                "📊 Re-executar auditoria completa após correções"
            ]
    
    def _generate_executive_summary(self, final_report: Dict[str, Any]) -> Dict[str, Any]:
        """Gera sumário executivo."""
        exec_summary = final_report["executive_summary"]
        
        return {
            "penin_omega_architectural_audit": {
                "timestamp": final_report["audit_metadata"]["timestamp"],
                "overall_status": exec_summary["overall_status"],
                "production_ready": exec_summary["overall_status"] in ["READY_FOR_PRODUCTION", "MOSTLY_READY"],
                "falcon_mamba_integration_ready": exec_summary["production_readiness_score"] >= 0.7,
                "scores": {
                    "production_readiness": f"{exec_summary['production_readiness_score']:.1%}",
                    "compliance": f"{exec_summary['compliance_score']:.1%}"
                },
                "findings_summary": exec_summary["findings_by_category"],
                "key_systems": exec_summary["key_metrics"],
                "critical_actions_required": len(final_report["findings_by_category"]["CRITICAL"]) > 0,
                "recommendations_count": len(final_report["recommendations"]),
                "next_steps_count": len(final_report["next_steps"])
            }
        }
    
    def _display_console_summary(self, executive_summary: Dict[str, Any]):
        """Exibe sumário no console."""
        summary = executive_summary["penin_omega_architectural_audit"]
        
        print("\n" + "=" * 80)
        print("🏛️  PENIN-Ω ARCHITECTURAL AUDIT - EXECUTIVE SUMMARY")
        print("=" * 80)
        
        print(f"📊 Overall Status: {summary['overall_status']}")
        print(f"🚀 Production Ready: {'✅ YES' if summary['production_ready'] else '❌ NO'}")
        print(f"🦅 Falcon-Mamba Ready: {'✅ YES' if summary['falcon_mamba_integration_ready'] else '❌ NO'}")
        
        print(f"\n📈 Scores:")
        print(f"   Production Readiness: {summary['scores']['production_readiness']}")
        print(f"   Compliance: {summary['scores']['compliance']}")
        
        print(f"\n🔍 Findings:")
        findings = summary['findings_summary']
        print(f"   🚨 Critical: {findings['critical']}")
        print(f"   ⚠️  High: {findings['high']}")
        print(f"   📋 Medium: {findings['medium']}")
        print(f"   📝 Low: {findings['low']}")
        print(f"   ✅ Info: {findings['info']}")
        
        print(f"\n🏗️  Key Systems:")
        systems = summary['key_systems']
        print(f"   Ω-State Integrity: {'✅' if systems['omega_state_integrity'] else '❌'}")
        print(f"   WORM Ledger: {'✅' if systems['worm_ledger_integrity'] else '❌'}")
        print(f"   Security Gates: {systems['security_gates_compliance']}/3")
        print(f"   Functional Modules: {systems['modules_functional']}/{systems['modules_total']}")
        
        if summary['critical_actions_required']:
            print(f"\n🚨 CRITICAL ACTIONS REQUIRED!")
            print(f"   Resolve {findings['critical']} critical finding(s) before proceeding")
        
        print("=" * 80)

# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

async def main():
    """Função principal."""
    executor = ArchitecturalAuditExecutor()
    await executor.execute_complete_audit()

if __name__ == "__main__":
    asyncio.run(main())
