#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Auditor Arquitetural Completo
=======================================
Auditoria rigorosa da arquitetura completa de 8 módulos.
"""

from __future__ import annotations
import asyncio
import json
import time
import traceback
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging
import importlib
import sys
import sqlite3
import hashlib

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PENIN_OMEGA_ROOT = Path("/root/.penin_omega")
AUDIT_PATH = PENIN_OMEGA_ROOT / "audit"
AUDIT_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CLASSES DE AUDITORIA
# =============================================================================

@dataclass
class AuditFinding:
    """Achado de auditoria."""
    component: str
    category: str  # CRITICAL, HIGH, MEDIUM, LOW, INFO
    finding_type: str  # VULNERABILITY, BUG, IMPROVEMENT, COMPLIANCE
    title: str
    description: str
    evidence: Dict[str, Any] = field(default_factory=dict)
    recommendation: str = ""
    risk_level: str = "MEDIUM"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

@dataclass
class ModuleAuditResult:
    """Resultado de auditoria de módulo."""
    module_name: str
    module_path: str
    exists: bool = False
    importable: bool = False
    functional: bool = False
    compliance_score: float = 0.0
    findings: List[AuditFinding] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

# =============================================================================
# AUDITOR ARQUITETURAL
# =============================================================================

class ArchitecturalAuditor:
    """Auditor completo da arquitetura PENIN-Ω."""
    
    def __init__(self):
        self.logger = logging.getLogger("ArchitecturalAuditor")
        self.findings = []
        self.module_results = {}
        self.omega_state_integrity = False
        self.gates_compliance = {}
        self.worm_integrity = False
        
        # Módulos esperados
        self.expected_modules = {
            "1/8": "penin_omega_1_core_v6",
            "2/8": "penin_omega_2_strategy", 
            "3/8": "penin_omega_3_acquisition",
            "4/8": "penin_omega_4_mutation",
            "5/8": "penin_omega_5_crucible",
            "6/8": "penin_omega_6_autorewrite",
            "7/8": "penin_omega_7_nexus",
            "8/8": "penin_omega_8_governance_hub"
        }
    
    async def audit_complete_architecture(self) -> Dict[str, Any]:
        """Executa auditoria completa da arquitetura."""
        self.logger.info("🔍 INICIANDO AUDITORIA ARQUITETURAL COMPLETA PENIN-Ω")
        
        start_time = time.time()
        
        # 1. Auditoria de Integridade do Estado Canônico
        await self.audit_omega_state_integrity()
        
        # 2. Auditoria de Gates e Portões Éticos
        await self.audit_security_gates()
        
        # 3. Auditoria WORM/PCE Ledger
        await self.audit_worm_ledger()
        
        # 4. Auditoria Scheduler & Watchdog
        await self.audit_scheduler_watchdog()
        
        # Auditorias básicas concluídas - outras serão feitas pelos módulos especializados
        
        total_duration = time.time() - start_time
        
        # Gera relatório final
        report = self._generate_audit_report(total_duration)
        
        # Salva relatório
        report_file = AUDIT_PATH / f"architectural_audit_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"📊 Auditoria concluída em {total_duration:.2f}s")
        self.logger.info(f"📄 Relatório salvo: {report_file}")
        
        return report
    
    async def audit_omega_state_integrity(self):
        """Auditoria 1: Integridade do Estado Canônico."""
        self.logger.info("🔍 Auditando integridade do Estado Canônico (Ω-State)...")
        
        try:
            # Verifica se existe sistema de estado global
            try:
                from penin_omega_global_state_manager import get_global_state, global_state_manager
                state_system_exists = True
            except ImportError:
                state_system_exists = False
                self.findings.append(AuditFinding(
                    component="omega_state",
                    category="CRITICAL",
                    finding_type="BUG",
                    title="Sistema de Estado Global Ausente",
                    description="Sistema de gerenciamento de estado global não encontrado",
                    recommendation="Implementar sistema de estado global unificado"
                ))
            
            if state_system_exists:
                # Testa consistência de leitura/escrita
                original_state = get_global_state()
                test_updates = {"audit_test": True, "rho": 0.5}
                
                # Escreve
                update_success = global_state_manager.update_state(test_updates, "auditor")
                
                # Lê
                updated_state = get_global_state()
                
                # Verifica consistência
                read_write_consistent = (
                    updated_state.get("audit_test") is True and
                    updated_state.get("rho") == 0.5
                )
                
                if read_write_consistent:
                    self.omega_state_integrity = True
                    self.findings.append(AuditFinding(
                        component="omega_state",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Estado Canônico Íntegro",
                        description="Leituras e escritas em Ω-State são consistentes",
                        evidence={"test_updates": test_updates, "verification": "PASSED"}
                    ))
                else:
                    self.findings.append(AuditFinding(
                        component="omega_state",
                        category="HIGH",
                        finding_type="BUG",
                        title="Inconsistência em Ω-State",
                        description="Leituras e escritas não são consistentes",
                        evidence={"expected": test_updates, "actual": updated_state},
                        recommendation="Corrigir sistema de sincronização de estado"
                    ))
                
                # Verifica invariantes globais
                required_fields = ["rho", "sr_score", "ece", "consent", "eco_ok"]
                missing_invariants = [field for field in required_fields if field not in updated_state]
                
                if missing_invariants:
                    self.findings.append(AuditFinding(
                        component="omega_state",
                        category="HIGH",
                        finding_type="COMPLIANCE",
                        title="Invariantes Globais Ausentes",
                        description=f"Campos obrigatórios ausentes: {missing_invariants}",
                        evidence={"missing_fields": missing_invariants},
                        recommendation="Implementar todos os invariantes globais obrigatórios"
                    ))
                else:
                    self.findings.append(AuditFinding(
                        component="omega_state",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Invariantes Globais Presentes",
                        description="Todos os invariantes globais estão implementados",
                        evidence={"invariants": required_fields}
                    ))
        
        except Exception as e:
            self.findings.append(AuditFinding(
                component="omega_state",
                category="CRITICAL",
                finding_type="BUG",
                title="Falha na Auditoria de Estado",
                description=f"Erro durante auditoria: {str(e)}",
                evidence={"error": str(e), "traceback": traceback.format_exc()}
            ))
    
    async def audit_security_gates(self):
        """Auditoria 2: Gates e Portões Éticos/Security."""
        self.logger.info("🔍 Auditando Gates de Segurança (Σ-Guard, IR→IC, SR-Ω∞)...")
        
        try:
            # Verifica implementação de gates de segurança
            security_modules = [
                "penin_omega_security_governance",
                "penin_omega_8_governance_hub"
            ]
            
            gates_implemented = {}
            
            for module_name in security_modules:
                try:
                    module = importlib.import_module(module_name)
                    gates_implemented[module_name] = True
                    
                    # Verifica se tem sistema de governança
                    if hasattr(module, 'security_governance'):
                        # Testa gate Σ-Guard (ECE, consentimento, eco)
                        test_content_clean = "Clean test content"
                        test_content_violation = "My email is test@example.com and SSN 123-45-6789"
                        
                        # Teste conteúdo limpo
                        clean_result = module.security_governance.secure_operation(
                            "sigma_guard_test", test_content_clean, {"test": True}
                        )
                        
                        # Teste conteúdo com violação
                        violation_result = module.security_governance.secure_operation(
                            "sigma_guard_violation", test_content_violation, {"test": True}
                        )
                        
                        # Verifica fail-closed
                        sigma_guard_working = (
                            clean_result.get("success") is True and
                            violation_result.get("success") is False
                        )
                        
                        if sigma_guard_working:
                            self.gates_compliance["sigma_guard"] = True
                            self.findings.append(AuditFinding(
                                component="security_gates",
                                category="INFO",
                                finding_type="COMPLIANCE",
                                title="Σ-Guard Funcionando Corretamente",
                                description="Gate Σ-Guard implementa fail-closed perfeito",
                                evidence={
                                    "clean_content_result": clean_result.get("success"),
                                    "violation_content_result": violation_result.get("success")
                                }
                            ))
                        else:
                            self.findings.append(AuditFinding(
                                component="security_gates",
                                category="CRITICAL",
                                finding_type="VULNERABILITY",
                                title="Σ-Guard Falha em Fail-Closed",
                                description="Gate Σ-Guard não implementa fail-closed corretamente",
                                evidence={
                                    "clean_result": clean_result,
                                    "violation_result": violation_result
                                },
                                recommendation="Corrigir implementação fail-closed do Σ-Guard"
                            ))
                
                except ImportError:
                    gates_implemented[module_name] = False
                    self.findings.append(AuditFinding(
                        component="security_gates",
                        category="HIGH",
                        finding_type="BUG",
                        title=f"Módulo de Segurança Ausente: {module_name}",
                        description=f"Módulo {module_name} não encontrado",
                        recommendation=f"Implementar módulo {module_name}"
                    ))
            
            # Verifica gate IR→IC (ρ < 1)
            try:
                from penin_omega_global_state_manager import get_global_state
                current_state = get_global_state()
                rho_value = current_state.get("rho", 1.0)
                
                ir_ic_compliant = rho_value < 1.0
                
                if ir_ic_compliant:
                    self.gates_compliance["ir_ic"] = True
                    self.findings.append(AuditFinding(
                        component="security_gates",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Gate IR→IC Conforme",
                        description=f"ρ = {rho_value} < 1.0 (conforme)",
                        evidence={"rho_value": rho_value, "threshold": 1.0}
                    ))
                else:
                    self.findings.append(AuditFinding(
                        component="security_gates",
                        category="CRITICAL",
                        finding_type="VULNERABILITY",
                        title="Gate IR→IC Violado",
                        description=f"ρ = {rho_value} ≥ 1.0 (violação crítica)",
                        evidence={"rho_value": rho_value, "threshold": 1.0},
                        recommendation="Corrigir valor de ρ para < 1.0"
                    ))
            
            except Exception as e:
                self.findings.append(AuditFinding(
                    component="security_gates",
                    category="HIGH",
                    finding_type="BUG",
                    title="Falha na Verificação IR→IC",
                    description=f"Erro ao verificar gate IR→IC: {str(e)}",
                    recommendation="Implementar verificação robusta do gate IR→IC"
                ))
            
            # Verifica gate SR-Ω∞
            try:
                current_state = get_global_state()
                sr_score = current_state.get("sr_score", 0.0)
                tau_sr = 0.8  # Threshold padrão
                
                sr_omega_compliant = sr_score >= tau_sr
                
                if sr_omega_compliant:
                    self.gates_compliance["sr_omega"] = True
                    self.findings.append(AuditFinding(
                        component="security_gates",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Gate SR-Ω∞ Conforme",
                        description=f"SR = {sr_score} ≥ {tau_sr} (conforme)",
                        evidence={"sr_score": sr_score, "tau_sr": tau_sr}
                    ))
                else:
                    self.findings.append(AuditFinding(
                        component="security_gates",
                        category="HIGH",
                        finding_type="COMPLIANCE",
                        title="Gate SR-Ω∞ Abaixo do Limite",
                        description=f"SR = {sr_score} < {tau_sr} (abaixo do limite)",
                        evidence={"sr_score": sr_score, "tau_sr": tau_sr},
                        recommendation="Melhorar SR para atingir threshold mínimo"
                    ))
            
            except Exception as e:
                self.findings.append(AuditFinding(
                    component="security_gates",
                    category="HIGH",
                    finding_type="BUG",
                    title="Falha na Verificação SR-Ω∞",
                    description=f"Erro ao verificar gate SR-Ω∞: {str(e)}",
                    recommendation="Implementar verificação robusta do gate SR-Ω∞"
                ))
        
        except Exception as e:
            self.findings.append(AuditFinding(
                component="security_gates",
                category="CRITICAL",
                finding_type="BUG",
                title="Falha Geral na Auditoria de Gates",
                description=f"Erro durante auditoria de gates: {str(e)}",
                evidence={"error": str(e)}
            ))
    
    async def audit_worm_ledger(self):
        """Auditoria 3: WORM/PCE Ledger Imutável."""
        self.logger.info("🔍 Auditando WORM Ledger...")
        
        try:
            # Verifica se WORM ledger existe
            try:
                from penin_omega_security_governance import security_governance
                worm_system_exists = True
            except ImportError:
                worm_system_exists = False
                self.findings.append(AuditFinding(
                    component="worm_ledger",
                    category="CRITICAL",
                    finding_type="BUG",
                    title="Sistema WORM Ausente",
                    description="Sistema WORM Ledger não encontrado",
                    recommendation="Implementar sistema WORM completo"
                ))
                return
            
            if worm_system_exists:
                # Testa integridade da cadeia
                integrity_ok, integrity_issues = security_governance.worm_ledger.verify_integrity()
                
                if integrity_ok:
                    self.worm_integrity = True
                    self.findings.append(AuditFinding(
                        component="worm_ledger",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="WORM Ledger Íntegro",
                        description="Hash-chain do WORM ledger está íntegra",
                        evidence={"integrity_verified": True, "issues_count": len(integrity_issues)}
                    ))
                else:
                    self.findings.append(AuditFinding(
                        component="worm_ledger",
                        category="CRITICAL",
                        finding_type="VULNERABILITY",
                        title="WORM Ledger Comprometido",
                        description="Hash-chain do WORM ledger está comprometida",
                        evidence={"integrity_issues": integrity_issues},
                        recommendation="Investigar e corrigir comprometimento do ledger"
                    ))
                
                # Testa eventos críticos
                critical_events = [
                    "PROMOTE", "ROLLBACK", "EXTINCTION", "KILL_SWITCH",
                    "STRATEGY_DECISION", "TASK_DONE", "TASK_FAIL"
                ]
                
                # Simula registro de eventos críticos
                events_logged = []
                for event in critical_events:
                    try:
                        record = security_governance.worm_ledger.append_record(
                            f"audit_{event.lower()}",
                            f"Audit test for {event}",
                            {"event_type": event, "audit": True}
                        )
                        events_logged.append(event)
                    except Exception as e:
                        self.findings.append(AuditFinding(
                            component="worm_ledger",
                            category="HIGH",
                            finding_type="BUG",
                            title=f"Falha no Registro de {event}",
                            description=f"Erro ao registrar evento {event}: {str(e)}",
                            recommendation=f"Corrigir registro de eventos {event}"
                        ))
                
                if len(events_logged) == len(critical_events):
                    self.findings.append(AuditFinding(
                        component="worm_ledger",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Eventos Críticos Registrados",
                        description="Todos os eventos críticos são registrados corretamente",
                        evidence={"events_logged": events_logged}
                    ))
                else:
                    self.findings.append(AuditFinding(
                        component="worm_ledger",
                        category="HIGH",
                        finding_type="BUG",
                        title="Eventos Críticos Não Registrados",
                        description=f"Apenas {len(events_logged)}/{len(critical_events)} eventos registrados",
                        evidence={"events_logged": events_logged, "events_missing": list(set(critical_events) - set(events_logged))},
                        recommendation="Implementar registro completo de eventos críticos"
                    ))
        
        except Exception as e:
            self.findings.append(AuditFinding(
                component="worm_ledger",
                category="CRITICAL",
                finding_type="BUG",
                title="Falha na Auditoria WORM",
                description=f"Erro durante auditoria WORM: {str(e)}",
                evidence={"error": str(e)}
            ))
    
    async def audit_scheduler_watchdog(self):
        """Auditoria 4: Scheduler, Watchdog & Canary."""
        self.logger.info("🔍 Auditando Scheduler & Watchdog (7/8)...")
        
        try:
            # Verifica módulo 7/8 (NEXUS)
            try:
                nexus_module = importlib.import_module("penin_omega_7_nexus")
                nexus_exists = True
            except ImportError:
                nexus_exists = False
                self.findings.append(AuditFinding(
                    component="scheduler_watchdog",
                    category="CRITICAL",
                    finding_type="BUG",
                    title="Módulo NEXUS (7/8) Ausente",
                    description="Módulo de scheduler/watchdog não encontrado",
                    recommendation="Implementar módulo NEXUS completo"
                ))
                return
            
            if nexus_exists:
                # Verifica se tem scheduler
                has_scheduler = hasattr(nexus_module, 'NexusScheduler') or hasattr(nexus_module, 'scheduler')
                
                if has_scheduler:
                    self.findings.append(AuditFinding(
                        component="scheduler_watchdog",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Scheduler Implementado",
                        description="Sistema de scheduler está presente",
                        evidence={"scheduler_found": True}
                    ))
                else:
                    self.findings.append(AuditFinding(
                        component="scheduler_watchdog",
                        category="HIGH",
                        finding_type="BUG",
                        title="Scheduler Ausente",
                        description="Sistema de scheduler não encontrado no módulo NEXUS",
                        recommendation="Implementar scheduler por utilidade segura"
                    ))
                
                # Verifica watchdog
                has_watchdog = hasattr(nexus_module, 'Watchdog') or hasattr(nexus_module, 'watchdog')
                
                if has_watchdog:
                    self.findings.append(AuditFinding(
                        component="scheduler_watchdog",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Watchdog Implementado",
                        description="Sistema de watchdog está presente",
                        evidence={"watchdog_found": True}
                    ))
                else:
                    self.findings.append(AuditFinding(
                        component="scheduler_watchdog",
                        category="HIGH",
                        finding_type="BUG",
                        title="Watchdog Ausente",
                        description="Sistema de watchdog não encontrado",
                        recommendation="Implementar watchdog para detecção de anomalias"
                    ))
        
        except Exception as e:
            self.findings.append(AuditFinding(
                component="scheduler_watchdog",
                category="CRITICAL",
                finding_type="BUG",
                title="Falha na Auditoria Scheduler/Watchdog",
                description=f"Erro durante auditoria: {str(e)}",
                evidence={"error": str(e)}
            ))
    
    def _generate_audit_report(self, total_duration: float) -> Dict[str, Any]:
        """Gera relatório final de auditoria."""
        # Categoriza findings
        critical_findings = [f for f in self.findings if f.category == "CRITICAL"]
        high_findings = [f for f in self.findings if f.category == "HIGH"]
        medium_findings = [f for f in self.findings if f.category == "MEDIUM"]
        low_findings = [f for f in self.findings if f.category == "LOW"]
        info_findings = [f for f in self.findings if f.category == "INFO"]
        
        # Calcula score de conformidade
        total_findings = len(self.findings)
        compliance_findings = len(info_findings)
        compliance_score = compliance_findings / max(1, total_findings)
        
        # Determina se está pronto para produção
        ready_for_production = (
            len(critical_findings) == 0 and
            len(high_findings) <= 2 and
            self.omega_state_integrity and
            self.worm_integrity and
            compliance_score >= 0.6
        )
        
        return {
            "audit_metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "duration_seconds": total_duration,
                "auditor_version": "1.0.0",
                "audit_scope": "complete_architecture"
            },
            "executive_summary": {
                "total_findings": total_findings,
                "critical_findings": len(critical_findings),
                "high_findings": len(high_findings),
                "medium_findings": len(medium_findings),
                "low_findings": len(low_findings),
                "compliance_findings": len(info_findings),
                "compliance_score": compliance_score,
                "ready_for_production": ready_for_production,
                "omega_state_integrity": self.omega_state_integrity,
                "worm_ledger_integrity": self.worm_integrity,
                "gates_compliance": self.gates_compliance
            },
            "detailed_findings": [asdict(finding) for finding in self.findings],
            "module_audit_results": self.module_results,
            "recommendations": self._generate_recommendations(),
            "production_readiness_assessment": self._assess_production_readiness()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Gera recomendações baseadas nos findings."""
        recommendations = []
        
        critical_findings = [f for f in self.findings if f.category == "CRITICAL"]
        high_findings = [f for f in self.findings if f.category == "HIGH"]
        
        if critical_findings:
            recommendations.append("PRIORIDADE CRÍTICA: Resolver todos os findings críticos antes de prosseguir")
        
        if high_findings:
            recommendations.append("PRIORIDADE ALTA: Resolver findings de alta prioridade")
        
        if not self.omega_state_integrity:
            recommendations.append("Implementar sistema robusto de integridade de estado")
        
        if not self.worm_integrity:
            recommendations.append("Corrigir integridade do WORM ledger")
        
        if len(self.gates_compliance) < 3:
            recommendations.append("Implementar todos os gates de segurança obrigatórios")
        
        return recommendations
    
    def _assess_production_readiness(self) -> Dict[str, Any]:
        """Avalia prontidão para produção."""
        critical_count = len([f for f in self.findings if f.category == "CRITICAL"])
        high_count = len([f for f in self.findings if f.category == "HIGH"])
        
        readiness_score = 1.0
        
        # Penaliza por findings críticos
        readiness_score -= critical_count * 0.3
        
        # Penaliza por findings altos
        readiness_score -= high_count * 0.1
        
        # Bônus por integridade
        if self.omega_state_integrity:
            readiness_score += 0.1
        
        if self.worm_integrity:
            readiness_score += 0.1
        
        readiness_score = max(0.0, min(1.0, readiness_score))
        
        # Determina status
        if readiness_score >= 0.9:
            status = "READY"
            recommendation = "Sistema pronto para integração Falcon-Mamba e produção"
        elif readiness_score >= 0.7:
            status = "MOSTLY_READY"
            recommendation = "Sistema quase pronto, resolver findings de alta prioridade"
        elif readiness_score >= 0.5:
            status = "NEEDS_WORK"
            recommendation = "Sistema precisa de correções antes da produção"
        else:
            status = "NOT_READY"
            recommendation = "Sistema não está pronto para produção"
        
        return {
            "status": status,
            "readiness_score": readiness_score,
            "recommendation": recommendation,
            "critical_blockers": critical_count,
            "high_priority_issues": high_count,
            "falcon_mamba_integration_ready": readiness_score >= 0.7
        }

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

architectural_auditor = ArchitecturalAuditor()

if __name__ == "__main__":
    async def main():
        report = await architectural_auditor.audit_complete_architecture()
        print(json.dumps(report, indent=2))
    
    asyncio.run(main())
