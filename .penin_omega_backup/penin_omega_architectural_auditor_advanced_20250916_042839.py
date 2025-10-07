#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Auditor Arquitetural - Extensões Avançadas
===================================================
Auditorias avançadas: Budget, Performance, Observabilidade, Validação.
"""

import asyncio
import psutil
import time
import json
import sqlite3
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import logging
import subprocess
import threading

from penin_omega_architectural_auditor import AuditFinding

# =============================================================================
# AUDITOR AVANÇADO
# =============================================================================

class AdvancedAuditor:
    """Auditor para funcionalidades avançadas."""
    
    def __init__(self):
        self.logger = logging.getLogger("AdvancedAuditor")
        self.performance_metrics = {}
        self.budget_status = {}
        self.observability_status = {}
    
    async def audit_budget_circuit_breaker(self) -> List[AuditFinding]:
        """Auditoria 5: Budget & Circuit Breaker."""
        self.logger.info("🔍 Auditando Budget & Circuit Breaker...")
        
        findings = []
        
        try:
            # Verifica se existe sistema de budget
            try:
                from penin_omega_global_state_manager import get_global_state
                current_state = get_global_state()
                
                # Verifica campos de budget
                budget_fields = ["budget_used", "budget_limit", "cost_per_operation"]
                budget_present = all(field in current_state for field in budget_fields)
                
                if budget_present:
                    findings.append(AuditFinding(
                        component="budget_system",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Sistema de Budget Presente",
                        description="Campos de budget implementados no estado global",
                        evidence={"budget_fields": budget_fields}
                    ))
                    
                    # Verifica se budget está sendo respeitado
                    budget_used = current_state.get("budget_used", 0)
                    budget_limit = current_state.get("budget_limit", 1000)
                    budget_utilization = budget_used / budget_limit if budget_limit > 0 else 0
                    
                    if budget_utilization < 0.9:
                        findings.append(AuditFinding(
                            component="budget_system",
                            category="INFO",
                            finding_type="COMPLIANCE",
                            title="Budget Dentro do Limite",
                            description=f"Utilização: {budget_utilization:.1%} < 90%",
                            evidence={"budget_used": budget_used, "budget_limit": budget_limit}
                        ))
                    else:
                        findings.append(AuditFinding(
                            component="budget_system",
                            category="HIGH",
                            finding_type="COMPLIANCE",
                            title="Budget Próximo do Limite",
                            description=f"Utilização: {budget_utilization:.1%} ≥ 90%",
                            evidence={"budget_used": budget_used, "budget_limit": budget_limit},
                            recommendation="Monitorar utilização de budget"
                        ))
                
                else:
                    findings.append(AuditFinding(
                        component="budget_system",
                        category="HIGH",
                        finding_type="BUG",
                        title="Sistema de Budget Ausente",
                        description="Campos de budget não encontrados no estado global",
                        evidence={"missing_fields": budget_fields},
                        recommendation="Implementar sistema completo de budget"
                    ))
            
            except Exception as e:
                findings.append(AuditFinding(
                    component="budget_system",
                    category="HIGH",
                    finding_type="BUG",
                    title="Falha na Verificação de Budget",
                    description=f"Erro ao verificar budget: {str(e)}",
                    recommendation="Implementar sistema robusto de budget"
                ))
            
            # Verifica circuit breaker
            try:
                # Procura por implementações de circuit breaker
                circuit_breaker_modules = [
                    "penin_omega_robust_multi_api",
                    "penin_omega_7_nexus"
                ]
                
                circuit_breaker_found = False
                
                for module_name in circuit_breaker_modules:
                    try:
                        module = __import__(module_name)
                        
                        # Verifica se tem circuit breaker
                        has_circuit_breaker = (
                            hasattr(module, 'CircuitBreaker') or
                            hasattr(module, 'circuit_breaker') or
                            'circuit' in str(module.__dict__).lower()
                        )
                        
                        if has_circuit_breaker:
                            circuit_breaker_found = True
                            findings.append(AuditFinding(
                                component="circuit_breaker",
                                category="INFO",
                                finding_type="COMPLIANCE",
                                title=f"Circuit Breaker em {module_name}",
                                description=f"Circuit breaker implementado em {module_name}",
                                evidence={"module": module_name}
                            ))
                            break
                    
                    except ImportError:
                        continue
                
                if not circuit_breaker_found:
                    findings.append(AuditFinding(
                        component="circuit_breaker",
                        category="MEDIUM",
                        finding_type="BUG",
                        title="Circuit Breaker Ausente",
                        description="Sistema de circuit breaker não encontrado",
                        recommendation="Implementar circuit breaker para APIs externas"
                    ))
            
            except Exception as e:
                findings.append(AuditFinding(
                    component="circuit_breaker",
                    category="MEDIUM",
                    finding_type="BUG",
                    title="Falha na Verificação de Circuit Breaker",
                    description=f"Erro ao verificar circuit breaker: {str(e)}"
                ))
        
        except Exception as e:
            findings.append(AuditFinding(
                component="budget_circuit_breaker",
                category="CRITICAL",
                finding_type="BUG",
                title="Falha Geral na Auditoria Budget/Circuit Breaker",
                description=f"Erro durante auditoria: {str(e)}",
                evidence={"error": str(e)}
            ))
        
        return findings
    
    async def audit_security_robustness(self) -> List[AuditFinding]:
        """Auditoria 7: Segurança e Robustez."""
        self.logger.info("🔍 Auditando Segurança e Robustez...")
        
        findings = []
        
        try:
            # Verifica DLP (Data Loss Prevention)
            try:
                from penin_omega_security_governance import security_governance
                
                # Testa DLP scanner
                test_content_clean = "This is clean content for testing"
                test_content_pii = "My email is john@example.com and SSN is 123-45-6789"
                
                clean_scan = security_governance.dlp_scanner.scan_content(test_content_clean)
                pii_scan = security_governance.dlp_scanner.scan_content(test_content_pii)
                
                dlp_working = (
                    len(clean_scan.get("violations", [])) == 0 and
                    len(pii_scan.get("violations", [])) > 0
                )
                
                if dlp_working:
                    findings.append(AuditFinding(
                        component="security_robustness",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="DLP Scanner Funcionando",
                        description="Scanner DLP detecta PII corretamente",
                        evidence={
                            "clean_violations": len(clean_scan.get("violations", [])),
                            "pii_violations": len(pii_scan.get("violations", []))
                        }
                    ))
                else:
                    findings.append(AuditFinding(
                        component="security_robustness",
                        category="HIGH",
                        finding_type="VULNERABILITY",
                        title="DLP Scanner Não Funcional",
                        description="Scanner DLP não detecta PII corretamente",
                        evidence={"clean_scan": clean_scan, "pii_scan": pii_scan},
                        recommendation="Corrigir detecção de PII no scanner DLP"
                    ))
            
            except ImportError:
                findings.append(AuditFinding(
                    component="security_robustness",
                    category="CRITICAL",
                    finding_type="VULNERABILITY",
                    title="Sistema DLP Ausente",
                    description="Sistema de prevenção de perda de dados não encontrado",
                    recommendation="Implementar sistema DLP completo"
                ))
            
            # Verifica criptografia
            crypto_modules = ["hashlib", "hmac", "secrets"]
            crypto_available = []
            
            for crypto_module in crypto_modules:
                try:
                    __import__(crypto_module)
                    crypto_available.append(crypto_module)
                except ImportError:
                    pass
            
            if len(crypto_available) >= 2:
                findings.append(AuditFinding(
                    component="security_robustness",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title="Módulos Criptográficos Disponíveis",
                    description="Módulos de criptografia estão disponíveis",
                    evidence={"available_modules": crypto_available}
                ))
            else:
                findings.append(AuditFinding(
                    component="security_robustness",
                    category="HIGH",
                    finding_type="VULNERABILITY",
                    title="Módulos Criptográficos Insuficientes",
                    description="Módulos de criptografia insuficientes",
                    evidence={"available_modules": crypto_available},
                    recommendation="Instalar módulos criptográficos necessários"
                ))
            
            # Verifica validação de entrada
            validation_patterns = [
                "validate", "sanitize", "clean", "escape", "filter"
            ]
            
            # Procura por funções de validação nos módulos
            validation_found = False
            
            try:
                import penin_omega_security_governance
                module_content = str(penin_omega_security_governance.__dict__)
                
                for pattern in validation_patterns:
                    if pattern in module_content.lower():
                        validation_found = True
                        break
                
                if validation_found:
                    findings.append(AuditFinding(
                        component="security_robustness",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Validação de Entrada Presente",
                        description="Funções de validação de entrada encontradas"
                    ))
                else:
                    findings.append(AuditFinding(
                        component="security_robustness",
                        category="MEDIUM",
                        finding_type="VULNERABILITY",
                        title="Validação de Entrada Limitada",
                        description="Funções de validação de entrada não encontradas",
                        recommendation="Implementar validação robusta de entrada"
                    ))
            
            except ImportError:
                findings.append(AuditFinding(
                    component="security_robustness",
                    category="HIGH",
                    finding_type="VULNERABILITY",
                    title="Módulo de Segurança Ausente",
                    description="Módulo de segurança não encontrado para validação"
                ))
        
        except Exception as e:
            findings.append(AuditFinding(
                component="security_robustness",
                category="CRITICAL",
                finding_type="BUG",
                title="Falha na Auditoria de Segurança",
                description=f"Erro durante auditoria de segurança: {str(e)}",
                evidence={"error": str(e)}
            ))
        
        return findings
    
    async def audit_observability(self) -> List[AuditFinding]:
        """Auditoria 8: Observabilidade."""
        self.logger.info("🔍 Auditando Observabilidade...")
        
        findings = []
        
        try:
            # Verifica sistema de logging
            logging_modules = [
                "penin_omega_structured_logging",
                "logging"
            ]
            
            structured_logging_found = False
            
            for module_name in logging_modules:
                try:
                    module = __import__(module_name)
                    
                    if "structured" in module_name:
                        structured_logging_found = True
                        findings.append(AuditFinding(
                            component="observability",
                            category="INFO",
                            finding_type="COMPLIANCE",
                            title="Logging Estruturado Presente",
                            description="Sistema de logging estruturado implementado",
                            evidence={"module": module_name}
                        ))
                        break
                
                except ImportError:
                    continue
            
            if not structured_logging_found:
                findings.append(AuditFinding(
                    component="observability",
                    category="MEDIUM",
                    finding_type="IMPROVEMENT",
                    title="Logging Estruturado Ausente",
                    description="Sistema de logging estruturado não encontrado",
                    recommendation="Implementar logging estruturado com JSON"
                ))
            
            # Verifica métricas de performance
            try:
                from penin_omega_performance_optimizer import performance_optimizer
                
                # Testa coleta de métricas
                metrics = performance_optimizer.get_current_metrics()
                
                required_metrics = ["cpu_percent", "memory_percent", "thread_count"]
                metrics_present = all(metric in metrics for metric in required_metrics)
                
                if metrics_present:
                    findings.append(AuditFinding(
                        component="observability",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Métricas de Performance Presentes",
                        description="Sistema coleta métricas essenciais",
                        evidence={"metrics": list(metrics.keys())}
                    ))
                else:
                    missing_metrics = [m for m in required_metrics if m not in metrics]
                    findings.append(AuditFinding(
                        component="observability",
                        category="MEDIUM",
                        finding_type="BUG",
                        title="Métricas de Performance Incompletas",
                        description=f"Métricas ausentes: {missing_metrics}",
                        evidence={"missing_metrics": missing_metrics},
                        recommendation="Implementar coleta completa de métricas"
                    ))
            
            except ImportError:
                findings.append(AuditFinding(
                    component="observability",
                    category="HIGH",
                    finding_type="BUG",
                    title="Sistema de Performance Ausente",
                    description="Otimizador de performance não encontrado",
                    recommendation="Implementar sistema de monitoramento de performance"
                ))
            
            # Verifica alertas e notificações
            alert_keywords = ["alert", "notify", "warning", "threshold", "alarm"]
            alerts_found = False
            
            try:
                # Procura em módulos relevantes
                search_modules = [
                    "penin_omega_7_nexus",
                    "penin_omega_performance_optimizer"
                ]
                
                for module_name in search_modules:
                    try:
                        module = __import__(module_name)
                        module_content = str(module.__dict__).lower()
                        
                        for keyword in alert_keywords:
                            if keyword in module_content:
                                alerts_found = True
                                break
                        
                        if alerts_found:
                            break
                    
                    except ImportError:
                        continue
                
                if alerts_found:
                    findings.append(AuditFinding(
                        component="observability",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Sistema de Alertas Presente",
                        description="Sistema de alertas e notificações implementado"
                    ))
                else:
                    findings.append(AuditFinding(
                        component="observability",
                        category="MEDIUM",
                        finding_type="IMPROVEMENT",
                        title="Sistema de Alertas Ausente",
                        description="Sistema de alertas não encontrado",
                        recommendation="Implementar sistema de alertas para anomalias"
                    ))
            
            except Exception as e:
                findings.append(AuditFinding(
                    component="observability",
                    category="MEDIUM",
                    finding_type="BUG",
                    title="Falha na Verificação de Alertas",
                    description=f"Erro ao verificar alertas: {str(e)}"
                ))
        
        except Exception as e:
            findings.append(AuditFinding(
                component="observability",
                category="CRITICAL",
                finding_type="BUG",
                title="Falha na Auditoria de Observabilidade",
                description=f"Erro durante auditoria: {str(e)}",
                evidence={"error": str(e)}
            ))
        
        return findings
    
    async def audit_performance(self) -> List[AuditFinding]:
        """Auditoria 9: Performance."""
        self.logger.info("🔍 Auditando Performance...")
        
        findings = []
        
        try:
            # Coleta métricas atuais do sistema
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            self.performance_metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
            
            # Verifica CPU
            if cpu_percent < 80:
                findings.append(AuditFinding(
                    component="performance",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title="CPU Utilização Normal",
                    description=f"CPU: {cpu_percent:.1f}% < 80%",
                    evidence={"cpu_percent": cpu_percent}
                ))
            else:
                findings.append(AuditFinding(
                    component="performance",
                    category="HIGH",
                    finding_type="COMPLIANCE",
                    title="CPU Utilização Alta",
                    description=f"CPU: {cpu_percent:.1f}% ≥ 80%",
                    evidence={"cpu_percent": cpu_percent},
                    recommendation="Investigar alta utilização de CPU"
                ))
            
            # Verifica memória
            if memory.percent < 85:
                findings.append(AuditFinding(
                    component="performance",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title="Memória Utilização Normal",
                    description=f"Memória: {memory.percent:.1f}% < 85%",
                    evidence={"memory_percent": memory.percent}
                ))
            else:
                findings.append(AuditFinding(
                    component="performance",
                    category="HIGH",
                    finding_type="COMPLIANCE",
                    title="Memória Utilização Alta",
                    description=f"Memória: {memory.percent:.1f}% ≥ 85%",
                    evidence={"memory_percent": memory.percent},
                    recommendation="Investigar alta utilização de memória"
                ))
            
            # Verifica disco
            if disk.percent < 90:
                findings.append(AuditFinding(
                    component="performance",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title="Disco Utilização Normal",
                    description=f"Disco: {disk.percent:.1f}% < 90%",
                    evidence={"disk_percent": disk.percent}
                ))
            else:
                findings.append(AuditFinding(
                    component="performance",
                    category="HIGH",
                    finding_type="COMPLIANCE",
                    title="Disco Utilização Alta",
                    description=f"Disco: {disk.percent:.1f}% ≥ 90%",
                    evidence={"disk_percent": disk.percent},
                    recommendation="Liberar espaço em disco"
                ))
            
            # Testa performance de operações básicas
            start_time = time.time()
            
            # Operação de I/O
            test_file = Path("/tmp/penin_omega_perf_test.txt")
            test_content = "Performance test content" * 1000
            
            with open(test_file, 'w') as f:
                f.write(test_content)
            
            with open(test_file, 'r') as f:
                read_content = f.read()
            
            test_file.unlink()
            
            io_duration = time.time() - start_time
            
            if io_duration < 0.1:
                findings.append(AuditFinding(
                    component="performance",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title="Performance I/O Normal",
                    description=f"I/O test: {io_duration:.3f}s < 0.1s",
                    evidence={"io_duration": io_duration}
                ))
            else:
                findings.append(AuditFinding(
                    component="performance",
                    category="MEDIUM",
                    finding_type="COMPLIANCE",
                    title="Performance I/O Lenta",
                    description=f"I/O test: {io_duration:.3f}s ≥ 0.1s",
                    evidence={"io_duration": io_duration},
                    recommendation="Investigar performance de I/O"
                ))
        
        except Exception as e:
            findings.append(AuditFinding(
                component="performance",
                category="CRITICAL",
                finding_type="BUG",
                title="Falha na Auditoria de Performance",
                description=f"Erro durante auditoria: {str(e)}",
                evidence={"error": str(e)}
            ))
        
        return findings
    
    async def audit_complete_validation(self) -> List[AuditFinding]:
        """Auditoria 10: Validação Completa."""
        self.logger.info("🔍 Executando Validação Completa...")
        
        findings = []
        
        try:
            # Executa teste de pipeline completo
            try:
                from penin_omega_automated_testing import automated_tester
                
                # Executa testes automatizados
                test_results = automated_tester.run_all_tests()
                
                total_tests = test_results.get("total_tests", 0)
                passed_tests = test_results.get("passed_tests", 0)
                success_rate = passed_tests / total_tests if total_tests > 0 else 0
                
                if success_rate >= 0.8:
                    findings.append(AuditFinding(
                        component="complete_validation",
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Testes Automatizados Passando",
                        description=f"Taxa de sucesso: {success_rate:.1%} ≥ 80%",
                        evidence={"success_rate": success_rate, "passed": passed_tests, "total": total_tests}
                    ))
                elif success_rate >= 0.6:
                    findings.append(AuditFinding(
                        component="complete_validation",
                        category="MEDIUM",
                        finding_type="COMPLIANCE",
                        title="Testes Automatizados Parcialmente Passando",
                        description=f"Taxa de sucesso: {success_rate:.1%} (60-80%)",
                        evidence={"success_rate": success_rate, "passed": passed_tests, "total": total_tests},
                        recommendation="Corrigir testes falhando para atingir 80%"
                    ))
                else:
                    findings.append(AuditFinding(
                        component="complete_validation",
                        category="HIGH",
                        finding_type="BUG",
                        title="Testes Automatizados Falhando",
                        description=f"Taxa de sucesso: {success_rate:.1%} < 60%",
                        evidence={"success_rate": success_rate, "passed": passed_tests, "total": total_tests},
                        recommendation="Corrigir falhas críticas nos testes"
                    ))
            
            except ImportError:
                findings.append(AuditFinding(
                    component="complete_validation",
                    category="HIGH",
                    finding_type="BUG",
                    title="Sistema de Testes Ausente",
                    description="Framework de testes automatizados não encontrado",
                    recommendation="Implementar framework de testes automatizados"
                ))
            
            # Valida integridade do sistema completo
            system_components = [
                "penin_omega_global_state_manager",
                "penin_omega_security_governance",
                "penin_omega_unified_classes"
            ]
            
            components_working = 0
            total_components = len(system_components)
            
            for component in system_components:
                try:
                    __import__(component)
                    components_working += 1
                except ImportError:
                    pass
            
            system_integrity = components_working / total_components
            
            if system_integrity >= 0.8:
                findings.append(AuditFinding(
                    component="complete_validation",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title="Integridade do Sistema Alta",
                    description=f"Componentes funcionais: {system_integrity:.1%}",
                    evidence={"working_components": components_working, "total_components": total_components}
                ))
            else:
                findings.append(AuditFinding(
                    component="complete_validation",
                    category="HIGH",
                    finding_type="BUG",
                    title="Integridade do Sistema Baixa",
                    description=f"Componentes funcionais: {system_integrity:.1%} < 80%",
                    evidence={"working_components": components_working, "total_components": total_components},
                    recommendation="Corrigir componentes não funcionais"
                ))
        
        except Exception as e:
            findings.append(AuditFinding(
                component="complete_validation",
                category="CRITICAL",
                finding_type="BUG",
                title="Falha na Validação Completa",
                description=f"Erro durante validação: {str(e)}",
                evidence={"error": str(e)}
            ))
        
        return findings

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

advanced_auditor = AdvancedAuditor()

if __name__ == "__main__":
    async def main():
        findings = []
        findings.extend(await advanced_auditor.audit_budget_circuit_breaker())
        findings.extend(await advanced_auditor.audit_security_robustness())
        findings.extend(await advanced_auditor.audit_observability())
        findings.extend(await advanced_auditor.audit_performance())
        findings.extend(await advanced_auditor.audit_complete_validation())
        
        print(f"Total findings: {len(findings)}")
        for finding in findings:
            print(f"[{finding.category}] {finding.title}")
    
    asyncio.run(main())
