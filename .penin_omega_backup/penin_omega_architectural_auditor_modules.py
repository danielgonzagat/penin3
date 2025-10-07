#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Auditor Arquitetural - Módulos Individuais
===================================================
Auditoria específica dos 8 módulos individuais.
"""

import asyncio
import importlib
import inspect
import ast
import traceback
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
import logging

from penin_omega_architectural_auditor import AuditFinding, ModuleAuditResult

# =============================================================================
# AUDITOR DE MÓDULOS INDIVIDUAIS
# =============================================================================

class ModuleAuditor:
    """Auditor específico para módulos individuais."""
    
    def __init__(self):
        self.logger = logging.getLogger("ModuleAuditor")
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
    
    async def audit_individual_modules(self) -> Dict[str, ModuleAuditResult]:
        """Auditoria completa dos módulos individuais."""
        self.logger.info("🔍 Auditando módulos individuais (1/8 → 8/8)...")
        
        results = {}
        
        for module_id, module_name in self.expected_modules.items():
            self.logger.info(f"🔍 Auditando {module_id}: {module_name}")
            result = await self.audit_single_module(module_id, module_name)
            results[module_id] = result
        
        return results
    
    async def audit_single_module(self, module_id: str, module_name: str) -> ModuleAuditResult:
        """Auditoria de um módulo específico."""
        result = ModuleAuditResult(
            module_name=module_name,
            module_path=f"/root/{module_name}.py"
        )
        
        try:
            # 1. Verifica se arquivo existe
            module_path = Path(f"/root/{module_name}.py")
            result.module_path = str(module_path)
            result.exists = module_path.exists()
            
            if not result.exists:
                result.findings.append(AuditFinding(
                    component=module_id,
                    category="CRITICAL",
                    finding_type="BUG",
                    title=f"Módulo {module_id} Ausente",
                    description=f"Arquivo {module_name}.py não encontrado",
                    recommendation=f"Implementar módulo {module_id}"
                ))
                return result
            
            # 2. Verifica se é importável
            try:
                module = importlib.import_module(module_name)
                result.importable = True
                
                result.findings.append(AuditFinding(
                    component=module_id,
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title=f"Módulo {module_id} Importável",
                    description=f"Módulo {module_name} importa sem erros"
                ))
                
            except ImportError as e:
                result.importable = False
                result.findings.append(AuditFinding(
                    component=module_id,
                    category="CRITICAL",
                    finding_type="BUG",
                    title=f"Módulo {module_id} Não Importável",
                    description=f"Erro de importação: {str(e)}",
                    evidence={"import_error": str(e)},
                    recommendation=f"Corrigir erros de importação em {module_name}"
                ))
                return result
            
            # 3. Auditoria específica por módulo
            await self.audit_module_specific(module_id, module, result)
            
            # 4. Auditoria de conformidade geral
            await self.audit_module_compliance(module_id, module, result)
            
            # 5. Calcula score de conformidade
            compliance_findings = len([f for f in result.findings if f.category == "INFO"])
            total_findings = len(result.findings)
            result.compliance_score = compliance_findings / max(1, total_findings)
            
            # 6. Determina se é funcional
            critical_findings = len([f for f in result.findings if f.category == "CRITICAL"])
            result.functional = critical_findings == 0 and result.importable
            
        except Exception as e:
            result.findings.append(AuditFinding(
                component=module_id,
                category="CRITICAL",
                finding_type="BUG",
                title=f"Falha na Auditoria de {module_id}",
                description=f"Erro durante auditoria: {str(e)}",
                evidence={"error": str(e), "traceback": traceback.format_exc()}
            ))
        
        return result
    
    async def audit_module_specific(self, module_id: str, module: Any, result: ModuleAuditResult):
        """Auditoria específica baseada no tipo de módulo."""
        
        if module_id == "1/8":  # CORE
            await self.audit_core_module(module, result)
        elif module_id == "2/8":  # STRATEGY
            await self.audit_strategy_module(module, result)
        elif module_id == "3/8":  # ACQUISITION
            await self.audit_acquisition_module(module, result)
        elif module_id == "4/8":  # MUTATION
            await self.audit_mutation_module(module, result)
        elif module_id == "5/8":  # CRUCIBLE
            await self.audit_crucible_module(module, result)
        elif module_id == "6/8":  # AUTOREWRITE
            await self.audit_autorewrite_module(module, result)
        elif module_id == "7/8":  # NEXUS
            await self.audit_nexus_module(module, result)
        elif module_id == "8/8":  # GOVERNANCE
            await self.audit_governance_module(module, result)
    
    async def audit_core_module(self, module: Any, result: ModuleAuditResult):
        """Auditoria específica do módulo CORE (1/8)."""
        
        # Verifica classes essenciais
        essential_classes = ["Candidate", "PlanOmega", "OmegaState"]
        
        for class_name in essential_classes:
            if hasattr(module, class_name):
                result.findings.append(AuditFinding(
                    component="1/8",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title=f"Classe {class_name} Presente",
                    description=f"Classe essencial {class_name} está implementada"
                ))
            else:
                result.findings.append(AuditFinding(
                    component="1/8",
                    category="HIGH",
                    finding_type="BUG",
                    title=f"Classe {class_name} Ausente",
                    description=f"Classe essencial {class_name} não encontrada",
                    recommendation=f"Implementar classe {class_name}"
                ))
        
        # Verifica funções de inicialização
        if hasattr(module, 'initialize_omega_state'):
            result.findings.append(AuditFinding(
                component="1/8",
                category="INFO",
                finding_type="COMPLIANCE",
                title="Inicialização de Estado Presente",
                description="Função de inicialização de estado implementada"
            ))
        else:
            result.findings.append(AuditFinding(
                component="1/8",
                category="MEDIUM",
                finding_type="BUG",
                title="Inicialização de Estado Ausente",
                description="Função de inicialização não encontrada",
                recommendation="Implementar função initialize_omega_state"
            ))
    
    async def audit_strategy_module(self, module: Any, result: ModuleAuditResult):
        """Auditoria específica do módulo STRATEGY (2/8)."""
        
        # Verifica estratégias implementadas
        strategy_functions = ["strategy_f2", "strategy_decision", "strategy_evaluation"]
        
        for func_name in strategy_functions:
            if hasattr(module, func_name):
                result.findings.append(AuditFinding(
                    component="2/8",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title=f"Estratégia {func_name} Implementada",
                    description=f"Função de estratégia {func_name} presente"
                ))
            else:
                result.findings.append(AuditFinding(
                    component="2/8",
                    category="MEDIUM",
                    finding_type="BUG",
                    title=f"Estratégia {func_name} Ausente",
                    description=f"Função {func_name} não encontrada",
                    recommendation=f"Implementar função {func_name}"
                ))
    
    async def audit_acquisition_module(self, module: Any, result: ModuleAuditResult):
        """Auditoria específica do módulo ACQUISITION (3/8)."""
        
        # Verifica funções de aquisição
        acquisition_functions = ["acquisition_f3", "acquire_candidates", "validate_acquisition"]
        
        for func_name in acquisition_functions:
            if hasattr(module, func_name):
                result.findings.append(AuditFinding(
                    component="3/8",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title=f"Aquisição {func_name} Implementada",
                    description=f"Função de aquisição {func_name} presente"
                ))
    
    async def audit_mutation_module(self, module: Any, result: ModuleAuditResult):
        """Auditoria específica do módulo MUTATION (4/8)."""
        
        # Verifica funções de mutação
        mutation_functions = ["mutation_f4", "mutate_candidates", "validate_mutations"]
        
        for func_name in mutation_functions:
            if hasattr(module, func_name):
                result.findings.append(AuditFinding(
                    component="4/8",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title=f"Mutação {func_name} Implementada",
                    description=f"Função de mutação {func_name} presente"
                ))
    
    async def audit_crucible_module(self, module: Any, result: ModuleAuditResult):
        """Auditoria específica do módulo CRUCIBLE (5/8)."""
        
        # Verifica funções do crucible
        crucible_functions = ["crucible_f5", "evaluate_candidates", "select_best"]
        
        for func_name in crucible_functions:
            if hasattr(module, func_name):
                result.findings.append(AuditFinding(
                    component="5/8",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title=f"Crucible {func_name} Implementado",
                    description=f"Função do crucible {func_name} presente"
                ))
        
        # Verifica se usa classes unificadas
        if hasattr(module, 'Candidate'):
            # Tenta instanciar para verificar compatibilidade
            try:
                test_candidate = module.Candidate(
                    id="test",
                    content="test content",
                    score=0.5,
                    metadata={}
                )
                result.findings.append(AuditFinding(
                    component="5/8",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title="Classes Unificadas Compatíveis",
                    description="Módulo usa classes unificadas corretamente"
                ))
            except Exception as e:
                result.findings.append(AuditFinding(
                    component="5/8",
                    category="HIGH",
                    finding_type="BUG",
                    title="Incompatibilidade de Classes",
                    description=f"Erro ao usar classes unificadas: {str(e)}",
                    recommendation="Atualizar para usar classes unificadas"
                ))
    
    async def audit_autorewrite_module(self, module: Any, result: ModuleAuditResult):
        """Auditoria específica do módulo AUTOREWRITE (6/8)."""
        
        # Verifica funções de autorewrite
        autorewrite_functions = ["autorewrite_f6", "rewrite_code", "validate_rewrite"]
        
        for func_name in autorewrite_functions:
            if hasattr(module, func_name):
                result.findings.append(AuditFinding(
                    component="6/8",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title=f"Autorewrite {func_name} Implementado",
                    description=f"Função de autorewrite {func_name} presente"
                ))
    
    async def audit_nexus_module(self, module: Any, result: ModuleAuditResult):
        """Auditoria específica do módulo NEXUS (7/8)."""
        
        # Verifica componentes do nexus
        nexus_components = ["NexusScheduler", "Watchdog", "CanarySystem"]
        
        for component_name in nexus_components:
            if hasattr(module, component_name):
                result.findings.append(AuditFinding(
                    component="7/8",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title=f"Componente {component_name} Presente",
                    description=f"Componente do nexus {component_name} implementado"
                ))
            else:
                result.findings.append(AuditFinding(
                    component="7/8",
                    category="HIGH",
                    finding_type="BUG",
                    title=f"Componente {component_name} Ausente",
                    description=f"Componente crítico {component_name} não encontrado",
                    recommendation=f"Implementar componente {component_name}"
                ))
    
    async def audit_governance_module(self, module: Any, result: ModuleAuditResult):
        """Auditoria específica do módulo GOVERNANCE (8/8)."""
        
        # Verifica componentes de governança
        governance_components = ["GovernanceHub", "SecurityGates", "ComplianceMonitor"]
        
        for component_name in governance_components:
            if hasattr(module, component_name):
                result.findings.append(AuditFinding(
                    component="8/8",
                    category="INFO",
                    finding_type="COMPLIANCE",
                    title=f"Governança {component_name} Presente",
                    description=f"Componente de governança {component_name} implementado"
                ))
        
        # Verifica integração com WORM ledger
        if hasattr(module, 'worm_ledger') or hasattr(module, 'security_governance'):
            result.findings.append(AuditFinding(
                component="8/8",
                category="INFO",
                finding_type="COMPLIANCE",
                title="Integração WORM Presente",
                description="Módulo integrado com sistema WORM"
            ))
        else:
            result.findings.append(AuditFinding(
                component="8/8",
                category="HIGH",
                finding_type="BUG",
                title="Integração WORM Ausente",
                description="Módulo não integrado com sistema WORM",
                recommendation="Implementar integração com WORM ledger"
            ))
    
    async def audit_module_compliance(self, module_id: str, module: Any, result: ModuleAuditResult):
        """Auditoria de conformidade geral do módulo."""
        
        # Verifica docstring
        if module.__doc__:
            result.findings.append(AuditFinding(
                component=module_id,
                category="INFO",
                finding_type="COMPLIANCE",
                title="Documentação Presente",
                description="Módulo possui docstring"
            ))
        else:
            result.findings.append(AuditFinding(
                component=module_id,
                category="LOW",
                finding_type="IMPROVEMENT",
                title="Documentação Ausente",
                description="Módulo não possui docstring",
                recommendation="Adicionar documentação ao módulo"
            ))
        
        # Verifica logging
        module_members = inspect.getmembers(module)
        has_logging = any("log" in name.lower() for name, _ in module_members)
        
        if has_logging:
            result.findings.append(AuditFinding(
                component=module_id,
                category="INFO",
                finding_type="COMPLIANCE",
                title="Sistema de Logging Presente",
                description="Módulo implementa logging"
            ))
        else:
            result.findings.append(AuditFinding(
                component=module_id,
                category="MEDIUM",
                finding_type="IMPROVEMENT",
                title="Sistema de Logging Ausente",
                description="Módulo não implementa logging",
                recommendation="Implementar logging estruturado"
            ))
        
        # Verifica tratamento de erros
        try:
            source_file = inspect.getsourcefile(module)
            if source_file:
                with open(source_file, 'r') as f:
                    source_code = f.read()
                
                has_try_except = "try:" in source_code and "except" in source_code
                
                if has_try_except:
                    result.findings.append(AuditFinding(
                        component=module_id,
                        category="INFO",
                        finding_type="COMPLIANCE",
                        title="Tratamento de Erros Presente",
                        description="Módulo implementa tratamento de erros"
                    ))
                else:
                    result.findings.append(AuditFinding(
                        component=module_id,
                        category="MEDIUM",
                        finding_type="IMPROVEMENT",
                        title="Tratamento de Erros Limitado",
                        description="Módulo tem tratamento de erros limitado",
                        recommendation="Implementar tratamento robusto de erros"
                    ))
        except Exception:
            # Não conseguiu analisar o código fonte
            pass

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

module_auditor = ModuleAuditor()

if __name__ == "__main__":
    async def main():
        results = await module_auditor.audit_individual_modules()
        for module_id, result in results.items():
            print(f"\n=== {module_id}: {result.module_name} ===")
            print(f"Exists: {result.exists}")
            print(f"Importable: {result.importable}")
            print(f"Functional: {result.functional}")
            print(f"Compliance Score: {result.compliance_score:.2f}")
            print(f"Findings: {len(result.findings)}")
    
    asyncio.run(main())
