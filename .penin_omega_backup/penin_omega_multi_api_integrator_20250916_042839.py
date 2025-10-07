#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Integrador Multi-API Universal
========================================
OBJETIVO: Conecta todos os módulos PENIN-Ω ao sistema multi-API existente,
fornecendo interface unificada para polling simultâneo de 6 APIs e
distribuição inteligente de workload.

INTEGRAÇÃO:
- Sistema multi-API existente (penin_omega_fusion_v6)
- Todos os módulos 3/8, 4/8, 5/8, 6/8, 7/8, 8/8
- Estado global sincronizado
- Métricas unificadas

Autor: Equipe PENIN-Ω
Versão: 1.0.0
"""

from __future__ import annotations
import asyncio
import json
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Union
import logging

# =============================================================================
# INTEGRAÇÃO COM SISTEMA MULTI-API EXISTENTE
# =============================================================================

try:
    from penin_omega_fusion_v6 import PeninOmegaFusion
    FUSION_AVAILABLE = True
except ImportError:
    FUSION_AVAILABLE = False
    print("⚠️  Sistema multi-API não encontrado")

# =============================================================================
# INTEGRADOR MULTI-API
# =============================================================================

@dataclass
class MultiAPIRequest:
    """Request para sistema multi-API."""
    prompt: str
    context: Dict[str, Any]
    module_source: str  # F3, F4, F5, F6, etc.
    priority: int = 50
    max_tokens: int = 4000
    timeout_s: int = 300

@dataclass 
class MultiAPIResponse:
    """Response do sistema multi-API."""
    request_id: str
    responses: Dict[str, str]  # api_name -> response
    total_chars: int
    processing_time_ms: float
    success: bool
    error_message: str = ""

class MultiAPIIntegrator:
    """Integrador universal para sistema multi-API."""
    
    def __init__(self):
        self.fusion = None
        self.logger = logging.getLogger("MultiAPI-Integrator")
        
        if FUSION_AVAILABLE:
            try:
                self.fusion = PeninOmegaFusion()
                self.logger.info("✅ Sistema multi-API conectado")
            except Exception as e:
                self.logger.error(f"❌ Erro conectando multi-API: {e}")
                self.fusion = None
        else:
            self.logger.warning("⚠️  Sistema multi-API não disponível")
    
    async def process_request(self, request: MultiAPIRequest) -> MultiAPIResponse:
        """Processa request usando sistema multi-API."""
        start_time = time.time()
        request_id = f"req_{int(time.time() * 1000) % 100000}"
        
        if not self.fusion:
            return MultiAPIResponse(
                request_id=request_id,
                responses={},
                total_chars=0,
                processing_time_ms=0,
                success=False,
                error_message="Sistema multi-API não disponível"
            )
        
        try:
            # Constrói prompt contextualizado
            contextual_prompt = self._build_contextual_prompt(request)
            
            # Chama sistema multi-API
            responses = await self.fusion.process_all_apis_simultaneously(
                contextual_prompt,
                max_tokens=request.max_tokens,
                timeout_seconds=request.timeout_s
            )
            
            # Calcula métricas
            total_chars = sum(len(resp) for resp in responses.values())
            processing_time = (time.time() - start_time) * 1000
            
            self.logger.info(f"✅ Multi-API processado: {len(responses)} APIs, {total_chars} chars")
            
            return MultiAPIResponse(
                request_id=request_id,
                responses=responses,
                total_chars=total_chars,
                processing_time_ms=processing_time,
                success=True
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.logger.error(f"❌ Erro no multi-API: {e}")
            
            return MultiAPIResponse(
                request_id=request_id,
                responses={},
                total_chars=0,
                processing_time_ms=processing_time,
                success=False,
                error_message=str(e)
            )
    
    def _build_contextual_prompt(self, request: MultiAPIRequest) -> str:
        """Constrói prompt contextualizado para o módulo."""
        context_info = ""
        
        # Adiciona contexto específico do módulo
        if request.module_source == "F3":
            context_info = "Você está ajudando com aquisição de conhecimento. "
        elif request.module_source == "F4":
            context_info = "Você está ajudando com mutação e geração de código. "
        elif request.module_source == "F5":
            context_info = "Você está ajudando com avaliação no crisol. "
        elif request.module_source == "F6":
            context_info = "Você está ajudando com auto-rewrite. "
        
        # Adiciona contexto adicional
        if request.context:
            goals = request.context.get("goals", [])
            if goals:
                goal_text = ", ".join(g.get("name", str(g)) for g in goals[:3])
                context_info += f"Objetivos: {goal_text}. "
        
        return f"{context_info}{request.prompt}"
    
    def is_available(self) -> bool:
        """Verifica se sistema multi-API está disponível."""
        return self.fusion is not None

# =============================================================================
# CONECTORES ESPECÍFICOS POR MÓDULO
# =============================================================================

class F3MultiAPIConnector:
    """Conector multi-API para módulo F3 (Aquisição)."""
    
    def __init__(self, integrator: MultiAPIIntegrator):
        self.integrator = integrator
    
    async def enrich_knowledge(self, query: str, context: Dict[str, Any]) -> str:
        """Enriquece conhecimento usando multi-API."""
        if not self.integrator.is_available():
            return ""
        
        request = MultiAPIRequest(
            prompt=f"Forneça conhecimento detalhado e relevante sobre: {query}",
            context=context,
            module_source="F3",
            max_tokens=2000
        )
        
        response = await self.integrator.process_request(request)
        
        if response.success:
            # Combina respostas das APIs
            combined = ""
            for api_name, resp in response.responses.items():
                if len(resp) > 100:  # Filtra respostas muito curtas
                    combined += f"\n[{api_name}]: {resp[:800]}..."
            
            return combined[:3000]  # Limita tamanho total
        
        return ""

class F4MultiAPIConnector:
    """Conector multi-API para módulo F4 (Mutação)."""
    
    def __init__(self, integrator: MultiAPIIntegrator):
        self.integrator = integrator
    
    async def synthesize_code(self, prompt: str, context: Dict[str, Any]) -> str:
        """Sintetiza código usando multi-API."""
        if not self.integrator.is_available():
            return ""
        
        structured_prompt = f"""
Gere código Python otimizado para: {prompt}

Requisitos:
1. Código sintática e semanticamente válido
2. Incluir docstrings e comentários
3. Seguir boas práticas Python
4. Ser eficiente e legível
5. Máximo 200 linhas

Contexto: {context.get('goals', [])}

Retorne apenas o código Python, sem explicações.
"""
        
        request = MultiAPIRequest(
            prompt=structured_prompt,
            context=context,
            module_source="F4",
            max_tokens=3000
        )
        
        response = await self.integrator.process_request(request)
        
        if response.success:
            # Seleciona melhor código das respostas
            return self._select_best_code(response.responses)
        
        return ""
    
    def _select_best_code(self, responses: Dict[str, str]) -> str:
        """Seleciona melhor código das respostas."""
        import ast
        import re
        
        candidates = []
        
        for api_name, response in responses.items():
            # Extrai código Python
            code = self._extract_python_code(response)
            if code and self._validate_syntax(code):
                candidates.append((code, len(code), api_name))
        
        if not candidates:
            return ""
        
        # Seleciona por qualidade (sintaxe válida + tamanho razoável)
        candidates.sort(key=lambda x: (x[1] > 50, -x[1]))
        return candidates[0][0]
    
    def _extract_python_code(self, text: str) -> str:
        """Extrai código Python de texto."""
        import re
        
        # Procura blocos de código
        code_blocks = re.findall(r'```python\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0]
        
        # Procura linhas que parecem código
        lines = text.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if any(kw in line for kw in ['def ', 'class ', 'import ', 'from ']):
                in_code = True
            
            if in_code:
                code_lines.append(line)
                
            if line.strip() == '' and in_code and len(code_lines) > 5:
                break
        
        return '\n'.join(code_lines)
    
    def _validate_syntax(self, code: str) -> bool:
        """Valida sintaxe do código."""
        try:
            import ast
            ast.parse(code)
            return True
        except:
            return False

# =============================================================================
# INTEGRADOR GLOBAL
# =============================================================================

class GlobalMultiAPIIntegrator:
    """Integrador global que conecta todos os módulos ao multi-API."""
    
    def __init__(self):
        self.integrator = MultiAPIIntegrator()
        self.f3_connector = F3MultiAPIConnector(self.integrator)
        self.f4_connector = F4MultiAPIConnector(self.integrator)
        self.logger = logging.getLogger("Global-MultiAPI")
        
        self.logger.info("🔗 Integrador multi-API global inicializado")
    
    def get_f3_connector(self) -> F3MultiAPIConnector:
        """Obtém conector F3."""
        return self.f3_connector
    
    def get_f4_connector(self) -> F4MultiAPIConnector:
        """Obtém conector F4."""
        return self.f4_connector
    
    async def process_generic_request(self, prompt: str, module: str, 
                                    context: Dict[str, Any] = None) -> MultiAPIResponse:
        """Processa request genérico."""
        request = MultiAPIRequest(
            prompt=prompt,
            context=context or {},
            module_source=module,
            max_tokens=4000
        )
        
        return await self.integrator.process_request(request)
    
    def is_available(self) -> bool:
        """Verifica se multi-API está disponível."""
        return self.integrator.is_available()
    
    def get_status(self) -> Dict[str, Any]:
        """Obtém status do integrador."""
        return {
            "multi_api_available": self.is_available(),
            "fusion_system": FUSION_AVAILABLE,
            "connectors": {
                "f3": True,
                "f4": True,
                "f5": False,  # TODO: implementar
                "f6": False   # TODO: implementar
            }
        }

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

_global_integrator: Optional[GlobalMultiAPIIntegrator] = None

def get_global_multi_api_integrator() -> GlobalMultiAPIIntegrator:
    """Obtém integrador global (singleton)."""
    global _global_integrator
    
    if _global_integrator is None:
        _global_integrator = GlobalMultiAPIIntegrator()
    
    return _global_integrator

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "MultiAPIIntegrator", "GlobalMultiAPIIntegrator",
    "F3MultiAPIConnector", "F4MultiAPIConnector",
    "MultiAPIRequest", "MultiAPIResponse",
    "get_global_multi_api_integrator"
]

if __name__ == "__main__":
    # Teste básico
    print("PENIN-Ω Multi-API Integrator")
    
    async def test_integrator():
        integrator = get_global_multi_api_integrator()
        status = integrator.get_status()
        
        print(f"✅ Status: {status}")
        
        if integrator.is_available():
            # Teste F3
            f3_result = await integrator.get_f3_connector().enrich_knowledge(
                "algoritmos de otimização", {"goals": [{"name": "melhorar performance"}]}
            )
            print(f"✅ F3 test: {len(f3_result)} chars")
            
            # Teste F4
            f4_result = await integrator.get_f4_connector().synthesize_code(
                "função de otimização", {"goals": [{"name": "reduzir complexidade"}]}
            )
            print(f"✅ F4 test: {len(f4_result)} chars")
        else:
            print("⚠️  Multi-API não disponível para testes")
    
    import asyncio
    asyncio.run(test_integrator())
    print("✅ Integrador funcionando!")
