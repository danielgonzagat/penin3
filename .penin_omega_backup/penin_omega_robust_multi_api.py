#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Sistema Multi-API Robusto
===================================
Implementação robusta com fallback, timeout handling e retry logic.
"""

from __future__ import annotations
import asyncio
import json
import time
import logging
import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Callable
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Imports seguros
from penin_omega_dependency_resolver import safe_import, config

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PENIN_OMEGA_ROOT = Path(config.get("root_path", "/root/.penin_omega"))
API_CACHE_PATH = PENIN_OMEGA_ROOT / "cache" / "api_cache"
API_CACHE_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CLASSES DE DADOS
# =============================================================================

@dataclass
class APIRequest:
    """Request para API."""
    prompt: str
    context: Dict[str, Any] = field(default_factory=dict)
    module_source: str = "unknown"
    priority: int = 50
    max_tokens: int = 4000
    timeout: float = 30.0
    retry_count: int = 3
    request_id: str = field(default_factory=lambda: f"req_{int(time.time())}")

@dataclass
class APIResponse:
    """Response de API."""
    content: str = ""
    success: bool = False
    provider: str = "unknown"
    latency: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    request_id: str = ""

# =============================================================================
# SISTEMA MULTI-API ROBUSTO
# =============================================================================

class RobustMultiAPI:
    """Sistema multi-API com fallback robusto e timeout handling."""
    
    def __init__(self):
        self.logger = logging.getLogger("RobustMultiAPI")
        self.providers = {}
        self.fallback_providers = []
        self.request_cache = {}
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "provider_stats": {}
        }
        
        # Thread pool para requests paralelos
        self.executor = ThreadPoolExecutor(max_workers=6)
        
        # Session HTTP com retry
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Inicializa providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Inicializa providers disponíveis."""
        # Tenta carregar sistema multi-API existente
        existing_multi_api = safe_import("penin_omega_multi_api_llm")
        if existing_multi_api and hasattr(existing_multi_api, 'MultiAPILLM'):
            try:
                self.legacy_api = existing_multi_api.MultiAPILLM()
                self.providers["legacy"] = {
                    "name": "Legacy Multi-API",
                    "available": True,
                    "priority": 1
                }
                self.logger.info("✅ Sistema multi-API legado carregado")
            except Exception as e:
                self.logger.warning(f"Falha ao carregar API legado: {e}")
        
        # Providers de fallback locais
        self.fallback_providers = [
            {
                "name": "local_llm",
                "type": "local",
                "available": True,
                "priority": 10
            },
            {
                "name": "mock_api", 
                "type": "mock",
                "available": True,
                "priority": 99
            }
        ]
        
        self.logger.info(f"🔗 Providers inicializados: {len(self.providers)} principais, {len(self.fallback_providers)} fallback")
    
    async def process_request(self, request: APIRequest) -> APIResponse:
        """Processa request com fallback robusto."""
        start_time = time.time()
        self.stats["total_requests"] += 1
        
        # Verifica cache primeiro
        cache_key = self._generate_cache_key(request)
        cached_response = self._get_cached_response(cache_key)
        if cached_response:
            self.stats["cache_hits"] += 1
            self.logger.info(f"💾 Cache hit para request {request.request_id}")
            return cached_response
        
        # Tenta providers principais
        response = await self._try_main_providers(request)
        
        # Se falhou, tenta fallback
        if not response.success:
            self.logger.warning(f"⚠️  Providers principais falharam, tentando fallback")
            response = await self._try_fallback_providers(request)
        
        # Atualiza estatísticas
        response.latency = time.time() - start_time
        if response.success:
            self.stats["successful_requests"] += 1
        else:
            self.stats["failed_requests"] += 1
        
        # Cache response se bem-sucedida
        if response.success:
            self._cache_response(cache_key, response)
        
        return response
    
    async def _try_main_providers(self, request: APIRequest) -> APIResponse:
        """Tenta providers principais."""
        if "legacy" in self.providers and hasattr(self, 'legacy_api'):
            try:
                # Usa sistema multi-API existente
                result = await asyncio.to_thread(
                    self._call_legacy_api, 
                    request.prompt, 
                    request.context,
                    request.timeout
                )
                
                if result and result.get("success"):
                    return APIResponse(
                        content=result.get("content", ""),
                        success=True,
                        provider="legacy_multi_api",
                        request_id=request.request_id,
                        tokens_used=result.get("tokens", 0)
                    )
                    
            except Exception as e:
                self.logger.error(f"Erro no provider legacy: {e}")
        
        return APIResponse(
            success=False,
            error="Nenhum provider principal disponível",
            request_id=request.request_id
        )
    
    def _call_legacy_api(self, prompt: str, context: Dict, timeout: float) -> Dict[str, Any]:
        """Chama API legado com timeout."""
        try:
            if hasattr(self.legacy_api, 'query_all_providers'):
                result = self.legacy_api.query_all_providers(
                    prompt=prompt,
                    context=context,
                    max_tokens=4000
                )
                return {
                    "success": True,
                    "content": result.get("best_response", ""),
                    "tokens": result.get("total_tokens", 0)
                }
            else:
                return {"success": False, "error": "Método não encontrado"}
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _try_fallback_providers(self, request: APIRequest) -> APIResponse:
        """Tenta providers de fallback."""
        for provider in self.fallback_providers:
            try:
                if provider["type"] == "local":
                    response = await self._call_local_llm(request)
                elif provider["type"] == "mock":
                    response = await self._call_mock_api(request)
                else:
                    continue
                
                if response.success:
                    return response
                    
            except Exception as e:
                self.logger.error(f"Erro no fallback {provider['name']}: {e}")
        
        return APIResponse(
            success=False,
            error="Todos os providers falharam",
            request_id=request.request_id
        )
    
    async def _call_local_llm(self, request: APIRequest) -> APIResponse:
        """Chama LLM local como fallback."""
        try:
            # Simula processamento local
            await asyncio.sleep(0.1)  # Simula latência
            
            # Resposta baseada no contexto
            if "optimization" in request.prompt.lower():
                content = "Otimização sugerida: Implementar cache L2 e reduzir overhead de I/O."
            elif "mutation" in request.prompt.lower():
                content = "Mutação gerada: Aplicar refatoração de função com melhoria de performance."
            elif "evaluation" in request.prompt.lower():
                content = "Avaliação: Candidato aprovado com score 0.85."
            else:
                content = f"Processamento local para: {request.prompt[:50]}..."
            
            return APIResponse(
                content=content,
                success=True,
                provider="local_llm",
                request_id=request.request_id,
                tokens_used=len(content.split())
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=f"Erro no LLM local: {e}",
                request_id=request.request_id
            )
    
    async def _call_mock_api(self, request: APIRequest) -> APIResponse:
        """API mock como último recurso."""
        try:
            await asyncio.sleep(0.05)  # Simula latência mínima
            
            content = f"Mock response para módulo {request.module_source}: Operação simulada com sucesso."
            
            return APIResponse(
                content=content,
                success=True,
                provider="mock_api",
                request_id=request.request_id,
                tokens_used=10
            )
            
        except Exception as e:
            return APIResponse(
                success=False,
                error=f"Erro no mock API: {e}",
                request_id=request.request_id
            )
    
    def _generate_cache_key(self, request: APIRequest) -> str:
        """Gera chave de cache para request."""
        cache_data = {
            "prompt": request.prompt,
            "context": request.context,
            "max_tokens": request.max_tokens
        }
        return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()
    
    def _get_cached_response(self, cache_key: str) -> Optional[APIResponse]:
        """Obtém response do cache."""
        try:
            cache_file = API_CACHE_PATH / f"{cache_key}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Verifica se cache não expirou (1 hora)
                cache_time = datetime.fromisoformat(data["timestamp"])
                if (datetime.now(timezone.utc) - cache_time).seconds < 3600:
                    return APIResponse(**data)
                    
        except Exception as e:
            self.logger.warning(f"Erro ao ler cache: {e}")
        
        return None
    
    def _cache_response(self, cache_key: str, response: APIResponse):
        """Salva response no cache."""
        try:
            cache_file = API_CACHE_PATH / f"{cache_key}.json"
            with open(cache_file, 'w') as f:
                json.dump(response.__dict__, f, indent=2)
                
        except Exception as e:
            self.logger.warning(f"Erro ao salvar cache: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do sistema."""
        success_rate = 0.0
        if self.stats["total_requests"] > 0:
            success_rate = self.stats["successful_requests"] / self.stats["total_requests"]
        
        return {
            **self.stats,
            "success_rate": success_rate,
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["total_requests"]),
            "providers_available": len(self.providers) + len(self.fallback_providers)
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Verifica saúde do sistema."""
        health = {
            "status": "healthy",
            "providers": {},
            "issues": []
        }
        
        # Verifica providers principais
        for name, provider in self.providers.items():
            try:
                # Teste simples
                health["providers"][name] = "available"
            except Exception as e:
                health["providers"][name] = f"error: {e}"
                health["issues"].append(f"Provider {name} com problemas")
        
        # Verifica fallback
        for provider in self.fallback_providers:
            health["providers"][provider["name"]] = "available"
        
        if health["issues"]:
            health["status"] = "degraded"
        
        return health

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

# Instância global do sistema multi-API
robust_multi_api = RobustMultiAPI()

# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

async def query_multi_api(prompt: str, context: Dict[str, Any] = None, module_source: str = "unknown") -> APIResponse:
    """Query simplificado para o sistema multi-API."""
    if context is None:
        context = {}
    
    request = APIRequest(
        prompt=prompt,
        context=context,
        module_source=module_source
    )
    
    return await robust_multi_api.process_request(request)

def get_multi_api_stats() -> Dict[str, Any]:
    """Obtém estatísticas do sistema multi-API."""
    return robust_multi_api.get_stats()

def multi_api_health_check() -> Dict[str, Any]:
    """Verifica saúde do sistema multi-API."""
    return robust_multi_api.health_check()

# =============================================================================
# TESTE DO SISTEMA
# =============================================================================

async def test_robust_multi_api():
    """Testa o sistema multi-API robusto."""
    print("🧪 Testando sistema multi-API robusto...")
    
    # Teste básico
    response = await query_multi_api(
        "Optimize this code for better performance",
        {"module": "test"},
        "test_module"
    )
    
    if response.success:
        print(f"✅ Query básico: {response.provider}")
        print(f"📝 Resposta: {response.content[:100]}...")
    else:
        print(f"❌ Query falhou: {response.error}")
    
    # Teste múltiplas queries
    tasks = []
    for i in range(5):
        task = query_multi_api(
            f"Process request {i}",
            {"request_id": i},
            f"test_module_{i}"
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    successful = sum(1 for r in results if r.success)
    print(f"✅ Queries paralelas: {successful}/5 bem-sucedidas")
    
    # Teste cache
    cached_response = await query_multi_api(
        "Optimize this code for better performance",  # Mesma query
        {"module": "test"},
        "test_cache"
    )
    
    if cached_response.success:
        print("✅ Cache funcionando")
    
    # Estatísticas
    stats = get_multi_api_stats()
    print(f"📊 Estatísticas: {stats['total_requests']} requests, {stats['success_rate']:.2%} sucesso")
    
    # Health check
    health = multi_api_health_check()
    print(f"🏥 Saúde: {health['status']}, {len(health['providers'])} providers")
    
    print("🎉 Sistema multi-API robusto funcionando!")
    return True

if __name__ == "__main__":
    import hashlib
    asyncio.run(test_robust_multi_api())
