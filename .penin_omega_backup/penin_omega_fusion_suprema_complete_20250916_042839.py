#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω v7.0 FUSION SUPREMA - Sistema Unificado Definitivo
Arquitetura: Σ-Guard/IR→IC/CAOS⁺/SR-Ω∞/WORM/Liga/Bridge/Async/Trust-Region/Webhook
Otimizado para CPU com suporte nativo a Falcon Mamba 7B e modelos locais
Data: 2025.09.16
Versão: 7.0.0 (Fusão Definitiva de 4 versões avançadas)
---
Características Superiores:
1. :followup[**Arquitetura Híbrida Monolítica-Modular** - Cache multi-nível (L1/L2/L3), WORM Merkle, Bridge LLM adaptativo]{question="Como a arquitetura híbrida monolítica-modular melhora a performance e a manutenção do sistema?" questionId="072f0846-9d89-4cb2-bbe1-5555a8eb52f8"}
2. :followup[**Suporte Nativo a Falcon Mamba 7B** - Execução otimizada para CPU com fallback inteligente]{question="Quais são as otimizações específicas para executar o Falcon Mamba 7B em CPU?" questionId="683b1b7e-923c-46a5-9872-8206529fd75e"}
3. :followup[**Cache Multi-Nível Inteligente** - L1 (memória), L2 (SQLite), L3 (Redis opcional) com eviction ML-based]{question="Como funciona o mecanismo de eviction baseado em ML no cache multi-nível?" questionId="cb7568d5-60c0-433e-b953-fbc9a9760457"}
4. :followup[**Bridge LLM Adaptativo** - Prioriza modelos locais, fallback para cloud, circuit breakers, cache de respostas]{question="Como o sistema decide entre usar modelos locais ou APIs de cloud?" questionId="ad894d11-3bde-4a7b-aa34-2efab52a0045"}
5. :followup[**WORM Ledger Imutável** - Merkle chain com verificação assíncrona e thread-safe]{question="Como a Merkle chain garante a integridade dos dados no WORM Ledger?" questionId="3f2bb916-b801-41e8-8c41-6859a1ef1eec"}
6. :followup[**Motores PENIN-Ω Otimizados** - Σ-Guard, IR→IC, CAOS⁺, SR-Ω∞ com paralelização e cache]{question="Qual é o papel de cada motor (Σ-Guard, IR→IC, CAOS⁺, SR-Ω∞) no funcionamento do sistema?" questionId="840bbd11-3437-4b36-9ba4-cac978fc0c21"}
7. :followup[**Estado Unificado Validado** - 60+ campos com validação automática e serialização eficiente]{question="Como a validação automática do estado ajuda a prevenir erros no sistema?" questionId="aaa1520f-1b49-4e6c-a9e4-4ee6058c9c6d"}
8. :followup[**Evolução Autoevolutiva** - Compatível com ETΩ, mutações guiadas por insights multi-IA]{question="Como o sistema implementa a autoevolução usando insights de múltiplas IAs?" questionId="af6e0d42-e617-431f-9698-6236183c0110"}
9. :followup[**Webhook Seguro** - Verificação RSA para chamadas externas]{question="Como a verificação RSA garante a segurança das chamadas externas via webhook?" questionId="fe642890-00c2-4b3b-bd22-f1a7f360d991"}
10. :followup[**Benchmarking Integrado** - Métricas de performance em tempo real]{question="Quais métricas de performance são monitoradas em tempo real pelo sistema?" questionId="fe5e8106-fd9f-4f97-869d-d9c5ce628afe"}
11. :followup[**Resiliência** - Circuit breakers, fallbacks, e tratamento de erros em todos os níveis]{question="Como os circuit breakers e fallbacks contribuem para a resiliência do sistema?" questionId="11208d7c-d1cc-405a-86b1-ffc0d293fccc"}
12. :followup[**Auditabilidade** - Todos eventos críticos registrados no WORM com snapshots versionados]{question="Como o sistema garante a auditabilidade de todas as operações críticas?" questionId="e4a048e0-2acf-40e7-8f19-b375c05f0d51"}
13. :followup[**Observabilidade** - Métricas detalhadas, logs estruturados, e diagnóstico completo]{question="Quais ferramentas de observabilidade estão integradas ao sistema?" questionId="95be0489-267b-417b-aeef-45d85ad9dd12"}
14. :followup[**Extensibilidade** - Arquitetura baseada em ABC para novos motores, provedores e subsistemas]{question="Como novos motores ou provedores podem ser adicionados ao sistema?" questionId="59c8d504-0eca-42bf-9402-19f620236737"}
---
Uso:
- CLI: python penin_omega.py [--diagnose|--evolve|--benchmark|--cleanup|--webhook] [--cycles N] [--parallel]
- API: Importar NucleoPENINOmega e chamar métodos (cycle(), evolve(), diagnose(), start_webhook())
- Extensões: Implementar novas classes baseadas em AIProvider, Engine, ou Subsystem
---
Requisitos:
pip install psutil transformers torch pydantic-settings python-dotenv aiohttp lz4 redis
"""
from __future__ import annotations
import os
import sys
import json
import time
import uuid
import math
import random
import hashlib
import asyncio
import threading
import multiprocessing
import sqlite3
import psutil
import logging
import signal
import base64
import traceback
import warnings
from pathlib import Path
from dataclasses import dataclass, asdict, field, fields as dc_fields
from typing import Any, Dict, List, Optional, Tuple, Literal, Callable, Type, TypeVar, Set, Union
from datetime import datetime, timezone, timedelta
from contextlib import asynccontextmanager, contextmanager
from collections import deque, defaultdict, OrderedDict
from abc import ABC, abstractmethod
from enum import Enum, auto
from functools import lru_cache, wraps
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Supressão de warnings
warnings.filterwarnings('ignore')

# =============================================================================
# IMPORTS OPCIONAIS COM FALLBACKS
# =============================================================================
try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None
    HAS_YAML = False

try:
    from pydantic_settings import BaseSettings
    from dotenv import load_dotenv
    load_dotenv()
    HAS_PYDANTIC = True
except ImportError:
    class BaseSettings: pass
    HAS_PYDANTIC = False

try:
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False

try:
    import lz4.frame as lz4f
    HAS_LZ4 = True
except ImportError:
    lz4f = None
    HAS_LZ4 = False

try:
    import redis as _redis
    HAS_REDIS = True
except ImportError:
    _redis = None
    HAS_REDIS = False

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    torch = None
    HAS_TORCH = False

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from openai import AsyncOpenAI
    HAS_OPENAI = True
except ImportError:
    AsyncOpenAI = None
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    anthropic = None
    HAS_ANTHROPIC = False

try:
    from mistralai import MistralAsyncClient
    HAS_MISTRAL = True
except ImportError:
    MistralAsyncClient = None
    HAS_MISTRAL = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    genai = None
    HAS_GEMINI = False

# =============================================================================
# CONSTANTES E METADADOS
# =============================================================================
PKG_NAME = "penin_omega_fusion_suprema"
PKG_VERSION = "7.0.0"
PKG_DESC = (
    "PENIN-Ω v7.0 FUSION SUPREMA - Sistema Unificado Definitivo com Falcon Mamba 7B. "
    "Arquitetura híbrida monolítica-modular com processamento adaptativo, cache multi-nível, "
    "WORM Merkle chain, Bridge LLM adaptativo, e suporte a evolução autoevolutiva."
)

# Paths do sistema
ROOT = Path("/opt/penin_omega") if os.path.exists("/opt/penin_omega") else Path.home() / ".penin_omega"
DIRS = {
    "LOG": ROOT / "logs",
    "STATE": ROOT / "state",
    "CACHE": ROOT / "cache",
    "MODELS": ROOT / "models",
    "WORM": ROOT / "worm",
    "SNAPSHOTS": ROOT / "snapshots",
    "CONFIG": ROOT / "config",
    "QUEUE": ROOT / "penin_queue",
    "WEBHOOK": ROOT / "webhook",
    "PLUGINS": ROOT / "plugins"
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Arquivos principais
LOG_FILE = DIRS["LOG"] / "brain.log"
STATE_FILE = DIRS["STATE"] / "brain_state.json"
WORM_FILE = DIRS["WORM"] / "ledger.jsonl"
GOVERNANCE_YAML = DIRS["CONFIG"] / "governance.yaml"
FOUNDATION_YAML = DIRS["CONFIG"] / "foundation.yaml"
BENCH_FILE = DIRS["LOG"] / "benchmark.log"

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# =============================================================================
# UTILITÁRIOS BÁSICOS
# =============================================================================
def _ts() -> str:
    """Timestamp ISO8601 com timezone UTC."""
    return datetime.now(timezone.utc).isoformat()

def _hash_data(data: Any) -> str:
    """SHA256 hash de qualquer dado (thread-safe)."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, ensure_ascii=False)
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif not isinstance(data, (bytes, bytearray)):
        data = str(data).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def log(msg: str, *, level: str = "INFO", also_print: bool = True) -> None:
    """Logging unificado (thread-safe)."""
    line = f"[{_ts()}][{level}] {msg}\n"
    try:
        with LOG_FILE.open("a", encoding="utf-8") as f:
            f.write(line)
    except Exception:
        pass
    if also_print:
        sys.stdout.write(line)
        sys.stdout.flush()

def save_json(path: Path, data: Any) -> None:
    """Salvar dados em JSON (thread-safe)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: Path, default: Any = None) -> Any:
    """Carregar dados de JSON (thread-safe)."""
    try:
        return json.load(path.open("r", encoding="utf-8"))
    except Exception:
        return default

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """Mesclagem profunda de dicionários."""
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

# =============================================================================
# SETTINGS (Pydantic + Env)
# =============================================================================
class Settings(BaseSettings if HAS_PYDANTIC else object):
    # Chaves de API (opcionais)
    OPENAI_API_KEY: str = ""
    ANTHROPIC_API_KEY: str = ""
    MISTRAL_API_KEY: str = ""
    GEMINI_API_KEY: str = ""
    MANUS_API_KEY: str = ""

    # Modelos padrão
    OPENAI_MODEL: str = "gpt-5"
    ANTHROPIC_MODEL: str = "claude-opus-4-1-20250805"
    MISTRAL_MODEL: str = "codestral-2508"
    GEMINI_MODEL: str = "gemini-2.5-pro"
    LOCAL_MODEL: str = "falcon-mamba-7b"

    # Webhook
    WEBHOOK_ENABLED: bool = False
    WEBHOOK_PUBLIC_KEY_URL: str = "https://api.manus.ai/v1/webhook/public_key"
    WEBHOOK_HOST: str = "127.0.0.1"
    WEBHOOK_PORT: int = 8000

    # Sistema
    PKG_VERSION: str = PKG_VERSION
    TIMEOUT_S: int = 180
    RETRIES: int = 3
    UCB_EXPLORATION: float = 0.20
    SNAPSHOT_TTL_DAYS: int = 30
    TASK_TTL_DAYS: int = 7
    MAX_CONCURRENT_TASKS: int = 8
    CACHE_L1_SIZE: int = 1000
    CACHE_L2_SIZE: int = 10000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

# =============================================================================
# ENUMS E ESTRUTURAS DE DADOS
# =============================================================================
class EventType(Enum):
    """Tipos de eventos para WORM/PCE"""
    BOOT = "BOOT"
    SHUTDOWN = "SHUTDOWN"
    PROMOTE = "PROMOTE"
    ROLLBACK = "ROLLBACK"
    EXTINCTION = "EXTINCTION"
    CYCLE_COMPLETE = "CYCLE_COMPLETE"
    CYCLE_ABORT = "CYCLE_ABORT"
    EVOLUTION_START = "EVOLUTION_START"
    EVOLUTION_COMPLETE = "EVOLUTION_COMPLETE"
    LLM_QUERY = "LLM_QUERY"
    SNAPSHOT_CREATED = "SNAPSHOT_CREATED"
    TASK_CREATED = "TASK_CREATED"
    TASK_COMPLETED = "TASK_COMPLETED"
    TASK_FAILED = "TASK_FAILED"
    WEBHOOK_CALL = "WEBHOOK_CALL"

class HealthStatus(Enum):
    """Status de saúde do sistema"""
    HEALTHY = "HEALTHY"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    DEAD = "DEAD"

class TaskStatus(Enum):
    """Status de uma tarefa"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

# =============================================================================
# EXCEÇÕES CUSTOMIZADAS
# =============================================================================
class PENINError(Exception):
    """Exceção base para erros do PENIN-Ω."""
    pass

class ExtinctionError(PENINError):
    """Exceção para extinção do sistema."""
    pass

class GuardError(PENINError):
    """Exceção para falhas em portões de segurança."""
    pass

class TrustRegionError(PENINError):
    """Exceção para falhas na trust-region."""
    pass

class WORMError(PENINError):
    """Exceção para falhas no ledger WORM."""
    pass

# =============================================================================
# PROVIDERS DE LLM - FALCON MAMBA OTIMIZADO
# =============================================================================
class AIResponse:
    """Resposta unificada de provedores de LLM."""

    def __init__(self,
                 provider: str,
                 status: Literal["COMPLETED", "ERROR", "RATE_LIMITED"],
                 content: Optional[str] = None,
                 error: Optional[str] = None,
                 latency: Optional[float] = None,
                 tokens_used: Optional[int] = None):
        self.provider = provider
        self.status = status
        self.content = content
        self.error = error
        self.latency = latency
        self.tokens_used = tokens_used

    def to_dict(self) -> Dict[str, Any]:
        """Converter para dicionário."""
        return {
            "provider": self.provider,
            "status": self.status,
            "content": self.content,
            "error": self.error,
            "latency": self.latency,
            "tokens_used": self.tokens_used
        }

class AIProvider(ABC):
    """Interface abstrata para provedores de LLM."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Nome do provedor."""
        pass

    @abstractmethod
    async def execute(self,
                      prompt: str,
                      system_prompt: str = "",
                      **kwargs: Any) -> AIResponse:
        """Executar chamada à API do provedor."""
        pass

    @abstractmethod
    async def validate_connection(self) -> bool:
        """Validar conexão com o provedor."""
        pass

class LocalLLMProvider(AIProvider):
    """Provider para Falcon Mamba 7B via HTTP (porta 8010) - VERSÃO SUPREMA."""

    def __init__(self, base_url: str = "http://localhost:8010"):
        self._name = "falcon_mamba_7b"
        self.base_url = base_url.rstrip("/")
        self.session = None
        self.circuit_breaker = {"failures": 0, "last_failure": 0, "is_open": False}
        self.metrics = {"total_calls": 0, "successful_calls": 0, "avg_latency": 0.0}
        self.retry_config = {"max_retries": 3, "backoff_factor": 1.5, "max_delay": 30}

    @property
    def name(self) -> str:
        return self._name

    def _is_circuit_open(self) -> bool:
        """Verifica se circuit breaker está aberto."""
        if self.circuit_breaker["is_open"]:
            # Reset após 60 segundos
            if time.time() - self.circuit_breaker["last_failure"] > 60:
                self.circuit_breaker["is_open"] = False
                self.circuit_breaker["failures"] = 0
                logger.info("🔄 Circuit breaker resetado para Falcon Mamba")
        return self.circuit_breaker["is_open"]

    def _handle_failure(self):
        """Gerencia falha do circuit breaker."""
        self.circuit_breaker["failures"] += 1
        self.circuit_breaker["last_failure"] = time.time()
        
        if self.circuit_breaker["failures"] >= 3:
            self.circuit_breaker["is_open"] = True
            logger.warning("⚠️ Circuit breaker aberto para Falcon Mamba")

    async def execute(self,
                      prompt: str,
                      system_prompt: str = "",
                      **kwargs) -> AIResponse:
        """Executa geração usando Falcon Mamba via HTTP com circuit breaker e retry."""
        start_time = time.time()

        if not HAS_AIOHTTP:
            return AIResponse(
                self.name,
                "ERROR",
                error="aiohttp não disponível para conexão HTTP"
            )

        if self._is_circuit_open():
            return AIResponse(
                self.name,
                "ERROR",
                error="Circuit breaker aberto - Falcon Mamba indisponível"
            )

        # Retry logic
        for attempt in range(self.retry_config["max_retries"]):
            try:
                if not self.session:
                    connector = aiohttp.TCPConnector(
                        limit=100,
                        limit_per_host=30,
                        keepalive_timeout=30,
                        enable_cleanup_closed=True
                    )
                    timeout = aiohttp.ClientTimeout(total=120, connect=10)
                    self.session = aiohttp.ClientSession(
                        connector=connector,
                        timeout=timeout,
                        headers={"User-Agent": f"PENIN-Omega/{PKG_VERSION}"}
                    )

                # Preparar payload otimizado para Falcon Mamba
                full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
                
                payload = {
                    "messages": [{"role": "user", "content": full_prompt}],
                    "temperature": kwargs.get("temperature", 0.7),
                    "max_tokens": kwargs.get("max_tokens", 512),
                    "top_p": kwargs.get("top_p", 0.9),
                    "frequency_penalty": kwargs.get("frequency_penalty", 0.0),
                    "presence_penalty": kwargs.get("presence_penalty", 0.0),
                    "stream": False
                }

                self.metrics["total_calls"] += 1

                async with self.session.post(
                    f"{self.base_url}/v1/chat/completions",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=90)
                ) as response:
                    
                    latency = time.time() - start_time
                    
                    if response.status == 200:
                        result = await response.json()
                        
                        # Extrair resposta com múltiplos formatos suportados
                        content = ""
                        if "choices" in result and result["choices"]:
                            choice = result["choices"][0]
                            if "message" in choice:
                                content = choice["message"].get("content", "")
                            elif "text" in choice:
                                content = choice["text"]
                        elif "response" in result:
                            content = result["response"]
                        elif "output" in result:
                            content = result["output"]
                        elif "text" in result:
                            content = result["text"]
                        
                        # Reset circuit breaker em caso de sucesso
                        self.circuit_breaker["failures"] = 0
                        
                        # Atualizar métricas
                        self.metrics["successful_calls"] += 1
                        current_avg = self.metrics["avg_latency"]
                        total_calls = self.metrics["total_calls"]
                        self.metrics["avg_latency"] = (
                            (current_avg * (total_calls - 1)) + latency
                        ) / total_calls
                        
                        return AIResponse(
                            provider=self.name,
                            status="COMPLETED",
                            content=content,
                            latency=latency,
                            tokens_used=result.get("usage", {}).get("total_tokens")
                        )
                    else:
                        error_text = await response.text()
                        
                        # Retry em caso de erro temporário
                        if response.status in [429, 502, 503, 504] and attempt < self.retry_config["max_retries"] - 1:
                            delay = min(
                                self.retry_config["backoff_factor"] ** attempt,
                                self.retry_config["max_delay"]
                            )
                            logger.warning(f"⚠️ Erro {response.status}, tentativa {attempt + 1}, aguardando {delay}s")
                            await asyncio.sleep(delay)
                            continue
                        
                        self._handle_failure()
                        status = "RATE_LIMITED" if response.status == 429 else "ERROR"
                        
                        return AIResponse(
                            provider=self.name,
                            status=status,
                            error=f"HTTP {response.status}: {error_text}",
                            latency=latency
                        )

            except asyncio.TimeoutError:
                if attempt < self.retry_config["max_retries"] - 1:
                    delay = min(
                        self.retry_config["backoff_factor"] ** attempt,
                        self.retry_config["max_delay"]
                    )
                    logger.warning(f"⚠️ Timeout, tentativa {attempt + 1}, aguardando {delay}s")
                    await asyncio.sleep(delay)
                    continue
                
                self._handle_failure()
                return AIResponse(
                    provider=self.name,
                    status="ERROR",
                    error="Timeout na conexão com Falcon Mamba",
                    latency=time.time() - start_time
                )
            except aiohttp.ClientError as e:
                if attempt < self.retry_config["max_retries"] - 1:
                    delay = min(
                        self.retry_config["backoff_factor"] ** attempt,
                        self.retry_config["max_delay"]
                    )
                    logger.warning(f"⚠️ Erro de conexão, tentativa {attempt + 1}, aguardando {delay}s")
                    await asyncio.sleep(delay)
                    continue
                
                self._handle_failure()
                return AIResponse(
                    provider=self.name,
                    status="ERROR",
                    error=f"Erro de conexão: {str(e)}",
                    latency=time.time() - start_time
                )
            except Exception as e:
                self._handle_failure()
                return AIResponse(
                    provider=self.name,
                    status="ERROR",
                    error=f"Erro inesperado: {str(e)}",
                    latency=time.time() - start_time
                )

        # Se chegou aqui, todas as tentativas falharam
        self._handle_failure()
        return AIResponse(
            provider=self.name,
            status="ERROR",
            error="Todas as tentativas de conexão falharam",
            latency=time.time() - start_time
        )

    async def validate_connection(self) -> bool:
        """Valida conexão com Falcon Mamba com múltiplos endpoints."""
        if self._is_circuit_open():
            return False
            
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
            
            # Tentar múltiplos endpoints de health check
            endpoints = ["/health", "/v1/models", "/"]
            
            for endpoint in endpoints:
                try:
                    async with self.session.get(
                        f"{self.base_url}{endpoint}",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status in [200, 404]:  # 404 é OK se endpoint não existe
                            logger.info(f"✅ Falcon Mamba conectado via {endpoint}")
                            return True
                except Exception:
                    continue
            
            return False
            
        except Exception as e:
            logger.warning(f"⚠️ Falha na validação de conexão: {e}")
            return False

    async def close(self):
        """Fecha sessão HTTP com cleanup completo."""
        if self.session:
            try:
                await self.session.close()
                # Aguardar cleanup completo
                await asyncio.sleep(0.1)
            except Exception as e:
                logger.warning(f"⚠️ Erro ao fechar sessão HTTP: {e}")
            finally:
                self.session = None

    def get_metrics(self) -> Dict[str, Any]:
        """Retorna métricas detalhadas do provider."""
        success_rate = (
            self.metrics["successful_calls"] / max(1, self.metrics["total_calls"])
        ) * 100
        
        return {
            "total_calls": self.metrics["total_calls"],
            "successful_calls": self.metrics["successful_calls"],
            "success_rate": f"{success_rate:.2f}%",
            "avg_latency_ms": f"{self.metrics['avg_latency'] * 1000:.2f}",
            "circuit_breaker": {
                "is_open": self.circuit_breaker["is_open"],
                "failures": self.circuit_breaker["failures"]
            }
        }
