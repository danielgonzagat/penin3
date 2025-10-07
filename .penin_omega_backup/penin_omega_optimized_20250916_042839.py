#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω v7.0 FUSION SUPREMA - Sistema Unificado Definitivo
OTIMIZADO PARA FALCON MAMBA 7B VIA HTTP

Integração direta com Falcon Mamba 7B rodando na porta 8010
Arquitetura: Σ-Guard/IR→IC/CAOS⁺/SR-Ω∞/WORM/Liga/Bridge/Async/Trust-Region/Webhook
"""

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
    import aiohttp
    HAS_AIOHTTP = True
except ImportError:
    aiohttp = None
    HAS_AIOHTTP = False

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

# =============================================================================
# CONSTANTES E METADADOS
# =============================================================================
PKG_NAME = "penin_omega_fusion_suprema"
PKG_VERSION = "7.0.1"
PKG_DESC = "PENIN-Ω v7.0 FUSION SUPREMA - Otimizado para Falcon Mamba 7B via HTTP"

# Paths do sistema
ROOT = Path("/opt/penin_omega") if os.path.exists("/opt/penin_omega") else Path.home() / ".penin_omega"
DIRS = {
    "LOG": ROOT / "logs",
    "STATE": ROOT / "state", 
    "CACHE": ROOT / "cache",
    "WORM": ROOT / "worm",
    "SNAPSHOTS": ROOT / "snapshots",
    "CONFIG": ROOT / "config"
}

for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

# Arquivos principais
LOG_FILE = DIRS["LOG"] / "brain.log"
STATE_FILE = DIRS["STATE"] / "brain_state.json"
WORM_FILE = DIRS["WORM"] / "ledger.jsonl"

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

# =============================================================================
# ENUMS E ESTRUTURAS DE DADOS
# =============================================================================
class EventType(Enum):
    """Tipos de eventos para WORM"""
    BOOT = "BOOT"
    SHUTDOWN = "SHUTDOWN"
    PROMOTE = "PROMOTE"
    ROLLBACK = "ROLLBACK"
    EXTINCTION = "EXTINCTION"
    CYCLE_COMPLETE = "CYCLE_COMPLETE"
    CYCLE_ABORT = "CYCLE_ABORT"
    LLM_QUERY = "LLM_QUERY"

# =============================================================================
# WORM LEDGER SIMPLES
# =============================================================================
class WORMLedger:
    """Sistema WORM com Merkle chain simples"""

    def __init__(self, path: Path = WORM_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._last_hash = self._get_last_hash()

    def _get_last_hash(self) -> str:
        """Obtém hash do último evento"""
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "genesis"
        try:
            with self.path.open("rb") as f:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
                last = f.readline().decode("utf-8")
            return json.loads(last).get("hash", "genesis")
        except Exception:
            return "genesis"

    def record_event(self, etype: str, data: Dict[str, Any]) -> str:
        """Registra evento imutável no ledger"""
        with self.lock:
            event_id = str(uuid.uuid4())
            timestamp = _ts()

            event_dict = {
                "event_id": event_id,
                "type": etype,
                "data": data,
                "timestamp": timestamp,
                "prev_hash": self._last_hash
            }

            event_str = json.dumps(event_dict, sort_keys=True, ensure_ascii=False)
            event_hash = hashlib.sha256(event_str.encode()).hexdigest()

            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({**event_dict, "hash": event_hash}, ensure_ascii=False) + "\n")

            self._last_hash = event_hash
            return event_id

# =============================================================================
# ESTADO CANÔNICO UNIFICADO
# =============================================================================
@dataclass
class OmegaState:
    """Estado canônico unificado do PENIN-Ω v7.0"""

    # Identificação
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=_ts)
    version: str = PKG_VERSION

    # Ética e Governança (Σ-Guard)
    ece: float = 0.0
    rho_bias: float = 1.0
    consent: bool = True
    eco_ok: bool = True

    # Risco (IR→IC)
    rho: float = 0.5
    uncertainty: float = 0.5
    risk_contractions: int = 0

    # CAOS⁺
    C: float = 0.6  # Curiosidade
    A: float = 0.6  # Autonomia
    O: float = 0.6  # Ordem
    S: float = 0.6  # Singularidade
    caos_pre: float = 1.0
    caos_post: float = 1.0

    # SR-Ω∞
    sr_score: float = 1.0
    C_cal: float = 0.8
    E_ok: float = 1.0
    M: float = 0.7
    A_eff: float = 0.6

    # Equação da Morte
    A_t: bool = False  # Autoevolução
    C_t: bool = False  # Descoberta
    E_t: bool = True   # Vivo
    V_t: bool = True   # Portão vital
    extinction_reason: Optional[str] = None

    # Métricas evolutivas
    delta_linf: float = 0.0
    mdl_gain: float = 0.0
    ppl_ood: float = 100.0
    novelty_sim: float = 1.0

    # Controle
    kill_switch: bool = False
    trust_region_radius: float = 0.1
    cycle_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Converter para dicionário."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OmegaState':
        """Criar a partir de dicionário."""
        valid_fields = {f.name for f in dc_fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)

    def validate(self) -> Tuple[bool, List[str]]:
        """Validar consistência do estado."""
        errors = []
        if not (0 <= self.ece <= 1):
            errors.append(f"ece out of bounds: {self.ece}")
        if not (0 <= self.rho <= 1):
            errors.append(f"rho out of bounds: {self.rho}")
        if self.kill_switch and self.E_t:
            errors.append("kill_switch active but E_t=True")
        return len(errors) == 0, errors

# =============================================================================
# PROVIDERS DE LLM
# =============================================================================
class AIResponse:
    """Resposta unificada de provedores de LLM."""

    def __init__(self,
                 provider: str,
                 status: Literal["COMPLETED", "ERROR", "RATE_LIMITED"],
                 content: Optional[str] = None,
                 error: Optional[str] = None,
                 latency: Optional[float] = None):
        self.provider = provider
        self.status = status
        self.content = content
        self.error = error
        self.latency = latency

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

class LocalLLMProvider(AIProvider):
    """Provider para Falcon Mamba 7B via HTTP (porta 8010)."""

    def __init__(self, base_url: str = "http://localhost:8010"):
        self._name = "falcon_mamba_7b"
        self.base_url = base_url.rstrip("/")
        self.session = None

    @property
    def name(self) -> str:
        return self._name

    async def execute(self,
                      prompt: str,
                      system_prompt: str = "",
                      **kwargs) -> AIResponse:
        """Executa geração usando Falcon Mamba via HTTP."""
        start_time = time.time()

        if not HAS_AIOHTTP:
            return AIResponse(
                self.name,
                "ERROR",
                error="aiohttp não disponível"
            )

        try:
            if not self.session:
                self.session = aiohttp.ClientSession()

            # Preparar payload para o Falcon Mamba
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            payload = {
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": kwargs.get("temperature", 0.7),
                "max_tokens": kwargs.get("max_tokens", 512),
                "top_p": kwargs.get("top_p", 0.9)
            }

            async with self.session.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Extrair resposta do formato OpenAI-compatible
                    content = ""
                    if "choices" in result and result["choices"]:
                        content = result["choices"][0].get("message", {}).get("content", "")
                    elif "response" in result:
                        content = result["response"]
                    elif "output" in result:
                        content = result["output"]
                    
                    return AIResponse(
                        provider=self.name,
                        status="COMPLETED",
                        content=content,
                        latency=time.time() - start_time
                    )
                else:
                    error_text = await response.text()
                    return AIResponse(
                        provider=self.name,
                        status="ERROR",
                        error=f"HTTP {response.status}: {error_text}",
                        latency=time.time() - start_time
                    )

        except Exception as e:
            return AIResponse(
                provider=self.name,
                status="ERROR",
                error=str(e),
                latency=time.time() - start_time
            )

    async def validate_connection(self) -> bool:
        """Valida conexão com Falcon Mamba."""
        try:
            if not self.session:
                self.session = aiohttp.ClientSession()
                
            async with self.session.get(
                f"{self.base_url}/health",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                return response.status == 200
        except Exception:
            return False

    async def close(self):
        """Fecha sessão HTTP."""
        if self.session:
            await self.session.close()
            self.session = None

# =============================================================================
# MOTORES PENIN-Ω SIMPLIFICADOS
# =============================================================================
class SigmaGuard:
    """Σ-Guard: Portão ético não-compensatório."""
    
    def __init__(self):
        self.ece_max = 0.01
        self.rho_bias_max = 1.05

    def check(self, state: OmegaState) -> bool:
        """Verificar conformidade ética."""
        return (state.ece <= self.ece_max and 
                state.rho_bias <= self.rho_bias_max and
                state.consent and state.eco_ok)

class IRIC:
    """IR→IC: Restrição de Incerteza."""
    
    def __init__(self):
        self.rho_max = 0.95
        self.contraction_factor = 0.98

    def safe(self, state: OmegaState) -> bool:
        """Verificar segurança."""
        return state.rho < self.rho_max

    def contract(self, state: OmegaState) -> None:
        """Aplicar contração de risco."""
        state.rho *= self.contraction_factor
        state.uncertainty *= self.contraction_factor
        state.risk_contractions += 1
        
        if state.rho >= self.rho_max:
            state.kill_switch = True

class CAOSPlusEngine:
    """Motor CAOS⁺."""
    
    def compute(self, state: OmegaState) -> float:
        """Calcular CAOS⁺."""
        C = max(0.0, state.C)
        A = max(0.0, state.A)
        O = max(0.0, state.O)
        S = max(0.0, state.S)
        
        base = 1.0 + 2.0 * C * A
        exponent = max(0.05, min(2.0, O * S))
        
        caos_value = base ** exponent
        
        state.caos_pre = state.caos_post
        state.caos_post = caos_value
        
        return caos_value

class SROmegaInfinityEngine:
    """Motor SR-Ω∞."""
    
    def compute(self, state: OmegaState) -> float:
        """Calcular fator reflexivo SR."""
        components = [
            (max(1e-6, state.E_ok), 0.40),
            (max(1e-6, state.M), 0.30),
            (max(1e-6, state.C_cal), 0.20),
            (max(1e-6, state.A_eff), 0.10),
        ]
        
        denominator = sum(weight / value for value, weight in components)
        sr_score = 1.0 / max(1e-6, denominator)
        state.sr_score = sr_score
        
        return sr_score

    def check_gate(self, state: OmegaState) -> bool:
        """Verificar gate reflexivo."""
        return state.sr_score >= 0.80

class EquacaoDaMorte:
    """Equação da Morte: V_t ∧ (A_t ∨ C_t)."""

    def __init__(self, ledger: WORMLedger, sigma_guard: SigmaGuard, iric: IRIC):
        self.ledger = ledger
        self.sigma_guard = sigma_guard
        self.iric = iric

    def execute(self, state: OmegaState) -> Tuple[bool, bool, bool]:
        """Executar equação da morte."""
        # Verificar portões vitais
        V_t = self.sigma_guard.check(state) and self.iric.safe(state)
        
        if not V_t:
            return self._kill(state, "PORTAO_VITAL_FALHOU")

        # Verificar autoevolução (A_t)
        A_t = (state.delta_linf >= 0.01 and
               state.mdl_gain >= 0.02 and
               state.ppl_ood < 95.0)

        # Verificar descoberta (C_t)
        C_t = state.novelty_sim <= 0.10

        # Verificar sobrevivência (S_t = A_t ∨ C_t)
        S_t = A_t or C_t

        if not S_t:
            return self._kill(state, "SEM_AUTOEVOLUCAO_NEM_DESCOBERTA")

        state.A_t, state.C_t, state.E_t = A_t, C_t, True
        state.extinction_reason = None
        return (A_t, C_t, True)

    def _kill(self, state: OmegaState, reason: str) -> Tuple[bool, bool, bool]:
        """Registrar extinção no WORM."""
        state.E_t = False
        state.extinction_reason = reason
        
        event = {"reason": reason, "state": state.to_dict()}
        self.ledger.record_event(EventType.EXTINCTION.value, event)
        
        logger.critical(f"💀 EXTINÇÃO: {reason}")
        return (False, False, False)

class League:
    """Liga de decisão: PROMOTE/ROLLBACK."""

    def decide(self, before: OmegaState, after: OmegaState) -> str:
        """Decidir PROMOTE/ROLLBACK."""
        if not after.E_t or after.kill_switch:
            return "ROLLBACK"
        
        if after.sr_score < 0.80:
            return "ROLLBACK"
        
        if after.caos_post < 1.0:
            return "ROLLBACK"
        
        # Verificar melhoria
        performance_improved = (
            after.caos_post >= before.caos_post and
            after.ppl_ood <= before.ppl_ood and
            after.rho <= before.rho
        )
        
        return "PROMOTE" if performance_improved else "ROLLBACK"

# =============================================================================
# NÚCLEO PRINCIPAL PENIN-Ω
# =============================================================================
class NucleoPENINOmega:
    """Orquestrador central do sistema PENIN-Ω v7.0."""

    def __init__(self):
        logger.info("=" * 80)
        logger.info(f"🧠 PENIN-Ω v{PKG_VERSION} FUSION SUPREMA — Inicializando")
        logger.info("=" * 80)

        # Estado
        self.state = OmegaState()

        # Subsistemas
        self.ledger = WORMLedger()
        self.llm_provider = LocalLLMProvider()

        # Motores
        self.sigma_guard = SigmaGuard()
        self.iric = IRIC()
        self.caos = CAOSPlusEngine()
        self.sr_omega = SROmegaInfinityEngine()
        self.eq_morte = EquacaoDaMorte(self.ledger, self.sigma_guard, self.iric)
        self.league = League()

        # Registro de nascimento
        self._register_birth()

        logger.info("✅ Núcleo PENIN-Ω operacional")
        logger.info("=" * 80)

    def _register_birth(self) -> None:
        """Registrar nascimento do sistema."""
        birth_data = {
            "version": PKG_VERSION,
            "initial_state": self.state.to_dict(),
            "falcon_mamba_url": "http://localhost:8010"
        }

        self.ledger.record_event(EventType.BOOT.value, birth_data)

    async def cycle(self) -> Dict[str, Any]:
        """Executar um ciclo completo."""
        cycle_id = str(uuid.uuid4())[:8]
        logger.info(f"\n🔄 CICLO {cycle_id} iniciado")

        result = {
            "cycle_id": cycle_id,
            "timestamp_start": _ts(),
            "success": False,
            "decision": None,
            "metrics": {}
        }

        try:
            # Validar estado
            valid, errors = self.state.validate()
            if not valid:
                result["reason"] = "INVALID_STATE"
                result["errors"] = errors
                return result

            # Gates de segurança
            if not self.sigma_guard.check(self.state):
                result["reason"] = "SIGMA_GUARD_FAILED"
                return result

            if not self.iric.safe(self.state):
                self.iric.contract(self.state)
                if self.state.kill_switch:
                    result["reason"] = "KILL_SWITCH_ARMED"
                    return result

            # Equação da Morte
            A_t, C_t, E_t = self.eq_morte.execute(self.state)
            if not E_t:
                result["reason"] = "EXTINCTION"
                result["extinction_reason"] = self.state.extinction_reason
                return result

            # Atualizar motores
            caos_value = self.caos.compute(self.state)
            sr_score = self.sr_omega.compute(self.state)

            if not self.sr_omega.check_gate(self.state):
                result["reason"] = "SR_GATE_FAILED"
                return result

            # Simular atualização de estado
            before_state = OmegaState.from_dict(self.state.to_dict())
            
            # Pequenas melhorias
            self.state.ppl_ood = max(1.0, self.state.ppl_ood * 0.99)
            self.state.rho = max(0.0, self.state.rho * 0.98)
            
            # Liga decide
            decision = self.league.decide(before_state, self.state)

            if decision == "PROMOTE":
                self.ledger.record_event(EventType.PROMOTE.value, {
                    "cycle_id": cycle_id,
                    "sr_score": self.state.sr_score,
                    "caos_post": self.state.caos_post
                })
                result["success"] = True
            else:
                self.ledger.record_event(EventType.ROLLBACK.value, {
                    "cycle_id": cycle_id,
                    "sr_score": self.state.sr_score,
                    "caos_post": self.state.caos_post
                })

            result["decision"] = decision
            self.state.cycle_count += 1

            result["metrics"] = {
                "sr_score": self.state.sr_score,
                "caos_post": self.state.caos_post,
                "rho": self.state.rho,
                "A_t": A_t,
                "C_t": C_t,
                "cycles": self.state.cycle_count
            }

            self.ledger.record_event(EventType.CYCLE_COMPLETE.value, {
                "cycle_id": cycle_id,
                "metrics": result["metrics"],
                "decision": decision
            })

        except Exception as e:
            result["reason"] = "UNEXPECTED_ERROR"
            result["error"] = str(e)
            logger.error(f"❌ Erro no ciclo {cycle_id}: {e}")

        result["timestamp_end"] = _ts()
        return result

    async def query_llm(self, prompt: str, **kwargs) -> str:
        """Consultar Falcon Mamba."""
        response = await self.llm_provider.execute(prompt, **kwargs)
        
        self.ledger.record_event(EventType.LLM_QUERY.value, {
            "provider": response.provider,
            "status": response.status,
            "prompt_length": len(prompt),
            "response_length": len(response.content or "")
        })
        
        return response.content or ""

    async def diagnose(self) -> Dict[str, Any]:
        """Diagnóstico do sistema."""
        valid, errors = self.state.validate()
        falcon_ok = await self.llm_provider.validate_connection()
        
        health = "HEALTHY"
        if not valid or not falcon_ok:
            health = "CRITICAL"
        elif self.state.rho > 0.8:
            health = "WARNING"

        return {
            "timestamp": _ts(),
            "version": PKG_VERSION,
            "health": health,
            "state": {
                "alive": self.state.E_t,
                "cycles": self.state.cycle_count,
                "risk": self.state.rho,
                "caos": self.state.caos_post,
                "sr": self.state.sr_score,
                "valid": valid,
                "errors": errors
            },
            "falcon_mamba": {
                "connected": falcon_ok,
                "url": "http://localhost:8010"
            }
        }

    async def shutdown(self):
        """Desligar sistema."""
        logger.info("\n⚠️ Desligando com segurança...")
        
        save_json(STATE_FILE, self.state.to_dict())
        
        self.ledger.record_event(EventType.SHUTDOWN.value, {
            "state": self.state.to_dict(),
            "cycles_completed": self.state.cycle_count
        })
        
        await self.llm_provider.close()
        
        logger.info("💤 Desligado.")

# =============================================================================
# CLI
# =============================================================================
async def main():
    import argparse
    
    parser = argparse.ArgumentParser(description=PKG_DESC)
    parser.add_argument("--diagnose", action="store_true", help="Executa diagnóstico")
    parser.add_argument("--cycles", type=int, default=3, help="Número de ciclos")
    parser.add_argument("--query", type=str, help="Consulta ao Falcon Mamba")
    args = parser.parse_args()

    core = NucleoPENINOmega()

    def signal_handler(sig, frame):
        logger.info("\n⏹️ Interrompido pelo usuário")
        asyncio.create_task(core.shutdown())
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)

    try:
        if args.diagnose:
            print(json.dumps(await core.diagnose(), ensure_ascii=False, indent=2))
            await core.shutdown()
            return

        if args.query:
            response = await core.query_llm(args.query)
            print(f"\n🤖 Falcon Mamba: {response}")
            await core.shutdown()
            return

        # Execução padrão: N ciclos
        for i in range(args.cycles):
            logger.info(f"\n{'='*42}\nCICLO {i+1}/{args.cycles}\n{'='*42}")
            result = await core.cycle()
            
            if not result["success"] and result.get("reason") == "EXTINCTION":
                logger.critical("💀 Sistema extinto. Encerrando.")
                break
                
            await asyncio.sleep(0.5)

        print("\n" + "="*64 + "\nDIAGNÓSTICO FINAL\n" + "="*64)
        print(json.dumps(await core.diagnose(), ensure_ascii=False, indent=2))
        
        await core.shutdown()

    except KeyboardInterrupt:
        logger.warning("⏹️ Interrompido")
        await core.shutdown()
    except Exception as e:
        logger.critical(f"❌ Erro fatal: {e}")
        traceback.print_exc()
        await core.shutdown()

if __name__ == "__main__":
    asyncio.run(main())
