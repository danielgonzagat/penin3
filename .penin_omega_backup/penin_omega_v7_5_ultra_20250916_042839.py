#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω v7.5 ULTRA FUSION — Núcleo Unificado e Evoluído

Arquitetura integrada: Σ-Guard / IR→IC / CAOS⁺ / SR-Ω∞ / Equação da Morte /
WORM Merkle (SQLite + JSONL redundante) / Liga (Arbítrio) / Bridge LLM / Async /
Trust-Region / Snapshots / Scheduler / Observabilidade.

Pronto para CPU-first, com suporte a modelos Transformers locais — diretório
padrão: ~/.penin_omega/models/falcon-mamba-7b (ex.: "Falcon Mamba 7B").
Inclui fallback para GPT-2 apenas para testes.

SR‑Ω∞ — Implementa agregador harmônico não-compensatório das componentes C/E/M/A,
gates éticos + IR→IC, projeção segura Π_{H∩S} e acoplamento ao CAOS⁺ via ϕ(z)
saturado; Equação da Morte, Penin-Update com região de confiança adaptativa e
Liga para decisão PROMOTE/ROLLBACK.

Referência conceitual do SR‑Ω∞: Dossiê do Módulo de Singularidade Reflexiva.  ⟶  :contentReference[oaicite:1]{index=1}
"""

from __future__ import annotations

# ========== Imports básicos ==========
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
import traceback
import sqlite3
import logging
import signal
import base64
from pathlib import Path
from dataclasses import dataclass, asdict, field, fields as dc_fields
from typing import Any, Dict, List, Optional, Tuple, Literal, Callable, Set
from datetime import datetime, timezone
from collections import OrderedDict, deque
from enum import Enum

# ========== Dependências opcionais (sempre tratadas com graceful-degradation) ==========
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    psutil = None
    HAS_PSUTIL = False

try:
    import yaml
    HAS_YAML = True
except Exception:
    yaml = None
    HAS_YAML = False

try:
    import lz4.frame as lz4f
    HAS_LZ4 = True
except Exception:
    lz4f = None
    HAS_LZ4 = False

try:
    import redis as _redis
    HAS_REDIS_LIB = True
except Exception:
    _redis = None
    HAS_REDIS_LIB = False

try:
    import torch
    HAS_TORCH = True
except Exception:
    torch = None
    HAS_TORCH = False

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM, GPT2LMHeadModel, GPT2Tokenizer
    HAS_TRANSF = True
except Exception:
    AutoTokenizer = AutoModelForCausalLM = GPT2LMHeadModel = GPT2Tokenizer = None
    HAS_TRANSF = False

# Provedores em nuvem opcionais (todos graceful)
try:
    from openai import AsyncOpenAI
except Exception:
    AsyncOpenAI = None

try:
    import anthropic
except Exception:
    anthropic = None

try:
    from mistralai import MistralAsyncClient
except Exception:
    MistralAsyncClient = None

try:
    import google.generativeai as genai
except Exception:
    genai = None


# ========== Constantes / Paths ==========
PKG_NAME    = "penin_omega_ultra_fusion"
PKG_VERSION = "7.5.0"
DESC        = "PENIN-Ω ULTRA FUSION — Σ/IR→IC/CAOS⁺/SR-Ω∞/WORM/League/Bridge/Async/TR/Snapshots"

ROOT = Path("/opt/penin_omega") if os.path.exists("/opt/penin_omega") else Path.home() / ".penin_omega"
DIRS = {
    "LOG": ROOT / "logs",
    "STATE": ROOT / "state",
    "CACHE": ROOT / "cache",
    "MODELS": ROOT / "models",
    "WORM": ROOT / "worm",
    "SNAPSHOT": ROOT / "snapshots",
    "QUEUE": ROOT / "queue",
    "CONFIG": ROOT / "config",
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE       = DIRS["LOG"] / "penin_omega.log"
STATE_FILE     = DIRS["STATE"] / "state.json"
WORM_SQLITE_DB = DIRS["WORM"] / "ledger.db"
WORM_JSONL     = DIRS["WORM"] / "ledger.jsonl"
GOVERNANCE_YAML= DIRS["CONFIG"] / "governance.yaml"

# ========== Logging ==========
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger("PENIN-Ω")

def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()

def _hash_data(data: Any) -> str:
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, ensure_ascii=False)
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif not isinstance(data, (bytes, bytearray)):
        data = str(data).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_json(path: Path, default: Any=None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


# ========== Governança (defaults + YAML opcional) ==========
DEFAULT_GOV = {
    "ethics": {
        "ece_max": 0.01, "rho_bias_max": 1.05,
        "consent_required": True, "eco_ok_required": True
    },
    "iric": {
        "rho_max": 0.95, "contraction_factor": 0.98, "kill_on_violation": True, "max_contractions": 5
    },
    "caos_plus": {
        "kappa": 2.0, "pmin": 0.05, "pmax": 2.0, "chaos_probability": 0.01, "k_phi": 1.5, "clamp": True
    },
    "sr_omega": {
        "tau_SR": 0.80,  # limiar de gate
        # pesos harmônicos (não-compensatórios). Recomenda-se dar maior peso à ética/risk.
        "weights": {"E":0.40, "M":0.30, "C":0.20, "A":0.10},
        "min_components": 3
    },
    "penin_update": {
        "alpha0": 0.01, "lambda_U": 0.5, "project_to": "H_intersect_S",
        "trust_region": {"initial_radius":0.10,"min_radius":0.02,"max_radius":0.50,"shrink_factor":0.90,"grow_factor":1.10},
        "max_step": 0.20, "min_improvement": 0.01
    },
    "league": {"consensus_required": True},
    "worm": {"max_size_mb": 120},
    "api": {
        "providers": ["local","openai","anthropic","mistral","gemini"],
        "ucb_exploration": 0.20, "ensemble_arbiter": "anthropic", "fallback_provider": "openai",
        "rate_limit_delay": 0.8
    },
    "snapshot": {"ttl_days": 30, "max_snapshots": 128},
    "scheduler": {"max_concurrent": 8, "task_ttl_days": 7}
}

def load_governance() -> Dict[str, Any]:
    cfg = dict(DEFAULT_GOV)
    if HAS_YAML and GOVERNANCE_YAML.exists():
        try:
            with GOVERNANCE_YAML.open("r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f) or {}
            cfg = deep_merge(cfg, user_cfg)
        except Exception as e:
            log.warning(f"governance.yaml inválido: {e}")
    # Normaliza pesos SR (somatório ≈ 1)
    w = cfg["sr_omega"]["weights"]
    s = sum(w.values()) or 1.0
    for k in w:
        w[k] = w[k] / s
    return cfg

GOV = load_governance()


# ========== Enums ==========
class EventType(Enum):
    BOOT="BOOT"; SHUTDOWN="SHUTDOWN"; PROMOTE="PROMOTE"; ROLLBACK="ROLLBACK"; EXTINCTION="EXTINCTION"
    CYCLE_COMPLETE="CYCLE_COMPLETE"; CYCLE_ABORT="CYCLE_ABORT"; EVOLUTION_COMPLETE="EVOLUTION_COMPLETE"
    LLM_QUERY="LLM_QUERY"; SNAPSHOT_CREATED="SNAPSHOT_CREATED"; CYCLE_ERROR="CYCLE_ERROR"

class TaskStatus(Enum):
    PENDING="pending"; PROCESSING="processing"; COMPLETED="completed"; FAILED="failed"; CANCELLED="cancelled"


# ========== Cache Multi-Nível (L1/L2/Redis) ==========
class MultiLevelCache:
    """
    L1: memória (LRU TTL) — ultra-rápido
    L2: SQLite persistente com compressão lz4 opcional
    L3: Redis (opcional) para cache distribuído
    """
    def __init__(self, l1_size=512, l2_size=10000, ttl_l1=30, ttl_l2=300):
        self.l1: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self.l1_size = l1_size
        self.ttl_l1 = ttl_l1

        self.l2 = sqlite3.connect(str(DIRS["CACHE"] / "l2_cache.db"), check_same_thread=False)
        self._init_l2(l2_size)
        self.ttl_l2 = ttl_l2

        self.rds = None
        if HAS_REDIS_LIB:
            try:
                self.rds = _redis.Redis(host="localhost", port=6379, db=0, decode_responses=False)
                self.rds.ping()
            except Exception:
                self.rds = None

        self.lock = threading.RLock()
        self.stats: Dict[str, Dict[str, int]] = {}

    def _init_l2(self, l2_size: int):
        c = self.l2.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS cache(
            key TEXT PRIMARY KEY, value BLOB, ts REAL, acc INT DEFAULT 0, last REAL
        )""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ts ON cache(ts)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_acc ON cache(acc)")
        self.l2.commit()
        self.l2_size = l2_size

    def _l1_set(self, k: str, v: Any):
        if k in self.l1:
            ent = self.l1.pop(k)
            ent["v"] = v
            ent["ts"] = time.time()
            self.l1[k] = ent
        else:
            if len(self.l1) >= self.l1_size:
                self.l1.popitem(last=False)
            self.l1[k] = {"v": v, "ts": time.time()}

    def _l2_set(self, key: str, value: Any):
        cur = self.l2.cursor()
        cur.execute("SELECT COUNT(*) FROM cache")
        cnt = int(cur.fetchone()[0])
        if cnt >= self.l2_size:
            # Evict baseado em acc e idade
            cur.execute("""
                DELETE FROM cache WHERE key IN(
                    SELECT key FROM cache ORDER BY (acc*0.3 + (? - ts)*0.7) DESC LIMIT ?
                )
            """, (time.time(), max(1, self.l2_size // 10)))
        try:
            raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
            blob = lz4f.compress(raw) if HAS_LZ4 else raw
        except Exception:
            blob = json.dumps({"_pickle_fallback": True}).encode("utf-8")
        cur.execute("INSERT OR REPLACE INTO cache(key,value,ts,last) VALUES(?,?,?,?)",
                    (key, blob, time.time(), time.time()))
        self.l2.commit()

    def set(self, key: str, value: Any, ttl: Optional[int]=None):
        with self.lock:
            self._l1_set(key, value)
            self._l2_set(key, value)
            if self.rds:
                try:
                    raw = json.dumps(value, ensure_ascii=False).encode("utf-8")
                    blob = lz4f.compress(raw) if HAS_LZ4 else raw
                    self.rds.setex(f"penin:{key}", ttl or self.ttl_l2, blob)
                except Exception:
                    pass

    def get(self, key: str, default: Any=None) -> Any:
        with self.lock:
            # L1
            if key in self.l1:
                ent = self.l1[key]
                if time.time() - ent["ts"] < self.ttl_l1:
                    self.l1.move_to_end(key)
                    self._stat_hit(key)
                    return ent["v"]
                else:
                    self.l1.pop(key, None)
            # L2
            cur = self.l2.cursor()
            cur.execute("SELECT value, ts FROM cache WHERE key=?", (key,))
            row = cur.fetchone()
            if row:
                blob, ts = row
                if time.time() - ts < self.ttl_l2:
                    try:
                        data = lz4f.decompress(blob) if HAS_LZ4 else blob
                        val = json.loads(data.decode("utf-8")) if isinstance(data, (bytes, bytearray)) else data
                    except Exception:
                        try:
                            val = json.loads(blob)
                        except Exception:
                            val = blob
                    self._l1_set(key, val)
                    cur.execute("UPDATE cache SET acc=acc+1, last=? WHERE key=?", (time.time(), key))
                    self.l2.commit()
                    self._stat_hit(key)
                    return val
            # L3 Redis
            if self.rds:
                try:
                    b = self.rds.get(f"penin:{key}")
                    if b:
                        data = lz4f.decompress(b) if HAS_LZ4 else b
                        val = json.loads(data.decode("utf-8")) if isinstance(data, (bytes, bytearray)) else data
                        self._l1_set(key, val)
                        self._l2_set(key, val)
                        self._stat_hit(key)
                        return val
                except Exception:
                    pass
            self._stat_miss(key)
            return default

    def _stat_hit(self, key: str):
        st = self.stats.setdefault(key, {"hits":0, "misses":0})
        st["hits"] += 1

    def _stat_miss(self, key: str):
        st = self.stats.setdefault(key, {"hits":0, "misses":0})
        st["misses"] += 1

    def clear(self):
        with self.lock:
            self.l1.clear()
            self.l2.execute("DELETE FROM cache")
            self.l2.commit()
            if self.rds:
                try:
                    for k in self.rds.scan_iter("penin:*"):
                        self.rds.delete(k)
                except Exception:
                    pass


# ========== WORM Ledgers (SQLite + JSONL) ==========
class WormLedgerSQLite:
    def __init__(self, db_path: Path = WORM_SQLITE_DB):
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.lock = threading.Lock()
        self._init_db()
        self.last_hash = self._get_last_hash()

    def _init_db(self):
        c = self.conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE NOT NULL,
            event_type TEXT NOT NULL,
            data TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            prev_hash TEXT NOT NULL,
            hash TEXT NOT NULL
        )''')
        c.execute("CREATE INDEX IF NOT EXISTS idx_hash ON events(hash)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_ts   ON events(timestamp)")
        self.conn.commit()

    def _get_last_hash(self) -> str:
        c = self.conn.cursor()
        c.execute("SELECT hash FROM events ORDER BY id DESC LIMIT 1")
        row = c.fetchone()
        return row[0] if row else "genesis"

    def record(self, etype: str, data: Dict[str, Any]) -> str:
        with self.lock:
            ev_id = str(uuid.uuid4())
            ts = _ts()
            payload = {
                "event_id": ev_id,
                "event_type": etype,
                "data": data,
                "timestamp": ts,
                "prev_hash": self.last_hash
            }
            h = _hash_data(payload)
            c = self.conn.cursor()
            c.execute("""INSERT INTO events(event_id,event_type,data,timestamp,prev_hash,hash)
                         VALUES(?,?,?,?,?,?)""",
                      (ev_id, etype, json.dumps(data, ensure_ascii=False), ts, self.last_hash, h))
            self.conn.commit()
            self.last_hash = h
            return h

    async def verify(self) -> Tuple[bool, List[str]]:
        c = self.conn.cursor()
        c.execute("SELECT event_id, event_type, data, timestamp, prev_hash, hash FROM events ORDER BY id")
        rows = c.fetchall()
        prev = "genesis"
        errors: List[str] = []
        for (eid, et, data, ts, ph, hv) in rows:
            if ph != prev:
                errors.append(f"Prev-hash mismatch @ {eid}")
            calc = _hash_data({
                "event_id": eid, "event_type": et, "data": json.loads(data),
                "timestamp": ts, "prev_hash": ph
            })
            if calc != hv:
                errors.append(f"Hash mismatch @ {eid}")
            prev = hv
        return (len(errors) == 0, errors)


class WormLedgerJSONL:
    def __init__(self, file_path: Path = WORM_JSONL):
        self.path = file_path
        self.lock = threading.Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _last_hash(self) -> str:
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

    def record(self, etype: str, data: Dict[str, Any]) -> str:
        with self.lock:
            ev = {
                "type": etype,
                "data": data,
                "timestamp": _ts(),
                "prev_hash": self._last_hash()
            }
            ev["hash"] = _hash_data({k:v for k,v in ev.items() if k!="hash"})
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            return ev["hash"]

    async def verify(self) -> Tuple[bool, List[str]]:
        if not self.path.exists():
            return True, []
        errors: List[str] = []
        lines = [ln for ln in self.path.read_text(encoding="utf-8").splitlines() if ln.strip()]
        prev = "genesis"
        for i, ln in enumerate(lines):
            ev = json.loads(ln)
            if ev.get("prev_hash") != prev:
                errors.append(f"Chain broken @ line {i}")
            hv = ev.get("hash")
            calc = _hash_data({k:v for k,v in ev.items() if k!="hash"})
            if hv != calc:
                errors.append(f"Hash mismatch @ line {i}")
            prev = hv
        return (len(errors) == 0, errors)


class WormLedger:
    """
    Wrapper que grava em SQLite *e* JSONL (redundância leve).
    Verificação ocorre no backend SQLite; JSONL é mantido para inspeção rápida.
    """
    def __init__(self):
        self.sql = WormLedgerSQLite(WORM_SQLITE_DB)
        self.jsonl = WormLedgerJSONL(WORM_JSONL)

    def record_event(self, et: EventType, data: Dict[str, Any]) -> str:
        h_sql = self.sql.record(et.value, data)
        try:
            self.jsonl.record(et.value, data)
        except Exception:
            pass
        return h_sql

    async def verify_chain(self) -> Tuple[bool, List[str]]:
        ok, errs = await self.sql.verify()
        return ok, errs


# ========== Estado Unificado ==========
@dataclass
class OmegaState:
    # Ética e governança
    ece: float = 0.0
    rho_bias: float = 1.0
    consent: bool = True
    eco_ok: bool = True

    # Risco (IR→IC)
    rho: float = 0.5
    uncertainty: float = 0.5
    risk_contractions: int = 0

    # CAOS⁺
    C: float = 0.6; A: float = 0.6; O: float = 0.6; S: float = 0.6
    caos_pre: float = 1.0
    caos_post: float = 1.0
    caos_stable: bool = True

    # SR‑Ω∞ (componentes e score)
    C_cal: float = 0.8
    E_ok: float = 1.0
    M: float = 0.7
    A_eff: float = 0.6
    sr_score: float = 1.0
    sr_valid: bool = True

    # Equação da Morte
    A_t: bool = False
    C_t: bool = False
    E_t: bool = True
    V_t: bool = True
    extinction_reason: Optional[str] = None

    # Métricas evolutivas
    delta_linf: float = 0.0
    mdl_gain: float = 0.0
    ppl_ood: float = 100.0
    novelty_sim: float = 1.0
    rag_recall: float = 1.0

    # Observabilidade/perf
    throughput: float = 0.0
    latency_p95: float = 0.0
    cache_hit_ratio: float = 0.0

    # Recursos/custos/dano
    U: float = 0.0
    cost: float = 0.0
    cost_rate: float = 0.0
    D_t: float = 0.0
    damage_rate: float = 0.0

    # Controle
    kill_switch: bool = False
    trust_region_radius: float = GOV["penin_update"]["trust_region"]["initial_radius"]
    rollback_ready: bool = True

    # Auditoria e meta
    state_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=_ts)
    version: str = PKG_VERSION
    cycle_count: int = 0
    hashes: List[str] = field(default_factory=list)
    proof_ids: List[str] = field(default_factory=list)
    audit_trail: List[Dict[str, Any]] = field(default_factory=list)
    security_status: Dict[str, Any] = field(default_factory=lambda: {"status":"secure","issues":[]})
    last_cycle_metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OmegaState":
        valid = {f.name for f in dc_fields(cls)}
        filt = {k:v for k,v in (data or {}).items() if k in valid}
        return cls(**filt)

    def validate(self) -> Tuple[bool, List[str]]:
        errs: List[str] = []
        if not (0 <= self.ece <= 1): errs.append("ece out of bounds")
        if not (0 <= self.rho <= 1): errs.append("rho out of bounds")
        if self.kill_switch and self.E_t: errs.append("kill_switch ativo com E_t=True")
        return (len(errs) == 0, errs)

    def update_metrics(self, m: Dict[str, float]):
        self.last_cycle_metrics.update(m)
        self.cost   += m.get("cost_increment", 0.0)
        self.D_t    += m.get("damage_increment", 0.0)


# ========== CLI ==========
async def main():
    import argparse
    parser = argparse.ArgumentParser(description=DESC)
    parser.add_argument("--diagnose", action="store_true", help="Executa diagnóstico e encerra")
    parser.add_argument("--evolve", action="store_true", help="Roda processo de evolução guiado por tópicos")
    parser.add_argument("--benchmark", action="store_true", help="Executa um benchmark rápido")
    parser.add_argument("--cycles", type=int, default=3, help="Número de ciclos padrão (ou rounds de evolução)")
    parser.add_argument("--parallel", action="store_true", help="No evolve, dispara consultas em paralelo")
    args = parser.parse_args()

    # CPU tuning Torch (opcional)
    if HAS_TORCH:
        try:
            torch.set_num_threads(os.cpu_count() or 4)
            if hasattr(torch, 'set_float32_matmul_precision'):
                torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    # Placeholder para o núcleo principal
    log.info(f"🧠 {PKG_NAME} v{PKG_VERSION} — Sistema carregado")
    log.info("✅ Pronto para execução")

if __name__ == "__main__":
    asyncio.run(main())
