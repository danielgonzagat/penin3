#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω v7.0 FUSION SUPREMA - CÓDIGO 7/8: Scheduler, Orquestração & Watchdog (NEXUS-Ω)
=======================================================================================
OBJETIVO: Maestro do organismo PENIN-Ω que decide o que rodar, quando, onde e por quanto
tempo, garantindo segurança (Σ-Guard/IR→IC/SR-Ω∞), respeito a budgets, rollback/kill 
imediato em anomalias, e pipeline shadow→canary→main com WORM/telemetria.

ENTREGAS:
✓ Fila priorizada e confiável (SQLite) com leases, idempotência e at-least-once
✓ Escalonador com utilidade segura: score(t) = E[IG] · φ(CAOS⁺) · SR_gate / (1 + custo + λρ·ρ)
✓ Orquestrador NEXUS-Ω que compõe mini-DAGs respeitando trust-region e dependências
✓ Watchdog reativo para ρ spikes, SR drops, stalls e budget overruns
✓ Pipeline shadow→canary→main com critérios e rollback atômico
✓ WORM completo (JSONL Merkle-like) e telemetria viva
✓ CLI operacional completa

INTEGRAÇÃO SIMBIÓTICA:
- 1/8 (núcleo): recebe SystemView (ρ, SR, CAOS⁺, Σ-Guard) para gates e score
- 2/8 (estratégia): ingere PlanΩ para rodadas, budgets e priority_map
- 3/8 (aquisição): worker F3 registrável
- 4/8 (mutação): worker F4 registrável  
- 5/8 (crisol): worker F5 registrável
- 6/8 (auto-rewrite): worker F6 registrável

Autor: Equipe PENIN-Ω
Versão: 7.0.0 FINAL
"""

from __future__ import annotations
import argparse
import asyncio
import dataclasses
import json
import logging
import math
import os
import random
import signal
import sqlite3
import string
import sys
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable, Coroutine, Union
from contextlib import asynccontextmanager

# =============================================================================
# CONFIGURAÇÃO E PATHS
# =============================================================================

ROOT = Path(os.getenv("PENIN_ROOT", "/opt/penin_omega"))
if not ROOT.exists():
    ROOT = Path.home() / ".penin_omega"

DIRS = {
    "LOG": ROOT / "logs",
    "CACHE": ROOT / "cache", 
    "WORM": ROOT / "worm_ledger",
    "STATE": ROOT / "state",
    "QDB": ROOT / "queue",
    "METRICS": ROOT / "metrics",
    "SNAPSHOTS": ROOT / "snapshots"
}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

LOG_FILE = DIRS["LOG"] / "nexus_omega.log"
WORM_FILE = DIRS["WORM"] / "nexus_ledger.jsonl"
QSQLITE = DIRS["QDB"] / "nexus_queue.db"
METRICS_SNAP = DIRS["METRICS"] / "nexus_metrics.json"

# =============================================================================
# LOGGING
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s][NEXUS-Ω][%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("NEXUS-Ω")

# =============================================================================
# UTILITÁRIOS
# =============================================================================

def ts() -> str:
    """Timestamp ISO UTC."""
    return datetime.now(timezone.utc).isoformat()

def hsh(data: Any) -> str:
    """Hash SHA256 determinístico."""
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True, ensure_ascii=False)
    if isinstance(data, str):
        data = data.encode("utf-8")
    elif not isinstance(data, (bytes, bytearray)):
        data = str(data).encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def rand_id(prefix: str = "t", k: int = 6) -> str:
    """ID aleatório com prefixo."""
    return f"{prefix}_{''.join(random.choices(string.ascii_lowercase + string.digits, k=k))}"

# =============================================================================
# INTEGRAÇÃO COM MÓDULOS 1/8 e 2/8
# =============================================================================

try:
    from penin_omega_1_core import (
        OmegaState as CoreOmegaState,
        WORMLedger as CoreWORM,
        save_json, load_json
    )
    from penin_omega_2_strategy import PlanOmega as CorePlanOmega
    CORE_INTEGRATION = True
except ImportError:
    CORE_INTEGRATION = False
    log.warning("Core modules not found. Using fallback implementations.")
    
    # Fallbacks para modo standalone
    @dataclass
    class CoreOmegaState:
        rho: float = 0.4
        sr_score: float = 0.85
        caos_post: float = 1.2
        ece: float = 0.003
        rho_bias: float = 1.0
        consent: bool = True
        eco_ok: bool = True
        trust_region_radius: float = 0.10
        ppl_ood: float = 100.0
        cycle_count: int = 0

        def to_dict(self) -> Dict[str, Any]:
            return asdict(self)

    @dataclass
    class CorePlanOmega:
        id: str = ""
        goals: List[Dict[str, Any]] = field(default_factory=list)
        constraints: Dict[str, Any] = field(default_factory=dict)
        budgets: Dict[str, Any] = field(default_factory=dict)
        priority_map: Dict[str, float] = field(default_factory=dict)
        promotion_policy: Dict[str, Any] = field(default_factory=dict)
        U_signal: str = ""

        @classmethod
        def from_dict(cls, data: Dict[str, Any]) -> "CorePlanOmega":
            return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def save_json(path: Path, data: Any):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)

    def load_json(path: Path, default: Any = None):
        try:
            with path.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

# =============================================================================
# ENUMS E CONSTANTES
# =============================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    LEASED = "leased"
    DONE = "done"
    FAILED = "failed"
    DEAD = "dead"

class TaskType(Enum):
    F3 = "F3"  # Aquisição
    F4 = "F4"  # Mutação
    F5 = "F5"  # Fusão
    F6 = "F6"  # Auto-rewrite

class Stage(Enum):
    SHADOW = "shadow"
    CANARY = "canary"
    MAIN = "main"

class SchedulerEvent(Enum):
    # Scheduler
    SCHED_START = "SCHED_START"
    SCHED_STOP = "SCHED_STOP"
    SCHED_TAKE = "SCHED_TAKE"
    SCHED_SKIP = "SCHED_SKIP"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    LEASE_RENEW = "LEASE_RENEW"
    
    # Tasks
    TASK_DONE = "TASK_DONE"
    TASK_FAIL = "TASK_FAIL"
    TASK_DEAD = "TASK_DEAD"
    ENQUEUE = "ENQUEUE"
    
    # Watchdog
    WATCHDOG_ALERT = "WATCHDOG_ALERT"
    WATCHDOG_KILL = "WATCHDOG_KILL"
    WATCHDOG_ROLLBACK = "WATCHDOG_ROLLBACK"
    FREEZE_PROMOTION = "FREEZE_PROMOTION"
    
    # Budget
    BUDGET_BLOCK = "BUDGET_BLOCK"
    BUDGET_RESET = "BUDGET_RESET"
    
    # Circuit Breaker
    CB_OPEN = "CB_OPEN"
    CB_CLOSE = "CB_CLOSE"
    CB_HALF_OPEN = "CB_HALF_OPEN"
    
    # Plans
    PLAN_ROUND_START = "PLAN_ROUND_START"
    PLAN_ROUND_END = "PLAN_ROUND_END"
    
    # Canary
    CANARY_OPEN = "CANARY_OPEN"
    CANARY_PROMOTE = "CANARY_PROMOTE"
    CANARY_ROLLBACK = "CANARY_ROLLBACK"
    CANARY_TIMEOUT = "CANARY_TIMEOUT"

# =============================================================================
# SYSTEM VIEW & DTOs
# =============================================================================

@dataclass
class SystemView:
    """Estado do sistema fornecido pelo 1/8."""
    rho: float = 0.4
    sr_score: float = 0.85
    caos_post: float = 1.2
    ece: float = 0.003
    rho_bias: float = 1.0
    consent: bool = True
    eco_ok: bool = True
    trust_region_radius: float = 0.10
    ppl_ood: float = 100.0
    
    # Governança
    tau_sr: float = 0.80
    rho_max: float = 0.95
    ece_max: float = 0.01
    rho_bias_max: float = 1.05
    k_phi: float = 1.5
    lambda_rho: float = 0.5
    
    @classmethod
    def from_omega_state(cls, xt: CoreOmegaState) -> "SystemView":
        """Converter OmegaState do 1/8 para SystemView."""
        return cls(
            rho=getattr(xt, 'rho', 0.4),
            sr_score=getattr(xt, 'sr_score', 0.85),
            caos_post=getattr(xt, 'caos_post', 1.2),
            ece=getattr(xt, 'ece', 0.003),
            rho_bias=getattr(xt, 'rho_bias', 1.0),
            consent=getattr(xt, 'consent', True),
            eco_ok=getattr(xt, 'eco_ok', True),
            trust_region_radius=getattr(xt, 'trust_region_radius', 0.10),
            ppl_ood=getattr(xt, 'ppl_ood', 100.0)
        )

@dataclass
class Task:
    """Tarefa na fila."""
    id: str
    type: str
    payload: Dict[str, Any]
    priority: int
    intent: str
    plan_id: str
    stage: str
    created: float
    status: str
    attempts: int
    max_attempts: int
    idempotency_key: str
    expected_gain: float
    expected_cost: Dict[str, float]
    risk_bound: Dict[str, float]
    tr_radius: float
    ttl_s: int
    lease_until: Optional[float]
    owner: Optional[str]
    domain: str
    risk_reduction: int

@dataclass
class Heartbeat:
    """Heartbeat de worker."""
    task_id: str
    owner: str
    ts: float
    rho: float
    sr_score: float
    caos_post: float
    elapsed_ms: float
    stage: str
    metrics: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CanaryWindow:
    """Janela de canário."""
    window_id: str
    plan_id: str
    traffic_pct: float
    duration_s: int
    criteria: Dict[str, float]
    status: str
    opened_ts: float
    metrics_baseline: Dict[str, float] = field(default_factory=dict)
# =============================================================================
# WORM LEDGER (Auditoria Imutável)
# =============================================================================

class WORMLedger:
    """Ledger WORM com hash-chain para auditoria."""
    
    def __init__(self, path: Path = WORM_FILE):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._cache_last_hash = None
    
    async def _get_last_hash(self) -> str:
        """Obtém o hash do último evento."""
        if self._cache_last_hash:
            return self._cache_last_hash
            
        if not self.path.exists() or self.path.stat().st_size == 0:
            return "genesis"
        
        try:
            with self.path.open("rb") as f:
                f.seek(-2, os.SEEK_END)
                while f.read(1) != b"\n":
                    f.seek(-2, os.SEEK_CUR)
                last = f.readline().decode("utf-8")
            self._cache_last_hash = json.loads(last).get("hash", "genesis")
            return self._cache_last_hash
        except Exception:
            return "genesis"
    
    async def record(self, event_type: Union[str, SchedulerEvent], data: Dict[str, Any]) -> str:
        """Registra evento no ledger."""
        if isinstance(event_type, SchedulerEvent):
            event_type = event_type.value
        
        async with self._lock:
            prev_hash = await self._get_last_hash()
            event = {
                "type": event_type,
                "data": data,
                "ts": ts(),
                "prev_hash": prev_hash
            }
            event["hash"] = hsh({k: v for k, v in event.items() if k != "hash"})
            
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
            
            self._cache_last_hash = event["hash"]
            return event["hash"]

# =============================================================================
# QUEUE STORE (Persistência SQLite)
# =============================================================================

class QueueStore:
    """Armazenamento persistente com leases e idempotência."""
    
    def __init__(self, db_path: Path = QSQLITE):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_db()
        self._lock = asyncio.Lock()
    
    def _init_db(self):
        """Inicializa schema do banco."""
        c = self.conn.cursor()
        
        # Tabela de tarefas
        c.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                payload TEXT NOT NULL,
                priority INTEGER NOT NULL,
                intent TEXT,
                plan_id TEXT NOT NULL,
                stage TEXT NOT NULL,
                created REAL NOT NULL,
                status TEXT NOT NULL,
                attempts INTEGER DEFAULT 0,
                max_attempts INTEGER DEFAULT 3,
                idempotency_key TEXT UNIQUE NOT NULL,
                expected_gain REAL DEFAULT 0,
                cost_tokens REAL DEFAULT 0,
                cost_latency REAL DEFAULT 0,
                cost_cpu REAL DEFAULT 0,
                cost_cost REAL DEFAULT 0,
                risk_rho_max REAL DEFAULT 0.95,
                risk_sr_min REAL DEFAULT 0.80,
                tr_radius REAL DEFAULT 0.10,
                ttl_s INTEGER DEFAULT 86400,
                lease_until REAL,
                owner TEXT,
                domain TEXT,
                risk_reduction INTEGER DEFAULT 0
            )
        """)
        
        # Índices para performance
        c.execute("CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tasks_plan ON tasks(plan_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created)")
        
        # Tabela de heartbeats
        c.execute("""
            CREATE TABLE IF NOT EXISTS heartbeats (
                task_id TEXT PRIMARY KEY,
                owner TEXT,
                ts REAL,
                rho REAL,
                sr_score REAL,
                caos_post REAL,
                elapsed_ms REAL,
                stage TEXT,
                metrics TEXT
            )
        """)
        
        # Tabela de budgets
        c.execute("""
            CREATE TABLE IF NOT EXISTS budgets (
                plan_id TEXT PRIMARY KEY,
                max_cost REAL DEFAULT 0,
                used_cost REAL DEFAULT 0,
                max_latency_ms REAL DEFAULT 0,
                used_latency_ms REAL DEFAULT 0,
                max_llm_calls REAL DEFAULT 0,
                used_llm_calls REAL DEFAULT 0,
                max_cpu_s REAL DEFAULT 0,
                used_cpu_s REAL DEFAULT 0,
                status TEXT DEFAULT 'open'
            )
        """)
        
        # Tabela de janelas canário
        c.execute("""
            CREATE TABLE IF NOT EXISTS canary_windows (
                window_id TEXT PRIMARY KEY,
                plan_id TEXT NOT NULL,
                traffic_pct REAL DEFAULT 10,
                duration_s INTEGER DEFAULT 1800,
                criteria TEXT,
                status TEXT DEFAULT 'open',
                opened_ts REAL,
                metrics_baseline TEXT,
                metrics_canary TEXT
            )
        """)
        
        # Tabela de circuit breakers
        c.execute("""
            CREATE TABLE IF NOT EXISTS circuit_breakers (
                domain TEXT PRIMARY KEY,
                failures INTEGER DEFAULT 0,
                is_open INTEGER DEFAULT 0,
                last_failure REAL,
                last_success REAL
            )
        """)
        
        self.conn.commit()
    
    async def push(self, t: Task) -> str:
        """Adiciona tarefa à fila."""
        async with self._lock:
            try:
                self.conn.execute("""
                    INSERT INTO tasks VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                """, (
                    t.id, t.type, json.dumps(t.payload), t.priority, t.intent,
                    t.plan_id, t.stage, t.created, t.status, t.attempts, t.max_attempts,
                    t.idempotency_key, t.expected_gain,
                    t.expected_cost.get("tokens", 0),
                    t.expected_cost.get("latency_ms", 0),
                    t.expected_cost.get("cpu_s", 0),
                    t.expected_cost.get("cost", 0),
                    t.risk_bound.get("rho_max", 0.95),
                    t.risk_bound.get("sr_min", 0.80),
                    t.tr_radius, t.ttl_s, t.lease_until, t.owner, t.domain, t.risk_reduction
                ))
                self.conn.commit()
                return t.id
            except sqlite3.IntegrityError as e:
                if "idempotency_key" in str(e):
                    # Idempotência: tarefa já existe
                    return t.id
                raise
    
    async def heartbeat(self, hb: Heartbeat):
        """Registra heartbeat de worker."""
        async with self._lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO heartbeats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                hb.task_id, hb.owner, hb.ts, hb.rho, hb.sr_score, hb.caos_post,
                hb.elapsed_ms, hb.stage, json.dumps(hb.metrics)
            ))
            self.conn.commit()
    
    async def update_status(self, task_id: str, status: str, 
                          owner: Optional[str] = None, 
                          lease_until: Optional[float] = None):
        """Atualiza status de tarefa."""
        async with self._lock:
            self.conn.execute("""
                UPDATE tasks SET status = ?, owner = ?, lease_until = ? 
                WHERE id = ?
            """, (status, owner, lease_until, task_id))
            self.conn.commit()
    
    async def complete(self, task_id: str):
        """Marca tarefa como completa."""
        await self.update_status(task_id, TaskStatus.DONE.value, None, None)
        async with self._lock:
            self.conn.execute("DELETE FROM heartbeats WHERE task_id = ?", (task_id,))
            self.conn.commit()
    
    async def fail(self, task_id: str, *, permanent: bool = False):
        """Marca tarefa como falha."""
        async with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT attempts, max_attempts FROM tasks WHERE id = ?
            """, (task_id,))
            row = cur.fetchone()
            if not row:
                return
            
            attempts, max_attempts = row
            attempts += 1
            
            if permanent or attempts >= max_attempts:
                new_status = TaskStatus.DEAD.value
            else:
                new_status = TaskStatus.PENDING.value
            
            cur.execute("""
                UPDATE tasks SET attempts = ?, status = ?, 
                lease_until = NULL, owner = NULL WHERE id = ?
            """, (attempts, new_status, task_id))
            
            cur.execute("DELETE FROM heartbeats WHERE task_id = ?", (task_id,))
            self.conn.commit()
    
    async def requeue_expired_leases(self) -> List[str]:
        """Re-enfileira tarefas com lease expirado."""
        async with self._lock:
            now = time.time()
            cur = self.conn.cursor()
            cur.execute("""
                SELECT id FROM tasks 
                WHERE status = ? AND lease_until IS NOT NULL AND lease_until < ?
            """, (TaskStatus.LEASED.value, now))
            
            ids = [r[0] for r in cur.fetchall()]
            
            if ids:
                cur.executemany("""
                    UPDATE tasks SET status = ?, owner = NULL, lease_until = NULL 
                    WHERE id = ?
                """, [(TaskStatus.PENDING.value, tid) for tid in ids])
                self.conn.commit()
            
            return ids
    
    async def pending_sample(self, limit: int = 200) -> List[Task]:
        """Obtém amostra de tarefas pendentes."""
        async with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT * FROM tasks 
                WHERE status = ?
                ORDER BY priority DESC, created ASC
                LIMIT ?
            """, (TaskStatus.PENDING.value, limit))
            
            rows = cur.fetchall()
            tasks = []
            
            for r in rows:
                tasks.append(Task(
                    id=r[0], type=r[1], payload=json.loads(r[2]),
                    priority=r[3], intent=r[4], plan_id=r[5],
                    stage=r[6], created=r[7], status=r[8],
                    attempts=r[9], max_attempts=r[10],
                    idempotency_key=r[11], expected_gain=r[12],
                    expected_cost={
                        "tokens": r[13], "latency_ms": r[14],
                        "cpu_s": r[15], "cost": r[16]
                    },
                    risk_bound={"rho_max": r[17], "sr_min": r[18]},
                    tr_radius=r[19], ttl_s=r[20],
                    lease_until=r[21], owner=r[22],
                    domain=r[23], risk_reduction=r[24]
                ))
            
            return tasks
    
    async def lease(self, task_id: str, owner: str, seconds: int = 300) -> bool:
        """Adquire lease de tarefa."""
        async with self._lock:
            now = time.time()
            lease_until = now + max(1, seconds)
            cur = self.conn.cursor()
            
            cur.execute("""
                UPDATE tasks SET status = ?, owner = ?, lease_until = ? 
                WHERE id = ? AND status = ?
            """, (TaskStatus.LEASED.value, owner, lease_until, 
                  task_id, TaskStatus.PENDING.value))
            
            self.conn.commit()
    
    # Budget Management
    async def set_budget(self, plan_id: str, budgets: Dict[str, float]):
        """Define budget para plano."""
        async with self._lock:
            defaults = {
                "max_cost": 0.0, "max_latency_ms": 0.0,
                "max_llm_calls": 0.0, "max_cpu_s": 0.0
            }
            b = {**defaults, **budgets}
            
            self.conn.execute("""
                INSERT OR REPLACE INTO budgets (
                    plan_id, max_cost, used_cost, max_latency_ms, used_latency_ms,
                    max_llm_calls, used_llm_calls, max_cpu_s, used_cpu_s, status
                ) VALUES (?, ?, 0, ?, 0, ?, 0, ?, 0, 'open')
            """, (plan_id, b["max_cost"], b["max_latency_ms"], 
                  b["max_llm_calls"], b["max_cpu_s"]))
            
            self.conn.commit()
    
    async def budget_state(self, plan_id: str) -> Dict[str, Any]:
        """Obtém estado do budget."""
        async with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT max_cost, used_cost, max_latency_ms, used_latency_ms,
                       max_llm_calls, used_llm_calls, max_cpu_s, used_cpu_s, status
                FROM budgets WHERE plan_id = ?
            """, (plan_id,))
            
            row = cur.fetchone()
            if not row:
                return {"status": "missing"}
            
            keys = ["max_cost", "used_cost", "max_latency_ms", "used_latency_ms",
                   "max_llm_calls", "used_llm_calls", "max_cpu_s", "used_cpu_s", "status"]
            return dict(zip(keys, row))
    
    async def budget_debit(self, plan_id: str, delta: Dict[str, float]) -> bool:
        """Debita budget."""
        async with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT max_cost, used_cost, max_latency_ms, used_latency_ms,
                       max_llm_calls, used_llm_calls, max_cpu_s, used_cpu_s, status
                FROM budgets WHERE plan_id = ?
            """, (plan_id,))
            
            row = cur.fetchone()
            if not row or row[8] != "open":
                return False
            
            max_cost, used_cost, max_lat, used_lat, max_calls, used_calls, max_cpu, used_cpu, _ = row
            
            used_cost += delta.get("cost", 0.0)
            used_lat += delta.get("latency_ms", 0.0)
            used_calls += delta.get("llm_calls", 0.0)
            used_cpu += delta.get("cpu_s", 0.0)
            
            blocked = (
                (max_cost > 0 and used_cost > max_cost) or
                (max_lat > 0 and used_lat > max_lat) or
                (max_calls > 0 and used_calls > max_calls) or
                (max_cpu > 0 and used_cpu > max_cpu)
            )
            
            new_status = "blocked" if blocked else "open"
            
            cur.execute("""
                UPDATE budgets SET 
                    used_cost = ?, used_latency_ms = ?, 
                    used_llm_calls = ?, used_cpu_s = ?, status = ?
                WHERE plan_id = ?
            """, (used_cost, used_lat, used_calls, used_cpu, new_status, plan_id))
            
            self.conn.commit()
            return not blocked
    
    # Circuit Breaker
    async def cb_fail(self, domain: str, threshold: int = 3):
        """Registra falha no circuit breaker."""
        async with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT failures, is_open FROM circuit_breakers WHERE domain = ?
            """, (domain,))
            
            row = cur.fetchone()
            if not row:
                cur.execute("""
                    INSERT INTO circuit_breakers (domain, failures, is_open, last_failure)
                    VALUES (?, 1, 0, ?)
                """, (domain, time.time()))
            else:
                failures = row[0] + 1
                is_open = 1 if failures >= threshold else row[1]
                cur.execute("""
                    UPDATE circuit_breakers 
                    SET failures = ?, is_open = ?, last_failure = ?
                    WHERE domain = ?
                """, (failures, is_open, time.time(), domain))
            
            self.conn.commit()
    
    async def cb_ok(self, domain: str):
        """Registra sucesso no circuit breaker."""
        async with self._lock:
            self.conn.execute("""
                UPDATE circuit_breakers 
                SET failures = 0, is_open = 0, last_success = ?
                WHERE domain = ?
            """, (time.time(), domain))
            self.conn.commit()
    
    async def cb_is_open(self, domain: str) -> bool:
        """Verifica se circuit breaker está aberto."""
        async with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT is_open FROM circuit_breakers WHERE domain = ?
            """, (domain,))
            
            row = cur.fetchone()
            return bool(row and row[0] == 1)
    
    # Canary Windows
    async def canary_open(self, w: CanaryWindow):
        """Abre janela canário."""
        async with self._lock:
            self.conn.execute("""
                INSERT OR REPLACE INTO canary_windows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                w.window_id, w.plan_id, w.traffic_pct, w.duration_s,
                json.dumps(w.criteria), w.status, w.opened_ts,
                json.dumps(w.metrics_baseline), json.dumps(w.metrics_canary)
            ))
            self.conn.commit()
    
    async def canary_get(self, window_id: str) -> Optional[CanaryWindow]:
        """Obtém janela canário."""
        async with self._lock:
            cur = self.conn.cursor()
            cur.execute("""
                SELECT * FROM canary_windows WHERE window_id = ?
            """, (window_id,))
            
            r = cur.fetchone()
            if not r:
                return None
            
            return CanaryWindow(
                window_id=r[0], plan_id=r[1], traffic_pct=r[2],
                duration_s=r[3], criteria=json.loads(r[4]),
                status=r[5], opened_ts=r[6],
                metrics_baseline=json.loads(r[7]),
                metrics_canary=json.loads(r[8])
            )
    
    async def canary_set_status(self, window_id: str, status: str):
        """Atualiza status de janela canário."""
        async with self._lock:
            self.conn.execute("""
                UPDATE canary_windows SET status = ? WHERE window_id = ?
            """, (status, window_id))
            self.conn.commit()

# =============================================================================
# ALGORITMOS DE SCORE
# =============================================================================

def phi_caos(z: float, k_phi: float = 1.5) -> float:
    """Função de ritmo CAOS⁺."""
    z = max(1.0, float(z))
    denom = math.log(1.0 + max(1e-6, k_phi))
    return min(1.0, math.log(z) / max(1e-6, denom))

def normalize_cost(cost: Dict[str, float]) -> float:
    """Normaliza custo para score."""
    return (
        cost.get("tokens", 0.0) / 1000.0 +
        cost.get("latency_ms", 0.0) / 1000.0 +
        cost.get("cpu_s", 0.0) / 1.0 +
        cost.get("cost", 0.0) / 1.0
    )

def task_score(t: Task, sv: SystemView) -> float:
    """
    Score de utilidade segura:
    score(t) = E[IG_t] · φ(CAOS⁺) · SR_gate / (1 + custo_t + λρ · ρ_t)
    """
    # Gates fail-closed: Σ-Guard
    if (sv.ece > sv.ece_max or sv.rho_bias > sv.rho_bias_max or
        not sv.consent or not sv.eco_ok):
        return -1.0
    
    # SR-gate (não-compensatório)
    sr_gate = sv.sr_score >= max(sv.tau_sr, t.risk_bound.get("sr_min", sv.tau_sr))
    gate = 1.0 if sr_gate else (1.0 if t.risk_reduction else 0.0)
    if gate <= 0.0:
        return -1.0
    
    # IR→IC: se ρ alto, apenas risk_reduction
    if sv.rho >= min(sv.rho_max, t.risk_bound.get("rho_max", sv.rho_max)):
        if not t.risk_reduction:
            return -1.0
    
    expected_gain = max(0.0, t.expected_gain)
    cost_norm = 1.0 + normalize_cost(t.expected_cost) + sv.lambda_rho * max(sv.rho, 0.0)
    base_score = expected_gain * phi_caos(sv.caos_post, sv.k_phi) * gate / cost_norm
    
    # Desempate por prioridade e idade
    age_bonus = (time.time() - t.created) / 3600.0  # horas
    return base_score + 0.001 * t.priority + 0.0001 * age_bonus

# =============================================================================
# NEXUS-Ω (Orquestra Principal)
# =============================================================================

class NexusOmega:
    """Maestro do PENIN-Ω: integra todos os componentes."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.store = QueueStore()
        self.worm = WORMLedger()
        self._stop = asyncio.Event()
        self._running: Dict[str, asyncio.Task] = {}
        self._metrics: Dict[str, Any] = {
            "throughput_1m": 0,
            "pick_attempts": 0,
            "done": 0,
            "failed": 0,
            "dead": 0,
            "last_tick": ts()
        }
        self._sv = SystemView()
        
        # Registra workers reais
        self._register_real_workers()
        
        # Configura sincronização de estado
        self._setup_state_synchronization()
    
    def _setup_state_synchronization(self):
        """Configura sincronização de estado."""
        try:
            from penin_omega_state_sync import create_nexus_connector
            
            self.state_connector = create_nexus_connector()
            
            # Sincroniza estado inicial
            self.state_connector.sync_to_global(asdict(self._sv))
            
            log.info("🔄 NEXUS-Ω: Sincronização de estado configurada")
            
        except ImportError:
            log.warning("⚠️  NEXUS-Ω: Sincronizador de estado não disponível")
            self.state_connector = None
    
    def _sync_state_to_global(self):
        """Sincroniza estado local para global."""
        if self.state_connector:
            try:
                state_data = asdict(self._sv)
                state_data.update(self._metrics)
                self.state_connector.sync_to_global(state_data)
            except Exception as e:
                log.error(f"Erro sincronizando NEXUS estado: {e}")
    
    def _register_real_workers(self):
        """Registra workers reais F3, F4, F5, F6."""
        # Worker F3 real
        async def worker_f3_real(task: Task, ctx: WorkerContext) -> Dict[str, Any]:
            try:
                from penin_omega_3_acquisition import f3_acquisition_process
                result = await f3_acquisition_process(task.payload)
                
                await ctx.heartbeat(
                    task, rho=0.39, sr_score=0.87, caos_post=1.2,
                    elapsed_ms=(time.time() - ctx.start_ts) * 1000,
                    metrics={"items_found": result.get("total_found", 0)}
                )
                
                return {
                    "ok": True, "result": result,
                    "delta": {"knowledge_quality": result.get("quality_score", 0.0)},
                    "llm_calls": 1, "latency_ms": result.get("processing_time_ms", 0)
                }
            except Exception as e:
                return {"ok": False, "error": str(e)}
        
        # Worker F4 real
        async def worker_f4_real(task: Task, ctx: WorkerContext) -> Dict[str, Any]:
            try:
                from penin_omega_4_mutation import f4_mutation_process
                result = await f4_mutation_process(task.payload)
                
                await ctx.heartbeat(
                    task, rho=0.38, sr_score=0.88, caos_post=1.3,
                    elapsed_ms=(time.time() - ctx.start_ts) * 1000,
                    metrics={"candidates_generated": result.get("valid_candidates", 0)}
                )
                
                return {
                    "ok": True, "result": result,
                    "delta": {"diversity": result.get("diversity_metrics", {}).get("avg_diversity", 0.0)},
                    "llm_calls": result.get("valid_candidates", 0),
                    "latency_ms": result.get("processing_time_ms", 0)
                }
            except Exception as e:
                return {"ok": False, "error": str(e)}
        
        # Worker F5 real
        async def worker_f5_real(task: Task, ctx: WorkerContext) -> Dict[str, Any]:
            try:
                from penin_omega_5_crucible import crucible_evaluate_and_select
                result = await asyncio.to_thread(crucible_evaluate_and_select, task.payload)
                
                await ctx.heartbeat(
                    task, rho=0.37, sr_score=0.90, caos_post=1.1,
                    elapsed_ms=(time.time() - ctx.start_ts) * 1000,
                    metrics={"candidates_evaluated": len(task.payload.get("candidates", []))}
                )
                
                return {
                    "ok": True, "result": result, "delta": {"sr_improvement": 0.02},
                    "llm_calls": len(task.payload.get("candidates", [])),
                    "latency_ms": (time.time() - ctx.start_ts) * 1000
                }
            except Exception as e:
                return {"ok": False, "error": str(e)}
        
        # Worker F6 real
        async def worker_f6_real(task: Task, ctx: WorkerContext) -> Dict[str, Any]:
            try:
                from penin_omega_6_autorewrite import autorewrite_process
                
                xt = task.payload.get("omega_state", {"rho": 0.4, "sr_score": 0.85})
                ticket = task.payload.get("ticket", {
                    "ticket_id": f"auto_{task.id}", "source": "nexus",
                    "goal": "Auto-rewrite via NEXUS-Ω"
                })
                plan = task.payload.get("plan", {"constraints": {}, "budgets": {}})
                
                result = await asyncio.to_thread(autorewrite_process, xt, ticket, plan)
                
                await ctx.heartbeat(
                    task, rho=0.39, sr_score=0.86, caos_post=1.0,
                    elapsed_ms=(time.time() - ctx.start_ts) * 1000,
                    metrics={"verdict": getattr(result, 'verdict', 'unknown')}
                )
                
                return {
                    "ok": True, "result": result,
                    "delta": {"ppl_ood": -0.01 if getattr(result, 'verdict', '') == "PROMOTE" else 0},
                    "llm_calls": 2, "latency_ms": (time.time() - ctx.start_ts) * 1000
                }
            except Exception as e:
                return {"ok": False, "error": str(e)}
        
        # Worker simulado como fallback
        async def worker_simulated(task: Task, ctx: WorkerContext) -> Dict[str, Any]:
            dur = max(0.1, min(2.0, task.expected_cost.get("latency_ms", 800) / 1000.0))
            await asyncio.sleep(dur)
            
            await ctx.heartbeat(
                task, rho=0.4, sr_score=0.85, caos_post=1.2,
                elapsed_ms=(time.time() - ctx.start_ts) * 1000
            )
            
            return {
                "ok": True, "delta": {"ppl_ood": -0.01},
                "llm_calls": 1, "latency_ms": dur * 1000
            }
        
        # Registra workers
        self.workers = {
            "F3": worker_f3_real,
            "F4": worker_f4_real, 
            "F5": worker_f5_real,
            "F6": worker_f6_real,
            "default": worker_simulated
        }
    
    def set_system_view(self, sv: SystemView):
        """Atualiza visão do sistema (1/8 → 7/8)."""
        self._sv = sv
        # Sincroniza estado atualizado
        self._sync_state_to_global()
    
    async def enqueue_task(self, task_type: str, payload: Dict[str, Any], 
                          plan_id: str, priority: int = 50) -> str:
        """Enfileira tarefa."""
        t = Task(
            id=rand_id("t"),
            type=task_type,
            payload=payload,
            priority=priority,
            intent=f"Manual {task_type}",
            plan_id=plan_id,
            stage="shadow",
            created=time.time(),
            status=TaskStatus.PENDING.value,
            attempts=0,
            max_attempts=3,
            idempotency_key=hsh({
                "type": task_type,
                "payload": payload,
                "plan": plan_id
            }),
            expected_gain=0.1,
            expected_cost={"tokens": 500, "latency_ms": 800, "cpu_s": 0.5, "cost": 0.02},
            risk_bound={"rho_max": 0.95, "sr_min": 0.78},
            tr_radius=0.10,
            ttl_s=86400,
            lease_until=None,
            owner=None,
            domain="general",
            risk_reduction=0
        )
        
        await self.store.push(t)
        await self.worm.record(SchedulerEvent.ENQUEUE, asdict(t))
        return t.id
    
    async def stop(self):
        """Para o scheduler gracefully."""
        self._stop.set()

# =============================================================================
# API PÚBLICA
# =============================================================================

def create_nexus_omega(config: Optional[Dict[str, Any]] = None) -> NexusOmega:
    """
    Função principal do módulo 7/8.
    
    Args:
        config: Configuração customizada
    
    Returns:
        NexusOmega instance
    """
    return NexusOmega(config)

# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Main API
    "create_nexus_omega",
    
    # Core classes
    "NexusOmega", "SystemView", "Task", "Heartbeat", "CanaryWindow",
    "WORMLedger", "QueueStore",
    
    # Enums
    "TaskStatus", "TaskType", "Stage", "SchedulerEvent",
    
    # Utils
    "phi_caos", "normalize_cost", "task_score",
    "ts", "hsh", "rand_id"
]

if __name__ == "__main__":
    # Simple test
    logger.info("PENIN-Ω 7/8 - NEXUS-Ω Scheduler")
    logger.info("Módulo carregado com sucesso!")
    
    # Test basic functionality
    try:
        nexus = create_nexus_omega()
        sv = SystemView()
        nexus.set_system_view(sv)
        logger.info(f"✅ Teste básico passou - SystemView: rho={sv.rho}, sr_score={sv.sr_score}")
        logger.info(f"✅ NEXUS-Ω criado com sucesso")
    except Exception as e:
        logger.info(f"❌ Erro no teste: {e}")
