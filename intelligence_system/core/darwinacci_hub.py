"""
Darwinacci Hub - Universal Connector
===================================

Single import to access a shared Darwinacci orchestrator instance and helper
methods for plugging into other subsystems (V7, PENIN³, RL, MNIST, DB, etc.).

Usage:
    from intelligence_system.core.darwinacci_hub import get_orchestrator, evolve_once

Notes:
- Lazy-inits a DarwinacciOrchestrator and reuses it.
- Provides a minimal, stable API; avoids heavy deps.
"""
from __future__ import annotations
import threading
from typing import Optional, Callable, Dict, Any
import json
from datetime import datetime
from pathlib import Path

# Import within package context (absolute) to avoid path issues
try:
    from intelligence_system.extracted_algorithms.darwin_engine_darwinacci import (
        DarwinacciOrchestrator,
    )
except Exception:  # fallback if same package context
    from extracted_algorithms.darwin_engine_darwinacci import DarwinacciOrchestrator

_lock = threading.Lock()
_ORCH: Optional[DarwinacciOrchestrator] = None

# Synapse file paths (shared across subsystems)
_TRANSFER_FILE = Path("/root/intelligence_system/data/darwin_transfer_latest.json")
_FEEDBACK_FILE = Path("/root/intelligence_system/data/v7_to_darwin_feedback.json")  # UPDATED: correct V7 feedback path
_BRAIN_SYNC_FILE = Path("/root/intelligence_system/data/brain_sync_signal.json")

# Ensure data directory exists
try:
    _TRANSFER_FILE.parent.mkdir(parents=True, exist_ok=True)
except Exception:
    pass


def get_orchestrator(activate: bool = True,
                     population_size: int = 50,
                     max_cycles: int = 5,
                     seed: int = 42,
                     fitness_fn: Optional[Callable[[Any], float]] = None) -> DarwinacciOrchestrator:
    global _ORCH
    with _lock:
        if _ORCH is None:
            _ORCH = DarwinacciOrchestrator(
                population_size=population_size,
                max_cycles=max_cycles,
                seed=seed,
                fitness_fn=fitness_fn,
            )
            if activate:
                _ORCH.activate(fitness_fn=fitness_fn)
        else:
            if activate and not _ORCH.active:
                _ORCH.activate(fitness_fn=fitness_fn)
    return _ORCH


def evolve_once(fitness_fn: Optional[Callable[[Any], float]] = None) -> Dict[str, Any]:
    orch = get_orchestrator(activate=True, fitness_fn=fitness_fn)
    return orch.evolve_generation(fitness_fn=fitness_fn)


def best_genome() -> Optional[Dict[str, Any]]:
    orch = get_orchestrator(activate=False)
    return orch.get_best_genome() if orch else None


def status() -> Dict[str, Any]:
    orch = get_orchestrator(activate=False)
    return orch.get_status() if orch else {"active": False}


# -------------------------
# Synapse I/O (file-based)
# -------------------------

def _now_iso() -> str:
    try:
        return datetime.utcnow().isoformat()
    except Exception:
        return ""


def _read_json_safely(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        text = path.read_text(encoding="utf-8", errors="ignore")
        if not text.strip():
            return None
        return json.loads(text)
    except Exception:
        return None


def _write_json_safely(path: Path, obj: Dict[str, Any]) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(path)
        return True
    except Exception:
        return False


def write_transfer(best: Optional[Dict[str, Any]] = None, stats: Optional[Dict[str, Any]] = None) -> bool:
    """Publish best genome + stats for V7 bridge consumption."""
    if best is None:
        best = best_genome()
    if not best:
        return False
    payload: Dict[str, Any] = {
        "source": "darwinacci",
        "timestamp": _now_iso(),
        "genome": best,
    }
    if isinstance(stats, dict):
        payload["stats"] = {
            k: v for k, v in stats.items()
            if isinstance(v, (int, float, str)) or k in ("generation", "population_size")
        }
    return _write_json_safely(_TRANSFER_FILE, payload)


def read_v7_feedback() -> Optional[Dict[str, Any]]:
    return _read_json_safely(_FEEDBACK_FILE)


def read_brain_sync_signal() -> Optional[Dict[str, Any]]:
    return _read_json_safely(_BRAIN_SYNC_FILE)


def apply_feedback_to_engine(feedback: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Apply light-weight feedback to the running Darwinacci engine.
    
    Now enhanced with V7 performance-based adaptation:
    - If transfer_helpful=True: Reduce mutation (exploit current direction)
    - If transfer_helpful=False: Increase mutation (explore more)
    - Adapt based on performance deltas

    Returns a dict with actions_applied metadata for logging/audit.
    """
    applied: Dict[str, Any] = {"actions": []}
    if not feedback:
        return applied

    orch = get_orchestrator(activate=False)
    if not orch or not getattr(orch, "engine", None):
        return applied

    # NEW: V7 performance-based adaptation
    try:
        transfer_helpful = bool(feedback.get("transfer_helpful", False))
        performance = feedback.get("performance", {})
        mnist_delta = float(performance.get("mnist_delta", 0.0))
        cart_delta = float(performance.get("cartpole_delta", 0.0))
        
        engine = orch.engine
        
        if transfer_helpful:
            # Transfer helped! Reduce mutation to exploit current direction
            old_mut = getattr(engine, "mutation_rate", 0.1)
            new_mut = max(0.05, old_mut * 0.9)  # Reduce by 10%, min 5%
            engine.mutation_rate = new_mut
            applied["actions"].append(f"mutation_rate: {old_mut:.3f} → {new_mut:.3f} (exploit)")
            
            # Also reduce crossover slightly (focus on current winners)
            old_cx = getattr(engine, "crossover_rate", 0.7)
            new_cx = max(0.5, old_cx * 0.95)
            engine.crossover_rate = new_cx
            applied["actions"].append(f"crossover_rate: {old_cx:.3f} → {new_cx:.3f}")
            
        else:
            # Transfer didn't help. Increase mutation to explore more
            old_mut = getattr(engine, "mutation_rate", 0.1)
            new_mut = min(0.3, old_mut * 1.1)  # Increase by 10%, max 30%
            engine.mutation_rate = new_mut
            applied["actions"].append(f"mutation_rate: {old_mut:.3f} → {new_mut:.3f} (explore)")
            
            # Increase crossover (mix more genomes)
            old_cx = getattr(engine, "crossover_rate", 0.7)
            new_cx = min(0.9, old_cx * 1.05)
            engine.crossover_rate = new_cx
            applied["actions"].append(f"crossover_rate: {old_cx:.3f} → {new_cx:.3f}")
        
        applied["v7_feedback"] = f"helpful={transfer_helpful}, MNIST Δ={mnist_delta:+.3f}%, CartPole Δ={cart_delta:+.1f}"
    except Exception as e:
        applied["v7_feedback_error"] = str(e)

    # 1) Exploration boost (legacy, kept for compatibility)
    if bool(feedback.get("explore_more", False)):
        old = getattr(orch, "max_cycles_per_call", 5)
        orch.max_cycles_per_call = max(1, min(10, int(old) + 1))
        applied["actions"].append({"type": "explore_more", "old": old, "new": orch.max_cycles_per_call})

    # 2) Focus on traits (best-effort): bias first few genomes
    focus = feedback.get("focus_on_traits")
    if isinstance(focus, dict):
        try:
            pop = getattr(orch.engine, "population", [])
            changed = 0
            for i in range(min(3, len(pop))):
                g = dict(pop[i])
                if "hidden_size" in focus:
                    try:
                        g["hidden_size"] = float(focus["hidden_size"])  # keep as float; engine accepts floats
                    except Exception:
                        pass
                if "learning_rate" in focus:
                    try:
                        g["learning_rate"] = float(focus["learning_rate"])
                    except Exception:
                        pass
                pop[i] = g
                changed += 1
            applied["actions"].append({"type": "focus_on_traits", "changed_genomes": changed})
        except Exception:
            # best-effort; ignore failures
            pass

    # 3) Brain sync hints
    brain = feedback.get("brain_hint")
    if isinstance(brain, dict) and brain.get("recommendation") == "increase_entropy":
        old = getattr(orch, "omega_boost", 0.0)
        orch.omega_boost = min(1.0, old + 0.05)
        applied["actions"].append({"type": "omega_boost", "old": old, "new": orch.omega_boost})

    # Persist feedback application for audit (best-effort)
    try:
        audit_dir = Path("/root/darwinacci_omega/data")
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_file = audit_dir / "feedback_applied.jsonl"
        line = json.dumps({
            "timestamp": _now_iso(),
            "feedback": feedback,
            "applied": applied,
        }, ensure_ascii=False)
        with audit_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass

    return applied

