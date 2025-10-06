from __future__ import annotations

import os
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import gzip, json, os

from darwinacci_omega.core.engine import DarwinacciEngine
from darwinacci_omega.plugins import toy
from darwinacci_omega.core.evaluator import EvaluatorPipeline
from darwinacci_omega.core.env_plugins import load_portfolio_preset


app = FastAPI(title="Darwinacci API")
_ENGINE: DarwinacciEngine | None = None


class StartRequest(BaseModel):
    cycles: int = 3
    pop: int = 32
    seed: int = 123


@app.get("/status")
def status():
    if _ENGINE is None:
        return {"active": False}
    e = _ENGINE
    return {
        'active': True,
        'pop_size': e.pop_size,
        'has_champion': e.arena.champion is not None,
    }


@app.post("/start")
def start(req: StartRequest):
    global _ENGINE
    # Ensure WORM paths use API-local defaults if unspecified
    os.environ.setdefault('DARWINACCI_WORM_PATH', 'data/worm.csv')
    os.environ.setdefault('DARWINACCI_WORM_HEAD', 'data/worm_head.txt')
    # Optional portfolio preset via env
    portfolio = os.getenv('DARWINACCI_PORTFOLIO_PRESET')
    if portfolio:
        fns, names = load_portfolio_preset(portfolio)
        pipe = EvaluatorPipeline(toy.evaluate, portfolio=fns, task_names=names)
        pipe.use_portfolio = True
        if os.getenv('DARWINACCI_CURRICULUM', '0') == '1':
            pipe.use_curriculum = True
        eval_fn = pipe.evaluate
    else:
        eval_fn = EvaluatorPipeline(toy.evaluate).evaluate
    _ENGINE = DarwinacciEngine(toy.init_genome, eval_fn, max_cycles=req.cycles, pop_size=req.pop, seed=req.seed)
    champ = _ENGINE.run(max_cycles=req.cycles)
    return {"champion": bool(champ is not None), "score": float(champ.score if champ else 0.0)}


@app.post("/import")
def import_ckpt(path: str):
    global _ENGINE
    if _ENGINE is None:
        eval_fn = EvaluatorPipeline(toy.evaluate).evaluate
        _ENGINE = DarwinacciEngine(toy.init_genome, eval_fn, max_cycles=1, pop_size=8, seed=1)
    cycle = _ENGINE.load_checkpoint_json(path)
    return {"restored_cycle": int(cycle)}


@app.get("/portfolio")
def portfolio():
    name = os.getenv('DARWINACCI_PORTFOLIO_PRESET', 'symbolic')
    fns, names = load_portfolio_preset(name)
    return {"preset": name, "tasks": names}


class ExportRequest(BaseModel):
    path: str
    note: Optional[str] = None


@app.post("/export")
def export_ckpt(req: ExportRequest):
    """Export current engine snapshot to a gzipped JSON checkpoint."""
    global _ENGINE
    if _ENGINE is None:
        return {"exported": False, "error": "engine_not_active"}
    # Build a minimal self-consistent payload; reuse engine's checkpoint structure
    tmp_dir = os.path.dirname(req.path) or '.'
    os.makedirs(tmp_dir, exist_ok=True)
    # Use internal metrics to compose a payload without re-evaluating
    payload = {
        'format_version': 1,
        'cycle': getattr(_ENGINE, '_current_cycle', 0) or 0,
        'rng_state': _ENGINE.rng.getstate(),
        'population': _ENGINE.population,
        'archive': [
            {
                'idx': int(idx),
                'best_score': float(cell.best_score),
                'behavior': list(cell.behavior),
                'genome': dict(getattr(cell, 'genome', {}) or {}),
            }
            for idx, cell in _ENGINE.archive.bests()
        ],
        'novelty_size': len(_ENGINE.novel.mem),
        'champion': {
            'genome': _ENGINE.arena.champion.genome if _ENGINE.arena.champion else None,
            'score': _ENGINE.arena.champion.score if _ENGINE.arena.champion else None,
            'behavior': _ENGINE.arena.champion.behavior if _ENGINE.arena.champion else None,
            'metrics': _ENGINE.arena.champion.metrics if _ENGINE.arena.champion else None,
        },
    }
    if req.note:
        payload['note'] = req.note
    # Optional HMAC similar to engine
    try:
        key = os.getenv('DARWINACCI_HMAC_KEY')
        if key:
            base = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            sig = __import__('hashlib').sha256((key + '|' + base).encode()).hexdigest()
            payload['hmac'] = sig
    except Exception:
        pass
    tmp = req.path + '.tmp'
    if req.path.endswith('.gz'):
        with gzip.open(tmp, 'wt') as f:
            json.dump(payload, f)
    else:
        with open(tmp, 'w') as f:
            json.dump(payload, f)
    os.replace(tmp, req.path)
    return {"exported": True, "path": req.path}


@app.post("/stop")
def stop():
    """Request cooperative stop of the running engine."""
    global _ENGINE
    if _ENGINE is None:
        return {"active": False}
    try:
        _ENGINE.stop()
        return {"active": True, "stopping": True}
    except Exception as e:
        return {"active": True, "stopping": False, "error": str(e)}


class ExportSkillsRequest(BaseModel):
    path: str
    k: int = 16


@app.post("/export-skills")
def export_skills(req: ExportSkillsRequest):
    global _ENGINE
    if _ENGINE is None:
        return {"exported": False, "error": "engine_not_active"}
    ok = _ENGINE.export_skills(req.path, k=req.k)
    return {"exported": bool(ok), "path": req.path, "k": req.k}
