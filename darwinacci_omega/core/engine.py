import random, math
import os
import logging
from dataclasses import dataclass
from typing import Callable, Dict, Any, List
from .constants import rng, clamp
from .worm import Worm
from .gates import SigmaGuard
from .golden_spiral import GoldenSpiralArchive
from .novelty_phi import Novelty
from .f_clock import TimeCrystal
from .darwin_ops import tournament, uniform_cx, gaussian_mut, prune_genes
from .godel_kick import godel_kick
from .champion import Arena, Champ
from .multiobj import agg, dominates
import json, gzip
from pathlib import Path
import statistics
from .schemas import CheckpointPayload
import concurrent.futures as cf

logger = logging.getLogger(__name__)

Genome = Dict[str,float]
EvalFn = Callable[[Genome, random.Random], Dict[str,Any]]
InitFn = Callable[[random.Random], Genome]

@dataclass
class Individual:
    genome: Genome
    metrics: Dict[str,Any]
    behavior: List[float]
    score: float

class DarwinacciEngine:
    """
    Híbrido Darwin + Fibonacci:
    - Darwin executa evolução local
    - Fibonacci cadencia budgets e faz superposição dos campeões
    - Gödel-kick quebra estagnação
    - Golden-Spiral Archive mantém QD/coverage
    """
    def __init__(self, init_fn:InitFn, eval_fn:EvalFn, max_cycles:int=7, pop_size:int=48, seed:int=123):
        # FIX 1.3: Validação de parâmetros
        if not callable(init_fn):
            raise ValueError("init_fn must be callable")
        if not callable(eval_fn):
            raise ValueError("eval_fn must be callable")
        if max_cycles <= 0:
            raise ValueError("max_cycles must be positive")
        if pop_size <= 0:
            raise ValueError("pop_size must be positive")
        if seed < 0:
            raise ValueError("seed must be non-negative")

        self.rng=random.Random(seed)
        self.init_fn=init_fn; self.eval_fn=eval_fn
        self.clock=TimeCrystal(max_cycles=max_cycles)
        self.guard=SigmaGuard()
        # Config via env/JSON minimal surface
        cfg_path = os.getenv('DARWINACCI_CONFIG_PATH')
        cfg = {}
        if cfg_path and os.path.exists(cfg_path):
            try:
                with open(cfg_path) as f:
                    cfg = json.load(f)
            except Exception:
                cfg = {}
        bins = int(os.getenv('DARWINACCI_BINS', str(cfg.get('bins', 89))))
        nov_k = int(os.getenv('DARWINACCI_NOVELTY_K', str(cfg.get('novelty_k', 7))))
        nov_max = int(os.getenv('DARWINACCI_NOVELTY_MAX', str(cfg.get('novelty_max', 2000))))
        arena_hist = int(os.getenv('DARWINACCI_ARENA_HIST', str(cfg.get('arena_hist', 8))))
        worm_path = os.getenv('DARWINACCI_WORM_PATH', cfg.get('worm_path', 'data/worm.csv'))
        worm_head = os.getenv('DARWINACCI_WORM_HEAD', cfg.get('worm_head', 'data/worm_head.txt'))

        self.worm=Worm(path=worm_path, head=worm_head)
        self.archive=GoldenSpiralArchive(bins=bins)
        self.novel=Novelty(k=nov_k,max_size=nov_max)
        self.arena=Arena(hist=arena_hist)
        self.pop_size=pop_size
        self.population: List[Genome] = [self.init_fn(self.rng) for _ in range(pop_size)]
        self.last_best=None; self.last_score=-1e9
        # Cooperative halt flag for external controllers (e.g., API /stop)
        self._halt: bool = False
        # Track current cycle for observability
        self._current_cycle: int = 0
        # Metrics history for simple anomaly detection (z-score)
        self._hist_best: List[float] = []
        self._hist_cov: List[float] = []
        # Allow overriding evaluation trials for heavy external fitness
        try:
            self._trials = int(os.getenv("DARWINACCI_TRIALS", "3"))
        except Exception:
            self._trials = 3

        # Optional Prometheus metrics
        self._prom = None
        try:
            if os.getenv('DARWINACCI_PROMETHEUS', '0') == '1':
                from prometheus_client import start_http_server, Gauge
                port = int(os.getenv('DARWINACCI_PROM_PORT', '8011'))
                start_http_server(port)
                self._prom = {
                    'best_score': Gauge('darwinacci_best_score', 'Best score per cycle'),
                    'coverage': Gauge('darwinacci_coverage', 'Archive coverage (0-1)'),
                    'novelty_archive_size': Gauge('darwinacci_novelty_size', 'Size of novelty archive'),
                    'mut': Gauge('darwinacci_mut', 'Mutation rate per cycle'),
                    'cx': Gauge('darwinacci_cx', 'Crossover rate per cycle'),
                    'elite': Gauge('darwinacci_elite', 'Elite size per cycle'),
                    # New gauges
                    'z_best': Gauge('darwinacci_z_best', 'Z-score anomaly for best score'),
                    'z_cov': Gauge('darwinacci_z_cov', 'Z-score anomaly for coverage'),
                    'accepted_rate': Gauge('darwinacci_accepted_rate', 'Acceptance rate windowed'),
                    'canary_pass_rate': Gauge('darwinacci_canary_pass_rate', 'Canary promotion pass rate windowed'),
                }
                logger.info(f"[Darwinacci] 📡 Prometheus on :{port}")
        except Exception:
            self._prom = None

        logger.info(
            f"[Darwinacci] Engine initialized: pop_size={pop_size}, max_cycles={max_cycles}, seed={seed}, "
            f"bins={bins}, novelty=(k={nov_k}, max={nov_max}), arena_hist={arena_hist}"
        )

    def _evaluate(self, g:Genome)->Individual:
        # Deterministic multi-trial evaluation for robustness
        trials=self._trials
        vals=[]; m_last=None
        seeds=[self.rng.randint(1, 10_000_000) for _ in range(trials)]
        parallel = os.getenv('DARWINACCI_PARALLEL_EVAL', '0') == '1' and trials > 1
        backend = os.getenv('DARWINACCI_EVAL_BACKEND', 'threads')
        if parallel and backend == 'threads':
            # Deterministic order: map seeds to indices and collect by index
            def run_one(idx_seed):
                idx, sd = idx_seed
                try:
                    local_rng = random.Random(sd)
                    res = self.eval_fn(g, local_rng)
                    return idx, res
                except Exception:
                    return idx, {"objective": 0.0}
            max_workers = max(1, min(trials, int(os.getenv('DARWINACCI_EVAL_WORKERS', '4'))))
            with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
                results = list(ex.map(run_one, list(enumerate(seeds))))
            # sort by original index to keep determinism
            results.sort(key=lambda t: t[0])
            metrics_list = [m for _, m in results]
            m_last = metrics_list[-1] if metrics_list else {"objective": 0.0}
            vals = [float(m.get("objective", 0.0)) for m in metrics_list]
        elif parallel and backend == 'process':
            # ProcessPool is optional; many eval_fns are not picklable. Fallback to threads if it fails.
            try:
                def run_one(sd):
                    local_rng = random.Random(sd)
                    return self.eval_fn(g, local_rng)
                max_workers = max(1, min(trials, int(os.getenv('DARWINACCI_EVAL_WORKERS', '4'))))
                with cf.ProcessPoolExecutor(max_workers=max_workers) as ex:
                    metrics_list = list(ex.map(run_one, seeds))
                m_last = metrics_list[-1] if metrics_list else {"objective": 0.0}
                vals = [float(m.get("objective", 0.0)) for m in metrics_list]
            except Exception:
                # Fallback to threads
                def run_one(idx_seed):
                    idx, sd = idx_seed
                    try:
                        local_rng = random.Random(sd)
                        res = self.eval_fn(g, local_rng)
                        return idx, res
                    except Exception:
                        return idx, {"objective": 0.0}
                max_workers = max(1, min(trials, int(os.getenv('DARWINACCI_EVAL_WORKERS', '4'))))
                with cf.ThreadPoolExecutor(max_workers=max_workers) as ex:
                    results = list(ex.map(run_one, list(enumerate(seeds))))
                results.sort(key=lambda t: t[0])
                metrics_list = [m for _, m in results]
                m_last = metrics_list[-1] if metrics_list else {"objective": 0.0}
                vals = [float(m.get("objective", 0.0)) for m in metrics_list]
        else:
            for sd in seeds:
                local_rng = random.Random(sd)
                m_last=self.eval_fn(g, local_rng)
                vals.append(float(m_last.get("objective", 0.0)))
        # aggregate objective as mean over trials
        if m_last is None:
            m={"objective":0.0}
        else:
            m=dict(m_last)
            m["objective_mean"]=sum(vals)/max(1,len(vals))
            # Use sample standard deviation when multiple trials (Bessel correction)
            denom = (len(vals)-1) if len(vals) > 1 else 1
            m["objective_std"]=float((sum((v - m["objective_mean"])**2 for v in vals)/denom)**0.5)
            # ✅ FIX C4: Safe objective with fallback
        if "objective" not in m or m.get("objective", 0.0) == 0.0:
            m["objective"] = m.get("objective_mean", 0.01)
        else:
            m["objective"] = m["objective_mean"]
        # Stabilize behavior fallback to two explicit, deterministic dimensions
        beh = m.get("behavior") or [
            float(g.get("hidden_size", 64)) / 256.0,
            float(g.get("learning_rate", 1e-3)) * 1000.0,
        ]
        nov=self.novel.score(beh); m["novelty"]=nov
        sc=agg(m)
        return Individual(genome=g, metrics=m, behavior=beh, score=sc)

    def run(self, max_cycles:int=7):
        best_global=None
        for c in range(1, max_cycles+1):
            # Allow external stop between cycles
            if self._halt:
                break
            self._current_cycle = c
            b=self.clock.budget(c)
            _t0 = __import__('time').time()
            # --- gerações internas por ciclo ---
            ind_objs=[self._evaluate(g) for g in self.population]
            for gen in range(b.generations):
                # Allow external stop between generations
                if self._halt:
                    break
                # seleção (torneio) + reprodução
                new=[]
                elite=sorted(ind_objs, key=lambda x:x.score, reverse=True)[:b.elite]
                new.extend([e.genome.copy() for e in elite])

                # QD-Lite diversification: inject archive elites into parent pool (~10%, escalate to 20% on stagnation)
                try:
                    qd_cells = list(self.archive.archive.values())
                    if qd_cells:
                        # Escalate injection when little improvement for >=3 generations
                        k = max(1, self.pop_size // 10)
                        try:
                            if gen >= 3:
                                prev_best = getattr(self, 'last_best', None)
                                curr_best = max(ind_objs, key=lambda x: x.score)
                                if (prev_best is not None) and (curr_best.score <= prev_best.score * (1 + 1e-4)):
                                    k = max(k, self.pop_size // 5)
                        except Exception:
                            pass
                        for _ in range(k):
                            if len(new) >= self.pop_size:
                                break
                            cell = random.choice(qd_cells)
                            # Prefer stored genome snapshot; reconstruct from behavior otherwise
                            qg = dict(getattr(cell, 'genome', {}) or {})
                            if not qg:
                                b0 = float(cell.behavior[0]) if len(cell.behavior) > 0 else 0.25
                                b1 = float(cell.behavior[1]) if len(cell.behavior) > 1 else 1.0
                                hidden_size = max(8, int(b0 * 256))
                                learning_rate = max(1e-5, float(b1 / 1000.0))
                                qg = {
                                    'hidden_size': hidden_size,
                                    'learning_rate': learning_rate,
                                }
                            new.append(qg)
                except Exception:
                    pass
                while len(new)<self.pop_size:
                    p1=tournament(ind_objs, k=3, key=lambda x:x.score, rng=self.rng)
                    p2=tournament(ind_objs, k=3, key=lambda x:x.score, rng=self.rng)
                    child=uniform_cx(p1.genome, p2.genome, self.rng)
                    child=gaussian_mut(child, rate=b.mut, scale=0.2, rng=self.rng)
                    # control genome bloat
                    child=prune_genes(child, max_genes=128, rng=self.rng)
                    new.append(child)
                # avaliação
                ind_objs=[self._evaluate(g) for g in new]
                # anti-estagnação (Gödel-kick) se pouco ganho e novelty baixa
                if gen>2:
                    curr_best=max(ind_objs,key=lambda x:x.score)
                    if (self.last_best is not None) and (curr_best.score <= self.last_best.score*(1+1e-4)) and (curr_best.metrics.get("novelty",0.0)<0.02):
                        # aplica kick nos top-3 por score (ordenado)
                        top = sorted(ind_objs, key=lambda x: x.score, reverse=True)[:3]
                        for ind in top:
                            ind.genome = godel_kick(ind.genome, self.rng, severity=0.3, new_genes=1)
                            _new = self._evaluate(ind.genome)
                            # atualizar no lugar para manter a lista consistente
                            ind.metrics = _new.metrics
                            ind.behavior = _new.behavior
                            ind.score = _new.score

            # pós-ciclo
            best=max(ind_objs,key=lambda x:x.score)
            # gates
            ok,reasons=self.guard.evaluate(best.metrics); best.metrics["ethics_pass"]=ok
            accepted=False
            # QD: arquivar e medir coverage (salva snapshot de genome)
            for ind in ind_objs:
                self.archive.add(ind.behavior, ind.score, genome=ind.genome)
                self.novel.add(ind.behavior)
            coverage=self.archive.coverage()

            # promoção segura: se passar Σ-Guard e melhora o score do campeão atual
            # Canary staged promotion with rollback window
            if ok:
                cand=Champ(genome=best.genome, score=best.score, behavior=best.behavior, metrics=best.metrics)
                promote = False
                if self.arena.champion is None:
                    promote = True
                else:
                    # Stage 1: small improvement threshold
                    thresh = float(os.getenv('DARWINACCI_CANARY_THRESH', '1e-6'))
                    promote = best.score > (self.arena.champion.score * (1 + thresh))
                if promote:
                    accepted=self.arena.consider(cand)
                else:
                    accepted=False

            # superposição Fibonacci + viés de skills injeta viés nos próximos pais
            sup=self.arena.superpose()
            if sup:
                # mistura leve na população (5% substituídos)
                k=max(1,self.pop_size//20)
                for i in range(k): 
                    self.population[i]= {**self.population[i], **{kk:(self.population[i].get(kk,0.0)*0.7 + vv*0.3) for kk,vv in sup.items()} }
            # Skills bias: usar top comportamentos para injetar chaves/valores simples em subset da população
            try:
                top_cells = self.archive.bests()[: max(1, self.pop_size // 10)]
                for idx, cell in top_cells:
                    if not cell.behavior:
                        continue
                    # Mapear behavior em 2 chaves canônicas
                    b0 = float(cell.behavior[0]) if len(cell.behavior) > 0 else 0.0
                    b1 = float(cell.behavior[1]) if len(cell.behavior) > 1 else 0.0
                    key0 = 'skill_b0'; key1 = 'skill_b1'
                    # Injetar leve viés em indivíduos aleatórios
                    for _ in range(2):
                        j = self.rng.randint(0, self.pop_size - 1)
                        g = dict(self.population[j])
                        g[key0] = g.get(key0, 0.0) * 0.8 + b0 * 0.2
                        g[key1] = g.get(key1, 0.0) * 0.8 + b1 * 0.2
                        self.population[j] = g
            except Exception:
                pass
            else:
                # caso inicial: reseta com elite
                self.population=[x.genome for x in sorted(ind_objs, key=lambda x:x.score, reverse=True)[:self.pop_size]]

            # limpeza automática de memória (FIX 1.2)
            self._auto_memory_cleanup()

            # ledger & timings
            h=self.worm.append({
                "cycle":c, "accepted":bool(accepted), "checkpoint":bool(b.checkpoint),
                "best_score":float(best.score), "objective":float(best.metrics.get("objective",0.0)),
                "novelty":float(best.metrics.get("novelty",0.0)), "coverage":float(coverage),
                "mut":float(b.mut), "cx":float(b.cx), "elite":int(b.elite),
                "reasons":{k:float(v) for k,v in reasons.items()}
            })

            logger.info(f"[Darwinacci] c={c:02d} best={best.score:.4f} acc={accepted} "
                        f"cov={coverage:.2f} mut={b.mut:.3f} cx={b.cx:.3f} elite={b.elite} worm={h[:10]}")
            # Update metrics if enabled
            if self._prom is not None:
                try:
                    self._prom['best_score'].set(float(best.score))
                    self._prom['coverage'].set(float(coverage))
                    self._prom['novelty_archive_size'].set(float(len(self.novel.mem)))
                    self._prom['mut'].set(float(b.mut))
                    self._prom['cx'].set(float(b.cx))
                    self._prom['elite'].set(float(b.elite))
                except Exception:
                    pass
            self.last_best=best

            # Emit JSONL metrics and simple anomaly alerts
            try:
                dur = __import__('time').time() - _t0
                self._emit_metrics(
                    cycle=c,
                    best_score=float(best.score),
                    coverage=float(coverage),
                    novelty_size=len(self.novel.mem),
                    accepted=bool(accepted),
                    checkpoint=bool(b.checkpoint),
                    mut=float(b.mut),
                    cx=float(b.cx),
                    elite=int(b.elite),
                    objective=float(best.metrics.get("objective", 0.0)),
                    duration_s=float(dur),
                )
            except Exception:
                pass

            # Optional checkpoint
            try:
                if bool(os.getenv('DARWINACCI_CHECKPOINT', '1') == '1') and bool(b.checkpoint):
                    ck_dir = Path(os.getenv('DARWINACCI_CKPT_DIR', 'data/checkpoints'))
                    ck_dir.mkdir(parents=True, exist_ok=True)
                    ck_path = ck_dir / f"cycle_{c:04d}.json.gz"
                    # Build and validate payload using Pydantic schema
                    arch = [
                        {
                            'idx': int(idx),
                            'best_score': float(cell.best_score),
                            'behavior': list(cell.behavior),
                            'genome': dict(getattr(cell, 'genome', {}) or {}),
                        }
                        for idx, cell in self.archive.bests()
                    ]
                    try:
                        payload_model = CheckpointPayload(
                            format_version=1,
                            cycle=c,
                            rng_state=self.rng.getstate(),
                            population=self.population,
                            archive=arch,
                            novelty_size=len(self.novel.mem),
                            champion={
                                'genome': self.arena.champion.genome if self.arena.champion else None,
                                'score': self.arena.champion.score if self.arena.champion else None,
                                'behavior': self.arena.champion.behavior if self.arena.champion else None,
                                'metrics': self.arena.champion.metrics if self.arena.champion else None,
                            },
                        )
                        payload = payload_model.model_dump()
                    except Exception:
                        # Fallback to permissive payload to guarantee checkpoint
                        payload = {
                            'format_version': 1,
                            'cycle': c,
                            'rng_state': self.rng.getstate(),
                            'population': self.population,
                            'archive': arch,
                            'novelty_size': len(self.novel.mem),
                            'champion': {
                                'genome': self.arena.champion.genome if self.arena.champion else None,
                                'score': self.arena.champion.score if self.arena.champion else None,
                                'behavior': self.arena.champion.behavior if self.arena.champion else None,
                            },
                        }
                    # Optional HMAC over payload
                    try:
                        key = os.getenv('DARWINACCI_HMAC_KEY')
                        if key:
                            base = json.dumps(payload, sort_keys=True, ensure_ascii=False)
                            sig = __import__('hashlib').sha256((key + '|' + base).encode()).hexdigest()
                            payload['hmac'] = sig
                    except Exception:
                        pass
                    tmp = ck_path.with_suffix('.json.gz.tmp')
                    with gzip.open(tmp, 'wt') as f:
                        json.dump(payload, f)
                    os.replace(tmp, ck_path)
            except Exception:
                pass

        return self.arena.champion

    # --------------- Skills export utility ---------------
    def export_skills(self, path: str, k: int = 16) -> bool:
        """Export top skills from the archive for external reuse."""
        try:
            return bool(self.archive.export_skills_json(path, k=k))
        except Exception:
            return False

    # ---------------- Control -----------------
    def stop(self) -> None:
        """Request a cooperative stop. The engine halts after current inner loop boundary."""
        self._halt = True

    def is_running(self) -> bool:
        """Best-effort signal: running if not halted and cycles remain."
        """
        return not self._halt

    # ---------------- Metrics & Alerts -----------------
    def _emit_metrics(self, cycle:int, best_score:float, coverage:float, novelty_size:int,
                      accepted:bool, checkpoint:bool, mut:float, cx:float, elite:int, objective:float,
                      duration_s: float | None = None) -> None:
        if os.getenv('DARWINACCI_METRICS', '1') != '1':
            return
        mpath = os.getenv('DARWINACCI_METRICS_PATH', 'data/metrics.jsonl')
        parent = os.path.dirname(mpath) or '.'
        os.makedirs(parent, exist_ok=True)
        payload = {
            'cycle': cycle,
            'best_score': best_score,
            'objective': objective,
            'coverage': coverage,
            'novelty_archive_size': int(novelty_size),
            'accepted': bool(accepted),
            'checkpoint': bool(checkpoint),
            'mut': mut,
            'cx': cx,
            'elite': elite,
            'duration_s': float(duration_s) if duration_s is not None else None,
        }
        try:
            with open(mpath, 'a') as f:
                f.write(json.dumps(payload) + "\n")
        except Exception:
            pass

        # Update local history and detect anomalies via z-score
        self._hist_best.append(best_score)
        self._hist_cov.append(coverage)
        window = int(os.getenv('DARWINACCI_ALERT_WINDOW', '10'))
        thresh = float(os.getenv('DARWINACCI_ALERT_Z', '3.0'))
        if len(self._hist_best) >= max(5, window):
            hist_b = self._hist_best[-window:]
            hist_c = self._hist_cov[-window:]
            try:
                mu_b = statistics.fmean(hist_b)
                sd_b = statistics.pstdev(hist_b) or 1e-9
                z_b = abs(best_score - mu_b) / sd_b
            except Exception:
                z_b = 0.0
            try:
                mu_c = statistics.fmean(hist_c)
                sd_c = statistics.pstdev(hist_c) or 1e-9
                z_c = abs(coverage - mu_c) / sd_c
            except Exception:
                z_c = 0.0
            if (z_b >= thresh) or (z_c >= thresh):
                self._emit_alert(cycle=cycle, z_best=z_b, z_cov=z_c, best=best_score, cov=coverage)

            # Update Prometheus z-scores and rates
            if self._prom is not None:
                try:
                    self._prom['z_best'].set(float(z_b))
                    self._prom['z_cov'].set(float(z_c))
                    # Windowed acceptance rates
                    acc_win = [1.0 if (i == len(self._hist_best)-1 and accepted) else 0.0 for i in range(len(self._hist_best))][-window:]
                    canary_win = acc_win  # canary ~= accepted in staged promotion
                    self._prom['accepted_rate'].set(float(sum(acc_win)/max(1,len(acc_win))))
                    self._prom['canary_pass_rate'].set(float(sum(canary_win)/max(1,len(canary_win))))
                except Exception:
                    pass

    def _emit_alert(self, cycle:int, z_best:float, z_cov:float, best:float, cov:float) -> None:
        apath = os.getenv('DARWINACCI_ALERTS_PATH', 'data/alerts.jsonl')
        parent = os.path.dirname(apath) or '.'
        os.makedirs(parent, exist_ok=True)
        alert = {
            'cycle': cycle,
            'type': 'zscore_anomaly',
            'z_best': z_best,
            'z_coverage': z_cov,
            'best_score': best,
            'coverage': cov,
        }
        try:
            with open(apath, 'a') as f:
                f.write(json.dumps(alert) + "\n")
        except Exception:
            pass

    def _auto_memory_cleanup(self):
        """Limpeza automática de memória para prevenir vazamentos"""
        # Limpar população antiga se muito grande
        max_pop_size = self.pop_size * 3  # Permite até 3x o tamanho normal
        if len(self.population) > max_pop_size:
            # Mantém apenas os melhores sem reavaliar cada indivíduo
            try:
                prev_len = len(self.population)
                # Cache scores to avoid repeated _evaluate side-effects
                scored = [(g, self._evaluate(g).score) for g in self.population]
                scored.sort(key=lambda t: t[1], reverse=True)
                self.population = [g for g,_ in scored[:self.pop_size]]
            except Exception:
                # Fallback: best-effort keep first N
                self.population = self.population[:self.pop_size]
            logger.debug(f"[Darwinacci] Memory cleanup: reduced population from {prev_len} to {self.pop_size}")

        # Limpar novelty memory se muito grande
        if len(self.novel.mem) > self.novel.max_size:
            # Remove entradas antigas (FIFO)
            excess = len(self.novel.mem) - self.novel.max_size
            self.novel.mem = self.novel.mem[excess:]
            logger.debug(f"[Darwinacci] Memory cleanup: reduced novelty memory by {excess} entries")

    # ---------------- Checkpoint Restore -----------------
    def _to_tuple(self, obj):
        """Recursively convert lists to tuples for RNG state restoration."""
        if isinstance(obj, list):
            return tuple(self._to_tuple(x) for x in obj)
        if isinstance(obj, dict):
            return {k: self._to_tuple(v) for k, v in obj.items()}
        return obj

    def load_checkpoint_json(self, path: str) -> int:
        """Load a gzipped JSON checkpoint created by this engine and restore state.

        Returns the cycle number encoded in the checkpoint.
        """
        # Read JSON (supports .gz and plain .json)
        if path.endswith('.gz'):
            with gzip.open(path, 'rt') as f:
                data = json.load(f)
        else:
            with open(path, 'r') as f:
                data = json.load(f)

        # Validate against schema
        # Optional HMAC verification
        try:
            if os.getenv('DARWINACCI_VERIFY_HMAC', '0') == '1' and 'hmac' in data:
                key = os.getenv('DARWINACCI_HMAC_KEY')
                if key:
                    base = dict(data)
                    sig = str(base.pop('hmac'))
                    base_str = json.dumps(base, sort_keys=True, ensure_ascii=False)
                    calc = __import__('hashlib').sha256((key + '|' + base_str).encode()).hexdigest()
                    if calc != sig:
                        raise ValueError('Checkpoint HMAC verification failed')
        except Exception:
            # If verification requested and fails, propagate; otherwise continue
            if os.getenv('DARWINACCI_VERIFY_HMAC', '0') == '1':
                raise
        payload = CheckpointPayload.model_validate(data)

        # Restore RNG state (convert JSON lists back to tuples)
        try:
            self.rng.setstate(self._to_tuple(payload.rng_state))
        except Exception:
            # If RNG restore fails, keep current RNG but proceed
            pass

        # Restore population
        self.population = list(payload.population)

        # Rebuild archive from entries
        self.archive = GoldenSpiralArchive(bins=self.archive.bins)
        for entry in payload.archive:
            # entry is dict-like after model_dump; ensure access
            idx = entry.get('idx') if isinstance(entry, dict) else entry.idx  # not used directly
            behavior = entry.get('behavior') if isinstance(entry, dict) else entry.behavior
            best_score = entry.get('best_score') if isinstance(entry, dict) else entry.best_score
            try:
                self.archive.add(behavior=behavior, score=float(best_score))
            except Exception:
                continue

        # Restore champion (best-effort)
        ch = payload.champion
        if ch and (ch.get('genome') if isinstance(ch, dict) else ch.genome):
            genome = ch.get('genome') if isinstance(ch, dict) else ch.genome
            score = ch.get('score') if isinstance(ch, dict) else ch.score
            behavior = ch.get('behavior') if isinstance(ch, dict) else ch.behavior
            try:
                self.arena.champion = Champ(genome=genome, score=float(score or 0.0), behavior=behavior or [], metrics=None)
                # seed history with current champion for superposition
                self.arena.history = [self.arena.champion]
            except Exception:
                self.arena.champion = None

        # Optional skills discovery dump
        try:
            if os.getenv('DARWINACCI_SKILLS', '0') == '1':
                skills_path = os.getenv('DARWINACCI_SKILLS_PATH', 'data/skills.jsonl')
                os.makedirs(os.path.dirname(skills_path) or '.', exist_ok=True)
                top = self.archive.bests()[: min(10, len(self.archive.archive))]
                skill = {
                    'cycle': int(payload.cycle),
                    'top_behaviors': [cell.behavior for _, cell in top],
                    'top_scores': [float(cell.best_score) for _, cell in top],
                }
                with open(skills_path, 'a') as f:
                    f.write(json.dumps(skill) + "\n")
        except Exception:
            pass

        return int(payload.cycle)