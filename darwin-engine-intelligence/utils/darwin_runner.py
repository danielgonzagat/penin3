#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DARWIN Runner - IA³ (Inteligência ao Cubo)
- Loop darwiniano com Equação da Morte:
  E(t+1) = 1 se A(t)=1 (ΔL∞>0 ∧ CAOS≥1.0 ∧ I≥I_min ∧ P≥P_min) ou C(t)=1 (novelty ≥ novelty_min)
           0 caso contrário
- WORM: /root/darwin_worm.log (cadeado por hash)
- Métricas Prometheus: :9092/metrics
- Manifest: /root/agents_active.json
- Nasce 1 novo agente a cada X mortes (deaths_per_birth)
"""
import os, sys, json, time, re, hashlib, signal, threading, traceback, random
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer

try:
    import yaml  # opcional
except Exception:
    yaml = None

DEFAULTS = {
    "paths": {
        "worm_log": "/root/darwin_worm.log",
        "manifest": "/root/agents_active.json",
        "heritage_dir": "/root/heritage",
        "agents_dir": "/root/agents",
        "teis_worm_log": "/root/teis_worm.log"
    },
    "thresholds": {
        "delta_linf_min": 0.0,
        "caos_ratio_min": 1.0,
        "I_min": 0.60,
        "P_min": 0.01,
        "novelty_min": 0.02,
    },
    "births": {"deaths_per_birth": 10},
    "metrics": {"port": 9092},
    "operational": {
        "interval_seconds": 15,
        "dry_run": False,
        "fail_closed": True,
        "max_agents": 64,
        "seed": None,
    },
}

LOCK = threading.Lock()

async def now_iso():
    return await datetime.utcnow().isoformat() + "Z"

# ----------------------------- Config Loader ----------------------------- #
async def load_config(path):
    cfg = json.loads(json.dumps(DEFAULTS))  # deep copy
    if os.path.exists(path):
        try:
            if path.endswith(".yaml") or path.endswith(".yml"):
                if yaml is None:
                    raise RuntimeError("PyYAML ausente; use JSON")
                with open(path, "r") as f:
                    cfg_from_file = yaml.safe_load(f) or {}
            else:
                with open(path, "r") as f:
                    cfg_from_file = json.load(f)
            # overlay raso
            async def merge(dst, src):
                for k,v in src.items():
                    if isinstance(v, dict) and k in dst and isinstance(dst[k], dict):
                        merge(dst[k], v)
                    else:
                        dst[k] = v
            merge(cfg, cfg_from_file)
        except Exception as e:
            logger.info(f"[DARWIN] Falha ao carregar config {path}: {e}", file=sys.stderr)
    return await cfg

# ------------------------------- WORM ----------------------------------- #
async def _last_hash(log_path):
    if not os.path.exists(log_path):
        return await hashlib.sha256(b"DARWIN-GENESIS").hexdigest()
    h = None
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            chunk = 4096
            data = b""
            while size > 0:
                step = min(chunk, size)
                size -= step
                f.seek(size)
                data = f.read(step) + data
                if b"HASH:" in data:
                    break
        # pega última ocorrência
        lines = data.decode(errors="ignore").strip().splitlines()
        for line in reversed(lines):
            if line.startswith("HASH:"):
                h = line.split("HASH:",1)[1].strip()
                break
    except Exception:
        h = None
    return await h or hashlib.sha256(b"DARWIN-GENESIS").hexdigest()

async def worm_append(log_path, event):
    """Escreve evento encadeado: EVENT:{json}\nHASH:{sha256(prev+json)}"""
    with LOCK:
        prev = _last_hash(log_path)
        event["previous_hash"] = prev
        payload = json.dumps(event, sort_keys=True, ensure_ascii=False)
        curr = hashlib.sha256((prev + payload).encode("utf-8")).hexdigest()
        with open(log_path, "a", encoding="utf-8") as f:
            f.write("EVENT:" + payload + "\n")
            f.write("HASH:" + curr + "\n")
        return await curr

# ----------------------------- Metrics ---------------------------------- #
class Metrics:
    async def __init__(self):
        self.decisions_total = 0
        self.kills_total = 0
        self.births_total = 0
        self.promotions_total = 0
        self.errors_total = 0
        self.death_counter_window = 0
        self.loop_seconds = 0.0
        self.last_decision_ts = 0.0
        self._lock = threading.Lock()

    async def scrape(self):
        with self._lock:
            lines = []
            async def m(name, mtype, help_):
                lines.append(f"# HELP {name} {help_}")
                lines.append(f"# TYPE {name} {mtype}")
            m("darwin_decisions_total","counter","Total de decisões Darwin")
            lines.append(f"darwin_decisions_total {self.decisions_total}")
            m("darwin_kills_total","counter","Total de mortes Darwin")
            lines.append(f"darwin_kills_total {self.kills_total}")
            m("darwin_births_total","counter","Total de nascimentos Darwin")
            lines.append(f"darwin_births_total {self.births_total}")
            m("darwin_promotions_total","counter","Total de promoções Darwin")
            lines.append(f"darwin_promotions_total {self.promotions_total}")
            m("darwin_errors_total","counter","Total de erros")
            lines.append(f"darwin_errors_total {self.errors_total}")
            m("darwin_death_counter_window","gauge","Contador de mortes na janela")
            lines.append(f"darwin_death_counter_window {self.death_counter_window}")
            m("darwin_loop_seconds","gauge","Duração do último loop")
            lines.append(f"darwin_loop_seconds {self.loop_seconds}")
            m("darwin_last_decision_ts","gauge","Epoch da última decisão")
            lines.append(f"darwin_last_decision_ts {self.last_decision_ts}")
            return await "\n".join(lines) + "\n"

METRICS = Metrics()

class MetricsHandler(BaseHTTPRequestHandler):
    async def do_GET(self):
        if self.path == "/metrics":
            data = METRICS.scrape().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type","text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404); self.end_headers()
    async def log_message(self, format, *args):
        pass  # Silencia logs HTTP

async def start_http_server(port):
    srv = HTTPServer(("0.0.0.0", port), MetricsHandler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    logger.info(f"[DARWIN] Métricas em :{port}/metrics")

# --------------------------- Manifest ----------------------------------- #
async def _default_manifest():
    return await {"agents": [], "next_id": 0, "death_counter": 0}

async def load_manifest(path):
    if not os.path.exists(path):
        return await _default_manifest()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return await json.load(f)
    except Exception:
        return await _default_manifest()

async def save_manifest(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

async def ensure_bootstrap(manifest, cfg):
    if not manifest["agents"]:
        # cria um primeiro agente
        aid = f"agent_{manifest['next_id']}"
        manifest["next_id"] += 1
        manifest["agents"].append({"id": aid, "born_ts": time.time(), "meta": {"origin":"bootstrap"}})
        save_manifest(cfg["paths"]["manifest"], manifest)

# ------------------------ Métricas do TEIS (opcional) -------------------- #
async def _parse_teis_worm_line(line):
    # aceita EVENT:{json} ou json puro
    try:
        if line.startswith("EVENT:"):
            line = line.split("EVENT:",1)[1]
        data = json.loads(line)
        return await data
    except Exception:
        return await None

async def read_latest_teix_metrics(teis_worm_log):
    """Tenta extrair últimas métricas úteis do TEIS WORM ou arquivo JSON direto."""
    # Primeiro tentar arquivo JSON direto do TEIS IA³
    teis_metrics_file = "/root/teis_darwin_metrics.json"
    if os.path.exists(teis_metrics_file):
        try:
            with open(teis_metrics_file, "r") as f:
                metrics = json.load(f)
                return await {
                    "delta_linf": metrics.get("delta_linf", 0.0),
                    "caos_ratio": metrics.get("caos_ratio", 1.0),
                    "I": metrics.get("I", 0.5),
                    "P": metrics.get("P", 0.0),
                    "novelty": metrics.get("novelty", 0.0),
                    "source": "teis_ia3_json",
                    "ia3_score": metrics.get("ia3_score", 0.0),
                    "system_health": metrics.get("system_health", 0.5)
                }
        except Exception as e:
            logger.info(f"[DARWIN] Erro lendo métricas TEIS JSON: {e}")

    # Fallback para worm log
    if not os.path.exists(teis_worm_log):
        return await None
    try:
        with open(teis_worm_log, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            step = 8192
            buf = b""
            while size > 0 and len(buf) < 200000:
                size = max(0, size - step)
                f.seek(size)
                buf = f.read(step) + buf
                if buf.count(b"\n") > 500:
                    break
        lines = buf.decode(errors="ignore").strip().splitlines()
        for line in reversed(lines):
            if '"delta_linf"' in line or '"delta_Linf"' in line or '"caos_ratio"' in line or '"I":' in line or '"novelty"' in line:
                ev = _parse_teis_worm_line(line)
                if not ev: continue
                # normaliza chaves
                m = {}
                # delta_linf
                if "delta_linf" in ev: m["delta_linf"] = ev["delta_linf"]
                elif "delta_Linf" in ev: m["delta_linf"] = ev["delta_Linf"]
                # caos_ratio
                if "caos_ratio" in ev: m["caos_ratio"] = ev["caos_ratio"]
                # I/P/novelty
                for k in ("I","P","novelty","oci","ece","rho"):
                    if k in ev: m[k] = ev[k]
                    elif "metrics" in ev and k in ev["metrics"]:
                        m[k] = ev["metrics"][k]
                # às vezes P está em invariants/lemni
                if "lemni_P" in ev: m["P"] = ev["lemni_P"]
                # se tiver bloco "scores", mantém como referência
                if "scores_post" in ev: m["scores_post"] = ev["scores_post"]
                if m: return await m
        return await None
    except Exception:
        return await None

# ------------------ Avaliação Darwin (A(t), C(t), E(t+1)) ---------------- #
async def compute_integrity_like(oci=0.67, lyapunov_ok=True, ece=0.04, rho=0.67, cost_ok=True):
    """Fallback de I (imitando lemniscata_gate.compute_integrity)."""
    I_oci = oci
    I_lyap = 1.0 if lyapunov_ok else 0.0
    I_ece = max(0.0, 1.0 - (ece/0.05))  # ece<=0.03 ideal => ~0.4/0.25 escala, aqui simplificado
    I_rho = max(0.0, 1.0 - rho)         # rho<1 melhor
    return await 0.25 * (I_oci + I_lyap + I_ece + I_rho)

async def evaluate_agent_metrics(agent, cfg):
    """
    Estratégia:
    1) Tenta ler métricas reais do TEIS WORM.
    2) Se falhar, usa gerador determinístico (seed a partir do id + tempo discretizado).
    """
    # 1) TEIS
    teix = read_latest_teix_metrics(cfg["paths"]["teis_worm_log"])
    if teix:
        delta = float(teix.get("delta_linf", 0.0))
        caos = float(teix.get("caos_ratio", 1.0))
        I = float(teix.get("I", compute_integrity_like(
            oci=float(teix.get("oci", 0.67)),
            lyapunov_ok=True,
            ece=float(teix.get("ece", 0.04)),
            rho=float(teix.get("rho", 0.67)),
            cost_ok=True
        )))
        P = float(teix.get("P", max(0.0, float(teix.get("novelty", 0.0)) - float(teix.get("iN", 0.0)) if "iN" in teix else float(teix.get("novelty", 0.0)))))
        novelty = float(teix.get("novelty", 0.0))
        return await {"delta_linf": delta, "caos_ratio": caos, "I": I, "P": P, "novelty": novelty, "source":"teis_worm"}
    # 2) Fallback determinístico seguro
    base_seed = str(cfg["operational"].get("seed")) if cfg["operational"].get("seed") is not None else ""
    bucket = int(time.time() // (cfg["operational"]["interval_seconds"]))  # discretiza no tempo
    seed = f"{agent['id']}|{bucket}|{base_seed}"
    rnd = random.Random(hashlib.sha256(seed.encode()).hexdigest())
    delta = max(0.0, rnd.gauss(0.03, 0.03))  # maioria ligeiramente > 0
    caos  = max(0.95, rnd.gauss(1.03, 0.03))
    novelty = max(0.0, rnd.gauss(0.03, 0.02))
    I = min(1.0, max(0.4, rnd.gauss(0.62, 0.06)))
    P = max(0.0, rnd.gauss(0.05, 0.03))
    return await {"delta_linf": delta, "caos_ratio": caos, "I": I, "P": P, "novelty": novelty, "source":"fallback"}

async def darwin_verdict(metrics, cfg):
    th = cfg["thresholds"]
    A = (metrics["delta_linf"] > th["delta_linf_min"]) and \
        (metrics["caos_ratio"] >= th["caos_ratio_min"]) and \
        (metrics["I"] >= th["I_min"]) and \
        (metrics["P"] >= th["P_min"])
    C = (metrics["novelty"] >= th["novelty_min"])
    E = bool(A or C)
    return await {"A": bool(A), "C": bool(C), "E_next": E}

# ----------------------------- Hooks ------------------------------------ #
async def promote_hook(agent, metrics, cfg):
    evt = {
        "ts": now_iso(), "event": "darwin_promote_hook",
        "agent": agent["id"], "metrics": metrics
    }
    worm_append(cfg["paths"]["worm_log"], evt)

async def kill_hook(agent, metrics, cfg, reason="no_evolution"):
    evt = {
        "ts": now_iso(), "event": "darwin_kill_hook",
        "agent": agent["id"], "metrics": metrics, "reason": reason
    }
    worm_append(cfg["paths"]["worm_log"], evt)

async def spawn_agent(manifest, cfg, heritage_note="periodic_birth"):
    aid = f"agent_{manifest['next_id']}"
    manifest["next_id"] += 1
    agent = {"id": aid, "born_ts": time.time(), "meta": {"origin":"darwin_birth", "note": heritage_note}}
    manifest["agents"].append(agent)
    # registra herança agregada (placeholder)
    heritage = {
        "ts": now_iso(), "born": aid, "note": heritage_note
    }
    hpath = os.path.join(cfg["paths"]["heritage_dir"], f"heritage_{int(time.time())}_{aid}.json")
    with open(hpath, "w", encoding="utf-8") as f:
        json.dump(heritage, f, ensure_ascii=False)
    evt = {"ts": now_iso(), "event":"darwin_spawn_hook", "agent": aid, "heritage": hpath}
    worm_append(cfg["paths"]["worm_log"], evt)
    return await agent

# ------------------------------ Runner ---------------------------------- #
STOP = False
async def handle_sig(sig, frm):
    global STOP
    STOP = True

async def run_once(cfg):
    t0 = time.time()
    try:
        manifest = load_manifest(cfg["paths"]["manifest"])
        ensure_bootstrap(manifest, cfg)
        # escolha simples do próximo agente (round-robin)
        if not manifest["agents"]:
            return
        agent = manifest["agents"][0]

        metrics = evaluate_agent_metrics(agent, cfg)
        verdict = darwin_verdict(metrics, cfg)

        # log de decisão
        decision = {
            "ts": now_iso(),
            "event": "darwin_decision",
            "agent": agent["id"],
            "verdict": verdict,
            "metrics": metrics
        }
        worm_append(cfg["paths"]["worm_log"], decision)

        METRICS.decisions_total += 1
        METRICS.last_decision_ts = time.time()

        if verdict["E_next"]:
            # sobrevive / promove
            promote_hook(agent, metrics, cfg)
            METRICS.promotions_total += 1
            # move agente para o fim (simples revezamento)
            manifest["agents"].append(manifest["agents"].pop(0))
        else:
            # morre
            kill_hook(agent, metrics, cfg, reason="A=0 and C=0")
            METRICS.kills_total += 1
            manifest["agents"].pop(0)
            manifest["death_counter"] = manifest.get("death_counter", 0) + 1
            # nascimento condicionado
            if manifest["death_counter"] >= cfg["births"]["deaths_per_birth"]:
                spawn_agent(manifest, cfg, heritage_note=f"{manifest['death_counter']}_deaths")
                METRICS.births_total += 1
                manifest["death_counter"] = 0

        METRICS.death_counter_window = manifest.get("death_counter", 0)

        # persistência (a não ser em dry_run)
        if not cfg["operational"]["dry_run"]:
            save_manifest(cfg["paths"]["manifest"], manifest)

    except Exception as e:
        METRICS.errors_total += 1
        err = {
            "ts": now_iso(),
            "event": "darwin_error",
            "error": str(e),
            "trace": traceback.format_exc()
        }
        worm_append(cfg["paths"]["worm_log"], err)
    finally:
        METRICS.loop_seconds = time.time() - t0

async def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/darwin_policy.yaml")
    ap.add_argument("--once", action="store_true", help="Executa um ciclo e sai")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # seed global (opcional)
    seed = cfg["operational"].get("seed")
    if seed is not None:
        random.seed(seed)

    # inicia métricas
    start_http_server(int(cfg["metrics"]["port"]))

    # registra bootstrap no WORM
    worm_append(cfg["paths"]["worm_log"], {"ts": now_iso(), "event": "darwin_boot", "config": args.config})

    # garante manifest
    manifest = load_manifest(cfg["paths"]["manifest"])
    ensure_bootstrap(manifest, cfg)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    if args.once:
        run_once(cfg)
        return

    itv = float(cfg["operational"]["interval_seconds"])
    while not STOP:
        run_once(cfg)
        time.sleep(itv)

    worm_append(cfg["paths"]["worm_log"], {"ts": now_iso(), "event": "darwin_stop"})

if __name__ == "__main__":
    main()
