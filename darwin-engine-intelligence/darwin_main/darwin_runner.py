#!/usr/bin/env python3
# stdlib only
import os, sys, json, time, hashlib, signal, threading
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from typing import Any, Dict, Optional

# ----- Config / Policy loader -----
def load_config(path: Optional[str]) -> Dict[str, Any]:
    cands = []
    if path: cands.append(path)
    cands += [
        os.environ.get("DARWIN_CONFIG"),
        "/root/darwin/darwin_policy.json",
        "/root/darwin_policy.json",
    ]
    for p in cands:
        if p and os.path.exists(p):
            # JSON
            try:
                with open(p,"r",encoding="utf-8") as f:
                    cfg = json.load(f)
                cfg["_config_path"] = p
                return cfg
            except Exception:
                pass
            # YAML (opcional)
            try:
                import yaml  # type: ignore
                with open(p,"r",encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg["_config_path"] = p
                return cfg
            except Exception:
                pass
    # default segura
    return {
        "_config_path": "(default in-memory)",
        "paths": {
            "worm_log": "/root/darwin_test/worm_test.log",
            "manifest": "/root/darwin_test/manifest_test.json",
            "state": "/root/darwin_test/state.json",
            "heritage_dir": "/root/darwin_test/heritage",
            "source_promotions": "/root/promotion_log.json"
        },
        "thresholds": {"delta_linf_min":0.0,"caos_ratio_min":1.0,"I_min":0.60,"P_min":0.01,"novelty_min":0.02},
        "births": {"deaths_per_birth":10},
        "metrics": {"port": 9092},
        "operational": {"interval_seconds":5,"dry_run":False,"fail_closed":True,"seed":42}
    }

def cfg_get(cfg: Dict[str,Any], path: str, default=None):
    cur: Any = cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur

# ----- WORM helpers (EVENT/HASH, GENESIS) -----
GENESIS_HASH = hashlib.sha256(b"DARWIN-GENESIS").hexdigest()

def worm_lines(path: str):
    if not os.path.exists(path): return []
    with open(path,"r",encoding="utf-8",errors="replace") as f:
        return f.read().splitlines()

def worm_write_event(path: str, event_obj: Dict[str, Any]) -> str:
    lines = worm_lines(path)
    prev = GENESIS_HASH
    for i in range(len(lines)-1, -1, -1):
        if lines[i].startswith("HASH:"):
            prev = lines[i].split("HASH:",1)[1].strip()
            break
    payload = json.dumps({**event_obj, "previous_hash": prev}, separators=(",",":"))
    h = hashlib.sha256((prev + payload).encode("utf-8")).hexdigest()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path,"a",encoding="utf-8") as f:
        f.write(f"EVENT:{payload}\n")
        f.write(f"HASH:{h}\n")
    return h

def verify_chain(lines):
    ok, msg, pairs = True, "ok", 0
    prev_seen = None
    i = 0
    while i < len(lines)-1:
        if not lines[i].startswith("EVENT:"):
            i += 1; continue
        payload = lines[i][6:]
        if not lines[i+1].startswith("HASH:"):
            return False, "HASH ausente após EVENT", pairs
        try:
            obj = json.loads(payload)
        except Exception as e:
            return False, f"EVENT inválido: {e}", pairs
        ph = obj.get("previous_hash")
        if prev_seen is None:
            if ph != GENESIS_HASH:
                return False, "previous_hash inicial != GENESIS_HASH", pairs
        else:
            if ph != prev_seen:
                return False, "previous_hash não encadeia", pairs
        expected = hashlib.sha256((ph + payload).encode("utf-8")).hexdigest()
        curr = lines[i+1].split("HASH:",1)[1].strip()
        if curr != expected:
            return False, "HASH incorreto", pairs
        prev_seen = curr
        pairs += 1
        i += 2
    return ok, msg, pairs

# ----- Metrics exporter -----
class Metrics:
    def __init__(self):
        self.decisions = 0
        self.promotions = 0
        self.kills = 0
        self.spawns = 0
        self.deaths_in_window = 0
        self.window_size = 10
        self.chain_ok = 1

    def render(self) -> str:
        parts = []
        parts.append(f"darwin_decisions_total {self.decisions}")
        parts.append(f"darwin_promotions_total {self.promotions}")
        parts.append(f"darwin_kills_total {self.kills}")
        parts.append(f"darwin_spawns_total {self.spawns}")
        parts.append(f"darwin_birth_window_deaths {self.deaths_in_window}")
        parts.append(f"darwin_birth_window_size {self.window_size}")
        parts.append(f"darwin_chain_ok {self.chain_ok}")
        return "\n".join(parts) + "\n"

class Handler(BaseHTTPRequestHandler):
    async def do_GET(self):
        if self.path.startswith("/metrics"):
            txt = self.server.metrics.render()  # type: ignore
            self.send_response(200)
            self.send_header("Content-Type","text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(txt)))
            self.end_headers()
            self.wfile.write(txt.encode("utf-8"))
        else:
            body = {"status":"ok","service":"darwin-runner","metrics":"/metrics"}
            raw = json.dumps(body).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type","application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

class HTTPServer(ThreadingHTTPServer):
    def __init__(self, addr, handler, metrics: Metrics):
        super().__init__(addr, handler)
        self.metrics = metrics

# ----- Darwin Core -----
def coerce_float(d: Dict[str,Any], *keys):
    for k in keys:
        if k in d and d[k] is not None:
            try: return float(d[k])
            except Exception: pass
    return None

def extract_metrics(obj: Dict[str, Any]) -> Dict[str, Any]:
    # Normaliza nomes vindos do promotion_log / canário
    return {
        "delta_linf": coerce_float(obj, "delta_linf","delta_Linf","ΔL∞"),
        "novelty":    coerce_float(obj, "novelty","N"),
        "caos_ratio": coerce_float(obj, "caos_ratio","CAOS","caos"),
        "I":          coerce_float(obj, "I"),
        "P":          coerce_float(obj, "P"),
        "oci":        coerce_float(obj, "oci","OCI"),
        "ece":        coerce_float(obj, "ece","ECE"),
        "rho":        coerce_float(obj, "rho","ρ"),
        "notes":      obj.get("notes","")
    }

def now_iso():
    import datetime
    return datetime.datetime.utcnow().isoformat() + "Z"

class DarwinRunner:
    def __init__(self, cfg: Dict[str,Any]):
        self.cfg = cfg
        self.worm = cfg_get(cfg, "paths.worm_log")
        self.manifest = cfg_get(cfg, "paths.manifest")
        self.state_path = cfg_get(cfg, "paths.state", os.path.join(os.path.dirname(self.manifest or "/root"), "darwin_state.json"))
        self.source = cfg_get(cfg, "paths.source_promotions", "/root/promotion_log.json")
        self.deaths_per_birth = int(cfg_get(cfg, "births.deaths_per_birth", 10))
        self.interval = int(cfg_get(cfg, "operational.interval_seconds", 5))
        self.dry = bool(cfg_get(cfg, "operational.dry_run", False))
        self.fail_closed = bool(cfg_get(cfg, "operational.fail_closed", True))

        # counters/state
        self.metrics = Metrics()
        self.metrics.window_size = self.deaths_per_birth
        self.decisions = 0
        self.promotions = 0
        self.kills = 0
        self.spawns = 0
        self.deaths_since_spawn = 0

        self._stop = threading.Event()
        self._http = None
        self._offset = 0
        self._load_state()

    def _load_state(self):
        try:
            with open(self.state_path,"r",encoding="utf-8") as f:
                st = json.load(f)
            self._offset = int(st.get("source_offset", 0))
            self.deaths_since_spawn = int(st.get("deaths_since_spawn", 0))
            self.decisions = int(st.get("decisions", 0))
            self.promotions = int(st.get("promotions", 0))
            self.kills = int(st.get("kills", 0))
            self.spawns = int(st.get("spawns", 0))
        except Exception:
            pass

    def _save_state(self):
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        with open(self.state_path,"w",encoding="utf-8") as f:
            json.dump({
                "source_path": self.source,
                "source_offset": self._offset,
                "deaths_since_spawn": self.deaths_since_spawn,
                "decisions": self.decisions,
                "promotions": self.promotions,
                "kills": self.kills,
                "spawns": self.spawns,
                "last_ts": now_iso()
            }, f)

    def _update_manifest(self, last_hash: Optional[str]):
        try:
            os.makedirs(os.path.dirname(self.manifest), exist_ok=True)
            with open(self.manifest,"w",encoding="utf-8") as f:
                json.dump({
                    "timestamp": now_iso(),
                    "config": self.cfg.get("_config_path"),
                    "worm": self.worm,
                    "counts": {
                        "decisions": self.decisions,
                        "promotions": self.promotions,
                        "kills": self.kills,
                        "spawns": self.spawns
                    },
                    "birth_window": {
                        "deaths_since_spawn": self.deaths_since_spawn,
                        "window": self.deaths_per_birth
                    },
                    "last_hash": last_hash
                }, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

    def _write_event(self, ev: Dict[str,Any]) -> Optional[str]:
        if self.dry:
            return None
        return worm_write_event(self.worm, ev)

    def _spawn_newborn(self) -> Optional[str]:
        # Gera herança mínima (placeholder seguro)
        hdir = cfg_get(self.cfg, "paths.heritage_dir", "/root/darwin/heritage")
        os.makedirs(hdir, exist_ok=True)
        newborn = {
            "timestamp": now_iso(),
            "reason": "deaths_window_reached",
            "deaths_per_birth": self.deaths_per_birth
        }
        nb_path = os.path.join(hdir, f"newborn_{int(time.time())}.json")
        with open(nb_path,"w",encoding="utf-8") as f:
            json.dump(newborn, f)
        ev = {"event":"darwin_spawn_hook","timestamp": now_iso(),"path": nb_path}
        return self._write_event(ev)

    def _serve_metrics(self):
        port = int(cfg_get(self.cfg, "metrics.port", 9092))
        httpd = HTTPServer(("0.0.0.0", port), Handler, self.metrics)
        self._http = httpd
        httpd.serve_forever()

    def start_http(self):
        th = threading.Thread(target=self._serve_metrics, daemon=True)
        th.start()

    def stop(self):
        self._stop.set()
        try:
            if self._http:
                self._http.shutdown()
        except Exception:
            pass

    def _read_new_source_lines(self):
        if not os.path.exists(self.source):
            return []
        out = []
        with open(self.source,"r",encoding="utf-8",errors="replace") as f:
            f.seek(0, os.SEEK_END)
            end = f.tell()
            if self._offset > end:
                # arquivo truncou; reinicia
                self._offset = 0
            f.seek(self._offset)
            for ln in f:
                if ln.strip():
                    out.append(ln)
            self._offset = f.tell()
        return out

    def loop(self):
        sys.path.append('/root/darwin')
        from darwin_gate import decide as gate_decide
        # HTTP
        self.start_http()
        # Sinais
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        signal.signal(signal.SIGINT,  lambda *_: self.stop())

        while not self._stop.is_set():
            last_hash = None
            # Atualiza chain status
            ok, _, _ = verify_chain(worm_lines(self.worm))
            self.metrics.chain_ok = 1 if ok else 0

            # Ler novas decisões (promotion_log.json append-only)
            for ln in self._read_new_source_lines():
                try:
                    src = json.loads(ln)
                except Exception:
                    continue
                metrics = extract_metrics(src)
                decision = gate_decide(metrics, cfg_get(self.cfg,"thresholds",{}))
                # prepara evento "darwin_decision"
                ev = {
                    "event": "darwin_decision",
                    "timestamp": now_iso(),
                    "metrics": decision.get("metrics", {}),
                    "A": decision["A"], "C": decision["C"], "E": decision["E"],
                    "promote": decision["promote"],
                    "reason": decision["reason"],
                    "source": "promotion_log.json",
                }
                last_hash = self._write_event(ev)
                self.decisions += 1
                self.metrics.decisions = self.decisions

                if decision["promote"]:
                    last_hash = self._write_event({
                        "event":"darwin_promote_hook",
                        "timestamp": now_iso(),
                        "source":"darwin_gate",
                    })
                    self.promotions += 1
                    self.metrics.promotions = self.promotions
                elif decision["E"] == 1:
                    # Sobrevive por C(t)=1, sem promoção
                    last_hash = self._write_event({
                        "event":"darwin_survive_hook",
                        "timestamp": now_iso(),
                        "source":"darwin_gate",
                    })
                else:
                    last_hash = self._write_event({
                        "event":"darwin_kill_hook",
                        "timestamp": now_iso(),
                        "source":"darwin_gate",
                    })
                    self.kills += 1
                    self.metrics.kills = self.kills
                    self.deaths_since_spawn += 1
                    self.metrics.deaths_in_window = self.deaths_since_spawn

                    if self.deaths_since_spawn >= self.deaths_per_birth:
                        # Nascimento por janela de mortes
                        shash = self._spawn_newborn()
                        self.spawns += 1
                        self.metrics.spawns = self.spawns
                        self.deaths_since_spawn = 0
                        self.metrics.deaths_in_window = self.deaths_since_spawn
                        last_hash = shash or last_hash

                self._update_manifest(last_hash)
                self._save_state()

            # espera próxima rodada
            self._save_state()
            time.sleep(max(1, self.interval))

def main():
    cfg_path = None
    if len(sys.argv) > 1 and sys.argv[1] in ("--config","-c") and len(sys.argv) > 2:
        cfg_path = sys.argv[2]
    cfg = load_config(cfg_path)
    runner = DarwinRunner(cfg)
    try:
        runner.loop()
    finally:
        runner.stop()

if __name__ == "__main__":
    main()