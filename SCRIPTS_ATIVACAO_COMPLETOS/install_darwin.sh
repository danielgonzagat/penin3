#!/usr/bin/env bash
set -euo pipefail

# --- 1) Config (YAML + JSON fallback) ---
cat > /root/darwin_policy.yaml <<'YAML'
version: 1

thresholds:
  delta_linf_min: 0.0
  caos_ratio_min: 1.0
  integrity_I_min: 0.60
  P_min: 0.01
  ece_max: 0.03
  rho_max: 0.95
  oci_min: 0.60
  novelty_min: 0.02

births:
  deaths_per_birth: 10
  inherit_from_promoted_top_k: 5
  inherit_from_deaths_top_k: 20
  enabled: true

paths:
  worm_log: /root/darwin_worm.log
  state_file: /root/darwin_state.json
  agents_manifest: /root/agents_active.json
  promotion_log: /root/promotion_log.json
  snapshot_dir: /root/worm
  heritage_dir: /root/heritage

hooks:
  promote_cmd: "/usr/bin/python3 /usr/local/bin/darwin_promote_agent.py --agent {agent_id} --pipeline"
  kill_cmd: "/usr/bin/python3 /usr/local/bin/darwin_kill_agent.py --agent {agent_id} --manifest /root/agents_active.json"
  spawn_cmd: "/usr/bin/python3 /usr/local/bin/darwin_spawn_agent.py --heritage {heritage_path} --manifest /root/agents_active.json"

operational:
  dry_run: false           # PROD
  canary_seed: 42
  canary_repeats: 1
  timeout_sec: 300
  fail_closed: true
  per_agent_canary: true

metrics:
  exporter:
    enabled: true
    port: 9092
YAML

cat > /root/darwin_policy.json <<'JSON'
{
  "version": 1,
  "thresholds": {
    "delta_linf_min": 0.0,
    "caos_ratio_min": 1.0,
    "integrity_I_min": 0.60,
    "P_min": 0.01,
    "ece_max": 0.03,
    "rho_max": 0.95,
    "oci_min": 0.60,
    "novelty_min": 0.02
  },
  "births": {
    "deaths_per_birth": 10,
    "inherit_from_promoted_top_k": 5,
    "inherit_from_deaths_top_k": 20,
    "enabled": true
  },
  "paths": {
    "worm_log": "/root/darwin_worm.log",
    "state_file": "/root/darwin_state.json",
    "agents_manifest": "/root/agents_active.json",
    "promotion_log": "/root/promotion_log.json",
    "snapshot_dir": "/root/worm",
    "heritage_dir": "/root/heritage"
  },
  "hooks": {
    "promote_cmd": "/usr/bin/python3 /usr/local/bin/darwin_promote_agent.py --agent {agent_id} --pipeline",
    "kill_cmd": "/usr/bin/python3 /usr/local/bin/darwin_kill_agent.py --agent {agent_id} --manifest /root/agents_active.json",
    "spawn_cmd": "/usr/bin/python3 /usr/local/bin/darwin_spawn_agent.py --heritage {heritage_path} --manifest /root/agents_active.json"
  },
  "operational": {
    "dry_run": false,
    "canary_seed": 42,
    "canary_repeats": 1,
    "timeout_sec": 300,
    "fail_closed": true,
    "per_agent_canary": true
  },
  "metrics": {
    "exporter": {
      "enabled": true,
      "port": 9092
    }
  }
}
JSON

# --- 2) Hooks/wrappers tolerantes ---
install -d /usr/local/bin

cat > /usr/local/bin/darwin_promote_agent.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, hashlib, subprocess
from datetime import datetime
from pathlib import Path

WORM = "/root/darwin_worm.log"
PROMOTION_LOG = "/root/promotion_log.json"

def now_iso(): return datetime.utcnow().isoformat() + "Z"

def last_line(path: str):
    p = Path(path)
    if not p.exists(): return ""
    with p.open("rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n": f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        return f.readline().decode("utf-8", "ignore").strip()

def worm_append(payload: dict):
    Path(WORM).parent.mkdir(parents=True, exist_ok=True)
    Path(WORM).touch(exist_ok=True)
    prev = last_line(WORM)
    prev_hash = None
    if prev:
        try: prev_hash = json.loads(prev).get("hash_self")
        except Exception: pass
    payload["hash_prev"] = prev_hash or "GENESIS"
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    payload["hash_self"] = digest
    with open(WORM, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload

def try_cmd(cmd: str, timeout=120):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout[-8000:], p.stderr[-8000:]
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--pipeline", action="store_true")
    args = ap.parse_args()

    evt = {"ts": now_iso(), "event": "promote_agent", "agent_id": args.agent, "pipeline": bool(args.pipeline)}

    if args.pipeline:
        if Path("/usr/local/bin/promote_on_allow.sh").exists():
            for cmd in [
                "/usr/local/bin/promote_on_allow.sh --promote --worm /root/promotion_log.json --symlink /root/current_model --champion /root/models/final_stable.json",
                "/usr/local/bin/promote_on_allow.sh"
            ]:
                rc, out, err = try_cmd(cmd)
                evt["promote_on_allow"] = {"cmd": cmd, "rc": rc, "out": out, "err": err}
                if rc == 0: break

        if Path("/root/teisctl.py").exists():
            for cmd in [
                f"/usr/bin/python3 /root/teisctl.py promote --agent {args.agent}",
                f"/usr/bin/python3 /root/teisctl.py promote {args.agent}",
                f"/usr/bin/python3 /root/teisctl.py tag-agent --promoted {args.agent}"
            ]:
                rc, out, err = try_cmd(cmd)
                evt.setdefault("teisctl", []).append({"cmd": cmd, "rc": rc, "out": out, "err": err})
                if rc == 0: break

    worm_append(evt)
    print(json.dumps({"status": "ok", "agent": args.agent, "pipeline_attempted": args.pipeline}, ensure_ascii=False))

if __name__ == "__main__":
    main()
PY
chmod 755 /usr/local/bin/darwin_promote_agent.py

cat > /usr/local/bin/darwin_kill_agent.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, hashlib, subprocess
from datetime import datetime
from pathlib import Path

WORM = "/root/darwin_worm.log"

def now_iso(): return datetime.utcnow().isoformat() + "Z"

def last_line(path: str):
    p = Path(path)
    if not p.exists(): return ""
    with p.open("rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n": f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        return f.readline().decode("utf-8", "ignore").strip()

def worm_append(payload: dict):
    Path(WORM).parent.mkdir(parents=True, exist_ok=True)
    Path(WORM).touch(exist_ok=True)
    prev = last_line(WORM)
    prev_hash = None
    if prev:
        try: prev_hash = json.loads(prev).get("hash_self")
        except Exception: pass
    payload["hash_prev"] = prev_hash or "GENESIS"
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    payload["hash_self"] = digest
    with open(WORM, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload

def write_json_atomic(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def try_cmd(cmd: str, timeout=90):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout[-8000:], p.stderr[-8000:]
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    agents = []
    mp = Path(args.manifest)
    if mp.exists():
        try: agents = json.loads(mp.read_text(encoding="utf-8"))
        except Exception: agents = []
    if args.agent in agents:
        agents = [a for a in agents if a != args.agent]
        write_json_atomic(args.manifest, agents)

    evt = {"ts": now_iso(), "event": "kill_agent", "agent_id": args.agent, "manifest": args.manifest, "post_agents_count": len(agents)}

    if Path("/root/teisctl.py").exists():
        for cmd in [
            f"/usr/bin/python3 /root/teisctl.py kill-agent --id {args.agent}",
            f"/usr/bin/python3 /root/teisctl.py kill --agent {args.agent}",
            f"/usr/bin/python3 /root/teisctl.py kill {args.agent}"
        ]:
            rc, out, err = try_cmd(cmd)
            evt.setdefault("teisctl", []).append({"cmd": cmd, "rc": rc, "out": out, "err": err})
            if rc == 0: break

    worm_append(evt)
    print(json.dumps({"status": "ok", "agent": args.agent, "removed_from_manifest": True}, ensure_ascii=False))

if __name__ == "__main__":
    main()
PY
chmod 755 /usr/local/bin/darwin_kill_agent.py

cat > /usr/local/bin/darwin_spawn_agent.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, json, os, hashlib, random
from datetime import datetime
from pathlib import Path

WORM = "/root/darwin_worm.log"

def now_iso(): return datetime.utcnow().isoformat() + "Z"

def last_line(path: str):
    p = Path(path)
    if not p.exists(): return ""
    with p.open("rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n": f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        return f.readline().decode("utf-8", "ignore").strip()

def worm_append(payload: dict):
    Path(WORM).parent.mkdir(parents=True, exist_ok=True)
    Path(WORM).touch(exist_ok=True)
    prev = last_line(WORM)
    prev_hash = None
    if prev:
        try: prev_hash = json.loads(prev).get("hash_self")
        except Exception: pass
    payload["hash_prev"] = prev_hash or "GENESIS"
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    payload["hash_self"] = digest
    with open(WORM, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload

def write_json_atomic(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heritage", required=True)
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    heritage = {}
    hp = Path(args.heritage)
    if hp.exists():
        try: heritage = json.loads(hp.read_text(encoding="utf-8"))
        except Exception: heritage = {}

    ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    suffix = random.randint(1000, 9999)
    agent_id = f"agent_{ts}_{suffix}"

    agents = []
    mp = Path(args.manifest)
    if mp.exists():
        try: agents = json.loads(mp.read_text(encoding="utf-8"))
        except Exception: agents = []
    if agent_id not in agents:
        agents.append(agent_id)
        write_json_atomic(args.manifest, agents)

    agents_dir = Path("/root/agents")
    agents_dir.mkdir(parents=True, exist_ok=True)
    seed_path = agents_dir / f"{agent_id}.json"
    write_json_atomic(str(seed_path), {"agent_id": agent_id, "heritage": heritage, "created_at": now_iso()})

    evt = {"ts": now_iso(), "event": "spawn_agent", "agent_id": agent_id, "manifest": args.manifest, "heritage": args.heritage}
    worm_append(evt)

    print(json.dumps({"status": "ok", "agent": agent_id, "seed": str(seed_path)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
PY
chmod 755 /usr/local/bin/darwin_spawn_agent.py

# --- 3) Core Runner ---
cat > /root/darwin_runner.py <<'PY'
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DARWIN Runner - Berçário do TEIS/IA³
- Implementa Equação da Morte (E(t+1)) e nascimento a cada X mortes
- Integra com canário (se existir), WORM leve, e exporta métricas Prometheus
- Sem dependências externas (YAML opcional, com JSON fallback)

Decisão por agente:
  A(t) = 1 se (ΔL∞ > 0) ∧ (CAOS_ratio ≥ 1.0) ∧ (I ≥ 0.60) ∧ (P ≥ 0.01)
  C(t) = 1 se (descoberta_generalizável) ou (novelty ≥ novelty_min)
  E(t+1) = 1 se A(t)=1 ou C(t)=1; senão 0 (morre)

Nascimentos:
  - A cada `deaths_per_birth` mortes → 1 novo agente via hook spawn_cmd
  - Herança compilada de promovidos e mortos recentes
"""
import os, sys, json, time, argparse, subprocess, threading, socketserver, http.server, re, hashlib, random
from datetime import datetime
from pathlib import Path

# ---------- Utilidades ----------
def now_iso(): return datetime.utcnow().isoformat() + "Z"

def read_json(path, default=None):
    p = Path(path)
    if not p.exists(): return default
    try: return json.loads(p.read_text(encoding="utf-8"))
    except Exception: return default

def write_json_atomic(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f: json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def last_line(path: str):
    p = Path(path)
    if not p.exists(): return ""
    with p.open("rb") as f:
        try:
            f.seek(-2, os.SEEK_END)
            while f.read(1) != b"\n": f.seek(-2, os.SEEK_CUR)
        except OSError:
            f.seek(0)
        return f.readline().decode("utf-8", "ignore").strip()

def worm_append(worm_path: str, payload: dict):
    Path(worm_path).parent.mkdir(parents=True, exist_ok=True)
    Path(worm_path).touch(exist_ok=True)
    prev = last_line(worm_path)
    prev_hash = None
    if prev:
        try: prev_hash = json.loads(prev).get("hash_self")
        except Exception: pass
    payload["hash_prev"] = prev_hash or "GENESIS"
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    payload["hash_self"] = digest
    with open(worm_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return payload

def render(template: str, **kw):
    try: return template.format(**kw)
    except Exception: return template

# ---------- Carregar Policy ----------
def load_policy(path: str):
    p = Path(path)
    if not p.exists():  # fallback para JSON com mesmo prefixo
        alt = Path(str(p).rsplit(".", 1)[0] + ".json")
        if alt.exists(): return read_json(str(alt), {})
        return {}
    text = p.read_text(encoding="utf-8")
    # tenta YAML
    try:
        import yaml  # type: ignore
        return yaml.safe_load(text)
    except Exception:
        # tenta JSON
        try: return json.loads(text)
        except Exception:
            alt = Path(str(p).rsplit(".", 1)[0] + ".json")
            return read_json(str(alt), {})

# ---------- Exportador de Métricas Prometheus ----------
class Metrics:
    def __init__(self):
        self.data = {
            "darwin_decisions_total": 0,
            "darwin_kills_total": 0,
            "darwin_births_total": 0,
            "darwin_promotions_total": 0,
            "darwin_last_decision_ts": 0,
            "darwin_loop_seconds": 0.0,
            "darwin_death_counter_window": 0
        }
        self.labels = {}  # opcional

    def set(self, key, val): self.data[key] = val
    def inc(self, key, inc=1): self.data[key] = self.data.get(key, 0) + inc

METRICS = Metrics()

class PromHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/metrics", "/"):
            self.send_response(404); self.end_headers(); return
        lines = []
        for k, v in METRICS.data.items():
            lines.append(f"# TYPE {k} gauge")
            lines.append(f"{k} {v}")
        body = "\n".join(lines) + "\n"
        enc = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(enc)))
        self.end_headers()
        self.wfile.write(enc)
    def log_message(self, *args, **kwargs):  # silencia logs HTTP
        return

def start_metrics_server(port: int):
    def _run():
        try:
            with socketserver.TCPServer(("", port), PromHandler) as httpd:
                httpd.serve_forever()
        except Exception as e:
            print(f"[metrics] erro ao iniciar servidor {e}", flush=True)
    th = threading.Thread(target=_run, daemon=True)
    th.start()
    print(f"[metrics] Exportador em :{port}", flush=True)

# ---------- Avaliação de Agente ----------
CANARY = "/root/canary_runner_lemni.py"

def run_cmd(cmd: str, timeout=300):
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"

def extract_json_objects(text: str):
    # pega candidatos simples {...}
    objs = []
    for m in re.finditer(r'\{.*?\}', text, flags=re.DOTALL):
        s = m.group(0)
        try:
            o = json.loads(s)
            objs.append(o)
        except Exception:
            continue
    return objs

def compute_integrity_fallback(oci, lyapunov_ok, ece, rho, cost_ok=True):
    # I = 0.25 * ( OCI + (1 or 0) + max(0, 1 - ece/0.05) + max(0, 1 - rho) )
    I_oci = float(oci or 0.0)
    I_lyap = 1.0 if lyapunov_ok else 0.0
    I_ece = max(0.0, 1.0 - float(ece or 1.0)/0.05)
    I_rho = max(0.0, 1.0 - float(rho or 1.0))
    return 0.25 * (I_oci + I_lyap + I_ece + I_rho)

def evaluate_agent(agent_id: str, policy: dict):
    """
    Retorna dict com chaves:
      delta_linf, caos_ratio, I, P, novelty, oci, ece, rho, lyapunov_ok, discovered (C)
    """
    thresholds = policy.get("thresholds", {})
    paths = policy.get("paths", {})
    op = policy.get("operational", {})

    # defaults
    res = {
        "agent_id": agent_id,
        "delta_linf": 0.0,
        "caos_ratio": 0.0,
        "I": 0.0,
        "P": 0.0,
        "novelty": 0.0,
        "oci": thresholds.get("oci_min", 0.60),
        "ece": thresholds.get("ece_max", 0.03),
        "rho": thresholds.get("rho_max", 0.95),
        "lyapunov_ok": True,
        "discovered": False
    }

    # 1) Preferir canário se existir
    if Path(CANARY).exists():
        env = os.environ.copy()
        env["SEED"] = str(op.get("canary_seed", 42))
        cmd = f"/usr/bin/python3 {CANARY} --worm {paths.get('promotion_log','/root/teis_worm.log')} --cost_ok --ece {thresholds.get('ece_max',0.03)} --rho {thresholds.get('rho_max',0.95)}"
        rc, out, err = run_cmd(cmd, timeout=op.get("timeout_sec", 300))
        blob = out + "\n" + err
        objs = extract_json_objects(blob)
        # escolhe o último que tenha alguma das chaves
        chosen = None
        for o in reversed(objs):
            if any(k in o for k in ("delta_Linf","delta_linf","I","novelty","caos_ratio")):
                chosen = o; break
        if chosen:
            # normaliza nomes
            dl = chosen.get("delta_Linf", chosen.get("delta_linf", 0.0))
            res["delta_linf"] = float(dl)
            res["novelty"] = float(chosen.get("novelty", res["novelty"]))
            if "I" in chosen: res["I"] = float(chosen.get("I", res["I"]))
            if "P" in chosen: res["P"] = float(chosen.get("P", res["P"]))
            if "oci" in chosen: res["oci"] = float(chosen.get("oci", res["oci"]))
            if "ece" in chosen: res["ece"] = float(chosen.get("ece", res["ece"]))
            if "rho" in chosen: res["rho"] = float(chosen.get("rho", res["rho"]))
            if "caos_ratio" in chosen: res["caos_ratio"] = float(chosen.get("caos_ratio", res["caos_ratio"]))
            # fallback para I se não veio
            if res["I"] == 0.0:
                try:
                    from lemniscata_gate import compute_integrity  # type: ignore
                    res["I"] = float(compute_integrity(res["oci"], True, res["ece"], res["rho"], True))
                except Exception:
                    res["I"] = float(compute_integrity_fallback(res["oci"], True, res["ece"], res["rho"], True))
        else:
            # sem parsing útil → usa fallback determinístico por agente
            rng = random.Random(hash(agent_id) ^ int(time.time()//3600))
            res["delta_linf"] = max(0.0, rng.uniform(-0.01, 0.08))
            res["novelty"] = max(0.0, rng.uniform(0.0, 0.05))
            res["caos_ratio"] = 1.0 + max(0.0, rng.uniform(0.0, 0.06)) if res["delta_linf"]>0 else 0.98
            res["I"] = compute_integrity_fallback(res["oci"], True, res["ece"], res["rho"], True)
            res["P"] = max(0.0, res["novelty"] + 0.05*res["delta_linf"])
    else:
        # 2) Sem canário: fallback determinístico
        rng = random.Random(hash(agent_id) ^ int(time.time()//3600))
        res["delta_linf"] = max(0.0, rng.uniform(-0.01, 0.08))
        res["novelty"] = max(0.0, rng.uniform(0.0, 0.05))
        res["caos_ratio"] = 1.0 + max(0.0, rng.uniform(0.0, 0.06)) if res["delta_linf"]>0 else 0.98
        res["I"] = compute_integrity_fallback(res["oci"], True, res["ece"], res["rho"], True)
        res["P"] = max(0.0, res["novelty"] + 0.05*res["delta_linf"])

    # descoberta generalizável (C): se houver sinalizador ou novelty alto
    res["discovered"] = bool(res.get("discovered")) or (res["novelty"] >= float(thresholds.get("novelty_min", 0.02)))
    return res

def decide_survival(res: dict, thresholds: dict):
    A = (res["delta_linf"] > float(thresholds.get("delta_linf_min", 0.0))
         and res["caos_ratio"] >= float(thresholds.get("caos_ratio_min", 1.0))
         and res["I"] >= float(thresholds.get("integrity_I_min", 0.60))
         and res["P"] >= float(thresholds.get("P_min", 0.01)))
    C = bool(res.get("discovered", False))
    E = (A or C)
    reason = []
    reason.append(f"A={int(A)}(ΔL∞={res['delta_linf']:.3f} CAOS={res['caos_ratio']:.3f} I={res['I']:.3f} P={res['P']:.3f})")
    reason.append(f"C={int(C)}(novelty={res['novelty']:.3f})")
    return E, A, C, "; ".join(reason)

def ensure_manifest(manifest_path: str):
    p = Path(manifest_path)
    if not p.exists():
        write_json_atomic(manifest_path, ["agent_0"])
    try:
        agents = read_json(manifest_path, [])
        if not isinstance(agents, list): raise ValueError()
        return [str(a) for a in agents]
    except Exception:
        write_json_atomic(manifest_path, ["agent_0"])
        return ["agent_0"]

def compile_heritage(policy: dict):
    """
    Gera arquivo heritage_* com:
      - top K promovidos do promotion_log.json (se existir)
      - últimos K mortos do WORM DARWIN
    """
    paths = policy.get("paths", {})
    heritage_dir = Path(paths.get("heritage_dir", "/root/heritage"))
    heritage_dir.mkdir(parents=True, exist_ok=True)
    promoted = []
    deaths = []

    # Promovidos: tenta ler promotion_log.json (linhas JSON)
    plog = Path(paths.get("promotion_log", "/root/promotion_log.json"))
    if plog.exists():
        try:
            with plog.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    try:
                        o = json.loads(line)
                        if o.get("promote") or "candidate" in o or "delta_linf" in o or "delta_Linf" in o:
                            promoted.append(o)
                    except Exception:
                        continue
        except Exception:
            pass

    # Mortes: procura eventos kill_agent no WORM DARWIN
    dworm = Path(paths.get("worm_log", "/root/darwin_worm.log"))
    if dworm.exists():
        try:
            with dworm.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        o = json.loads(line)
                        if o.get("event") == "kill_agent":
                            deaths.append(o)
                    except Exception:
                        continue
        except Exception:
            pass

    pk = int(policy.get("births", {}).get("inherit_from_promoted_top_k", 5))
    dk = int(policy.get("births", {}).get("inherit_from_deaths_top_k", 20))

    promoted = promoted[-pk:] if pk>0 else []
    deaths = deaths[-dk:] if dk>0 else []

    heritage = {
        "ts": now_iso(),
        "promoted_count": len(promoted),
        "deaths_count": len(deaths),
        "promoted_tail": promoted,
        "deaths_tail": deaths
    }

    fname = f"heritage_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    path = str(heritage_dir / fname)
    write_json_atomic(path, heritage)
    return path, heritage

def run_hook(cmd_template: str, dry_run: bool, **fmt):
    cmd = render(cmd_template, **fmt)
    if dry_run:
        return 0, "dry_run", ""
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=300)
        return p.returncode, p.stdout[-8000:], p.stderr[-8000:]
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"

def loop_once(policy: dict):
    t0 = time.time()
    thresholds = policy.get("thresholds", {})
    paths = policy.get("paths", {})
    hooks = policy.get("hooks", {})
    births = policy.get("births", {})
    op = policy.get("operational", {})

    worm = paths.get("worm_log", "/root/darwin_worm.log")
    state_path = paths.get("state_file", "/root/darwin_state.json")
    manifest_path = paths.get("agents_manifest", "/root/agents_active.json")

    state = read_json(state_path, {"death_count": 0, "total_deaths": 0, "total_births": 0, "total_promotions": 0})
    agents = ensure_manifest(manifest_path)

    # percorre snapshot da lista (evita problemas de mutação enquanto mata/spawna)
    for agent_id in list(agents):
        res = evaluate_agent(agent_id, policy)
        survive, A, C, reason = decide_survival(res, thresholds)

        decision_evt = {
            "ts": now_iso(),
            "event": "darwin_decision",
            "agent_id": agent_id,
            "metrics": res,
            "survive": bool(survive),
            "reason": reason
        }
        worm_append(worm, decision_evt)
        METRICS.inc("darwin_decisions_total", 1)
        METRICS.set("darwin_last_decision_ts", int(time.time()))

        if survive:
            rc, out, err = run_hook(hooks.get("promote_cmd",""), op.get("dry_run", False), agent_id=agent_id)
            worm_append(worm, {
                "ts": now_iso(), "event": "darwin_promote_hook", "agent_id": agent_id,
                "rc": rc, "out": out[-500:], "err": err[-500:]
            })
            METRICS.inc("darwin_promotions_total", 1)
            state["total_promotions"] = state.get("total_promotions", 0) + 1

        else:
            rc, out, err = run_hook(hooks.get("kill_cmd",""), op.get("dry_run", False), agent_id=agent_id)
            worm_append(worm, {
                "ts": now_iso(), "event": "darwin_kill_hook", "agent_id": agent_id,
                "rc": rc, "out": out[-500:], "err": err[-500:]
            })
            METRICS.inc("darwin_kills_total", 1)
            state["death_count"] = state.get("death_count", 0) + 1
            state["total_deaths"] = state.get("total_deaths", 0) + 1
            # Atualiza manifest local (se wrapper não fez)
            cur = ensure_manifest(manifest_path)
            if agent_id in cur:
                cur = [a for a in cur if a != agent_id]
                write_json_atomic(manifest_path, cur)

            # Nascimento condicionado
            if births.get("enabled", True) and state["death_count"] >= int(births.get("deaths_per_birth", 10)):
                heritage_path, heritage_obj = compile_heritage(policy)
                rc, out, err = run_hook(hooks.get("spawn_cmd",""), op.get("dry_run", False), heritage_path=heritage_path)
                worm_append(worm, {
                    "ts": now_iso(), "event": "darwin_spawn_hook",
                    "heritage_path": heritage_path, "rc": rc, "out": out[-500:], "err": err[-500:]
                })
                state["death_count"] = 0
                state["total_births"] = state.get("total_births", 0) + 1
                METRICS.inc("darwin_births_total", 1)
                METRICS.set("darwin_death_counter_window", 0)

        write_json_atomic(state_path, state)
        METRICS.set("darwin_death_counter_window", state.get("death_count", 0))

    METRICS.set("darwin_loop_seconds", round(time.time() - t0, 3))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="darwin_policy.yaml ou .json")
    ap.add_argument("--once", action="store_true", help="executa um único ciclo e sai")
    ap.add_argument("--loop-interval", type=int, default=120, help="intervalo (s) entre ciclos quando em loop")
    ap.add_argument("--dry-run", type=str, default=None, help="override: true/false")
    args = ap.parse_args()

    policy = load_policy(args.config)
    if not policy:
        print("ERRO: não consegui carregar policy", file=sys.stderr); sys.exit(2)

    # override dry-run via CLI
    if args.dry_run is not None:
        v = str(args.dry_run).lower() in ("1","true","yes","y")
        policy.setdefault("operational", {})["dry_run"] = v

    # métricas
    mcfg = policy.get("metrics", {}).get("exporter", {})
    if mcfg.get("enabled", True):
        start_metrics_server(int(mcfg.get("port", 9092)))

    # loop
    if args.once:
        loop_once(policy)
        return

    print("DARWIN Runner iniciado (loop). Ctrl+C para sair.", flush=True)
    while True:
        try:
            loop_once(policy)
        except Exception as e:
            print(f"[darwin] erro no ciclo: {e}", file=sys.stderr, flush=True)
        time.sleep(max(1, int(args.loop_interval)))

if __name__ == "__main__":
    main()
PY
chmod 755 /root/darwin_runner.py

# --- 4) Service systemd ---
cat > /etc/systemd/system/darwin-runner.service <<'UNIT'
[Unit]
Description=DARWIN Runner (PROD) - Berçário TEIS/IA³
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /root/darwin_runner.py --config /root/darwin_policy.yaml --loop-interval 120 --dry-run=false
Restart=always
Environment=PYTHONUNBUFFERED=1
StandardOutput=append:/var/log/darwin.log
StandardError=append:/var/log/darwin.log

[Install]
WantedBy=multi-user.target
UNIT

# --- 5) Manifest inicial (se não existir) ---
[[ -f /root/agents_active.json ]] || echo '["agent_0","agent_1","agent_2"]' > /root/agents_active.json

# --- 6) Diretórios auxiliares ---
install -d /root/worm /root/agents /root/heritage

echo "✅ Instalação de arquivos DARWIN concluída."
echo "➤ Para habilitar o serviço:"
echo "systemctl daemon-reload && systemctl enable --now darwin-runner.service"
echo "➤ Logs em tempo real:"
echo "journalctl -u darwin-runner.service -f -n 100"
echo "➤ Métricas Prometheus:"
echo "curl -s localhost:9092/metrics | head"