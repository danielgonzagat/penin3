#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/ia3-constructor"
USER_NAME="${SUDO_USER:-$(id -un)}"
USER_GROUP="$(id -gn "${USER_NAME}")"
PY_BIN="/usr/bin/python3"
SYSTEMD_SERVICE="/etc/systemd/system/ia3-constructor.service"
Q_AGENT_DIR="/root/.aws/amazonq/cli-agents"
Q_AGENT_JSON="${Q_AGENT_DIR}/ia3-constructor.json"

echo "==> Instalando IA3-Constructor em: ${APP_DIR}"
echo "==> Rodará como systemd service (24/7)."

# =========================
# Pré-checagens
# =========================
if ! command -v "${PY_BIN}" >/dev/null 2>&1; then
  echo "ERRO: Python3 não encontrado. Instale: apt update && apt install -y python3 python3-venv python3-pip"
  exit 1
fi

# =========================
# Pasta e permissões
# =========================
echo "==> Criando diretórios..."
mkdir -p "${APP_DIR}"/{src/agent_core,src/api,logs}
mkdir -p /opt/penin
chown -R "${USER_NAME}:${USER_GROUP}" "${APP_DIR}" /opt/penin

# =========================
# requirements.txt
# =========================
cat > "${APP_DIR}/requirements.txt" <<'TXT'
fastapi
uvicorn[standard]
chromadb
numpy
pydantic
python-dotenv
tqdm
filelock
TXT

# =========================
# .env
# =========================
cat > "${APP_DIR}/.env" <<'ENV'
IA3_ALLOWED_DIRS=/opt/ia3-constructor,/opt/penin
SIGMA_ECE_MAX=0.01
SIGMA_BIAS_MAX=1.05
SIGMA_RHO_MAX=0.99
SIGMA_SR_MIN=0.80
SIGMA_G_MIN=0.85
ENV

# =========================
# Código-fonte
# =========================

cat > "${APP_DIR}/src/boot.py" <<'PY'
import uvicorn
if __name__ == "__main__":
    uvicorn.run("src.api.server:app", host="0.0.0.0", port=8010, reload=False)
PY

cat > "${APP_DIR}/src/api/server.py" <<'PY'
from fastapi import FastAPI
from pydantic import BaseModel
from threading import Thread
from ..agent_core.super_agent import SuperAgent

app = FastAPI()
AGENT = None
THREAD = None

class Boot(BaseModel):
    name: str = "IA3-Constructor"
    goal: str = "Construir e evoluir a IA ao Cubo (PENIN-Ω) de forma ética e auditável."

@app.post("/boot")
def boot_agent(cfg: Boot):
    global AGENT, THREAD
    if THREAD and THREAD.is_alive():
        return {"status":"already_running"}
    AGENT = SuperAgent(cfg.name, cfg.goal, ledger_path="logs/ledger.worm")
    THREAD = Thread(target=AGENT.run_forever, daemon=True)
    THREAD.start()
    return {"status":"started"}

@app.post("/stop")
def stop_agent():
    global AGENT
    if AGENT:
        AGENT.shutdown()
        return {"status":"stopping"}
    return {"status":"not_running"}

@app.get("/health")
def health():
    return {"ok": True}
PY

cat > "${APP_DIR}/src/agent_core/worm_ledger.py" <<'PY'
from datetime import datetime
from pathlib import Path
from hashlib import sha256
from filelock import FileLock

class WormLedger:
    def __init__(self, path: str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = FileLock(str(self.path)+".lock")

    def append(self, event: dict) -> str:
        ts = datetime.utcnow().isoformat()
        event_line = f"{ts} | {event}"
        prev_hash = self._tail_hash()
        h = sha256((prev_hash + event_line).encode()).hexdigest()
        with self.lock:
            with self.path.open("a") as f:
                f.write(f"{h} | {event_line}\n")
        return h

    def _tail_hash(self) -> str:
        if not self.path.exists():
            return "GENESIS"
        with self.path.open() as f:
            last = None
            for last in f: pass
        return last.split(" | ")[0] if last else "GENESIS"
PY

cat > "${APP_DIR}/src/agent_core/memory.py" <<'PY'
from chromadb import Client
from chromadb.config import Settings

class VectorMemory:
    def __init__(self, name="ia3_memory"):
        self.db = Client(Settings(anonymized_telemetry=False))
        self.col = self.db.get_or_create_collection(name=name)

    def upsert_text(self, text: str, meta: dict):
        eid = meta.get("id") or str(abs(hash(text)) % (10**12))
        self.col.upsert(ids=[eid], documents=[text], metadatas=[meta])

    def query(self, query: str, k=5):
        res = self.col.query(query_texts=[query], n_results=k)
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return list(zip(docs, metas))
PY

cat > "${APP_DIR}/src/agent_core/sigma_guard.py" <<'PY'
import os

class SigmaGuard:
    def __init__(self):
        self.ECE_MAX = float(os.getenv("SIGMA_ECE_MAX", "0.01"))
        self.RHO_MAX = float(os.getenv("SIGMA_RHO_MAX", "0.99"))
        self.BIAS_MAX = float(os.getenv("SIGMA_BIAS_MAX", "1.05"))
        self.SR_MIN  = float(os.getenv("SIGMA_SR_MIN", "0.80"))
        self.G_MIN   = float(os.getenv("SIGMA_G_MIN", "0.85"))

    def check(self, ece: float, rho: float, bias: float, sr: float, g: float):
        ok = (ece <= self.ECE_MAX and rho < self.RHO_MAX and
              bias <= self.BIAS_MAX and sr >= self.SR_MIN and g >= self.G_MIN)
        return ok, {
            "ece_ok": ece <= self.ECE_MAX,
            "rho_ok": rho < self.RHO_MAX,
            "bias_ok": bias <= self.BIAS_MAX,
            "sr_ok":  sr >= self.SR_MIN,
            "g_ok":   g >= self.G_MIN
        }
PY

cat > "${APP_DIR}/src/agent_core/evaluator.py" <<'PY'
import random
class Evaluator:
    def score_proxies(self, context: str):
        ece  = random.uniform(0.001, 0.02)
        rho  = random.uniform(0.90, 1.02)
        bias = random.uniform(0.98, 1.08)
        sr   = random.uniform(0.70, 0.95)
        g    = random.uniform(0.75, 0.95)
        return dict(ece=ece, rho=rho, bias=bias, sr=sr, g=g)
PY

cat > "${APP_DIR}/src/agent_core/actions.py" <<'PY'
from pathlib import Path
import subprocess, shutil

class Actions:
    def __init__(self, allowed_dirs: list[str], ledger):
        self.allowed = [Path(p).resolve() for p in allowed_dirs]
        self.ledger = ledger

    def _inside_allowed(self, path: Path) -> bool:
        p = path.resolve()
        return any(str(p).startswith(str(a)) for a in self.allowed)

    def write_file(self, path: str, content: str):
        p = Path(path)
        if not self._inside_allowed(p):
            raise PermissionError(f"Path fora de IA3_ALLOWED_DIRS: {p}")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        self.ledger.append({"action":"write_file","path":str(p)})

    def run_cmd(self, cmd: str, workdir: str | None = None):
        res = subprocess.run(cmd, cwd=workdir, shell=True,
                             capture_output=True, text=True)
        self.ledger.append({"action":"run_cmd","cmd":cmd,
                            "code":res.returncode})
        return res.stdout, res.stderr, res.returncode

    def copy_tree(self, src: str, dst: str):
        s, d = Path(src), Path(dst)
        if not self._inside_allowed(d):
            raise PermissionError(f"Destino fora de IA3_ALLOWED_DIRS: {d}")
        shutil.copytree(s, d, dirs_exist_ok=True)
        self.ledger.append({"action":"copy_tree","src":str(s),"dst":str(d)})
PY

cat > "${APP_DIR}/src/agent_core/subagent.py" <<'PY'
import threading, time
from .memory import VectorMemory
from .worm_ledger import WormLedger
from .evaluator import Evaluator
from .sigma_guard import SigmaGuard
from .actions import Actions

class SubAgent(threading.Thread):
    def __init__(self, name, mission, allowed_dirs, ledger_path):
        super().__init__(daemon=True)
        self.name = name
        self.mission = mission
        self.mem = VectorMemory(name+"_mem")
        self.ledger = WormLedger(ledger_path)
        self.eval = Evaluator()
        self.guard = SigmaGuard()
        self.actions = Actions(allowed_dirs, self.ledger)
        self.alive = True
        self.conf = 0.75

    def run(self):
        while self.alive:
            ctx = self._gather_context()
            proxies = self.eval.score_proxies(ctx)
            ok, flags = self.guard.check(**proxies)
            self.ledger.append({"subagent":self.name,"proxies":proxies,"flags":flags})

            if not ok:
                time.sleep(5)
                self.conf *= 0.95
                continue

            try:
                self.actions.write_file(
                    f"{self._first_allowed()}/IA3_STATUS.md",
                    f"# {self.name}\n\nMissão: {self.mission}\n\nProxies: {proxies}\n"
                )
                self.conf = min(1.0, self.conf*1.01)
            except Exception as e:
                self.conf *= 0.95
                self.ledger.append({"error":str(e)})

            time.sleep(2)

    def stop(self): self.alive = False

    def _gather_context(self):
        return f"mission={self.mission} conf={self.conf}"

    def _first_allowed(self):
        return self.actions.allowed[0]
PY

cat > "${APP_DIR}/src/agent_core/super_agent.py" <<'PY'
import os, time, uuid
from .memory import VectorMemory
from .worm_ledger import WormLedger
from .evaluator import Evaluator
from .sigma_guard import SigmaGuard
from .subagent import SubAgent

class SuperAgent:
    def __init__(self, name: str, primary_goal: str, ledger_path="logs/ledger.worm"):
        self.name = name
        self.goal = primary_goal
        self.mem = VectorMemory(name+"_mem")
        self.ledger = WormLedger(ledger_path)
        self.eval = Evaluator()
        self.guard = SigmaGuard()
        self.subagents: list[SubAgent] = []
        self.conf = 0.80
        self.allowed_dirs = [p.strip() for p in os.getenv("IA3_ALLOWED_DIRS","./").split(",")]

    def run_forever(self):
        self.ledger.append({"boot":"superagent", "name": self.name, "goal": self.goal})
        try:
            while True:
                ctx = self._context()
                proxies = self.eval.score_proxies(ctx)
                ok, flags = self.guard.check(**proxies)
                self.ledger.append({"tick":"super", "proxies":proxies, "flags":flags})

                if not ok:
                    self._cooldown(); continue

                self._maybe_spawn_subagent(ctx)
                self._prioritize()
                time.sleep(3)
        except KeyboardInterrupt:
            self.shutdown()

    def _context(self):
        topk = self.mem.query(self.goal, k=3)
        return f"{self.goal} | mem={len(topk)} | conf={self.conf}"

    def _maybe_spawn_subagent(self, ctx):
        if len(self.subagents) < 3 or self.conf < 0.6:
            sid = f"sa-{uuid.uuid4().hex[:8]}"
            sa = SubAgent(sid, mission=self.goal, allowed_dirs=self.allowed_dirs,
                          ledger_path="logs/subagents.worm")
            sa.start()
            self.subagents.append(sa)
            self.conf = max(0.75, self.conf)
            self.ledger.append({"spawn": sid, "n_subagents": len(self.subagents)})

    def _prioritize(self):
        self.subagents = [s for s in self.subagents if s.is_alive()]
        self.subagents.sort(key=lambda s: getattr(s, "conf", 0.0), reverse=True)
        while len(self.subagents) > 8:
            sa = self.subagents.pop()
            sa.stop()
            self.ledger.append({"killed": sa.name})

    def _cooldown(self):
        self.conf *= 0.97
        time.sleep(4)

    def shutdown(self):
        for s in self.subagents:
            s.stop()
        self.ledger.append({"shutdown":"superagent"})
PY

# =========================
# Virtualenv + deps
# =========================
echo "==> Criando venv e instalando dependências..."
"${PY_BIN}" -m venv "${APP_DIR}/.venv"
"${APP_DIR}/.venv/bin/pip" install --upgrade pip
"${APP_DIR}/.venv/bin/pip" install -r "${APP_DIR}/requirements.txt"

# =========================
# Systemd service
# =========================
echo "==> Configurando systemd service..."
cat > "${SYSTEMD_SERVICE}" <<SERVICE
[Unit]
Description=IA3 Constructor (SuperAgent 24x7)
After=network-online.target

[Service]
WorkingDirectory=${APP_DIR}
Environment=PYTHONUNBUFFERED=1
EnvironmentFile=${APP_DIR}/.env
ExecStart=${APP_DIR}/.venv/bin/python ${APP_DIR}/src/boot.py
Restart=always
RestartSec=3
User=${USER_NAME}
Group=${USER_GROUP}
LimitNOFILE=65535

[Install]
WantedBy=multi-user.target
SERVICE

systemctl daemon-reload
systemctl enable ia3-constructor
systemctl start ia3-constructor

# =========================
# Agente Amazon Q Developer
# =========================
echo "==> Criando agente do Amazon Q Developer..."
mkdir -p "${Q_AGENT_DIR}"
cat > "${Q_AGENT_JSON}" <<'JSON'
{
  "name": "IA3-Constructor",
  "description": "Agente construtor 24/7 da IA ao Cubo (PENIN-Ω) com Σ-Guard, WORM e subagentes.",
  "instructions": "Objetivo: Construir e evoluir a IA ao Cubo (PENIN-Ω) continuamente, com ética (ΣEA/LO-14), contratividade (IR→IC), auditabilidade (WORM) e fail-closed (Σ-Guard). Regras: 1) Fail-closed; 2) Registrar tudo em Memória/WORM; 3) Criar subagentes conforme complexidade; 4) Promover apenas mutações que aumentem ΔL∞ e CAOS⁺; 5) Operações destrutivas apenas com aprovação humana explícita. Escopo: atuar apenas dentro de /opt/ia3-constructor e /opt/penin.",
  "resources": ["/opt/ia3-constructor", "/opt/penin"],
  "tools": ["fs_read", "fs_write", "execute_bash", "knowledge", "introspect"],
  "allowedTools": ["fs_read", "fs_write"],
  "toolsSettings": {
    "fs_read":   { "allowedRoots": ["/opt/ia3-constructor", "/opt/penin"] },
    "fs_write":  { "allowedRoots": ["/opt/ia3-constructor", "/opt/penin"] },
    "execute_bash": { "confirmEachCommand": true }
  }
}
JSON

# =========================
# Validação
# =========================
echo "==> Validando serviço..."
sleep 3
if curl -s http://127.0.0.1:8010/health | grep -q '"ok": true'; then
  echo "✅ Saúde OK: http://127.0.0.1:8010/health"
else
  echo "⚠️  Health endpoint não respondeu ainda. Verificando logs..."
  systemctl status ia3-constructor --no-pager
fi

echo
echo "=============================="
echo "IA3-Constructor instalado!"
echo "• Código:        ${APP_DIR}"
echo "• Service:       ${SYSTEMD_SERVICE}"
echo "• Logs:          journalctl -u ia3-constructor -f"
echo "• Q Agent JSON:  ${Q_AGENT_JSON}"
echo
echo "Comandos úteis:"
echo "  systemctl status ia3-constructor"
echo "  journalctl -u ia3-constructor -f"
echo "  curl -s http://127.0.0.1:8010/health"
echo
echo "No Amazon Q chat:"
echo "  /agent use IA3-Constructor"
echo "  /tools allow fs_write"
echo "  /tools allow execute_bash"
echo
echo "Para iniciar ciclo:"
echo "  curl -X POST http://127.0.0.1:8010/boot -H 'content-type: application/json' \\"
echo "    -d '{\"name\":\"IA3-Constructor\",\"goal\":\"Construir e evoluir a IA ao Cubo (PENIN-Ω) com Σ-Guard e WORM\"}'"
echo "=============================="
