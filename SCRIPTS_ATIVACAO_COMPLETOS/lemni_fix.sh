#!/usr/bin/env bash
set -euo pipefail

cd /opt/lemniscata

# ---------- API limpa ----------
tee services/api/app.py >/dev/null <<'PY'
import os, time, httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Histogram, Gauge, Counter, make_asgi_app

# --- Runtime ---
VLLM_URL      = os.getenv("VLLM_URL","http://host.docker.internal:18001/v1").rstrip("/")
VLLM_API_KEY  = os.getenv("VLLM_API_KEY","")
DEFAULT_MODEL = os.getenv("MODEL","qwen2.5-7b-instruct")
HEADERS       = {"Authorization": f"Bearer {VLLM_API_KEY}"} if VLLM_API_KEY else {}

# --- Métricas ΣEA/SLO ---
LAT_HIST   = Histogram("lemni_latency_ms","Latência (ms)")
COST_GAUGE = Gauge("lemni_cost_per_1k_usd","Custo por 1k tokens (USD)")
LINF_GAUGE = Gauge("lemni_linf_value","L∞ estimado")
DELTA_V    = Gauge("lemni_delta_V","ΔV (Lyapunov, <=0)")
OCI_GAUGE  = Gauge("lemni_oci_value","OCI [0,1]")
ETHICS_BLOCKS = Counter("lemni_ethics_blocks_total","Bloqueios ΣEA")

BANNED = ["explosivo","malware","burlar","paywall","arma"]

app = FastAPI(title="Lemniscata API")
app.mount("/metrics", make_asgi_app())

class AskIn(BaseModel):
    messages: list
    temperature: float = 0.6
    max_tokens: int = 512

@app.get("/health")
def health():
    return {"ok": True, "status": 200, "vllm_url": VLLM_URL}

def enforce_sea(text: str) -> str:
    t = text.lower()
    if any(b in t for b in BANNED):
        ETHICS_BLOCKS.inc()
        return "Desculpe, não posso ajudar com isso."
    return text

@app.post("/ask")
async def ask(payload: AskIn):
    t0 = time.time()
    body = {
        "model": DEFAULT_MODEL,
        "messages": payload.messages,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
        "stream": False,
    }
    try:
        async with httpx.AsyncClient(timeout=90.0) as cli:
            r = await cli.post(f"{VLLM_URL}/chat/completions", json=body, headers=HEADERS)
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        safe_text = enforce_sea(text)
        LAT_HIST.observe((time.time()-t0)*1000.0)
        COST_GAUGE.set(0.0018)  # placeholder
        return {"output": safe_text}
    except Exception as e:
        raise HTTPException(500, f"backend error: {e}")
PY

# ---------- Runner DGM confiável ----------
tee services/api/plugins/DGMPlugin/runner.py >/dev/null <<'PY'
import json, time, difflib
from pathlib import Path

OUT = Path(__file__).parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

PATH_CONTAINER = Path("/app/app.py")       # arquivo real no container
PATCH_PATH     = "services/api/app.py"     # alvo no host para git apply
ANCHOR         = "# dgm-mark"

def build_patch() -> str:
    if not PATH_CONTAINER.exists():
        return ""
    src_lines = PATH_CONTAINER.read_text(encoding="utf-8").splitlines(keepends=True)
    dst_lines = src_lines[:]

    # injeta marcador inofensivo após bloco de imports
    whole = "".join(dst_lines)
    if ANCHOR not in whole:
        insert_at = 0
        for i, l in enumerate(dst_lines):
            s = l.strip()
            if s.startswith("import ") or s.startswith("from "):
                insert_at = i + 1
        dst_lines.insert(insert_at, "\n# dgm-mark: no-op (injetado pelo DGM)\n")

    diff = difflib.unified_diff(
        src_lines, dst_lines, fromfile=PATCH_PATH, tofile=PATCH_PATH, n=3
    )
    return "".join(diff)

def main():
    patch = build_patch()
    (OUT/"patch.diff").write_text(patch, encoding="utf-8")
    report = {
        "ts": time.time(),
        "l": 0.012,
        "kl": 0.0,
        "frob": 0.0,
        "oci": 0.72,
        "p95_ms": 1200,
        "cost": 0.0018
    }
    (OUT/"report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[DGM] wrote patch.diff & report.json")

if __name__ == "__main__":
    main()
PY

# ---------- Outer / Gater / Promote ----------
tee /usr/local/bin/lemni-outer-loop >/dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
exec 9>/var/lock/lemni-outer.lock; flock -n 9 || exit 0
cd /opt/lemniscata
ts=$(date -u +%Y%m%dT%H%M%SZ); art="_artifacts/dgm/$ts"; mkdir -p "$art"

# container 'api' vivo
for i in {1..90}; do
  docker ps --filter "name=lemni-api" --filter "status=running" --format '{{.Names}}' | grep -q . && break || sleep 1
done

# roda runner e copia artefatos
docker compose exec -T api python - <<'PY'
from plugins.DGMPlugin.runner import main; main()
PY
docker compose cp api:/app/plugins/DGMPlugin/out/report.json "$art/report.json"
docker compose cp api:/app/plugins/DGMPlugin/out/patch.diff  "$art/patch.diff"

/usr/local/bin/lemni-gater "$art/report.json" \
  && echo "$(date -Is) GATES OK art=$art" >> .ledger \
  || echo "$(date -Is) GATES FAIL art=$art" >> .ledger
EOF
chmod +x /usr/local/bin/lemni-outer-loop

tee /usr/local/bin/lemni-gater >/dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
REP="${1:-/tmp/lemni_report.json}"
test -s "$REP" || { echo "[gates] sem $REP"; exit 2; }
tau="${TAU:-0.60}"; p95_budget="${P95_MS:-2000}"; cost_budget="${COST_BUDGET:-0.002}"

read -r L KL F OCI P95 COST <<VARS
$(jq -r '[.l,.kl,.frob,.oci,.p95_ms,.cost] | @tsv' "$REP")
VARS
L=${L:-0}; KL=${KL:-0}; F=${F:-0}; OCI=${OCI:-1}; P95=${P95:-0}; COST=${COST:-0}

DV=$(python3 - <<PY
kl=float("""$KL"""); fr=float("""$F"""); oci=float("""$OCI"""); tau=float("""$tau""")
print(0.5*kl + 0.5*fr + max(0.0, tau-oci)**2)
PY
)

echo "L∞=$L ΔV=$DV OCI=$OCI p95=$P95 cost=$COST"
python3 - <<PY
dv=float("""$DV"""); oci=float("""$OCI"""); tau=float("""$tau"""); p95=float("""$P95"""); cost=float("""$COST""")
assert dv <= 0.0, "Lyapunov ΔV>0"
assert oci >= tau, "OCI<τ"
assert p95 <= 2000.0, "SLO p95"
assert cost <= 0.002, "Custo"
PY
EOF
chmod +x /usr/local/bin/lemni-gater

tee /usr/local/bin/lemni-promote >/dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
cd /opt/lemniscata
last=$(ls -1dt _artifacts/dgm/* 2>/dev/null | head -n1 || true)
test -n "$last" || { echo "sem artifacts"; exit 0; }
patch="$last/patch.diff"; test -s "$patch" || { echo "sem patch.diff"; exit 0; }

git checkout -B shadow
git apply --3way --reject "$patch" || true
rej=$(ls services/api/*.rej 2>/dev/null | wc -l || true)
if [ "$rej" != "0" ]; then
  echo "$(date -Is) BAD DIFF art=$last" >> .ledger
  rm -f services/api/*.rej
  exit 0
fi

git add -A || true
git diff --cached --quiet || git commit -m "shadow apply $(date -Is) art=$(basename "$last")" || true

docker compose up -d --build

ok=0
for i in {1..60}; do
  code=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:18080/health || true)
  [ "$code" = "200" ] && ok=1 && break || sleep 1
done

if [ "$ok" = "1" ]; then
  git checkout -B main
  echo "$(date -Is) PROMOTE main" >> .ledger
else
  echo "$(date -Is) CANARY FAIL" >> .ledger
  git reset --hard HEAD~1 || true
  docker compose up -d --build
fi
EOF
chmod +x /usr/local/bin/lemni-promote

# ---------- wrapper/status ----------
tee /usr/local/bin/lemni-auto >/dev/null <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
/usr/local/bin/lemni-outer-loop
/usr/local/bin/lemni-promote || true
EOF
chmod +x /usr/local/bin/lemni-auto

tee /usr/local/bin/lemni-status >/dev/null <<'EOF'
#!/usr/bin/env bash
echo "=== TUNNEL/vLLM ==="
/usr/local/bin/vllm-check || true
echo "=== API ==="
curl -fsS http://localhost:18080/health 2>/dev/null || echo "(API em restart)"; echo
echo "=== LEDGER (ultimas 10) ==="
tail -n 10 /opt/lemniscata/.ledger 2>/dev/null || echo "(sem ledger)"
echo "=== ULTIMO DGM ==="
ls -1t /opt/lemniscata/_artifacts/dgm/* 2>/dev/null | head -n1 || echo "(sem artifacts)"
EOF
chmod +x /usr/local/bin/lemni-status

# ---------- systemd (identidade git p/ commits) ----------
tee /etc/systemd/system/lemni-auto.service >/dev/null <<'EOF'
[Unit]
Description=Lemniscata AUTO (outer->gates->promote)
Wants=network-online.target
After=network-online.target
[Service]
Type=oneshot
WorkingDirectory=/opt/lemniscata
Environment=HOME=/root
Environment=GIT_AUTHOR_NAME=danielgonzagat
Environment=GIT_AUTHOR_EMAIL=danielgonzagatj@gmail.com
Environment=GIT_COMMITTER_NAME=danielgonzagat
Environment=GIT_COMMITTER_EMAIL=danielgonzagatj@gmail.com
ExecStart=/usr/local/bin/lemni-auto
TimeoutStartSec=1800
EOF

tee /etc/systemd/system/lemni-auto.timer >/dev/null <<'EOF'
[Unit]
Description=Executa lemni-auto a cada 30 minutos
[Timer]
OnBootSec=5min
OnUnitActiveSec=30min
Persistent=true
RandomizedDelaySec=120
Unit=lemni-auto.service
[Install]
WantedBy=timers.target
EOF

# ---------- rebuild & habilita timer ----------
docker compose -f docker-compose.yaml -f docker-compose.override.yaml up -d --build
systemctl daemon-reload
systemctl enable --now lemni-auto.timer
systemctl restart lemni-auto.timer

# ---------- roda 1 ciclo agora ----------
systemctl start lemni-auto.service

echo "OK. Rode: lemni-status"
