#!/usr/bin/env bash
set -euo pipefail

TD="/root/darwin_test"
mkdir -p "$TD"/{heritage,agents}
: > "$TD/.gitkeep" || true

# ─────────────────────────────────────────────────────────────────────
# Configs de teste (JSON para evitar depender de PyYAML)
# ─────────────────────────────────────────────────────────────────────

# Config "geral" (usa portas e caminhos de teste)
cat > "$TD/darwin_policy.test.json" <<'JSON'
{
  "paths": {
    "worm_log": "/root/darwin_test/worm_test.log",
    "manifest": "/root/darwin_test/manifest_test.json",
    "heritage_dir": "/root/darwin_test/heritage",
    "agents_dir": "/root/darwin_test/agents",
    "teis_worm_log": "/root/darwin_test/teis_worm_fake.log"
  },
  "thresholds": {
    "delta_linf_min": 0.0,
    "caos_ratio_min": 1.0,
    "I_min": 0.60,
    "P_min": 0.01,
    "novelty_min": 0.02
  },
  "births": { "deaths_per_birth": 3 },
  "metrics": { "port": 9192 },
  "operational": { "interval_seconds": 1, "dry_run": false, "fail_closed": true, "max_agents": 16, "seed": 42 }
}
JSON

# Força promoção (limiares fáceis) — para testar caminho "vive"
cat > "$TD/darwin_policy.promote.json" <<'JSON'
{
  "paths": {
    "worm_log": "/root/darwin_test/worm_promote.log",
    "manifest": "/root/darwin_test/manifest_promote.json",
    "heritage_dir": "/root/darwin_test/heritage",
    "agents_dir": "/root/darwin_test/agents",
    "teis_worm_log": "/root/darwin_test/teis_worm_fake.log"
  },
  "thresholds": {
    "delta_linf_min": 0.0,
    "caos_ratio_min": 0.90,
    "I_min": 0.0,
    "P_min": 0.0,
    "novelty_min": 0.0
  },
  "births": { "deaths_per_birth": 2 },
  "metrics": { "port": 9193 },
  "operational": { "interval_seconds": 1, "dry_run": false, "fail_closed": true, "max_agents": 16, "seed": 7 }
}
JSON

# Força morte (limiares duros) — para testar janela de nascimentos
cat > "$TD/darwin_policy.kill.json" <<'JSON'
{
  "paths": {
    "worm_log": "/root/darwin_test/worm_kill.log",
    "manifest": "/root/darwin_test/manifest_kill.json",
    "heritage_dir": "/root/darwin_test/heritage",
    "agents_dir": "/root/darwin_test/agents",
    "teis_worm_log": "/root/darwin_test/teis_worm_fake.log"
  },
  "thresholds": {
    "delta_linf_min": 0.99,
    "caos_ratio_min": 1.20,
    "I_min": 0.99,
    "P_min": 0.50,
    "novelty_min": 0.99
  },
  "births": { "deaths_per_birth": 2 },
  "metrics": { "port": 9194 },
  "operational": { "interval_seconds": 1, "dry_run": false, "fail_closed": true, "max_agents": 16, "seed": 123 }
}
JSON

# TEIS fake (linhas simples de JSON; o parser aceita EVENT:... ou JSON puro)
cat > "$TD/teis_worm_fake.log" <<'EOF'
{"delta_linf": 0.123, "caos_ratio": 1.02, "I": 0.65, "P": 0.04, "novelty": 0.03, "oci": 0.67, "ece": 0.02, "rho": 0.85}
{"delta_linf": 0.051, "caos_ratio": 1.04, "I": 0.62, "P": 0.03, "novelty": 0.025, "oci": 0.66, "ece": 0.03, "rho": 0.88}
EOF

# ─────────────────────────────────────────────────────────────────────
# Testes unitários (python -m unittest)
# ─────────────────────────────────────────────────────────────────────
cat > "$TD/test_darwin.py" <<'PY'
import os, sys, json, time, signal, hashlib, subprocess, unittest, urllib.request

TD = "/root/darwin_test"
RUNNER = "/root/darwin_runner.py"

CFG_TEST     = f"{TD}/darwin_policy.test.json"
CFG_PROMOTE  = f"{TD}/darwin_policy.promote.json"
CFG_KILL     = f"{TD}/darwin_policy.kill.json"

WORM_TEST    = f"{TD}/worm_test.log"
WORM_PROMOTE = f"{TD}/worm_promote.log"
WORM_KILL    = f"{TD}/worm_kill.log"

MAN_TEST     = f"{TD}/manifest_test.json"
MAN_PROMOTE  = f"{TD}/manifest_promote.json"
MAN_KILL     = f"{TD}/manifest_kill.json"

def rm_safe(p):
    try: os.remove(p)
    except FileNotFoundError: pass

def run_once(cfg):
    # Executa um ciclo e retorna saída/rc
    proc = subprocess.run(["python3", RUNNER, "--config", cfg, "--once"], capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"runner --once falhou ({proc.returncode}):\n{proc.stderr}")
    time.sleep(0.2)  # tempo para flush no WORM
    return proc

def read_lines(p):
    if not os.path.exists(p): return []
    with open(p, "r", encoding="utf-8") as f:
        return f.read().splitlines()

def last_events(p, n=50):
    lines = read_lines(p)
    return [l for l in lines[-n:] if l.startswith("EVENT:")]

def start_bg(cfg):
    # roda o runner em background (para testar /metrics)
    p = subprocess.Popen(["python3", RUNNER, "--config", cfg], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    time.sleep(1.5)
    return p

def stop_bg(p):
    try:
        p.terminate()
        try:
            p.wait(timeout=3)
        except subprocess.TimeoutExpired:
            p.kill()
    except Exception:
        pass

class DarwinTests(unittest.TestCase):

    def setUp(self):
        for p in [WORM_TEST, WORM_PROMOTE, WORM_KILL, MAN_TEST, MAN_PROMOTE, MAN_KILL]:
            rm_safe(p)

    def test_001_decision_once_and_manifest(self):
        run_once(CFG_TEST)
        self.assertTrue(os.path.exists(MAN_TEST), "Manifest de teste não foi criado")
        evs = "\n".join(last_events(WORM_TEST, 10))
        self.assertIn("darwin_decision", evs, "Não registrou decisão no WORM")

    def test_002_worm_chain_integrity(self):
        # 2 decisões para termos >1 par EVENT/HASH
        run_once(CFG_TEST)
        run_once(CFG_TEST)
        lines = read_lines(WORM_TEST)
        # valida cadeia por previous_hash + payload -> HASH
        genesis = hashlib.sha256(b"DARWIN-GENESIS").hexdigest()
        prev_hash = None
        i = 0
        while i < len(lines)-1:
            if not lines[i].startswith("EVENT:"):
                i += 1; continue
            payload = lines[i][6:]
            self.assertTrue(lines[i+1].startswith("HASH:"), "Linha HASH ausente após EVENT")
            curr_hash = lines[i+1].split("HASH:",1)[1].strip()
            data = json.loads(payload)
            ph = data.get("previous_hash")
            if prev_hash is None:
                # primeira entrada deve apontar para genesis
                self.assertEqual(ph, genesis, "previous_hash inicial não bate com genesis")
            else:
                self.assertEqual(ph, prev_hash, "previous_hash não encadeia com hash anterior")
            expected = hashlib.sha256((ph + payload).encode("utf-8")).hexdigest()
            self.assertEqual(curr_hash, expected, "HASH atual não confere com sha256(prev+payload)")
            prev_hash = curr_hash
            i += 2

    def test_003_promote_path(self):
        run_once(CFG_PROMOTE)
        evs = "\n".join(last_events(WORM_PROMOTE, 20))
        self.assertIn("darwin_promote_hook", evs, "Não registrou promoção mesmo com limiares fáceis")

    def test_004_kill_then_birth_after_window(self):
        # deaths_per_birth=2 -> após 2 mortes, deve nascer 1
        run_once(CFG_KILL)  # mata o bootstrap
        run_once(CFG_KILL)  # mata o re-bootstrap => deve disparar spawn
        evs = "\n".join(last_events(WORM_KILL, 50))
        self.assertIn("darwin_kill_hook", evs, "Não registrou kill")
        self.assertIn("darwin_spawn_hook", evs, "Não houve nascimento após janela de mortes")

    def test_005_metrics_exporter(self):
        p = start_bg(CFG_PROMOTE)
        try:
            # lê /metrics
            with urllib.request.urlopen("http://127.0.0.1:9193/metrics", timeout=3) as r:
                body = r.read().decode("utf-8")
            self.assertIn("darwin_decisions_total", body, "Métrica darwin_decisions_total ausente")
        finally:
            stop_bg(p)

    def test_006_teis_integration_source_tag(self):
        # Garante que quando o arquivo TEIS existe, a decisão marca source=teis_worm
        run_once(CFG_TEST)
        # pega último EVENT: darwin_decision
        events = [json.loads(l.split("EVENT:",1)[1]) for l in last_events(WORM_TEST, 50)]
        decisions = [e for e in events if e.get("event")=="darwin_decision"]
        self.assertTrue(decisions, "Sem darwin_decision no WORM de teste")
        src = decisions[-1].get("metrics",{}).get("source")
        self.assertEqual(src, "teis_worm", "Decisão não marcou source=teis_worm (integração TEIS)")
PY

# Script de execução dos testes
cat > "$TD/run_tests.sh" <<'BASH2'
#!/usr/bin/env bash
set -euo pipefail
python3 -m unittest -v /root/darwin_test/test_darwin.py
BASH2
chmod +x "$TD/run_tests.sh"

echo "✅ Suíte de testes do DARWIN instalada em: $TD"
echo "▶ Para rodar:  python3 -m unittest -v $TD/test_darwin.py"
echo "   (ou)       $TD/run_tests.sh"