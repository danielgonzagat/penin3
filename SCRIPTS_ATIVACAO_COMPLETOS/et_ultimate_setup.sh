#!/usr/bin/env bash
# ET★★★★ ULTIMATE - SETUP COMPLETO (seguro, idempotente)
set -Eeuo pipefail
trap 'echo -e "\e[31m[✗] erro na linha $LINENO\e[0m"; exit 1' ERR

ET_DIR="/opt/et_ultimate"
LOG_DIR="/var/log/et_ultimate"
VENV="$ET_DIR/venv"

say() { printf "\033[1;34m==>\033[0m %s\n" "$*"; }
ok()  { printf "\033[1;32m[✓]\033[0m %s\n" "$*"; }
warn(){ printf "\033[1;33m[!]\033[0m %s\n" "$*"; }

wait_apt () {
  local tries=0
  while :; do
    if pgrep -x apt >/dev/null || pgrep -x apt-get >/dev/null || pgrep -x dpkg >/dev/null \
       || fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 \
       || fuser /var/cache/apt/archives/lock >/dev/null 2>&1; then
      ((tries++))
      if (( tries % 5 == 1 )); then warn "aguardando apt/dpkg liberar locks..."; fi
      sleep 2
      continue
    fi
    break
  done
  rm -f /var/lib/dpkg/lock-frontend /var/cache/apt/archives/lock 2>/dev/null || true
}

ensure_pkgs () {
  local need=()
  for p in "$@"; do
    if ! dpkg -s "$p" >/dev/null 2>&1; then need+=("$p"); fi
  done
  if (( ${#need[@]} )); then
    wait_apt
    DEBIAN_FRONTEND=noninteractive apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends "${need[@]}"
  fi
}

ensure_dirs () {
  mkdir -p "$ET_DIR" "$LOG_DIR"
  chmod 755 "$ET_DIR" "$LOG_DIR" || true
}

ensure_venv () {
  if [[ ! -x "$VENV/bin/python3" ]]; then
    python3 -m venv "$VENV"
  fi
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install "torch==2.3.1+cpu" --index-url https://download.pytorch.org/whl/cpu || {
    warn "fallback: tentando torch CPU genérica"
    "$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
  }
  "$VENV/bin/pip" install "transformers<5" safetensors tokenizers requests psutil
}

write_nginx () {
  say "configurando nginx (proxy 8080 -> 8090/8091)..."
  cat >/etc/nginx/sites-available/et_ultimate <<'NGX'
upstream llama_backend {
    least_conn;
    server 127.0.0.1:8090 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8091 max_fails=3 fail_timeout=30s;
    keepalive 32;
}
server {
    listen 8080;
    server_name _;
    proxy_connect_timeout 60s;
    proxy_send_timeout 60s;
    proxy_read_timeout 300s;

    proxy_buffer_size 128k;
    proxy_buffers 4 256k;
    proxy_busy_buffers_size 256k;

    location / {
        proxy_pass http://llama_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header Connection "";
        proxy_http_version 1.1;

        add_header Access-Control-Allow-Origin *;
        add_header Access-Control-Allow-Methods "GET, POST, OPTIONS";
        add_header Access-Control-Allow-Headers "Authorization, Content-Type";
        if ($request_method = 'OPTIONS') { return 204; }
    }
    location /health {
        access_log off;
        return 200 "ET Ultimate OK\n";
        add_header Content-Type text/plain;
    }
}
NGX
  ln -sf /etc/nginx/sites-available/et_ultimate /etc/nginx/sites-enabled/et_ultimate
  rm -f /etc/nginx/sites-enabled/default
  nginx -t
  systemctl enable --now nginx
  systemctl restart nginx
  ok "nginx ok na porta 8080"
}

fix_numa_if_any () {
  local f
  for f in /opt/llama-run-s0.sh /opt/llama-run-s1.sh \
           /etc/systemd/system/llama-s0.service /etc/systemd/system/llama-s1.service; do
    [[ -f "$f" ]] || continue
    sed -E -i 's/--numa[[:space:]]+[^ ]+/--numa distribute/g' "$f" || true
    ok "ajustado NUMA em $f"
  done
  systemctl daemon-reload || true
}

write_core_service () {
  # Core placeholder estável
  if [[ ! -f "$ET_DIR/et_ultimate_core.py" ]]; then
    cat >"$ET_DIR/et_ultimate_core.py" <<'PY'
#!/usr/bin/env python3
import time, logging, signal, sys
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
log = logging.getLogger("ET_CORE")
log.info("ET Ultimate Core placeholder iniciado.")
def _stop(*_): log.info("Encerrando..."); sys.exit(0)
signal.signal(signal.SIGTERM, _stop); signal.signal(signal.SIGINT, _stop)
while True:
    time.sleep(60)
PY
    chmod +x "$ET_DIR/et_ultimate_core.py"
  fi

  # Serviço systemd do core
  cat >/etc/systemd/system/et-ultimate.service <<UNIT
[Unit]
Description=ET Ultimate Core (placeholder)
After=network.target
[Service]
Type=simple
WorkingDirectory=$ET_DIR
ExecStart=$VENV/bin/python3 $ET_DIR/et_ultimate_core.py
Restart=always
RestartSec=5
[Install]
WantedBy=multi-user.target
UNIT
  systemctl daemon-reload
  systemctl enable --now et-ultimate
  ok "serviço et-ultimate ativo"
}

write_manager () {
  cat >"$ET_DIR/et_ultimate_manager.py" <<'PY'
#!/usr/bin/env python3
import argparse, os, sys, subprocess, time, json, shutil
from typing import Optional
ET_DIR = "/opt/et_ultimate"
VENV = os.path.join(ET_DIR, "venv")
ENV = os.environ.copy()

def sh(cmd:list[str], check=False) -> tuple[int,str]:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=ENV)
    if check and p.returncode != 0:
        print(p.stdout, file=sys.stderr)
        raise SystemExit(p.returncode)
    return p.returncode, p.stdout

def svc_exists(name:str)->bool:
    return sh(["systemctl","cat",name])[0]==0

def svc(action:str, name:str):
    if svc_exists(name):
        rc,out = sh(["systemctl",action,name])
        print(f"$ systemctl {action} {name}\n{out.strip()}")
    else:
        print(f"(serviço {name} não existe, ignorando)")

def ensure_nginx_site():
    path="/etc/nginx/sites-available/et_ultimate"
    if not os.path.exists(path):
        print("site nginx ausente — rode o setup novamente"); return
    os.symlink(path, "/etc/nginx/sites-enabled/et_ultimate") if not os.path.exists("/etc/nginx/sites-enabled/et_ultimate") else None
    sh(["nginx","-t"], check=True)
    sh(["systemctl","restart","nginx"])

def get(url:str, timeout=8, headers:Optional[dict]=None)->tuple[int,str]:
    try:
        import requests
        r = requests.get(url, timeout=timeout, headers=headers or {})
        return r.status_code, r.text
    except Exception as e:
        return 0, f"ERROR: {e}"

def post_json(url:str, payload:dict, timeout=30, headers:Optional[dict]=None)->tuple[int,str]:
    try:
        import requests
        r = requests.post(url, json=payload, timeout=timeout, headers=headers or {"Content-Type":"application/json"})
        return r.status_code, r.text
    except Exception as e:
        return 0, f"ERROR: {e}"

def cmd_start(_):
    for base in ("postgresql","redis-server","nginx"):
        svc("start", base)
    for llama in ("llama-s0","llama-s1"):
        svc("start", llama)
    svc("start","et-ultimate")

def cmd_stop(_):
    for name in ("et-ultimate","llama-s1","llama-s0"):
        svc("stop", name)

def cmd_restart(_):
    cmd_stop(_)
    time.sleep(1)
    cmd_start(_)

def cmd_status(_):
    for name in ("et-ultimate","nginx","redis-server","postgresql","llama-s0","llama-s1"):
        rc,out = sh(["systemctl","status","--no-pager","-n","0",name])
        head = "\n".join(out.splitlines()[:6])
        print(f"\n--- {name} ---\n{head}")

def fix_numa():
    changed=False
    for f in ("/opt/llama-run-s0.sh","/opt/llama-run-s1.sh",
              "/etc/systemd/system/llama-s0.service","/etc/systemd/system/llama-s1.service"):
        if os.path.exists(f):
            with open(f,"r",encoding="utf-8",errors="ignore") as fh: c=fh.read()
            import re
            new = re.sub(r"--numa\s+[^ ]+","--numa distribute",c)
            if new!=c:
                with open(f,"w") as fh: fh.write(new)
                print(f"ajustado NUMA em {f}")
                changed=True
    if changed: sh(["systemctl","daemon-reload"])

def cmd_fix(_):
    fix_numa()
    ensure_nginx_site()
    print("fix: ok")

def cmd_test(args):
    print("# Saúde via proxy (8080):")
    sc,body = get("http://127.0.0.1:8080/health")
    print(sc, body.strip())

    model_id=None
    for port in (8080,8090,8091):
        sc,body = get(f"http://127.0.0.1:{port}/v1/models")
        if sc==200 and '"data"' in body:
            try:
                j=json.loads(body)
                if j.get("data"):
                    model_id=j["data"][0]["id"]
                    print(f"model[{port}]: {model_id}")
                    break
            except Exception: pass

    if not model_id:
        print("não consegui obter model id (tente iniciar seus backends :8090/:8091)"); return

    payload={"model":model_id,"messages":[{"role":"user","content":"Diga oi"}],"max_tokens":32}
    sc,body = post_json("http://127.0.0.1:8080/v1/chat/completions", payload)
    print("chat:", sc)
    try:
        j=json.loads(body)
        if "choices" in j: print(j["choices"][0]["message"]["content"])
        else: print(body[:500])
    except Exception:
        print(body[:500])

def cmd_monitor(args):
    dur = int(getattr(args,"duration",60))
    try:
        import psutil
    except Exception:
        print("psutil não disponível no venv"); return
    t0=time.time()
    while time.time()-t0 < dur:
        cpu = psutil.cpu_percent(interval=1)
        mem = psutil.virtual_memory().percent
        procs = ", ".join(p.name() for p in psutil.process_iter(attrs=["name"]) if p.info["name"] in ("nginx","redis-server","postgres","llama","llama-server"))
        print(f"CPU:{cpu:5.1f}%  MEM:{mem:5.1f}%  PROC:[{procs}]")
    print("monitor: fim")

def cmd_dashboard(_):
    print("=== DASHBOARD (rápido) ===")
    cmd_status(_)
    cmd_test(_)

def cmd_powers(args):
    if not getattr(args,"i_am_sure",False):
        print("⚠️ Esta ação aplica tunings de kernel/limites. Rode com --i-am-sure para prosseguir.")
        return
    # Tunings seguros (sem mexer em GRUB/mitigations/setcap)
    sysctl="/etc/sysctl.d/98-et-ultimate.conf"
    with open(sysctl,"w") as f:
        f.write("\n".join([
            "vm.swappiness=10",
            "net.core.rmem_max=134217728",
            "net.core.wmem_max=134217728",
            "net.core.netdev_max_backlog=5000",
            "net.ipv4.tcp_congestion_control=bbr",
        ]))
    sh(["sysctl","-p",sysctl])
    print("poderes (seguros) aplicados.")

def cmd_protect(args):
    user = getattr(args,"user","daniel")
    nopw = getattr(args,"nopasswd",False)
    pubkey = getattr(args,"pubkey",None)
    # criar usuário se não existir (sem definir senha por padrão!)
    if subprocess.run(["id",user]).returncode!=0:
        subprocess.run(["useradd","-m","-s","/bin/bash",user], check=True)
        print(f"usuário {user} criado")
    os.makedirs(f"/home/{user}/.ssh", exist_ok=True)
    if pubkey:
        with open(f"/home/{user}/.ssh/authorized_keys","a") as f: f.write(pubkey.strip()+"\n")
        subprocess.run(["chown","-R",f"{user}:{user}",f"/home/{user}/.ssh"])
        subprocess.run(["chmod","700",f"/home/{user}/.ssh"])
        subprocess.run(["chmod","600",f"/home/{user}/.ssh/authorized_keys"])
        print("chave SSH adicionada")
    if nopw:
        path=f"/etc/sudoers.d/{user}_ultimate"
        with open(path,"w") as f: f.write(f"{user} ALL=(ALL) NOPASSWD:ALL\n")
        subprocess.run(["chmod","440",path])
        print("sudo NOPASSWD concedido")
    print("protect: ok")

def main():
    ap = argparse.ArgumentParser(prog="et-manager", description="ET Ultimate Manager")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("start").set_defaults(func=cmd_start)
    sub.add_parser("stop").set_defaults(func=cmd_stop)
    sub.add_parser("restart").set_defaults(func=cmd_restart)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("fix").set_defaults(func=cmd_fix)
    sub.add_parser("test").set_defaults(func=cmd_test)
    mon=sub.add_parser("monitor"); mon.add_argument("--duration", type=int, default=60); mon.set_defaults(func=cmd_monitor)
    sub.add_parser("dashboard").set_defaults(func=cmd_dashboard)
    pw=sub.add_parser("powers"); pw.add_argument("--i-am-sure", action="store_true"); pw.set_defaults(func=cmd_powers)
    pr=sub.add_parser("protect"); pr.add_argument("--user", default="daniel"); pr.add_argument("--nopasswd", action="store_true"); pr.add_argument("--pubkey"); pr.set_defaults(func=cmd_protect)

    args=ap.parse_args()
    args.func(args)

if __name__=="__main__":
    main()
PY
  chmod +x "$ET_DIR/et_ultimate_manager.py"

  # wrapper para PATH global
  cat >/usr/local/bin/et-manager <<SH
#!/usr/bin/env bash
set -Eeuo pipefail
exec "$VENV/bin/python3" "$ET_DIR/et_ultimate_manager.py" "\$@"
SH
  chmod +x /usr/local/bin/et-manager
}

smoke () {
  say "smoke tests..."
  curl -sS http://127.0.0.1:8080/health || true
  if curl -fsS http://127.0.0.1:8090/v1/models >/dev/null 2>&1; then
    echo -n "modelo[8090]: "
    curl -sS http://127.0.0.1:8090/v1/models | python3 - <<'PY' || true
import sys, json
try:
  j=json.load(sys.stdin); print(j["data"][0]["id"])
except: print("n/d")
PY
  fi
}

main () {
  [[ $EUID -eq 0 ]] || { echo "execute como root"; exit 1; }

  say "instalando pacotes base (python, nginx, redis, postgresql, numactl, jq, curl)..."
  ensure_pkgs python3 python3-venv python3-pip python3-requests python3-psutil \
              nginx redis-server postgresql postgresql-contrib numactl jq curl
  ok "pacotes base ok"

  ensure_dirs
  ensure_venv
  fix_numa_if_any
  write_nginx
  write_core_service
  write_manager
  smoke

  echo
  ok "setup concluído"
  echo "Use:  et-manager start|status|test|fix|monitor|dashboard|powers|protect"
  echo "Test: curl -s http://127.0.0.1:8080/health"
}
main "$@"
