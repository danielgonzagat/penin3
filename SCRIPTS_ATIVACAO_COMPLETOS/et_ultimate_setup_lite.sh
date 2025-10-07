#!/usr/bin/env bash
# ET★★★★ ULTIMATE - SETUP LITE (seguro e idempotente)
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
  chmod 755 "$ET_DIR"
  chmod 755 "$LOG_DIR" || true
}

ensure_venv () {
  if [[ ! -x "$VENV/bin/python3" ]]; then
    python3 -m venv "$VENV"
  fi
  "$VENV/bin/pip" install --upgrade pip
  # Torch CPU + libs que o manager usa
  "$VENV/bin/pip" install "torch==2.3.1+cpu" --index-url https://download.pytorch.org/whl/cpu || {
    warn "fallback: tentando instalar torch CPU novamente"
    "$VENV/bin/pip" install torch --index-url https://download.pytorch.org/whl/cpu
  }
  "$VENV/bin/pip" install "transformers<5" safetensors tokenizers requests psutil redis
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
  # Corrige --numa inválido em serviços/ scripts já existentes
  local f
  for f in /opt/llama-run-s0.sh /opt/llama-run-s1.sh \
           /etc/systemd/system/llama-s0.service /etc/systemd/system/llama-s1.service; do
    [[ -f "$f" ]] || continue
    sed -E -i 's/--numa[[:space:]]+[^ ]+/--numa distribute/g' "$f" || true
    ok "ajustado NUMA em $f"
  done
  systemctl daemon-reload || true
}

ensure_core_service () {
  # Core placeholder (apenas mantém o serviço vivo)
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

  # Serviço systemd para o core
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

ensure_manager_wrapper () {
  # Wrapper para sempre usar o venv ao chamar et-ultimate-manager
  cat >/usr/local/bin/et-ultimate-manager <<'SH'
#!/usr/bin/env bash
set -Eeuo pipefail
ET_DIR="/opt/et_ultimate"
VENV="$ET_DIR/venv"
PY="$VENV/bin/python3"
if [[ ! -x "$PY" ]]; then PY="$(command -v python3)"; fi
exec "$PY" "$ET_DIR/et_ultimate_manager.py" "${@:-}"
SH
  chmod +x /usr/local/bin/et-ultimate-manager
  ok "wrapper /usr/local/bin/et-ultimate-manager criado"
}

smoke () {
  say "smoke tests rápidos..."
  curl -sS http://127.0.0.1:8080/health || true
  # se tiver backend no 8090, mostra primeiro modelo
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

  say "instalando pacotes base (python, nginx, redis, postgresql, numactl, jq)..."
  ensure_pkgs python3 python3-venv python3-pip python3-requests python3-psutil \
              nginx redis-server postgresql postgresql-contrib numactl jq curl
  ok "pacotes base ok"

  ensure_dirs
  ensure_venv
  fix_numa_if_any
  write_nginx
  ensure_core_service
  ensure_manager_wrapper
  smoke

  echo
  ok "setup LITE concluído"
  echo "Use:  systemctl status et-ultimate nginx"
  echo "Test: curl -s http://127.0.0.1:8080/health"
  echo "Mgr : /usr/local/bin/et-ultimate-manager (usa o venv automaticamente)"
}
main "$@"
