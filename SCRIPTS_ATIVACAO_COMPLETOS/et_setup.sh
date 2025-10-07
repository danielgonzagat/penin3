#!/usr/bin/env bash
set -Eeuo pipefail

# =========================
# ET★★★★ ULTIMATE - SETUP SAFE
# =========================

# Cores
G='\033[0;32m'; Y='\033[1;33m'; R='\033[0;31m'; C='\033[0;36m'; N='\033[0m'
log(){ echo -e "${G}[$(date '+%F %T')] $*${N}"; }
warn(){ echo -e "${Y}[WARN] $*${N}"; }
err(){ echo -e "${R}[ERROR] $*${N}"; }

[[ $EUID -eq 0 ]] || { err "Execute como root"; exit 1; }

ET_DIR=/opt/et_ultimate
LOG_DIR=/var/log/et_ultimate

log "🔧 Preparando pastas"
mkdir -p "$ET_DIR"/{workspace,generated_ais,experiments,models,data,backups}
mkdir -p "$LOG_DIR"

log "📚 Instalando pacotes base"
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y --no-install-recommends \
  python3-venv python3-pip git curl jq numactl \
  nginx redis-server postgresql postgresql-contrib nmap

log "🐍 Criando venv e instalando libs"
python3 -m venv "$ET_DIR/venv"
source "$ET_DIR/venv/bin/activate"
pip install -q --upgrade pip
pip install -q "torch==2.3.1+cpu" --index-url https://download.pytorch.org/whl/cpu
pip install -q "transformers<5" safetensors tokenizers numpy requests psutil redis schedule aiohttp
deactivate

log "🧠 Gravando core minimal (log + heartbeat)"
cat > "$ET_DIR/et_ultimate_core.py" <<'PY'
#!/usr/bin/env python3
import logging, time, signal, sys, os
os.makedirs('/var/log/et_ultimate', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler('/var/log/et_ultimate/ultimate.log'),
              logging.StreamHandler()]
)
log = logging.getLogger("ET-ULTIMATE")
log.info("🚀 ET★★★★ Ultimate Core iniciado")
def handle(sig, frame):
    log.info("🛑 Encerrando com segurança..."); sys.exit(0)
signal.signal(signal.SIGINT, handle); signal.signal(signal.SIGTERM, handle)
while True:
    log.info("✅ heartbeat: ok")
    time.sleep(60)
PY
chmod +x "$ET_DIR/et_ultimate_core.py"

log "🧩 Service systemd (usa o venv)"
cat > /etc/systemd/system/et-ultimate.service <<'UNIT'
[Unit]
Description=ET★★★★ Ultimate Core (SAFE)
After=network.target
[Service]
Type=simple
WorkingDirectory=/opt/et_ultimate
ExecStart=/opt/et_ultimate/venv/bin/python /opt/et_ultimate/et_ultimate_core.py
Restart=always
RestartSec=3
[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable --now et-ultimate

log "🦙 Ajustando NUMA dos serviços llama (se existirem)"
for svc in llama-s0 llama-s1; do
  if [ -f "/etc/systemd/system/$svc.service" ]; then
    sed -i 's/--numa [^ ]*/--numa distribute/g' "/etc/systemd/system/$svc.service" || true
  fi
done
systemctl daemon-reload || true
systemctl restart llama-s0 2>/dev/null || true
systemctl restart llama-s1 2>/dev/null || true

log "🌐 Nginx 8080 → backends 8090/8091 + /health"
cat > /etc/nginx/sites-available/llama_api <<'NGX'
upstream llama_backends {
    least_conn;
    server 127.0.0.1:8090 max_fails=3 fail_timeout=15s;
    server 127.0.0.1:8091 max_fails=3 fail_timeout=15s;
    keepalive 64;
}
server {
    listen 8080;
    client_max_body_size 10m;

    location /health {
        access_log off;
        add_header Content-Type text/plain;
        return 200 "OK\n";
    }

    location / {
        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_read_timeout 300;
        proxy_send_timeout 300;
        proxy_next_upstream error timeout http_502 http_503 http_504;
        proxy_pass http://llama_backends;
    }
}
NGX
ln -sf /etc/nginx/sites-available/llama_api /etc/nginx/sites-enabled/llama_api
rm -f /etc/nginx/sites-enabled/default
nginx -t && systemctl reload nginx

log "🛠️ Atalho 'et-manager' (se o manager já existir)"
if [ -x /usr/local/bin/et-ultimate-manager ]; then
  ln -sf /usr/local/bin/et-ultimate-manager /usr/local/bin/et-manager
fi

log "🧪 Testes rápidos"
systemctl is-active --quiet et-ultimate && echo "✅ et-ultimate ativo" || echo "❌ et-ultimate inativo"
curl -fsS http://127.0.0.1:8080/health || true
echo
echo "Para testar modelos (se os backends estiverem rodando):"
echo "  curl -s http://127.0.0.1:8080/v1/models | head"
echo "Done."
