#!/usr/bin/env bash
# et_enforce_lockdown.sh — “modo guerra”: mata processos, mascara serviços, bloqueia egress, blinda SSH

set -euo pipefail
export LC_ALL=C

STAMP="$(date -u +%Y%m%d-%H%M%S)"
LOG="/root/et_enforce_lockdown_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

echo "== ET ENFORCE LOCKDOWN =="
echo "Host: $(hostname)  |  UTC: $(date -u)"
echo "Kernel: $(uname -a)"

# 0) Descobrir porta SSH atual (default 22) e IP do cliente (para info)
SSH_PORT="$(ss -ltnp | awk '/ssh/ && /LISTEN/ {print $4}' | awk -F: '{print $NF}' | sort -u | head -n1)"
SSH_PORT="${SSH_PORT:-22}"
ADMIN_IP="$(echo ${SSH_CLIENT:-unknown} | awk '{print $1}')"
echo "[i] SSH_PORT=$SSH_PORT  ADMIN_IP=$ADMIN_IP"

# 1) Backups e rollback helpers
ROLL="/root/et_enforce_rollback_${STAMP}.sh"
echo "#!/usr/bin/env bash" > "$ROLL"
echo "set -euo pipefail" >> "$ROLL"
chmod +x "$ROLL"

backup_file() {
  local f="$1"
  if [ -f "$f" ]; then
    cp -a "$f" "${f}.bak_${STAMP}"
    echo "cp -a '${f}.bak_${STAMP}' '${f}'" >> "$ROLL"
  fi
}

unmask_unit() {
  local u="$1"
  if systemctl is-enabled "$u" >/dev/null 2>&1 || systemctl is-enabled "$u" >/dev/null 2>&1; then true; fi
  echo "systemctl unmask '$u' || true" >> "$ROLL"
  echo "systemctl enable '$u' || true" >> "$ROLL"
}

# 2) Matar processos suspeitos (ssh -R/-L/-D, proxies/túneis)
echo "== Matando processos suspeitos =="
PATTERN='ssh .* -[RLD]|autossh|socat|chisel|frpc|frps|ngrok|gost|proxychains|torsocks|tor'
PIDS="$(ps auxww | egrep -i "$PATTERN" | egrep -v 'egrep|et_enforce_lockdown' | awk '{print $2}' || true)"
if [ -n "${PIDS:-}" ]; then
  echo "$PIDS" | xargs -r -I{} sh -c 'kill -9 {} || true'
fi

# 3) Desabilitar/Mascarar serviços (systemd)
echo "== Parando/mascarando serviços =="
SERVICES=(
  tor
  tor@default
  frpc frps
  ngrok
  et-supervisor
  et-capabilities
  et-brain
  et-brain-operacional
  et-evolver
  et-autonomy
  et-liga-copilotos
  et-mutation-orchestrator
  et-chat
)
for s in "${SERVICES[@]}"; do
  systemctl stop "$s" 2>/dev/null || true
  systemctl disable "$s" 2>/dev/null || true
  systemctl reset-failed "$s" 2>/dev/null || true
  systemctl mask "$s" 2>/dev/null || true
  unmask_unit "$s"
done

# 4) Blindar SSH contra forward/tunnel/agent/X11
echo "== Blindando sshd_config =="
SSHD="/etc/ssh/sshd_config"
backup_file "$SSHD"

ensure_cfg() {
  local key="$1"; shift
  local val="$*"
  if grep -qiE "^\s*${key}\b" "$SSHD"; then
    sed -i "s#^\s*${key}.*#${key} ${val}#I" "$SSHD"
  else
    echo "${key} ${val}" >> "$SSHD"
  fi
}
ensure_cfg "AllowTcpForwarding" "no"
ensure_cfg "PermitTunnel" "no"
ensure_cfg "GatewayPorts" "no"
ensure_cfg "X11Forwarding" "no"
ensure_cfg "AllowAgentForwarding" "no"
# trava porta atual explicitamente
ensure_cfg "Port" "${SSH_PORT}"

# reload seguro
systemctl reload sshd || systemctl restart ssh || true
echo "[ok] sshd reloaded"

# 5) Remover pacotes de túnel (mantém netcat para emergências se quiser)
echo "== Removendo pacotes de túnel =="
apt-get update || true
apt-get purge -y tor proxychains4 socat autossh chisel frpc frps ngrok gost || true
apt-get autoremove -y || true

# 6) nftables: política de egress DROP (sem NEW), INPUT restrito com SSH e loopback
echo "== Aplicando nftables hard lockdown =="
backup_file "/etc/nftables.conf"

NFT_TMP="/root/nft_hardlock_${STAMP}.nft"
cat > "$NFT_TMP" <<EOF
flush ruleset

table inet filter {
  chains {
    input {
      type filter hook input priority 0;
      policy drop;
      iif lo accept
      ct state established,related accept
      tcp dport ${SSH_PORT} accept
    }

    forward {
      type filter hook forward priority 0;
      policy drop;
    }

    output {
      type filter hook output priority 0;
      policy drop;
      oif lo accept
      ct state established,related accept
      # Bloqueia qualquer NEW: nada inicia conexões para fora.
      # (Se quiser liberar DNS/apt depois, eu ajusto.)
    }
  }
}
EOF

# Salvar permanentemente
nft -f "$NFT_TMP"
cp -a "$NFT_TMP" /etc/nftables.conf
echo "nft -f /etc/nftables.conf" >> "$ROLL"
systemctl enable nftables --now

echo "[ok] nftables aplicado. Saída NEW bloqueada; entrada só SSH/lo/established."

# 7) Opcional: travar sshd_config (anti-tamper). Se der problema, remova o immutable pelo rollback.
if command -v chattr >/dev/null 2>&1; then
  chattr +i "$SSHD" || true
  echo "chattr -i '$SSHD' || true" >> "$ROLL"
fi

# 8) Limpeza de timers/sockets suspeitos
echo "== Desativando timers/sockets suspeitos =="
systemctl list-timers --all | awk '{print $1}' | egrep -i 'et-|tor|frp|ngrok|gost' | xargs -r -I{} systemctl disable --now {} || true
systemctl list-sockets | awk '{print $1}' | egrep -i 'et-|tor|frp|ngrok|gost' | xargs -r -I{} systemctl disable --now {} || true

# 9) Registrar rollback básico (desmascarar serviços e reabrir egress)
cat >> "$ROLL" <<'EOF'
echo "== ROLLBACK =="
# Reabrir egress (output ACCEPT)
TMP="/root/nft_rollback.nft"
cat > "$TMP" <<'EONFT'
flush ruleset
table inet filter {
  chains {
    input { type filter hook input priority 0; policy drop;
      iif lo accept
      ct state established,related accept
      tcp dport 22 accept
    }
    forward { type filter hook forward priority 0; policy drop; }
    output { type filter hook output priority 0; policy accept; oif lo accept; }
  }
}
EONFT
nft -f "$TMP"
cp -a "$TMP" /etc/nftables.conf
systemctl reload nftables || systemctl restart nftables || true
echo "[ok] egress liberado (policy ACCEPT). Ajuste conforme necessário."

# Destravar sshd_config
if command -v chattr >/dev/null 2>&1; then
  chattr -i /etc/ssh/sshd_config || true
fi
systemctl reload sshd || true
echo "[ok] sshd_config destravado e sshd recarregado"

# Desmascarar serviços principais (ajuste os que precisar retomar)
for s in tor tor@default frpc frps ngrok et-supervisor et-capabilities et-brain et-brain-operacional et-evolver et-autonomy et-liga-copilotos et-mutation-orchestrator et-chat; do
  systemctl unmask "$s" 2>/dev/null || true
  systemctl enable "$s" 2>/dev/null || true
done
echo "[ok] serviços desmascarados (não necessariamente iniciados)."
EOF

echo
echo "== LOCKDOWN CONCLUÍDO =="
echo "Log: $LOG"
echo "Rollback script gerado: $ROLL"
