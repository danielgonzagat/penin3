AUD="/root/tunnel_audit_$(date +%F_%H%M%S)"; mkdir -p "$AUD"

# --- A. Processos/portas ao vivo ---
ss -panto > "$AUD/ss_panto.txt" 2>&1
lsof -nP -i > "$AUD/lsof_net.txt" 2>&1 || true
ps auxwwf > "$AUD/ps_auxwwf.txt"
# filtros rápidos
ps auxww | egrep -i 'socat|proxychains|torsocks|autossh|ssh .*-[RDL]|ncat|nc ' | grep -v egrep > "$AUD/ps_sus.txt"

# --- B. Persistência: systemd/cron/timers ---
systemctl list-units --type=service --all > "$AUD/systemd_services.txt"
systemctl list-timers --all > "$AUD/systemd_timers.txt"
journalctl -b -u tor -u ssh -u sshd > "$AUD/journal_tor_sshd_currentboot.txt" 2>&1 || true
crontab -l -u root > "$AUD/crontab_root.txt" 2>&1 || true
for u in $(awk -F: '{if($3>=1000 && $1!="nobody") print $1}' /etc/passwd) root; do
  crontab -l -u "$u" > "$AUD/crontab_$u.txt" 2>&1 || true
done
ls -l /etc/cron.*/* /var/spool/cron/crontabs/* 2>/dev/null > "$AUD/cron_files.txt" || true

# --- C. Logs e histórico (grep por padrões de túnel) ---
# auth.log (SSH) e syslog/journal compactados também
zgrep -hEi 'ssh.*(R |-R|D |-D|L |-L)|Accepted|Forward|reverse' /var/log/auth.log* 2>/dev/null > "$AUD/auth_tunnel_grep.txt" || true
zgrep -hEi 'socat|proxychains|torsocks|autossh|ncat|nc -e|ssh -R|ssh -D|ssh -L' /var/log/*log* /var/log/*/* 2>/dev/null > "$AUD/logs_tunnel_grep.txt" || true
journalctl --since "14 days ago" | egrep -i 'socat|proxychains|torsocks|autossh|ssh .*-[RDL]|ncat|nc -e' > "$AUD/journal_tunnel_grep_14d.txt" 2>&1 || true

# Históricos de shell mais comuns
for H in /root /home/*; do
  [ -d "$H" ] || continue
  for f in .bash_history .zsh_history .mysql_history .psql_history; do
    [ -f "$H/$f" ] && cp -a "$H/$f" "$AUD/history_${H##*/}_$f"
  done
done
# grep nos históricos copiados
grep -RniE 'socat|proxychains|torsocks|autossh|ssh .*-[RDL]|ncat|nc -e' "$AUD" > "$AUD/history_hits.txt" 2>/dev/null || true

# --- D. Configurações relevantes ---
# SSH client/server
egrep -ni 'ProxyCommand|ProxyJump|DynamicForward|LocalForward|RemoteForward|Include' /etc/ssh/ssh_config /etc/ssh/ssh_config.d/* 2>/dev/null > "$AUD/ssh_client_cfg_hits.txt" || true
egrep -ni 'AllowTcpForwarding|GatewayPorts|PermitOpen|Match|Include' /etc/ssh/sshd_config /etc/ssh/sshd_config.d/* 2>/dev/null > "$AUD/sshd_server_cfg_hits.txt" || true
# tor / proxychains
[ -f /etc/tor/torrc ] && cp -a /etc/tor/torrc "$AUD/torrc"
egrep -ni 'HiddenService|ControlPort|SocksPort|Include' /etc/tor/torrc* 2>/dev/null > "$AUD/torrc_hits.txt" || true
[ -f /etc/proxychains.conf ] && cp -a /etc/proxychains.conf "$AUD/proxychains.conf"
egrep -ni 'socks4|socks5|tor|127\.0\.0\.1:9050|9150' /etc/proxychains.conf 2>/dev/null >> "$AUD/proxychains_hits.txt" || true

# systemd units contendo comandos suspeitos
grep -RniE 'socat|proxychains|torsocks|autossh|ssh .*-[RDL]|ncat|nc -e' /etc/systemd/system /lib/systemd/system 2>/dev/null > "$AUD/systemd_exec_hits.txt" || true

# --- E. Rede/Firewall (regras que podem abrir portas ou NAT) ---
iptables-save > "$AUD/iptables_save.txt" 2>/dev/null || true
nft list ruleset > "$AUD/nft_ruleset.txt" 2>/dev/null || true
ufw status verbose > "$AUD/ufw_status.txt" 2>/dev/null || true

# --- F. Binaries e versões (se instalados) ---
which socat proxychains4 tor autossh ncat nc  > "$AUD/which_tools.txt" 2>&1 || true
for b in socat proxychains4 tor autossh ncat nc; do command -v "$b" >/dev/null 2>&1 && "$b" --version >/dev/null 2>&1 && "$b" --version | head -n1 >> "$AUD/tool_versions.txt"; done

# --- G. Sessões tmux/screen que podem esconder túneis ---
tmux ls > "$AUD/tmux_ls.txt" 2>&1 || true
screen -ls > "$AUD/screen_ls.txt" 2>&1 || true

echo "Coleta concluída em: $AUD"
