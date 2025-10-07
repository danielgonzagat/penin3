#!/usr/bin/env bash
set -euo pipefail

# Create and enable systemd units for core daemons

have_systemctl() {
  command -v systemctl >/dev/null 2>&1 || return 1
  # Ensure PID 1 is systemd
  [[ "$(ps -p 1 -o comm=)" == "systemd" ]] || return 1
  return 0
}

install_unit() {
  local name="$1"; shift
  local body="$1"; shift || true
  local path="/etc/systemd/system/${name}.service"
  echo "Installing ${path}"
  printf "%s" "$body" > "$path"
  chmod 644 "$path"
  systemctl daemon-reload
  systemctl enable --now "$name" || true
  systemctl status --no-pager "$name" || true
}

require_root() {
  if [[ $(id -u) -ne 0 ]]; then
    echo "Please run as root: sudo bash INSTALL_SYSTEMD_SERVICES.sh" >&2
    exit 1
  fi
}

require_root

# Unit bodies
read -r -d '' DARWINACCI_SERVICE <<'EOF'
[Unit]
Description=Darwinacci Daemon
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /root/DARWINACCI_DAEMON.py
Restart=always
Nice=10
IOSchedulingClass=2
IOSchedulingPriority=7

[Install]
WantedBy=multi-user.target
EOF

read -r -d '' PHASE4_SERVICE <<'EOF'
[Unit]
Description=Phase 4 Daemon
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /root/PHASE4_DAEMON.py
Restart=always
Nice=10
IOSchedulingClass=2
IOSchedulingPriority=7

[Install]
WantedBy=multi-user.target
EOF

read -r -d '' PHASE5_SERVICE <<'EOF'
[Unit]
Description=Phase 5 Daemon
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /root/PHASE5_DAEMON.py
Restart=always
Nice=10
IOSchedulingClass=2
IOSchedulingPriority=7

[Install]
WantedBy=multi-user.target
EOF

read -r -d '' CONNECTOR_SERVICE <<'EOF'
[Unit]
Description=System Connector
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /root/EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py 100 60
Restart=always
Nice=5
IOSchedulingClass=2
IOSchedulingPriority=4

[Install]
WantedBy=multi-user.target
EOF

read -r -d '' V7RUNNER_SERVICE <<'EOF'
[Unit]
Description=V7 Runner Daemon
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /root/V7_RUNNER_DAEMON.py
Restart=always
Nice=10
IOSchedulingClass=2
IOSchedulingPriority=7

[Install]
WantedBy=multi-user.target
EOF

read -r -d '' VALIDATOR_SERVICE <<'EOF'
[Unit]
Description=Auto Validator
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /root/AUTO_VALIDATOR.py
Restart=always
Nice=0
IOSchedulingClass=2
IOSchedulingPriority=4

[Install]
WantedBy=multi-user.target
EOF

read -r -d '' V7METRICS_SERVICE <<'EOF'
[Unit]
Description=V7 Prometheus Metrics Exporter
After=network.target

[Service]
Type=simple
ExecStart=/usr/bin/python3 /root/intelligence_system/metrics/prometheus_exporter.py 8012
Restart=always
Nice=10
IOSchedulingClass=2
IOSchedulingPriority=7

[Install]
WantedBy=multi-user.target
EOF

if have_systemctl; then
  install_unit darwinacci "$DARWINACCI_SERVICE"
  install_unit phase4 "$PHASE4_SERVICE"
  install_unit phase5 "$PHASE5_SERVICE"
  install_unit system_connector "$CONNECTOR_SERVICE"
  install_unit v7_runner "$V7RUNNER_SERVICE"
  install_unit auto_validator "$VALIDATOR_SERVICE"
  install_unit v7_metrics_exporter "$V7METRICS_SERVICE"
  echo "All systemd services installed and started."
else
  echo "systemd not available; installing @reboot cron entries as fallback..."
  # Merge-safe crontab update
  TMPCRON=$(mktemp)
  crontab -l 2>/dev/null > "$TMPCRON" || true
  # Remove existing our lines
  sed -i '/# I3-DAEMONS-START/,/# I3-DAEMONS-END/d' "$TMPCRON"
  {
    echo '# I3-DAEMONS-START'
    echo '@reboot nohup python3 /root/DARWINACCI_DAEMON.py > /root/darwinacci_daemon.log 2>&1 &'
    echo '@reboot nohup python3 /root/PHASE4_DAEMON.py > /root/phase4_daemon.log 2>&1 &'
    echo '@reboot nohup python3 /root/PHASE5_DAEMON.py > /root/phase5_daemon.log 2>&1 &'
    echo '@reboot nohup python3 /root/EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py 100 60 > /root/system_connector.log 2>&1 &'
    echo '@reboot nohup python3 /root/V7_RUNNER_DAEMON.py > /root/v7_runner_daemon.log 2>&1 &'
    echo '@reboot nohup python3 /root/intelligence_system/metrics/prometheus_exporter.py 8012 > /root/v7_metrics_exporter.log 2>&1 &'
    echo '@reboot nohup python3 /root/AUTO_VALIDATOR.py > /root/auto_validator.log 2>&1 &'
    echo '# I3-DAEMONS-END'
  } >> "$TMPCRON"
  crontab "$TMPCRON"
  rm -f "$TMPCRON"
  echo "Cron @reboot entries installed."
fi
