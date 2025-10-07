#!/bin/bash
# 🔧 SYSTEMD SERVICE INSTALLER
# Instala serviços systemd para operação 24/7

echo "🔧 Installing systemd services for Intelligence System..."

# 1. Brain Daemon Service
cat > /etc/systemd/system/brain-daemon.service << 'EOF'
[Unit]
Description=Unified Brain Daemon - Intelligence System
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root/UNIFIED_BRAIN
Environment="PYTHONUNBUFFERED=1"
Environment="DARWINACCI_WORM_PATH=/root/darwinacci_omega/data/worm.csv"
Environment="DARWINACCI_WORM_HEAD=/root/darwinacci_omega/data/worm_head.txt"
ExecStart=/usr/bin/python3 -u /root/UNIFIED_BRAIN/brain_daemon_real_env.py
Restart=always
RestartSec=10
StandardOutput=append:/root/logs/brain_daemon.log
StandardError=append:/root/logs/brain_daemon_error.log

[Install]
WantedBy=multi-user.target
EOF

# 2. Darwin Continuity Guardian
cat > /etc/systemd/system/darwin-guardian.service << 'EOF'
[Unit]
Description=Darwin Evolution Continuity Guardian
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
Environment="DARWINACCI_WORM_PATH=/root/darwinacci_omega/data/worm.csv"
Environment="DARWINACCI_WORM_HEAD=/root/darwinacci_omega/data/worm_head.txt"
ExecStart=/bin/bash /root/darwin_continuity_guardian.sh
Restart=always
RestartSec=30
StandardOutput=append:/root/logs/darwin_guardian.log
StandardError=append:/root/logs/darwin_guardian_error.log

[Install]
WantedBy=multi-user.target
EOF

# 3. Health Monitor
cat > /etc/systemd/system/health-monitor.service << 'EOF'
[Unit]
Description=Intelligence System Health Monitor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
Environment="DARWINACCI_WORM_PATH=/root/darwinacci_omega/data/worm.csv"
Environment="DARWINACCI_WORM_HEAD=/root/darwinacci_omega/data/worm_head.txt"
ExecStart=/bin/bash /root/health_monitor_phase1.sh
Restart=always
RestartSec=30
StandardOutput=append:/root/logs/health_monitor.log
StandardError=append:/root/logs/health_monitor_error.log

[Install]
WantedBy=multi-user.target
EOF

# 4. Emergence Monitor
cat > /etc/systemd/system/emergence-monitor.service << 'EOF'
[Unit]
Description=Intelligence Emergence Signal Monitor
After=network.target

[Service]
Type=simple
User=root
WorkingDirectory=/root
ExecStart=/bin/bash /root/monitor_emergencia.sh
Restart=always
RestartSec=30
StandardOutput=append:/root/logs/emergence_monitor.log
StandardError=append:/root/logs/emergence_monitor_error.log

[Install]
WantedBy=multi-user.target
EOF

# Create logs directory
mkdir -p /root/logs

# Reload systemd
systemctl daemon-reload

echo "✅ Systemd services installed!"
echo ""
echo "To enable and start services:"
echo "  systemctl enable brain-daemon darwin-guardian health-monitor emergence-monitor"
echo "  systemctl start brain-daemon darwin-guardian health-monitor emergence-monitor"
echo ""
echo "To check status:"
echo "  systemctl status brain-daemon"
echo "  systemctl status darwin-guardian"
echo "  systemctl status health-monitor"
echo "  systemctl status emergence-monitor"
echo ""
echo "To view logs:"
echo "  journalctl -u brain-daemon -f"
echo "  journalctl -u darwin-guardian -f"