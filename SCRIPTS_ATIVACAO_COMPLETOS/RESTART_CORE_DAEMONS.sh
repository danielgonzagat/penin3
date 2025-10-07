#!/bin/bash
set -euo pipefail

# Stop old instances (if any)
pkill -f EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py || true
pkill -f AUTO_VALIDATOR.py || true
pkill -f META_LEARNER_REALTIME.py || true
pkill -f CROSS_POLLINATION_AUTO_FIXED.py || true

# Start fresh
nohup python3 /root/EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py 200 60 > /root/system_connector.log 2>&1 &
nohup python3 /root/AUTO_VALIDATOR.py > /root/auto_validator.log 2>&1 &
nohup python3 /root/META_LEARNER_REALTIME.py > /root/meta_learner.log 2>&1 &
nohup python3 /root/CROSS_POLLINATION_AUTO_FIXED.py > /root/cross_pollination_auto_fixed.log 2>&1 &

sleep 3

# Quick status
pgrep -fl "EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR|AUTO_VALIDATOR|META_LEARNER_REALTIME|CROSS_POLLINATION_AUTO_FIXED" || true

# Show last lines of logs
for f in \
  /root/system_connector.log \
  /root/auto_validator.log \
  /root/meta_learner.log \
  /root/cross_pollination_auto_fixed.log; do
  echo "----- $f -----"
  [ -f "$f" ] && tail -n 50 "$f" || echo "(no log yet)"
  echo
done
