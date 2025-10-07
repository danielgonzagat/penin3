#!/bin/bash
# Watchdog anti-duplicação
while true; do
    V7_COUNT=$(pgrep -c "V7_RUNNER_DAEMON")
    if [ "$V7_COUNT" -gt 1 ]; then
        echo "$(date): $V7_COUNT V7_RUNNER detected, killing extras..."
        pgrep -f "V7_RUNNER_DAEMON" | head -n -1 | xargs -r kill -9
    fi
    sleep 300  # Check every 5 min
done
