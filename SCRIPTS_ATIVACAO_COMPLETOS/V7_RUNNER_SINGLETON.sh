#!/bin/bash
PIDFILE=/var/run/v7_runner.pid

if [ -f "$PIDFILE" ] && kill -0 $(cat "$PIDFILE") 2>/dev/null; then
    echo "✓ V7_RUNNER já rodando (PID: $(cat $PIDFILE))"
    exit 0
fi

echo "🚀 Iniciando V7_RUNNER (singleton)..."
python3 /root/V7_RUNNER_DAEMON.py &
NEW_PID=$!
echo $NEW_PID > "$PIDFILE"
echo "✓ V7_RUNNER iniciado (PID: $NEW_PID)"
