#!/usr/bin/env bash
echo "[$(date -u +%Y-%m-%dT%H:%M:%SZ)] ROLLBACK: reason=$REASON success=${ROLLBACK_SUCCESS:-0}" >> /root/ia3_darwin_hooks.log
echo "✅ Rollback notificado: $REASON"
