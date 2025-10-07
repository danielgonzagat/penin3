#!/bin/bash
# 💾 CHECKPOINT BACKUP
# BLOCO 2 - TAREFA 27

BACKUP_DIR="/root/checkpoint_backups"
mkdir -p "$BACKUP_DIR"

echo "[$(date)] 📦 Starting checkpoint backup..."

# Compress and backup
BACKUP_FILE="$BACKUP_DIR/checkpoints_$(date +%Y%m%d_%H%M%S).tar.gz"

tar -czf "$BACKUP_FILE" \
    /root/checkpoints/ \
    /root/UNIFIED_BRAIN/*.json \
    /root/intelligence_system/data/*.db \
    2>/dev/null

if [ $? -eq 0 ]; then
    SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
    echo "[$(date)] ✅ Backup completed: $BACKUP_FILE ($SIZE)"
else
    echo "[$(date)] ❌ Backup failed"
    exit 1
fi

# Keep only last 20 backups
BACKUPS=$(ls -t "$BACKUP_DIR"/checkpoints_*.tar.gz 2>/dev/null | tail -n +21)
if [ -n "$BACKUPS" ]; then
    echo "$BACKUPS" | xargs rm -f
    COUNT=$(echo "$BACKUPS" | wc -l)
    echo "[$(date)] 🗑️ Cleaned up $COUNT old backups"
fi

echo "[$(date)] 📊 Current backups: $(ls -1 "$BACKUP_DIR"/checkpoints_*.tar.gz 2>/dev/null | wc -l)"
