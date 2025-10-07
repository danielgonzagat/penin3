#!/usr/bin/env bash
set -euo pipefail

OUT_DIR=/root/intelligence_system/data/exports/backups
mkdir -p "$OUT_DIR"

TMPCRON=$(mktemp)
crontab -l 2>/dev/null > "$TMPCRON" || true
sed -i '/# I3-BACKUPS-START/,/# I3-BACKUPS-END/d' "$TMPCRON"
{
  echo '# I3-BACKUPS-START'
  echo '0 3 * * * /usr/bin/env bash -lc '\''ts=$(date +%Y%m%d_%H%M%S); mkdir -p /root/intelligence_system/data/exports/backups; sqlite3 /root/intelligence_system/data/intelligence.db "VACUUM"; cp /root/intelligence_system/data/intelligence.db /root/intelligence_system/data/exports/backups/intelligence_${ts}.db; gzip -f /root/intelligence_system/data/exports/backups/intelligence_${ts}.db; ls -1t /root/intelligence_system/data/exports/backups/intelligence_*.db.gz | tail -n +15 | xargs -r rm -f'\'''
  echo '5 3 * * * /usr/bin/env bash -lc '\''ts=$(date +%Y%m%d_%H%M%S); if [ -f /root/intelligence_system/data/unified_worm.db ]; then cp /root/intelligence_system/data/unified_worm.db /root/intelligence_system/data/exports/backups/unified_worm_${ts}.db; gzip -f /root/intelligence_system/data/exports/backups/unified_worm_${ts}.db; fi'\'''
  echo '# I3-BACKUPS-END'
} >> "$TMPCRON"
crontab "$TMPCRON"
rm -f "$TMPCRON"
echo "Backup cron installed."
