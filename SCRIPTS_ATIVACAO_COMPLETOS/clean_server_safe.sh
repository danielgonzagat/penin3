#!/usr/bin/env bash
# Clean Server SAFE Playbook (no-format) — v1.0
# Ubuntu/Debian oriented. Run as root. Default: DRY-RUN.

set -euo pipefail
DRY_RUN=0
CLEAN_APT=1
CLEAN_JOURNAL=1
CLEAN_LOGS=1
CLEAN_TMP=1
CLEAN_DOCKER=0
CLEAN_SNAP=0
CLEAN_LANG_CACHES=1
DISABLE_UNUSED_SERVICES=0
REPORT_LARGEST=1
timestamp(){ date +"%Y-%m-%d %H:%M:%S"; }
say(){ printf "\033[1;36m[%s]\033[0m %s\n" "$(timestamp)" "$*"; }
warn(){ printf "\033[1;33m[%s][WARN]\033[0m %s\n" "$(timestamp)" "$*"; }
run(){ if [[ "$DRY_RUN" -eq 1 ]]; then echo "DRY-RUN: $*"; else eval "$@"; fi; }
require_root(){ if [[ "$(id -u)" -ne 0 ]]; then warn "Run as root: sudo -i && bash clean_server_safe.sh"; exit 1; fi; }
preflight(){ say "Preflight"; uname -a || true; df -hT || true; ps aux --sort=-%mem | awk 'NR<=11{print $0}' || true; ss -tulpn || true; }
clean_apt(){ say "APT"; run "apt-get update -y"; run "apt-get -f install -y || true"; run "apt-get autoremove --purge -y"; run "apt-get autoclean -y"; run "apt-get clean -y"; RC=$(dpkg -l | awk '/^rc/{print $2}'); [[ -n "$RC" ]] && run "dpkg -P $RC"; CURRENT=$(uname -r | sed 's/-generic//'); INSTALLED=$(dpkg -l 'linux-image-*' | awk '/ii/{print $2}' | grep -E 'linux-image-[0-9]'); for k in $INSTALLED; do [[ "$k" == *"$CURRENT"* ]] && echo "Keeping $k" || run "apt-get purge -y $k || true"; done; }
clean_journal(){ say "journal"; run "journalctl --vacuum-size=1G || true"; run "journalctl --vacuum-time=7d || true"; }
clean_logs(){ say "logs"; run "logrotate -f /etc/logrotate.conf || true"; run "find /var/log -type f \\( -name '*.gz' -o -name '*.1' -o -name '*.old' -o -name '*-????????' \\) -print -delete"; }
clean_tmp(){ say "tmp"; run "rm -rf /tmp/* /var/tmp/*"; }
clean_docker(){ command -v docker >/dev/null 2>&1 || { say "no docker"; return; }; docker system df || true; run "docker system prune -a -f"; run "docker volume prune -f"; }
clean_snap(){ command -v snap >/dev/null 2>&1 || { say "no snap"; return; }; run "snap set system refresh.retain=2"; snap list --all | awk '/disabled/{print $1, $3}' | while read n r; do run "snap remove ${n} --revision=${r}"; done; }
clean_lang_caches(){ say "lang caches"; command -v pip >/dev/null 2>&1 && run "pip cache purge || true"; command -v conda >/dev/null 2>&1 && run "conda clean -a -y || true"; command -v npm >/dev/null 2>&1 && run "npm cache clean --force || true"; command -v yarn >/dev/null 2>&1 && run "yarn cache clean || true"; command -v pnpm >/dev/null 2>&1 && run "pnpm store prune || true"; command -v cargo >/dev/null 2>&1 && run "cargo cache -a || true"; command -v go >/dev/null 2>&1 && run "go clean -modcache || true"; command -v composer >/dev/null 2>&1 && run "composer clear-cache || true"; command -v pipx >/dev/null 2>&1 && run "pipx list --orphan | awk '/package/ {print $2}' | xargs -r pipx uninstall || true"; }
disable_unused_services(){ systemctl list-unit-files --type=service | sed -n '1,200p' || true; }
report_largest(){ say "largest"; du -xh --max-depth=1 /var 2>/dev/null | sort -h | tail -n 20 || true; find / -xdev \( -path /proc -o -path /sys -o -path /dev \) -prune -o -type f -size +100M -printf "%s\t%p\n" 2>/dev/null | sort -n | tail -n 20 || true; }
main(){ require_root; say "START (DRY_RUN=${DRY_RUN})"; preflight; [[ "$CLEAN_APT" -eq 1 ]] && clean_apt; [[ "$CLEAN_JOURNAL" -eq 1 ]] && clean_journal; [[ "$CLEAN_LOGS" -eq 1 ]] && clean_logs; [[ "$CLEAN_TMP" -eq 1 ]] && clean_tmp; [[ "$CLEAN_DOCKER" -eq 1 ]] && clean_docker; [[ "$CLEAN_SNAP" -eq 1 ]] && clean_snap; [[ "$CLEAN_LANG_CACHES" -eq 1 ]] && clean_lang_caches; [[ "$DISABLE_UNUSED_SERVICES" -eq 1 ]] && disable_unused_services; [[ "$REPORT_LARGEST" -eq 1 ]] && report_largest; say "DONE — re-run with DRY_RUN=0 to apply"; }
main "$@"
