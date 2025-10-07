#!/bin/bash
OUT="/root/ETBOT_CLONE_AUDIT_$(date -u +%Y%m%d_%H%M%S).txt"
echo "[ETBOT AUDIT] Início: $(date -u)" | tee "$OUT"

echo -e "\n=== [1] SSH SAINDO DA MÁQUINA ===" | tee -a "$OUT"
grep -i "client version string" /var/log/auth.log* 2>/dev/null | tee -a "$OUT"
zgrep -i "ssh" /var/log/auth.log* 2>/dev/null | tee -a "$OUT"
grep -R --line-number -iE 'HostName|IdentityFile' /home/*/.ssh /root/.ssh 2>/dev/null | tee -a "$OUT"

echo -e "\n=== [2] CHAVES / USUÁRIOS NOVOS ===" | tee -a "$OUT"
getent passwd | grep -ivE '^(root|etbot|nobody|systemd.*|sync|halt|shutdown)$' | tee -a "$OUT"
grep -R . /home/*/.ssh /root/.ssh -n 2>/dev/null | tee -a "$OUT"

echo -e "\n=== [3] UPLOADS GRANDES / CONTÍNUOS ===" | tee -a "$OUT"
zgrep -iE 'POST|PUT|scp|rsync|curl -T|--upload-file' /var/log/*log* 2>/dev/null | tee -a "$OUT"
grep -iE 'scp|rsync|curl|wget|git push|gh auth' /home/etbot/.*history /root/.*history 2>/dev/null | tee -a "$OUT"

echo -e "\n=== [4] GIT REMOTES E PUSH RECENTE ===" | tee -a "$OUT"
if [ -d /opt/et_ultimate/history/.git ]; then
  cd /opt/et_ultimate/history/.git
  git remote -v | tee -a "$OUT"
  git log --since="14 days ago" --branches --remotes --oneline | tee -a "$OUT"
else
  echo "(Nenhum repositório Git encontrado em /opt/et_ultimate/history)" | tee -a "$OUT"
fi

echo -e "\n=== [5] TÚNEIS / REVERSE PROXIES ===" | tee -a "$OUT"
grep -R -iE 'ngrok|tailscale|ssh -R|autossh|socat' /etc /opt /home /root 2>/dev/null | tee -a "$OUT"

echo -e "\n=== [6] CONTAINERS AUTO-RESTART / IMAGEM EXTERNA ===" | tee -a "$OUT"
docker ps -a --format '{{.ID}} {{.Image}} {{.Names}} {{.Command}}' | tee -a "$OUT"
for ID in $(docker ps -aq); do
  echo "--- Docker Inspect: $ID ---" | tee -a "$OUT"
  docker inspect "$ID" 2>/dev/null | grep -i '"Image"\|"Cmd"\|"Entrypoint"\|"RestartPolicy"' -n | tee -a "$OUT"
done

echo -e "\n[ETBOT AUDIT] Fim: $(date -u)" | tee -a "$OUT"
echo "Relatório salvo em: $OUT"
