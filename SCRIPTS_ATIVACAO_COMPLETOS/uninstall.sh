#!/usr/bin/env bash
set -euo pipefail

APP_DIR="/opt/ia3-constructor"
LAUNCHD_PLIST="/Library/LaunchDaemons/com.ia3.constructor.plist"
Q_AGENT_JSON="${HOME}/.aws/amazonq/cli-agents/ia3-constructor.json"

echo "==> Desinstalando IA3-Constructor..."

# Parar e descarregar LaunchDaemon
if [ -f "${LAUNCHD_PLIST}" ]; then
    echo "==> Parando LaunchDaemon..."
    sudo launchctl stop com.ia3.constructor || true
    sudo launchctl unload "${LAUNCHD_PLIST}" || true
    
    echo "==> Fazendo backup do plist..."
    sudo cp "${LAUNCHD_PLIST}" "${LAUNCHD_PLIST}.bak.$(date +%Y%m%d_%H%M%S)"
    sudo rm "${LAUNCHD_PLIST}"
fi

# Backup e remoção do código
if [ -d "${APP_DIR}" ]; then
    echo "==> Fazendo backup do código..."
    sudo tar -czf "/tmp/ia3-constructor-backup-$(date +%Y%m%d_%H%M%S).tar.gz" -C "$(dirname ${APP_DIR})" "$(basename ${APP_DIR})"
    echo "Backup salvo em: /tmp/ia3-constructor-backup-*.tar.gz"
    
    echo "==> Removendo código..."
    sudo rm -rf "${APP_DIR}"
fi

# Backup e remoção do agente Q
if [ -f "${Q_AGENT_JSON}" ]; then
    echo "==> Fazendo backup do agente Q..."
    cp "${Q_AGENT_JSON}" "${Q_AGENT_JSON}.bak.$(date +%Y%m%d_%H%M%S)"
    rm "${Q_AGENT_JSON}"
fi

echo "==> IA3-Constructor desinstalado com sucesso!"
echo "Backups criados em /tmp/ e ~/.aws/amazonq/cli-agents/"
