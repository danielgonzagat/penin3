set -euo pipefail
[ "$(id -u)" -eq 0 ] || { echo "🚫 Rode como root"; exit 1; }
export DEBIAN_FRONTEND=noninteractive

echo "==> 0) Removendo snap 'gemini' (não é o CLI oficial)…"
snap remove gemini >/dev/null 2>&1 || true

echo "==> 1) Pacotes base + utilitários de busca"
apt-get update -y
apt-get install -y curl ca-certificates gnupg lsb-release git jq unzip \
  ripgrep fd-find fzf build-essential

echo "==> 2) Swap 4 GiB (evita OOM/Terminated)"
if ! swapon --show | grep -q /swapfile; then
  fallocate -l 4G /swapfile || dd if=/dev/zero of=/swapfile bs=1M count=4096
  chmod 600 /swapfile
  mkswap /swapfile
  swapon /swapfile
  sysctl vm.swappiness=10
  echo "vm.swappiness=10" > /etc/sysctl.d/99-gemini.conf
  grep -q '/swapfile' /etc/fstab || echo '/swapfile none swap sw 0 0' >> /etc/fstab
fi

echo "==> 3) Node.js 20.x (oficial NodeSource) se necessário"
if ! command -v node >/dev/null 2>&1 || ! node -v | grep -qE '^v2[0-9]\.'; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt-get install -y nodejs
fi
echo "Node: $(node -v)  NPM: $(npm -v)"

echo "==> 4) PATH do npm global permanente"
NPM_PREFIX="$(npm config get prefix)"
echo "export PATH=\"$NPM_PREFIX/bin:\$PATH\"" >/etc/profile.d/npm_prefix.sh
export PATH="$NPM_PREFIX/bin:$PATH"

echo "==> 5) Instalar Gemini CLI oficial (leve, com menos RAM)"
export NODE_OPTIONS="--max-old-space-size=4096"
npm install -g @google/gemini-cli@latest --omit=optional --no-fund --no-audit \
  || npm install -g @google/gemini-cli@0.6.1 --omit=optional --no-fund --no-audit

command -v gemini >/dev/null 2>&1 || { echo "❌ gemini não entrou no PATH"; exit 1; }

echo "==> 6) Workspace seguro"
mkdir -p /opt/gemini-agent/{repo,logs}
# alias fd -> fdfind (Ubuntu)
[ -x /usr/bin/fdfind ] && ln -sf /usr/bin/fdfind /usr/local/bin/fd || true

# ignora lixo comum pra auditoria ser rápida
cat > /opt/gemini-agent/.rgignore <<'RG'
node_modules
venv
.env
*.log
*.zip
*.tar
*.tar.gz
.DS_Store
__pycache__
.git
RG

cat > /opt/gemini-agent/GEMINI.md <<'MD'
# Regras (GEMINI.md)
- Escopo de arquivos **restrito a `./repo`**.
- Se tentar ler fora de `./repo`, **ABORTE**.
- Use `rg`/`fd` para buscas.
- Saída preferencial: markdown conciso.

Comandos úteis:
/audit — auditoria técnica em ./repo
/hardening — plano de guardrails/sandbox
/pr-review — revisão de PR
MD

echo "==> 7) Slash-commands"
mkdir -p /root/.gemini/commands

cat > /root/.gemini/commands/audit.toml <<'TOML'
name = "audit"
description = "Auditoria técnica restrita a ./repo"
prompt = """
Você é um auditor técnico. REGRAS: 1) NUNCA ler fora de ./repo; 2) Use rg/fd; 3) Saída em markdown com 5 seções: Visão Geral, Pontos Fortes, Fragilidades, Riscos, Ações.
Audite ./repo e entregue um relatório conciso.
"""
TOML

cat > /root/.gemini/commands/hardening.toml <<'TOML'
name = "hardening"
description = "Gerar plano de segurança/guardrails"
prompt = """
Gere um plano de hardening cobrindo: isolamento (Docker/user), permissões mínimas, validação de entrada, execução segura de shell, limites de custo/token, observabilidade (logs/metrics), rollback, canário e testes automatizados.
Contexto: projeto atual em ./repo.
"""
TOML

cat > /root/.gemini/commands/pr-review.toml <<'TOML'
name = "pr-review"
description = "Revisão de PR"
prompt = """
Atue como revisor. Entregue: 1) Resumo do PR; 2) Comentários objetivos; 3) Checklist (segurança, performance, legibilidade, testes); 4) Veredito (approve/changes).
"""
TOML

echo "==> 8) Wrapper não-interativo (sem TUI)"
cat > /usr/local/bin/gemini-run <<'SH'
#!/usr/bin/env bash
set -euo pipefail
export PATH="$(npm config get prefix)/bin:$PATH"
export CI=1 TERM=dumb FORCE_COLOR=0
exec gemini "$@"
SH
chmod +x /usr/local/bin/gemini-run

echo "==> 9) Job 24/7 com systemd timer (loop curto de exemplo)"
cat > /opt/gemini-agent/loop.sh <<'SH'
#!/usr/bin/env bash
set -euo pipefail
export PATH="$(npm config get prefix)/bin:$PATH"
export CI=1 TERM=dumb FORCE_COLOR=0
cd /opt/gemini-agent
# exemplo: heartbeat de status (ajuste o prompt pra sua rotina real)
gemini -p "Resuma o status do projeto restrito a ./repo em 5 bullets." \
  --output-format text >> /opt/gemini-agent/logs/loop.log 2>&1
SH
chmod +x /opt/gemini-agent/loop.sh

cat > /etc/systemd/system/gemini-agent.service <<'UNIT'
[Unit]
Description=Execução periódica do agente Gemini (non-interactive)
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
Environment=PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ExecStart=/opt/gemini-agent/loop.sh
WorkingDirectory=/opt/gemini-agent
Nice=5
IOSchedulingClass=best-effort
IOSchedulingPriority=7

[Install]
WantedBy=multi-user.target
UNIT

cat > /etc/systemd/system/gemini-agent.timer <<'UNIT'
[Unit]
Description=Timer do Agente Gemini (executa a cada 5 min)

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
UNIT

systemctl daemon-reload
systemctl enable --now gemini-agent.timer

echo "==> 10) Exemplo de workflow do GitHub Actions (PR review)"
mkdir -p /opt/gemini-agent/examples
cat > /opt/gemini-agent/examples/github_actions_gemini_review.yml <<'YML'
name: Gemini PR Review
on:
  pull_request:
    types: [opened, synchronize, reopened]
jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pull-requests: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: '20'
      - run: npm install -g @google/gemini-cli@latest --omit=optional --no-fund --no-audit
      - name: Run review
        run: |
          gemini -p "Faça uma revisão concisa do PR (segurança, performance, legibilidade, testes). Saída em markdown." --output-format markdown > review.md
      - name: Comment on PR
        uses: marocchino/sticky-pull-request-comment@v2
        with:
          path: review.md
YML

echo
echo "✅ CONCLUÍDO."
echo "• Workspace:  /opt/gemini-agent   (coloque seus projetos dentro de ./repo)"
echo "• Interativo: cd /opt/gemini-agent && gemini"
echo "• Slash cmds: /audit  /hardening  /pr-review"
echo "• Non-TUI:    gemini-run -p 'OK' --output-format text"
echo "• Timer:      systemctl status gemini-agent.timer"
