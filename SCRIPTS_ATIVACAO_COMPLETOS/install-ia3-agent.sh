#!/usr/bin/env bash
set -euo pipefail

AGENT_NAME="IA3-Constructor"
GLOBAL_DIR="${HOME}/.aws/amazonq/cli-agents"
GLOBAL_JSON="${GLOBAL_DIR}/ia3-constructor.json"
PROJECT_DIR=".amazonq/cli-agents"
PROJECT_JSON="${PROJECT_DIR}/ia3-constructor.json"

ALLOWED_ROOTS=("/opt/ia3-constructor" "/opt/penin")

echo "==> Verificando Amazon Q Developer CLI (q)"
if ! command -v q >/dev/null 2>&1; then
  echo "ERRO: comando 'q' não encontrado. Instale/atualize o Amazon Q Developer CLI e rode novamente."
  echo "Dica: consulte a documentação oficial do Q CLI."
  exit 1
fi
q --version || true

echo "==> Criando diretório global de agentes: ${GLOBAL_DIR}"
mkdir -p "${GLOBAL_DIR}"

timestamp() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

backup_if_exists () {
  local f="$1"
  if [ -f "$f" ]; then
    local bk="${f}.bak.$(timestamp)"
    cp "$f" "$bk"
    echo "Backup criado: $bk"
  fi
}

echo "==> Escrevendo agente global em ${GLOBAL_JSON}"
backup_if_exists "${GLOBAL_JSON}"
cat > "${GLOBAL_JSON}" <<JSON
{
  "name": "${AGENT_NAME}",
  "description": "Agente construtor 24/7 da IA ao Cubo (PENIN-Ω) com Σ-Guard, WORM e subagentes.",
  "instructions": "Objetivo: Construir e evoluir a IA ao Cubo (PENIN-Ω) continuamente, com ética (ΣEA/LO-14), contratividade (IR→IC), auditabilidade (WORM) e fail-closed (Σ-Guard). Regras: 1) Fail-closed; 2) Registrar tudo em Memória/WORM; 3) Criar subagentes conforme complexidade; 4) Promover apenas mutações que aumentem ΔL∞ e CAOS⁺; 5) Operações destrutivas apenas com aprovação humana explícita. Escopo: atuar apenas dentro de /opt/ia3-constructor e /opt/penin.",
  "resources": [
    "${ALLOWED_ROOTS[0]}",
    "${ALLOWED_ROOTS[1]}"
  ],
  "tools": [
    "fs_read",
    "fs_write",
    "execute_bash",
    "knowledge",
    "introspect"
  ],
  "allowedTools": [
    "fs_read",
    "fs_write"
  ],
  "toolsSettings": {
    "fs_read": { "allowedRoots": ["${ALLOWED_ROOTS[0]}", "${ALLOWED_ROOTS[1]}"] },
    "fs_write": { "allowedRoots": ["${ALLOWED_ROOTS[0]}", "${ALLOWED_ROOTS[1]}"] },
    "execute_bash": { "confirmEachCommand": true }
  }
}
JSON

echo "==> (Opcional) Deseja também criar o agente LOCAL no projeto atual (${PROJECT_JSON})? [s/N]"
read -r REPLY_CREATE_LOCAL || true
if [[ "${REPLY_CREATE_LOCAL:-n}" =~ ^[sSyY]$ ]]; then
  echo "==> Criando agente local no projeto"
  mkdir -p "${PROJECT_DIR}"
  backup_if_exists "${PROJECT_JSON}"
  cp "${GLOBAL_JSON}" "${PROJECT_JSON}"
  echo "Agente local criado: ${PROJECT_JSON}"
else
  echo "Pulando agente local."
fi

echo "==> Validação rápida"
echo " - Listando agentes no Q chat:"
echo "   (abra o Q chat e rode: /agent )"
echo
echo "==> Dicas:"
echo "1) No Q chat:"
echo "   /agent                # lista agentes; verifique '${AGENT_NAME}'"
echo "   /agent use ${AGENT_NAME}"
echo "   /tools                # veja permissões ativas"
echo "   /tools allow fs_write"
echo "   /tools allow execute_bash"
echo
echo "2) Garanta que os diretórios permitidos existem:"
for r in "${ALLOWED_ROOTS[@]}"; do
  if [ ! -d "$r" ]; then
    echo "   sudo mkdir -p $r && sudo chown $(id -un):$(id -gn) $r"
  fi
done
echo
echo "3) Se estiver rodando seu serviço IA3-Constructor local:"
echo "   curl -X POST http://127.0.0.1:8010/boot -H 'content-type: application/json' \\"
echo "     -d '{\"name\":\"IA3-Constructor\",\"goal\":\"Construir e evoluir a IA ao Cubo (PENIN-Ω) com Σ-Guard e WORM\"}'"
echo
echo "Pronto. Agente '${AGENT_NAME}' instalado."
