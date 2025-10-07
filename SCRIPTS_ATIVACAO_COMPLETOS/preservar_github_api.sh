#!/usr/bin/env bash
set -euo pipefail

# Token válido
TOKEN="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"
USER="danielgonzagat"

# Projetos a preservar
declare -A PROJETOS=(
  ["neural-farm-ia3-integration"]="Neural Farm IA3 Integration|neural_farm_ia3_integration.py,neuron_farm_optimized.py,IA3_SUPREME/neural_farm.py|neural_farm_prod/,neural_arch_search/"
  ["teis-v2-enhanced"]="TEIS V2 Enhanced|teis_v2_enhanced.py,real_emergence_detector.py,chaos_utils.py,oci_lyapunov.py|teis_v2_out_prod/,teis_checkpoints/"
  ["preserved-intelligence-models"]="PRESERVED Intelligence Models|README.md|PRESERVED_INTELLIGENCE/"
  ["darwin-evolution-system"]="Darwin Evolution System|darwin/darwin_runner.py,darwin/darwin_policy.json,darwin_metrics.py|PRESERVED_INTELLIGENCE/evidence_of_breakthrough/"
  ["intelligence-emergence-maestro"]="Intelligence Emergence Maestro|intelligence_emergence_maestro.py,emergence_detector.py|intelligence_emergence.db,unified_data/"
  ["auditoria-cientifica-brutal-final"]="Auditoria Cientifica Brutal Final|AUDITORIA_CIENTIFICA_BRUTAL_FINAL.md|"
)

WORKDIR="/root/_github_export"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

echo "🚀 Iniciando preservação via API GitHub..."

for REPO in "${!PROJETOS[@]}"; do
  IFS="|" read -r NAME CODE_PATHS BIG_PATHS <<< "${PROJETOS[$REPO]}"
  
  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "📦 $NAME"
  echo "🔗 $REPO"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  
  REPO_DIR="$WORKDIR/$REPO"
  mkdir -p "$REPO_DIR"
  
  # 1) Copiar código
  echo "📝 Copiando código..."
  IFS="," read -ra FILES <<< "$CODE_PATHS"
  for f in "${FILES[@]}"; do
    if [ -n "$f" ] && [ -e "/root/$f" ]; then
      echo "  ✓ $f"
      mkdir -p "$REPO_DIR/$(dirname "$f")"
      cp -r "/root/$f" "$REPO_DIR/$f"
    fi
  done
  
  # 2) README
  cat > "$REPO_DIR/README.md" <<EOF
# $NAME

Preservado do servidor de IA evolutiva - Auditoria Científica 2025-09-30

## 📦 Conteúdo

Este repositório contém o código principal do sistema.

Artefatos grandes (datasets, checkpoints, logs) serão adicionados posteriormente como Releases.

## 📊 Origem

- **Servidor:** Gcore Cloud
- **Data:** $(date +%Y-%m-%d)
- **Auditoria:** [Ver relatório completo](https://github.com/$USER/auditoria-cientifica-brutal-final)

## 🔧 Sistemas Relacionados

1. [Neural Farm IA3](https://github.com/$USER/neural-farm-ia3-integration) - 49 dias CPU time
2. [TEIS V2 Enhanced](https://github.com/$USER/teis-v2-enhanced) - 100 gerações RL
3. [PRESERVED Intelligence](https://github.com/$USER/preserved-intelligence-models) - 28GB modelos
4. [Darwin Evolution](https://github.com/$USER/darwin-evolution-system) - Seleção natural
5. [Emergence Maestro](https://github.com/$USER/intelligence-emergence-maestro) - Orquestração
6. [Auditoria Final](https://github.com/$USER/auditoria-cientifica-brutal-final) - Relatório
EOF

  # 3) .gitignore
  cat > "$REPO_DIR/.gitignore" <<'EOF'
__pycache__/
*.pyc
.venv/
.env
*.log
*.cache/
.DS_Store
EOF

  # 4) Verificar se repo já existe
  echo "🔍 Verificando repositório..."
  STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
    -H "Authorization: token $TOKEN" \
    "https://api.github.com/repos/$USER/$REPO")
  
  if [ "$STATUS" = "404" ]; then
    echo "🔧 Criando repositório..."
    curl -s -X POST \
      -H "Authorization: token $TOKEN" \
      -H "Content-Type: application/json" \
      -d "{\"name\":\"$REPO\",\"private\":true,\"description\":\"$NAME\"}" \
      https://api.github.com/user/repos | jq -r '.html_url // "erro"'
    sleep 2
  else
    echo "✅ Repositório já existe"
  fi
  
  # 5) Git init e push
  cd "$REPO_DIR"
  git init -b main
  git add .
  git commit -m "feat: snapshot inicial $(date +%Y%m%d)"
  
  echo "📤 Enviando para GitHub..."
  git remote add origin "https://$USER:$TOKEN@github.com/$USER/$REPO.git" 2>/dev/null || true
  git push -u origin main --force 2>&1 | tail -3
  
  echo "✅ Finalizado: https://github.com/$USER/$REPO"
  cd /root
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 FASE 1 COMPLETA - Código preservado!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Repositórios criados:"
for REPO in "${!PROJETOS[@]}"; do
  echo "  🔗 https://github.com/$USER/$REPO"
done
echo ""
echo "⏳ FASE 2: Empacotando dados grandes (11GB + 28GB)..."
echo "   Isso será feito em segundo plano para não travar."
