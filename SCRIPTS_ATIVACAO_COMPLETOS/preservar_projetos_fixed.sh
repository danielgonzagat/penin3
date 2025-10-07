#!/usr/bin/env bash
set -euo pipefail

# =====================[ CONFIGURAÇÃO ]=====================
# Remover tokens inválidos do ambiente
unset GH_TOKEN 2>/dev/null || true
unset GITHUB_TOKEN 2>/dev/null || true

# Usar token válido do arquivo de configuração
export GITHUB_USER="danielgonzagat"

# Configuração dos projetos
PROJETOS=(
"neural-farm-ia3-integration|Neural Farm IA3 Integration|Evolução populacional com IA3 - 49 dias de CPU time, 17.3M métricas|neural_farm_ia3_integration.py,neuron_farm_optimized.py,IA3_SUPREME/neural_farm.py|neural_farm_prod/,neural_arch_search/"
"teis-v2-enhanced|TEIS V2 Enhanced|Sistema de RL com experience replay e policy gradients - 100 gerações|teis_v2_enhanced.py,real_emergence_detector.py,chaos_utils.py,oci_lyapunov.py|teis_v2_out_prod/,teis_checkpoints/"
"preserved-intelligence-models|PRESERVED Intelligence Models|Modelos históricos preservados - 28GB de checkpoints e evolução|README.md|PRESERVED_INTELLIGENCE/"
"darwin-evolution-system|Darwin Evolution System|Sistema evolutivo com seleção natural - 23K→254 neurônios|darwin/darwin_runner.py,darwin/darwin_policy.json,darwin_metrics.py|PRESERVED_INTELLIGENCE/evidence_of_breakthrough/"
"intelligence-emergence-maestro|Intelligence Emergence Maestro|Orquestrador de emergência multi-sistema|intelligence_emergence_maestro.py,intelligence_emergence_maestro_deterministic.py,emergence_detector.py|intelligence_emergence.db,intelligence_emergence_maestro_state.json,unified_data/"
"auditoria-cientifica-brutal-final|Auditoria Científica Brutal Final|Relatório completo de auditoria científica dos sistemas de IA|AUDITORIA_CIENTIFICA_BRUTAL_FINAL.md|"
)

VISIBILIDADE="private"
SPLIT_SIZE="1900m"

# Template .gitignore
read -r -d '' GITIGNORE_CONTENT <<'EOF' || true
__pycache__/
*.pyc
*.pyo
.venv/
venv/
.env
.env.*
*.tmp
*.log
*.cache/
output/
runs/
.DS_Store
EOF

# Padrões LFS
LFS_PATTERNS=(
"*.pt" "*.pth" "*.ckpt" "*.bin" "*.safetensors"
"*.onnx" "*.h5" "*.pb"
"*.npz" "*.npy"
"*.tar" "*.tar.gz" "*.tar.xz" "*.zip"
"*.db" "*.sqlite"
)

# =====================[ FUNÇÕES ]=========================

die(){ echo "❌ Erro: $*" >&2; exit 1; }

ensure_clean_tmp(){
  rm -rf "$1"
  mkdir -p "$1"
}

copy_if_exists(){
  local path="$1" dest="$2"
  echo "  📋 Copiando: $path"
  if [ -e "$path" ]; then
    if [ -d "$path" ]; then
      rsync -a --info=progress2 "$path" "$dest/" 2>&1 | tail -3
    else
      mkdir -p "$dest/$(dirname "$path")"
      rsync -a --info=progress2 "$path" "$dest/$path" 2>&1 | tail -3
    fi
  else
    echo "  ⚠️  Não encontrado: $path (pulando)"
  fi
}

write_gitignore(){
  echo "$GITIGNORE_CONTENT" > "$1/.gitignore"
}

init_git_repo(){
  local dir="$1"
  pushd "$dir" >/dev/null
  git init -b main
  git lfs install
  for pat in "${LFS_PATTERNS[@]}"; do git lfs track "$pat"; done
  git add .gitattributes 2>/dev/null || true
  popd >/dev/null
}

first_commit(){
  local dir="$1" msg="$2"
  pushd "$dir" >/dev/null
  git add .
  git commit -m "$msg" || echo "  ⚠️  Nada para commitar"
  popd >/dev/null
}

create_repo_remote(){
  local reponame="$1" desc="$2"
  echo "  🔧 Criando repositório: $GITHUB_USER/$reponame"
  if gh repo view "$GITHUB_USER/$reponame" >/dev/null 2>&1; then
    echo "  ✅ Repositório já existe"
  else
    gh repo create "$GITHUB_USER/$reponame" --"$VISIBILIDADE" -y \
      --description "$desc" 2>&1 | tail -3
    echo "  ✅ Criado: https://github.com/$GITHUB_USER/$reponame"
  fi
}

push_main(){
  local dir="$1" reponame="$2"
  echo "  📤 Push para GitHub..."
  pushd "$dir" >/dev/null
  git remote add origin "https://github.com/$GITHUB_USER/$reponame.git" 2>/dev/null || true
  git push -u origin main 2>&1 | tail -5
  popd >/dev/null
}

checksum_dir(){
  local src="$1" out="$2"
  echo "  🔐 Gerando checksums..."
  find "$src" -type f -print0 | sort -z | xargs -0 sha256sum > "$out" 2>&1
  echo "  ✅ Checksums salvos: $(wc -l < "$out") arquivos"
}

pack_big_assets(){
  local work="$1" outdir="$2" reponame="$3"
  echo "  📦 Empacotando assets grandes..."
  mkdir -p "$outdir"
  local tarball="$outdir/${reponame}_assets.tar"
  tar -cf "$tarball" -C "$work" . 2>&1 | tail -3
  echo "  🗜️  Comprimindo com xz..."
  xz -T0 -9 "$tarball" 2>&1
  rm -f "$tarball"
  local final="${tarball}.xz"
  echo "  ✅ Asset empacotado: $(du -h "$final" | cut -f1)"
  echo "$final"
}

split_file(){
  local filepath="$1" chunks_dir="$2"
  echo "  ✂️  Dividindo em partes de $SPLIT_SIZE..."
  mkdir -p "$chunks_dir"
  split -b "$SPLIT_SIZE" -d -a 3 "$filepath" "$chunks_dir/part_"
  local num_parts=$(ls -1 "$chunks_dir"/part_* | wc -l)
  echo "  ✅ Dividido em $num_parts partes"
  echo "$chunks_dir"
}

create_release_and_upload(){
  local reponame="$1" tag="$2" title="$3" notes_file="$4" assets_dir="$5"
  echo "  🚀 Criando Release $tag..."
  
  local assets=()
  if [ -d "$assets_dir" ]; then
    while IFS= read -r -d '' file; do
      assets+=("$file")
    done < <(find "$assets_dir" -type f -print0)
  fi
  
  if [ ${#assets[@]} -gt 0 ]; then
    gh release create "$tag" "${assets[@]}" \
      --repo "$GITHUB_USER/$reponame" \
      --title "$title" \
      --notes-file "$notes_file" 2>&1 | tail -5
  else
    gh release create "$tag" \
      --repo "$GITHUB_USER/$reponame" \
      --title "$title" \
      --notes-file "$notes_file" 2>&1 | tail -5
  fi
  echo "  ✅ Release publicada!"
}

mk_readme(){
  local dir="$1" title="$2" desc="$3"
  cat > "$dir/README.md" <<EOF
# $title

$desc

## 📦 Conteúdo

- **Código principal:** Versionado diretamente neste repositório Git
- **Artefatos grandes:** Datasets, checkpoints, logs → publicados como assets na Release \`v1.0-initial-snapshot\`
- **CHECKSUMS.txt:** SHA-256 para verificação de integridade

## 🔧 Como recompor os assets divididos

Baixe todas as partes da Release e rode:

\`\`\`bash
# Baixar release
gh release download v1.0-initial-snapshot --repo $GITHUB_USER/${title// /-}

# Recompor arquivo
cat part_* > assets.tar.xz

# Verificar integridade
sha256sum -c CHECKSUMS.txt

# Extrair
tar -xf assets.tar.xz
\`\`\`

## 📊 Origem

Preservado do servidor de IA evolutiva em $(date +%Y-%m-%d).

Ver: [Auditoria Científica Brutal Final](https://github.com/$GITHUB_USER/auditoria-cientifica-brutal-final)
EOF
}

mk_release_notes(){
  local tmpfile="$1" title="$2"
  cat > "$tmpfile" <<EOF
# $title — Snapshot Inicial

Este release contém os artefatos grandes (divididos em partes) e o arquivo CHECKSUMS.txt
para verificação de integridade.

## 📥 Recomposição

\`\`\`bash
# Baixar todos os assets
gh release download v1.0-initial-snapshot

# Recompor
cat part_* > assets.tar.xz

# Verificar integridade
sha256sum -c CHECKSUMS.txt

# Extrair
tar -xf assets.tar.xz
\`\`\`

**Data:** $(date +%Y-%m-%d\ %H:%M:%S)
**Origem:** Servidor de IA Evolutiva - Auditoria Científica
EOF
}

# =====================[ VERIFICAÇÃO INICIAL ]=============

echo "🔍 Verificando autenticação GitHub..."
if ! gh auth status >/dev/null 2>&1; then
  die "Erro: gh CLI não está autenticado. Execute: gh auth login"
fi
echo "✅ Autenticado como: $(gh api user --jq .login)"
echo ""

# =====================[ PIPELINE ]========================

echo "🚀 Iniciando preservação de projetos..."
echo "📂 Usuário GitHub: $GITHUB_USER"
echo ""

WORKROOT="/root/_export_preservacao"
ensure_clean_tmp "$WORKROOT"

DATE_TAG="$(date +%Y%m%d-%H%M%S)"

for entry in "${PROJETOS[@]}"; do
  IFS="|" read -r REPO NAME DESC CODE_CSV BIG_CSV <<< "$entry"

  echo ""
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  echo "📦 Projeto: $NAME"
  echo "🔗 Repo:    $REPO"
  echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  PROJ_DIR="$WORKROOT/$REPO"
  CODE_DIR="$PROJ_DIR/code"
  BIG_DIR="$PROJ_DIR/big"

  mkdir -p "$CODE_DIR" "$BIG_DIR"

  # 1) Coletar código
  echo "📝 Coletando código..."
  IFS="," read -ra CODE_PATHS <<< "${CODE_CSV:-}"
  for p in "${CODE_PATHS[@]}"; do
    [ -n "$p" ] && copy_if_exists "/root/$p" "$CODE_DIR"
  done

  # 2) Coletar dados grandes
  echo "💾 Coletando dados grandes..."
  IFS="," read -ra BIG_PATHS <<< "${BIG_CSV:-}"
  for p in "${BIG_PATHS[@]}"; do
    [ -n "$p" ] && copy_if_exists "/root/$p" "$BIG_DIR"
  done

  # 3) Checksums
  if [ -d "$BIG_DIR" ] && [ -n "$(find "$BIG_DIR" -type f -print -quit)" ]; then
    checksum_dir "$BIG_DIR" "$PROJ_DIR/CHECKSUMS.txt"
  else
    echo "  ℹ️  Sem artefatos grandes para $NAME"
  fi

  # 4) Repositório local
  echo "🗂️  Montando repositório..."
  REPO_DIR="$PROJ_DIR/repo"
  mkdir -p "$REPO_DIR"
  cp -a "$CODE_DIR/." "$REPO_DIR/" 2>/dev/null || true
  [ -f "$PROJ_DIR/CHECKSUMS.txt" ] && cp "$PROJ_DIR/CHECKSUMS.txt" "$REPO_DIR/"

  write_gitignore "$REPO_DIR"
  mk_readme "$REPO_DIR" "$NAME" "$DESC"

  init_git_repo "$REPO_DIR"
  first_commit "$REPO_DIR" "feat: snapshot inicial ($DATE_TAG)"

  # 5) Criar repo + push
  create_repo_remote "$REPO" "$DESC"
  push_main "$REPO_DIR" "$REPO"

  # 6) Empacotar e dividir
  NOTES_FILE="$PROJ_DIR/RELEASE_NOTES.md"
  mk_release_notes "$NOTES_FILE" "$NAME"

  if [ -d "$BIG_DIR" ] && [ -n "$(find "$BIG_DIR" -type f -print -quit)" ]; then
    ASSET_TAR_XZ="$(pack_big_assets "$BIG_DIR" "$PROJ_DIR/assets" "$REPO")"
    CHUNKS_DIR="$PROJ_DIR/chunks"
    split_file "$ASSET_TAR_XZ" "$CHUNKS_DIR"

    # Copiar checksums para chunks
    [ -f "$PROJ_DIR/CHECKSUMS.txt" ] && cp "$PROJ_DIR/CHECKSUMS.txt" "$CHUNKS_DIR/"

    # 7) Release
    TAG="v1.0-initial-snapshot"
    create_release_and_upload "$REPO" "$TAG" "$NAME — Initial Snapshot" "$NOTES_FILE" "$CHUNKS_DIR"

    # 8) Tag no repo
    pushd "$REPO_DIR" >/dev/null
    git tag -a "$TAG" -m "Initial snapshot ($DATE_TAG)"
    git push origin "$TAG" 2>&1 | tail -3
    popd >/dev/null
  else
    echo "  ℹ️  Sem grandes para empacotar - criando Release apenas com código"
    TAG="v1.0-initial-snapshot"
    TMP_ASSETS_DIR="$PROJ_DIR/empty_assets"
    mkdir -p "$TMP_ASSETS_DIR"
    [ -f "$PROJ_DIR/CHECKSUMS.txt" ] && cp "$PROJ_DIR/CHECKSUMS.txt" "$TMP_ASSETS_DIR/"
    create_release_and_upload "$REPO" "$TAG" "$NAME — Initial Snapshot" "$NOTES_FILE" "$TMP_ASSETS_DIR"

    pushd "$REPO_DIR" >/dev/null
    git tag -a "$TAG" -m "Initial snapshot ($DATE_TAG)"
    git push origin "$TAG" 2>&1 | tail -3
    popd >/dev/null
  fi

  echo "✅ FINALIZADO: https://github.com/$GITHUB_USER/$REPO"
  echo ""
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 SUCESSO! Todos os projetos foram preservados."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📋 Repositórios criados:"
for entry in "${PROJETOS[@]}"; do
  IFS="|" read -r REPO NAME _ _ _ <<< "$entry"
  echo "  🔗 https://github.com/$GITHUB_USER/$REPO"
done
echo ""
echo "✨ Tudo preservado com integridade verificável (checksums SHA-256)"