#!/bin/bash

echo "📥 CLONANDO TODOS OS REPOSITÓRIOS DE danielgonzagat"
echo "=================================================="

# Criar diretório para repositórios
REPO_DIR="/root/github_repos"
mkdir -p $REPO_DIR
cd $REPO_DIR

echo "📁 Diretório: $REPO_DIR"
echo ""

# Obter e clonar repositórios
curl -s "https://api.github.com/users/danielgonzagat/repos?per_page=100" | \
grep '"ssh_url"' | \
sed 's/.*"ssh_url": "\([^"]*\)".*/\1/' | \
while read repo; do
    repo_name=$(basename "$repo" .git)
    echo "📦 Clonando: $repo_name"
    
    if [ -d "$repo_name" ]; then
        echo "   ⚠️  Já existe, pulando..."
    else
        git clone "$repo" && echo "   ✅ Clonado com sucesso!"
    fi
    echo ""
done

echo "🎉 TODOS OS REPOSITÓRIOS CLONADOS!"
echo "📁 Localização: $REPO_DIR"
ls -la $REPO_DIR
