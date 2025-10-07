#!/bin/bash

# GitHub Repository Manager
# Script para gerenciar acesso aos repositórios GitHub

set -e

# Configurações
GITHUB_USER="danielgonzagat"
WORKSPACE_DIR="/root/github_workspace"
REPOS_FILE="/root/github_repos.txt"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Função para log
log() {
    echo -e "${BLUE}[$(date +'%Y-%m-%d %H:%M:%S')]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Função para criar workspace
setup_workspace() {
    log "Configurando workspace GitHub..."
    mkdir -p "$WORKSPACE_DIR"
    cd "$WORKSPACE_DIR"
    success "Workspace criado em: $WORKSPACE_DIR"
}

# Função para listar repositórios via API
list_repositories() {
    log "Listando repositórios do usuário $GITHUB_USER..."
    
    # Tentar usar GitHub API (requer token se for privado)
    if command -v curl >/dev/null 2>&1; then
        log "Tentando acessar via GitHub API..."
        curl -s "https://api.github.com/users/$GITHUB_USER/repos?per_page=100" | \
        jq -r '.[].clone_url' 2>/dev/null > "$REPOS_FILE" || {
            warning "Não foi possível acessar via API. Usando método alternativo..."
            list_repositories_manual
        }
    else
        list_repositories_manual
    fi
    
    if [ -f "$REPOS_FILE" ] && [ -s "$REPOS_FILE" ]; then
        success "Repositórios encontrados:"
        cat "$REPOS_FILE" | while read repo; do
            echo "  - $repo"
        done
    else
        warning "Nenhum repositório encontrado via API"
        list_repositories_manual
    fi
}

# Função para listar repositórios manualmente (se API falhar)
list_repositories_manual() {
    log "Listando repositórios conhecidos..."
    cat > "$REPOS_FILE" << EOF
git@github.com:danielgonzagat/monte-carlo-tree-search.git
EOF
    success "Repositórios conhecidos adicionados"
}

# Função para clonar repositório específico
clone_repo() {
    local repo_url="$1"
    local repo_name=$(basename "$repo_url" .git)
    
    if [ -z "$repo_url" ]; then
        error "URL do repositório não fornecida"
        return 1
    fi
    
    log "Clonando repositório: $repo_name"
    
    if [ -d "$repo_name" ]; then
        warning "Repositório $repo_name já existe. Atualizando..."
        cd "$repo_name"
        git pull origin main || git pull origin master
        cd ..
    else
        git clone "$repo_url"
        success "Repositório $repo_name clonado com sucesso"
    fi
}

# Função para sincronizar todos os repositórios
sync_all() {
    log "Sincronizando todos os repositórios..."
    
    if [ ! -f "$REPOS_FILE" ]; then
        list_repositories
    fi
    
    while IFS= read -r repo_url; do
        if [ -n "$repo_url" ]; then
            clone_repo "$repo_url"
        fi
    done < "$REPOS_FILE"
    
    success "Sincronização concluída"
}

# Função para pesquisar em todos os repositórios
search_repos() {
    local search_term="$1"
    
    if [ -z "$search_term" ]; then
        error "Termo de pesquisa não fornecido"
        return 1
    fi
    
    log "Pesquisando '$search_term' em todos os repositórios..."
    
    find "$WORKSPACE_DIR" -name "*.py" -o -name "*.js" -o -name "*.ts" -o -name "*.java" -o -name "*.cpp" -o -name "*.c" -o -name "*.h" | \
    xargs grep -l "$search_term" 2>/dev/null | while read file; do
        echo "Arquivo: $file"
        grep -n "$search_term" "$file" | head -5
        echo "---"
    done
}

# Função para mostrar status dos repositórios
status_repos() {
    log "Status dos repositórios:"
    
    for dir in "$WORKSPACE_DIR"/*; do
        if [ -d "$dir" ] && [ -d "$dir/.git" ]; then
            repo_name=$(basename "$dir")
            echo "Repositório: $repo_name"
            cd "$dir"
            git status --porcelain
            echo "Último commit: $(git log -1 --format='%h %s %ad' --date=short)"
            echo "---"
        fi
    done
}

# Função para mostrar ajuda
show_help() {
    echo "GitHub Repository Manager"
    echo ""
    echo "Uso: $0 [COMANDO] [ARGUMENTOS]"
    echo ""
    echo "Comandos disponíveis:"
    echo "  setup          - Configurar workspace"
    echo "  list           - Listar repositórios"
    echo "  clone <url>    - Clonar repositório específico"
    echo "  sync           - Sincronizar todos os repositórios"
    echo "  search <term>  - Pesquisar termo em todos os repositórios"
    echo "  status         - Mostrar status dos repositórios"
    echo "  help           - Mostrar esta ajuda"
    echo ""
    echo "Exemplos:"
    echo "  $0 setup"
    echo "  $0 list"
    echo "  $0 clone git@github.com:user/repo.git"
    echo "  $0 sync"
    echo "  $0 search 'function'"
    echo "  $0 status"
}

# Main
main() {
    case "${1:-help}" in
        setup)
            setup_workspace
            ;;
        list)
            list_repositories
            ;;
        clone)
            clone_repo "$2"
            ;;
        sync)
            sync_all
            ;;
        search)
            search_repos "$2"
            ;;
        status)
            status_repos
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            error "Comando desconhecido: $1"
            show_help
            exit 1
            ;;
    esac
}

# Executar função principal
main "$@"