#!/bin/bash

# GitHub Intelligent Search
# Script para pesquisar e analisar repositórios GitHub sem clonar todos

set -e

# Configurações
GITHUB_USER="danielgonzagat"
CACHE_DIR="/root/github_cache"
SEARCH_RESULTS="/root/github_search_results.json"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
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

info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

# Função para criar cache
setup_cache() {
    log "Configurando cache GitHub..."
    mkdir -p "$CACHE_DIR"
    success "Cache criado em: $CACHE_DIR"
}

# Função para buscar repositórios via API
fetch_repositories() {
    local cache_file="$CACHE_DIR/repositories.json"
    
    if [ -f "$cache_file" ] && [ $(find "$cache_file" -mmin -60 | wc -l) -gt 0 ]; then
        info "Usando cache de repositórios (menos de 1 hora)"
        return 0
    fi
    
    log "Buscando repositórios via GitHub API..."
    
    # Buscar repositórios públicos
    curl -s "https://api.github.com/users/$GITHUB_USER/repos?per_page=100&sort=updated" > "$cache_file"
    
    if [ $? -eq 0 ] && [ -s "$cache_file" ]; then
        success "Repositórios carregados: $(jq length "$cache_file")"
    else
        error "Falha ao carregar repositórios"
        return 1
    fi
}

# Função para buscar conteúdo de arquivo específico
fetch_file_content() {
    local repo="$1"
    local file_path="$2"
    local cache_file="$CACHE_DIR/${repo//\//_}_${file_path//\//_}.json"
    
    if [ -f "$cache_file" ] && [ $(find "$cache_file" -mmin -30 | wc -l) -gt 0 ]; then
        cat "$cache_file"
        return 0
    fi
    
    log "Buscando conteúdo do arquivo: $repo/$file_path"
    
    curl -s "https://api.github.com/repos/$repo/contents/$file_path" > "$cache_file"
    
    if [ $? -eq 0 ] && [ -s "$cache_file" ]; then
        cat "$cache_file"
    else
        echo "{}"
    fi
}

# Função para buscar código em repositórios
search_code() {
    local search_term="$1"
    local language="${2:-}"
    local repo_filter="${3:-}"
    
    log "Pesquisando código: '$search_term'"
    
    # Usar GitHub Code Search API
    local search_url="https://api.github.com/search/code?q=$search_term+user:$GITHUB_USER"
    
    if [ -n "$language" ]; then
        search_url="${search_url}+language:$language"
    fi
    
    if [ -n "$repo_filter" ]; then
        search_url="${search_url}+repo:$GITHUB_USER/$repo_filter"
    fi
    
    log "URL de busca: $search_url"
    
    local results_file="$CACHE_DIR/search_$(echo "$search_term" | tr ' ' '_').json"
    curl -s "$search_url" > "$results_file"
    
    if [ $? -eq 0 ] && [ -s "$results_file" ]; then
        success "Resultados salvos em: $results_file"
        cat "$results_file"
    else
        error "Falha na busca"
        echo "{}"
    fi
}

# Função para analisar repositório específico
analyze_repository() {
    local repo_name="$1"
    local repo_info_file="$CACHE_DIR/repo_${repo_name}.json"
    
    log "Analisando repositório: $repo_name"
    
    # Buscar informações do repositório
    curl -s "https://api.github.com/repos/$GITHUB_USER/$repo_name" > "$repo_info_file"
    
    if [ ! -s "$repo_info_file" ]; then
        error "Repositório não encontrado: $repo_name"
        return 1
    fi
    
    # Extrair informações importantes
    local description=$(jq -r '.description // "Sem descrição"' "$repo_info_file")
    local language=$(jq -r '.language // "N/A"' "$repo_info_file")
    local stars=$(jq -r '.stargazers_count' "$repo_info_file")
    local forks=$(jq -r '.forks_count' "$repo_info_file")
    local updated=$(jq -r '.updated_at' "$repo_info_file")
    local size=$(jq -r '.size' "$repo_info_file")
    
    echo "=== REPOSITÓRIO: $repo_name ==="
    echo "Descrição: $description"
    echo "Linguagem: $language"
    echo "Stars: $stars | Forks: $forks"
    echo "Tamanho: $size KB"
    echo "Última atualização: $updated"
    echo ""
    
    # Buscar arquivos principais
    log "Buscando arquivos principais..."
    curl -s "https://api.github.com/repos/$GITHUB_USER/$repo_name/contents" | \
    jq -r '.[] | select(.type == "file") | .name' | head -10 | while read file; do
        echo "  - $file"
    done
    echo ""
}

# Função para recomendar melhor repositório
recommend_repository() {
    local search_term="$1"
    local language="${2:-}"
    
    log "Recomendando melhor repositório para: '$search_term'"
    
    # Buscar repositórios relevantes
    local repos_file="$CACHE_DIR/repositories.json"
    if [ ! -f "$repos_file" ]; then
        fetch_repositories
    fi
    
    # Filtrar por linguagem se especificada
    local filtered_repos="$repos_file"
    if [ -n "$language" ]; then
        jq "[.[] | select(.language == \"$language\")]" "$repos_file" > "$CACHE_DIR/filtered_repos.json"
        filtered_repos="$CACHE_DIR/filtered_repos.json"
    fi
    
    # Ordenar por relevância (stars + forks + recente)
    local recommendations=$(jq -r '.[] | 
        {
            name: .name,
            description: .description,
            language: .language,
            stars: .stargazers_count,
            forks: .forks_count,
            updated: .updated_at,
            score: (.stargazers_count + .forks_count + (now - (.updated_at | fromdateiso8601) / 86400))
        } | 
        select(.description | test("'$search_term'"; "i") or .name | test("'$search_term'"; "i")) |
        .score' "$filtered_repos" | sort -nr | head -5)
    
    if [ -n "$recommendations" ]; then
        success "Top 5 repositórios recomendados:"
        echo "$recommendations"
    else
        warning "Nenhum repositório encontrado para '$search_term'"
    fi
}

# Função para mostrar estatísticas
show_stats() {
    log "Estatísticas dos repositórios:"
    
    local repos_file="$CACHE_DIR/repositories.json"
    if [ ! -f "$repos_file" ]; then
        fetch_repositories
    fi
    
    local total_repos=$(jq length "$repos_file")
    local total_stars=$(jq '[.[].stargazers_count] | add' "$repos_file")
    local total_forks=$(jq '[.[].forks_count] | add' "$repos_file")
    local languages=$(jq -r '.[].language // "Unknown" | select(. != null)' "$repos_file" | sort | uniq -c | sort -nr | head -10)
    
    echo "Total de repositórios: $total_repos"
    echo "Total de stars: $total_stars"
    echo "Total de forks: $total_forks"
    echo ""
    echo "Top linguagens:"
    echo "$languages"
}

# Função para mostrar ajuda
show_help() {
    echo "GitHub Intelligent Search"
    echo ""
    echo "Uso: $0 [COMANDO] [ARGUMENTOS]"
    echo ""
    echo "Comandos disponíveis:"
    echo "  setup                    - Configurar cache"
    echo "  search <term> [lang]      - Pesquisar código"
    echo "  analyze <repo>           - Analisar repositório específico"
    echo "  recommend <term> [lang]   - Recomendar melhor repositório"
    echo "  stats                    - Mostrar estatísticas"
    echo "  list [lang]              - Listar repositórios"
    echo "  help                     - Mostrar esta ajuda"
    echo ""
    echo "Exemplos:"
    echo "  $0 setup"
    echo "  $0 search 'machine learning'"
    echo "  $0 search 'neural network' python"
    echo "  $0 analyze monte-carlo-tree-search"
    echo "  $0 recommend 'AI agent' python"
    echo "  $0 stats"
    echo "  $0 list python"
}

# Função para listar repositórios
list_repositories() {
    local language="${1:-}"
    
    local repos_file="$CACHE_DIR/repositories.json"
    if [ ! -f "$repos_file" ]; then
        fetch_repositories
    fi
    
    if [ -n "$language" ]; then
        log "Listando repositórios em $language:"
        jq -r '.[] | select(.language == "'$language'") | "\(.name) - \(.description // "Sem descrição") (\(.stargazers_count) stars)"' "$repos_file"
    else
        log "Listando todos os repositórios:"
        jq -r '.[] | "\(.name) - \(.description // "Sem descrição") (\(.language // "N/A") - \(.stargazers_count) stars)"' "$repos_file" | head -20
    fi
}

# Main
main() {
    case "${1:-help}" in
        setup)
            setup_cache
            fetch_repositories
            ;;
        search)
            search_code "$2" "$3"
            ;;
        analyze)
            analyze_repository "$2"
            ;;
        recommend)
            recommend_repository "$2" "$3"
            ;;
        stats)
            show_stats
            ;;
        list)
            list_repositories "$2"
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