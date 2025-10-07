#!/bin/bash

# GitHub AI Assistant
# Script principal para interação inteligente com repositórios GitHub

set -e

# Configurações
GITHUB_USER="danielgonzagat"
WORKSPACE_DIR="/root/github_workspace"
CACHE_DIR="/root/github_cache"

# Cores para output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
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

highlight() {
    echo -e "${MAGENTA}[HIGHLIGHT]${NC} $1"
}

# Função para mostrar banner
show_banner() {
    echo -e "${CYAN}"
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    GitHub AI Assistant                       ║"
    echo "║                                                              ║"
    echo "║  🤖 Sistema inteligente para análise de repositórios GitHub ║"
    echo "║  📊 Pesquisa semântica e recomendações automáticas         ║"
    echo "║  🔍 Acesso direto sem necessidade de clonar tudo          ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

# Função para análise inteligente de consulta
analyze_query() {
    local query="$1"
    
    log "Analisando consulta: '$query'"
    
    # Detectar tipo de consulta
    if [[ "$query" =~ (melhor|best|recomend|sugest) ]]; then
        echo "recommendation"
    elif [[ "$query" =~ (analisar|analyze|examinar) ]]; then
        echo "analyze"
    elif [[ "$query" =~ \b(buscar|search|find|encontrar)\b ]]; then
        echo "search"
    elif [[ "$query" =~ (listar|list|mostrar|show) ]]; then
        echo "list"
    elif [[ "$query" =~ (estatística|stats|estatisticas) ]]; then
        echo "stats"
    else
        echo "general"
    fi
}

# Função para extrair termos de busca
extract_search_terms() {
    local query="$1"
    
    # Remover palavras de comando e extrair termos relevantes
    echo "$query" | sed -E 's/(melhor|best|recomend|sugest|buscar|search|find|encontrar|analisar|analyze|examinar|listar|list|mostrar|show|estatística|stats|estatisticas|para|for|em|in|com|with|de|of|da|do|das|dos)//g' | tr -s ' ' | sed 's/^[[:space:]]*//;s/[[:space:]]*$//'
}

# Função para detectar linguagem
detect_language() {
    local query="$1"
    
    if [[ "$query" =~ (python|py) ]]; then
        echo "python"
    elif [[ "$query" =~ (javascript|js|node) ]]; then
        echo "javascript"
    elif [[ "$query" =~ (java) ]]; then
        echo "java"
    elif [[ "$query" =~ \b(cpp|c\+\+|c\+\+)\b ]]; then
        echo "cpp"
    elif [[ "$query" =~ (go|golang) ]]; then
        echo "go"
    elif [[ "$query" =~ (rust) ]]; then
        echo "rust"
    elif [[ "$query" =~ (typescript|ts) ]]; then
        echo "typescript"
    else
        echo ""
    fi
}

# Função para processar consulta inteligente
process_intelligent_query() {
    local query="$1"
    local query_type=$(analyze_query "$query")
    local search_terms=$(extract_search_terms "$query")
    local language=$(detect_language "$query")
    
    highlight "Tipo de consulta detectado: $query_type"
    highlight "Termos de busca: '$search_terms'"
    if [ -n "$language" ]; then
        highlight "Linguagem detectada: $language"
    fi
    
    case "$query_type" in
        "recommendation")
            info "🔍 Buscando melhores repositórios para: $search_terms"
            /root/github_search.sh recommend "$search_terms" "$language"
            ;;
        "search")
            info "🔍 Pesquisando código: $search_terms"
            /root/github_search.sh search "$search_terms" "$language"
            ;;
        "analyze")
            info "📊 Analisando repositório: $search_terms"
            /root/github_search.sh analyze "$search_terms"
            ;;
        "list")
            info "📋 Listando repositórios"
            if [ -n "$language" ]; then
                /root/github_search.sh list "$language"
            else
                /root/github_search.sh list
            fi
            ;;
        "stats")
            info "📈 Mostrando estatísticas"
            /root/github_search.sh stats
            ;;
        "general")
            info "🤖 Processando consulta geral..."
            # Tentar busca geral primeiro
            /root/github_search.sh search "$search_terms" "$language"
            echo ""
            info "💡 Sugestões adicionais:"
            echo "  - Use 'melhor repositório para X' para recomendações"
            echo "  - Use 'analisar repositório Y' para análise detalhada"
            echo "  - Use 'listar repositórios em python' para filtrar por linguagem"
            ;;
    esac
}

# Função para modo interativo
interactive_mode() {
    show_banner
    
    success "Modo interativo ativado!"
    info "Digite suas consultas em linguagem natural"
    info "Exemplos:"
    echo "  - 'melhor repositório para machine learning'"
    echo "  - 'buscar algoritmos de otimização em python'"
    echo "  - 'analisar repositório monte-carlo-tree-search'"
    echo "  - 'listar repositórios em javascript'"
    echo "  - 'estatísticas dos meus repositórios'"
    echo ""
    info "Digite 'sair' ou 'quit' para encerrar"
    echo ""
    
    while true; do
        echo -n -e "${CYAN}🤖 GitHub AI> ${NC}"
        read -r query
        
        if [[ "$query" =~ ^(sair|quit|exit|q)$ ]]; then
            success "Até logo! 👋"
            break
        fi
        
        if [ -n "$query" ]; then
            echo ""
            process_intelligent_query "$query"
            echo ""
        fi
    done
}

# Função para mostrar ajuda
show_help() {
    show_banner
    echo ""
    echo "Uso: $0 [COMANDO] [ARGUMENTOS]"
    echo ""
    echo "Comandos disponíveis:"
    echo "  interactive              - Modo interativo com IA"
    echo "  query '<consulta>'       - Processar consulta específica"
    echo "  setup                    - Configurar sistema"
    echo "  stats                    - Estatísticas dos repositórios"
    echo "  help                     - Mostrar esta ajuda"
    echo ""
    echo "Exemplos de consultas:"
    echo "  $0 query 'melhor repositório para AI'"
    echo "  $0 query 'buscar machine learning em python'"
    echo "  $0 query 'analisar repositório monte-carlo-tree-search'"
    echo "  $0 query 'listar repositórios em javascript'"
    echo ""
    echo "Modo interativo:"
    echo "  $0 interactive"
    echo ""
    echo "Configuração inicial:"
    echo "  $0 setup"
}

# Função para configurar sistema
setup_system() {
    log "Configurando GitHub AI Assistant..."
    
    # Configurar workspace
    /root/github_manager.sh setup
    
    # Configurar cache
    /root/github_search.sh setup
    
    success "Sistema configurado com sucesso!"
    info "Agora você pode usar:"
    echo "  - Modo interativo: $0 interactive"
    echo "  - Consultas diretas: $0 query 'sua consulta'"
}

# Main
main() {
    case "${1:-help}" in
        interactive)
            interactive_mode
            ;;
        query)
            if [ -z "$2" ]; then
                error "Consulta não fornecida"
                echo "Uso: $0 query '<sua consulta>'"
                exit 1
            fi
            process_intelligent_query "$2"
            ;;
        setup)
            setup_system
            ;;
        stats)
            /root/github_search.sh stats
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