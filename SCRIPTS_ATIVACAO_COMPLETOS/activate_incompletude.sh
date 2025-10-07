#!/bin/bash
# Ativa Incompletude Infinita globalmente

echo "🚀 Ativando Incompletude Infinita..."

# Exporta para todos os shells
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:$PYTHONPATH"
export INCOMPLETUDE_ACTIVE=1

# Função helper para rodar com incompletude
run_with_incompletude() {
    python3 -c "from incompletude_infinita import apply_incompletude" 2>/dev/null
    python3 "$@"
}

alias python3_ii="run_with_incompletude"

echo "✓ Incompletude ativada!"
echo "  Use: python3_ii seu_script.py"
