#!/bin/bash
# Script para executar o Cursor Agent
# Uso: ./cursor-agent.sh [prompt...]

CURSOR_PATH="/root/.cursor-server/bin/3ccce8f55d8cca49f6d28b491a844c699b8719a0/bin/remote-cli/cursor"

if [ ! -f "$CURSOR_PATH" ]; then
    echo "Erro: Cursor não encontrado em $CURSOR_PATH"
    echo "Verifique se o Cursor está instalado corretamente."
    exit 1
fi

# Executa o Cursor Agent com os argumentos passados
exec "$CURSOR_PATH" agent "$@"
