#!/bin/bash
# Script para limpar processos duplicados
echo "🧹 Limpando processos duplicados..."

# Matar conectores extras
pkill -f "EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py" || true
sleep 2

# Matar outros possíveis duplicados (adapte conforme necessário)
pkill -f "multiple_instances" || true

# Limpar locks órfãos
rm -f /tmp/emergence_connector.lock

echo "✅ Limpeza concluída."
