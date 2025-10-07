#!/bin/bash
# Script para reconstruir weights do UNIFIED_BRAIN

echo "Reconstruindo real_env_weights.pt..."
cat weights_part_* > real_env_weights.pt
echo "✅ real_env_weights.pt reconstruído (17MB)"

echo "Reconstruindo real_env_weights_v3.pt..."
cat weights_v3_part_* > real_env_weights_v3.pt
echo "✅ real_env_weights_v3.pt reconstruído (8.1MB)"

echo ""
echo "Weights reconstruídos com sucesso!"
echo "Para usar:"
echo "  cp real_env_weights.pt /path/to/UNIFIED_BRAIN/"
echo "  cp real_env_weights_v3.pt /path/to/UNIFIED_BRAIN/"