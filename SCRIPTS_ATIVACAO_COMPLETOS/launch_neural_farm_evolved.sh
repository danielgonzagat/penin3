#!/bin/bash
# Lançar Neural Farm 2.0 com super-neurônios evoluídos

echo "🚀 Lançando Neural Farm 2.0 com 300 Super-Neurônios"
echo "=================================================="

# Criar diretórios
mkdir -p /root/neural_farm_evolved_output

# Executar em modo teste primeiro
echo "🧪 Executando teste inicial..."
python3 /root/neural_farm_evolved.py \
    --mode test \
    --out-dir /root/neural_farm_evolved_output \
    --db-path /root/neural_farm_evolved.db \
    --seed 42

echo ""
echo "✅ Teste concluído!"
echo ""
echo "Para executar em modo contínuo, use:"
echo "python3 /root/neural_farm_evolved.py --mode run"
