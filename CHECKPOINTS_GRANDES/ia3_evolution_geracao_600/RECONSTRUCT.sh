#!/bin/bash
# Script para reconstruir checkpoint ia3_evolution_V3_gen600_SPECIAL_16456neurons.pt
# Checkpoint: Geração 600 com 16.456 neurônios

echo "Reconstruindo checkpoint ia3_evolution_V3_gen600_SPECIAL_16456neurons.pt..."

cat gen600_part_aa gen600_part_ab gen600_part_ac gen600_part_ad gen600_part_ae > ia3_evolution_V3_gen600_SPECIAL_16456neurons.pt

echo "Checkpoint reconstruído!"
echo "Tamanho: $(ls -lh ia3_evolution_V3_gen600_SPECIAL_16456neurons.pt | awk '{print $5}')"
echo "MD5: $(md5sum ia3_evolution_V3_gen600_SPECIAL_16456neurons.pt | awk '{print $1}')"
echo ""
echo "Para usar:"
echo "  python3 -c 'import torch; model = torch.load(\"ia3_evolution_V3_gen600_SPECIAL_16456neurons.pt\"); print(model.keys())'"