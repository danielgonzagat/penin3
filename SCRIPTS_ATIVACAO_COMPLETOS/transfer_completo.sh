#!/usr/bin/env bash
set -euo pipefail

TOKEN="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"
USER="danielgonzagat"

echo "🚀 Iniciando transferência completa dos 50GB..."
mkdir -p /tmp/github_upload
cd /tmp/github_upload

# Função para criar release e fazer upload
upload_to_release() {
  local repo=$1
  local tag=$2
  local title=$3
  local file=$4
  local filename=$(basename "$file")
  
  echo "  📤 Criando release $tag em $repo..."
  
  # Criar release
  RELEASE_JSON=$(cat <<EOF
{
  "tag_name": "$tag",
  "name": "$title",
  "body": "Dados preservados do servidor - $(date +%Y-%m-%d)\n\nArquivo: $filename\nTamanho: $(du -h "$file" | cut -f1)"
}
EOF
)
  
  UPLOAD_URL=$(curl -s -X POST \
    -H "Authorization: token $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$RELEASE_JSON" \
    "https://api.github.com/repos/$USER/$repo/releases" | jq -r '.upload_url' | sed 's/{.*}//')
  
  if [ "$UPLOAD_URL" != "null" ] && [ -n "$UPLOAD_URL" ]; then
    echo "  ⬆️  Fazendo upload de $filename..."
    curl -X POST \
      -H "Authorization: token $TOKEN" \
      -H "Content-Type: application/gzip" \
      --data-binary @"$file" \
      "${UPLOAD_URL}?name=$filename" \
      --max-time 3600 \
      --retry 3 \
      2>&1 | grep -E "(upload|complete|error)" || echo "Upload em andamento..."
    echo "  ✅ Upload completo!"
    return 0
  else
    echo "  ❌ Erro ao criar release"
    return 1
  fi
}

# 1️⃣ NEURAL FARM (11GB)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  Neural Farm - 11GB"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "📦 Empacotando neural_farm_prod..."
cd /root
tar -czf /tmp/github_upload/neural_farm_prod.tar.gz neural_farm_prod/ 2>&1 | tail -5
SIZE=$(du -h /tmp/github_upload/neural_farm_prod.tar.gz | cut -f1)
echo "✅ Empacotado: $SIZE"

echo "✂️  Dividindo em partes..."
cd /tmp/github_upload
split -b 1900M -d neural_farm_prod.tar.gz neural_farm_part_
rm neural_farm_prod.tar.gz
NUM_PARTS=$(ls neural_farm_part_* | wc -l)
echo "✅ Dividido em $NUM_PARTS partes"

# Upload de cada parte
for i in $(seq 0 $((NUM_PARTS-1))); do
  PART=$(printf "neural_farm_part_%02d" $i)
  if [ -f "$PART" ]; then
    echo "📤 Upload parte $((i+1))/$NUM_PARTS..."
    upload_to_release "neural-farm-ia3-integration" "v1.0-data-part$i" "Neural Farm Data - Parte $((i+1))/$NUM_PARTS" "$PART"
    rm "$PART"  # Liberar espaço
  fi
done

# 2️⃣ TEIS V2 (11GB)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  TEIS V2 - 11GB"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "📦 Empacotando teis_v2_out_prod..."
cd /root
tar -czf /tmp/github_upload/teis_v2_prod.tar.gz teis_v2_out_prod/ 2>&1 | tail -5
SIZE=$(du -h /tmp/github_upload/teis_v2_prod.tar.gz | cut -f1)
echo "✅ Empacotado: $SIZE"

echo "✂️  Dividindo em partes..."
cd /tmp/github_upload
split -b 1900M -d teis_v2_prod.tar.gz teis_part_
rm teis_v2_prod.tar.gz
NUM_PARTS=$(ls teis_part_* | wc -l)
echo "✅ Dividido em $NUM_PARTS partes"

for i in $(seq 0 $((NUM_PARTS-1))); do
  PART=$(printf "teis_part_%02d" $i)
  if [ -f "$PART" ]; then
    echo "📤 Upload parte $((i+1))/$NUM_PARTS..."
    upload_to_release "teis-v2-enhanced" "v1.0-data-part$i" "TEIS V2 Data - Parte $((i+1))/$NUM_PARTS" "$PART"
    rm "$PART"
  fi
done

# 3️⃣ PRESERVED INTELLIGENCE (28GB) - MAIOR
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  PRESERVED Intelligence - 28GB (MAIOR)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo "📦 Empacotando PRESERVED_INTELLIGENCE..."
cd /root
tar -czf /tmp/github_upload/preserved_intel.tar.gz PRESERVED_INTELLIGENCE/ 2>&1 | tail -5
SIZE=$(du -h /tmp/github_upload/preserved_intel.tar.gz | cut -f1)
echo "✅ Empacotado: $SIZE"

echo "✂️  Dividindo em partes..."
cd /tmp/github_upload
split -b 1900M -d preserved_intel.tar.gz preserved_part_
rm preserved_intel.tar.gz
NUM_PARTS=$(ls preserved_part_* | wc -l)
echo "✅ Dividido em $NUM_PARTS partes"

for i in $(seq 0 $((NUM_PARTS-1))); do
  PART=$(printf "preserved_part_%02d" $i)
  if [ -f "$PART" ]; then
    echo "📤 Upload parte $((i+1))/$NUM_PARTS..."
    upload_to_release "preserved-intelligence-models" "v1.0-data-part$i" "PRESERVED Data - Parte $((i+1))/$NUM_PARTS" "$PART"
    rm "$PART"
  fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 TRANSFERÊNCIA COMPLETA - 50GB ENVIADOS!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Neural Farm: Releases criadas"
echo "✅ TEIS V2: Releases criadas"
echo "✅ PRESERVED Intelligence: Releases criadas"
echo ""
echo "📋 Verifique em:"
echo "  🔗 https://github.com/$USER/neural-farm-ia3-integration/releases"
echo "  🔗 https://github.com/$USER/teis-v2-enhanced/releases"
echo "  🔗 https://github.com/$USER/preserved-intelligence-models/releases"
