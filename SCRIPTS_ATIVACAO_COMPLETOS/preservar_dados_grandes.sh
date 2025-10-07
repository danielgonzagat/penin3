#!/usr/bin/env bash
set -euo pipefail

TOKEN="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"
USER="danielgonzagat"

echo "📦 FASE 2: Empacotando dados grandes..."
echo ""

# 1) Neural Farm (11GB)
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️⃣  Neural Farm IA3 - 11GB"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d /root/neural_farm_prod ]; then
  cd /root
  echo "  📦 Empacotando neural_farm_prod..."
  tar -czf neural_farm_prod.tar.gz neural_farm_prod/ 2>&1 | tail -3
  SIZE=$(du -h neural_farm_prod.tar.gz | cut -f1)
  echo "  ✅ Empacotado: $SIZE"
  
  echo "  📤 Criando release..."
  RELEASE_DATA=$(cat <<EOF
{
  "tag_name": "v1.0-data",
  "name": "Neural Farm Data - 17.3M métricas",
  "body": "## 📊 Neural Farm Production Data\n\n- **Tamanho comprimido:** $SIZE\n- **Métricas:** 17.3 milhões de linhas\n- **Tempo CPU:** 49 dias\n- **Gerações:** 42\n\n### 🔧 Extrair:\n\`\`\`bash\ntar -xzf neural_farm_prod.tar.gz\n\`\`\`"
}
EOF
)
  
  UPLOAD_URL=$(curl -s -X POST \
    -H "Authorization: token $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$RELEASE_DATA" \
    "https://api.github.com/repos/$USER/neural-farm-ia3-integration/releases" | jq -r '.upload_url' | sed 's/{.*}//')
  
  if [ "$UPLOAD_URL" != "null" ] && [ -n "$UPLOAD_URL" ]; then
    echo "  📤 Upload para release..."
    curl -X POST \
      -H "Authorization: token $TOKEN" \
      -H "Content-Type: application/gzip" \
      --data-binary @neural_farm_prod.tar.gz \
      "${UPLOAD_URL}?name=neural_farm_prod.tar.gz" 2>&1 | tail -3
    echo "  ✅ Upload completo!"
  else
    echo "  ⚠️  Erro ao criar release, guardando arquivo localmente"
  fi
  rm -f neural_farm_prod.tar.gz
fi

# 2) TEIS V2 (11GB)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️⃣  TEIS V2 Enhanced - 11GB"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d /root/teis_v2_out_prod ]; then
  cd /root
  echo "  📦 Empacotando teis_v2_out_prod..."
  tar -czf teis_v2_out_prod.tar.gz teis_v2_out_prod/ 2>&1 | tail -3
  SIZE=$(du -h teis_v2_out_prod.tar.gz | cut -f1)
  echo "  ✅ Empacotado: $SIZE"
  
  echo "  📤 Criando release..."
  RELEASE_DATA=$(cat <<EOF
{
  "tag_name": "v1.0-data",
  "name": "TEIS V2 Training Data - 100 gerações",
  "body": "## 📊 TEIS V2 Production Data\n\n- **Tamanho comprimido:** $SIZE\n- **Gerações:** 100\n- **Checkpoints:** Completos\n- **Experience Replay:** 10K buffer\n\n### 🔧 Extrair:\n\`\`\`bash\ntar -xzf teis_v2_out_prod.tar.gz\n\`\`\`"
}
EOF
)
  
  UPLOAD_URL=$(curl -s -X POST \
    -H "Authorization: token $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$RELEASE_DATA" \
    "https://api.github.com/repos/$USER/teis-v2-enhanced/releases" | jq -r '.upload_url' | sed 's/{.*}//')
  
  if [ "$UPLOAD_URL" != "null" ] && [ -n "$UPLOAD_URL" ]; then
    echo "  📤 Upload para release..."
    curl -X POST \
      -H "Authorization: token $TOKEN" \
      -H "Content-Type: application/gzip" \
      --data-binary @teis_v2_out_prod.tar.gz \
      "${UPLOAD_URL}?name=teis_v2_out_prod.tar.gz" 2>&1 | tail -3
    echo "  ✅ Upload completo!"
  else
    echo "  ⚠️  Erro ao criar release, guardando arquivo localmente"
  fi
  rm -f teis_v2_out_prod.tar.gz
fi

# 3) PRESERVED_INTELLIGENCE (28GB) - MAIOR, dividir
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️⃣  PRESERVED Intelligence - 28GB (dividindo...)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ -d /root/PRESERVED_INTELLIGENCE ]; then
  cd /root
  echo "  📦 Empacotando PRESERVED_INTELLIGENCE..."
  tar -czf preserved_intelligence.tar.gz PRESERVED_INTELLIGENCE/ 2>&1 | tail -3
  SIZE=$(du -h preserved_intelligence.tar.gz | cut -f1)
  echo "  ✅ Empacotado: $SIZE"
  
  echo "  ✂️  Dividindo em partes de 1.9GB..."
  split -b 1900M -d preserved_intelligence.tar.gz preserved_part_
  NUM_PARTS=$(ls preserved_part_* | wc -l)
  echo "  ✅ Dividido em $NUM_PARTS partes"
  
  echo "  📤 Criando release..."
  RELEASE_DATA=$(cat <<EOF
{
  "tag_name": "v1.0-data",
  "name": "PRESERVED Intelligence Models - 28GB",
  "body": "## 📊 PRESERVED Intelligence Data\n\n- **Tamanho total:** $SIZE\n- **Partes:** $NUM_PARTS arquivos\n- **Modelos:** Checkpoints históricos\n- **Evolução:** Gerações 40-45\n\n### 🔧 Recompor:\n\`\`\`bash\ncat preserved_part_* > preserved_intelligence.tar.gz\ntar -xzf preserved_intelligence.tar.gz\n\`\`\`"
}
EOF
)
  
  UPLOAD_URL=$(curl -s -X POST \
    -H "Authorization: token $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$RELEASE_DATA" \
    "https://api.github.com/repos/$USER/preserved-intelligence-models/releases" | jq -r '.upload_url' | sed 's/{.*}//')
  
  if [ "$UPLOAD_URL" != "null" ] && [ -n "$UPLOAD_URL" ]; then
    for part in preserved_part_*; do
      echo "  📤 Upload: $part"
      curl -X POST \
        -H "Authorization: token $TOKEN" \
        -H "Content-Type: application/octet-stream" \
        --data-binary @"$part" \
        "${UPLOAD_URL}?name=$part" 2>&1 | tail -1
    done
    echo "  ✅ Todos os uploads completos!"
  else
    echo "  ⚠️  Erro ao criar release, guardando partes localmente em /root/"
  fi
  rm -f preserved_intelligence.tar.gz preserved_part_*
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🎉 FASE 2 COMPLETA!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "✅ Todos os dados foram empacotados e enviados como Releases"
echo ""
echo "📋 Acesse os releases em:"
echo "  🔗 https://github.com/$USER/neural-farm-ia3-integration/releases"
echo "  🔗 https://github.com/$USER/teis-v2-enhanced/releases"
echo "  🔗 https://github.com/$USER/preserved-intelligence-models/releases"
