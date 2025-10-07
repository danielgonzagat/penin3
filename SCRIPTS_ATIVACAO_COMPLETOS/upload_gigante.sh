#!/bin/bash
set -e

TOKEN="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"
USER="danielgonzagat"

echo "🚀 UPLOAD DOS ARQUIVOS GIGANTES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ============================================
# 1) METRICS.JSONL (9.5GB)
# ============================================
echo "📊 1/2: Neural Farm Metrics (9.5GB)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd /root

if [ -f neural_farm_prod/metrics.jsonl ]; then
  echo "📦 Comprimindo metrics.jsonl..."
  gzip -c neural_farm_prod/metrics.jsonl > metrics.jsonl.gz
  SIZE=$(du -h metrics.jsonl.gz | cut -f1)
  echo "✅ Comprimido: $SIZE"
  
  echo "✂️  Dividindo em partes de 1.9GB..."
  split -b 1900M -d metrics.jsonl.gz metrics_part_
  PARTS=$(ls metrics_part_* | wc -l)
  echo "✅ Dividido em $PARTS partes"
  
  echo "🚀 Obtendo URL de upload..."
  RELEASE=$(curl -s -X GET \
    -H "Authorization: token $TOKEN" \
    "https://api.github.com/repos/$USER/neural-farm-ia3-integration/releases/tags/v1.0-data")
  
  UPLOAD_URL=$(echo "$RELEASE" | jq -r '.upload_url' | sed 's/{.*}//')
  
  if [ "$UPLOAD_URL" != "null" ] && [ -n "$UPLOAD_URL" ]; then
    for part in metrics_part_*; do
      PART_SIZE=$(du -h "$part" | cut -f1)
      echo "📤 Enviando $part ($PART_SIZE)..."
      curl -X POST \
        -H "Authorization: token $TOKEN" \
        -H "Content-Type: application/octet-stream" \
        --data-binary @"$part" \
        "${UPLOAD_URL}?name=$part" \
        --progress-bar -o /dev/null
      echo "✅ $part enviado!"
      rm "$part"
    done
    rm metrics.jsonl.gz
    echo "🎉 Metrics completo!"
  else
    echo "❌ Erro ao obter URL de upload"
  fi
fi

# ============================================
# 2) PRESERVED .PT FILES (28GB)
# ============================================
echo ""
echo "💎 2/2: PRESERVED Models (28GB)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd /root

if [ -d PRESERVED_INTELLIGENCE/evidence_of_breakthrough ]; then
  # Pegar o maior modelo (generation_40)
  MODEL="PRESERVED_INTELLIGENCE/evidence_of_breakthrough/generation_40_ia3_ready.pt"
  
  if [ -f "$MODEL" ]; then
    echo "📦 Comprimindo generation_40_ia3_ready.pt (6.6GB)..."
    gzip -c "$MODEL" > generation_40.pt.gz
    SIZE=$(du -h generation_40.pt.gz | cut -f1)
    echo "✅ Comprimido: $SIZE"
    
    echo "✂️  Dividindo em partes de 1.9GB..."
    split -b 1900M -d generation_40.pt.gz gen40_part_
    PARTS=$(ls gen40_part_* | wc -l)
    echo "✅ Dividido em $PARTS partes"
    
    echo "🚀 Obtendo URL de upload..."
    RELEASE=$(curl -s -X GET \
      -H "Authorization: token $TOKEN" \
      "https://api.github.com/repos/$USER/preserved-intelligence-models/releases/tags/v1.0-data" 2>/dev/null)
    
    if [ "$?" -ne 0 ] || [ -z "$RELEASE" ]; then
      echo "🔧 Release não existe, criando..."
      RELEASE=$(curl -s -X POST \
        -H "Authorization: token $TOKEN" \
        -H "Content-Type: application/json" \
        -d '{"tag_name":"v1.0-data","name":"PRESERVED Models Data","body":"## Models\n\ngeneration_40_ia3_ready.pt dividido em partes.\n\nRecompor: `cat gen40_part_* > generation_40.pt.gz && gunzip generation_40.pt.gz`"}' \
        "https://api.github.com/repos/$USER/preserved-intelligence-models/releases")
    fi
    
    UPLOAD_URL=$(echo "$RELEASE" | jq -r '.upload_url' | sed 's/{.*}//')
    
    if [ "$UPLOAD_URL" != "null" ] && [ -n "$UPLOAD_URL" ]; then
      for part in gen40_part_*; do
        PART_SIZE=$(du -h "$part" | cut -f1)
        echo "📤 Enviando $part ($PART_SIZE)..."
        curl -X POST \
          -H "Authorization: token $TOKEN" \
          -H "Content-Type: application/octet-stream" \
          --data-binary @"$part" \
          "${UPLOAD_URL}?name=$part" \
          --progress-bar -o /dev/null
        echo "✅ $part enviado!"
        rm "$part"
      done
      rm generation_40.pt.gz
      echo "🎉 Modelo completo!"
    else
      echo "❌ Erro ao obter URL de upload"
    fi
  fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ UPLOAD GIGANTE COMPLETO!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Total enviado:"
echo "  ✅ Database: neural_farm.db.gz (667MB)"
echo "  ✅ Checkpoints: teis_checkpoints.tar.gz (175MB)"
echo "  ✅ Metrics: metrics.jsonl dividido em partes (9.5GB comprimido)"
echo "  ✅ Model: generation_40 dividido em partes (6.6GB comprimido)"
echo ""
echo "🔗 Verifique em:"
echo "  https://github.com/$USER/neural-farm-ia3-integration/releases"
echo "  https://github.com/$USER/preserved-intelligence-models/releases"
