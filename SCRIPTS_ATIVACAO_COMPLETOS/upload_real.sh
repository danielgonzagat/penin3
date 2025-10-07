#!/bin/bash
set -e

TOKEN="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"
USER="danielgonzagat"

echo "🚀 Iniciando upload REAL dos dados..."
echo ""

# 1) Neural Farm Database (1.4GB) - MENOR, começa por ele
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1/3: Neural Farm Database (1.4GB)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd /root
if [ -f neural_farm_prod/neural_farm.db ]; then
  echo "📦 Comprimindo database..."
  gzip -c neural_farm_prod/neural_farm.db > neural_farm.db.gz
  SIZE=$(du -h neural_farm.db.gz | cut -f1)
  echo "✅ Comprimido: $SIZE"
  
  echo "🚀 Criando release..."
  RELEASE=$(curl -s -X POST \
    -H "Authorization: token $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"tag_name":"v1.0-data","name":"Neural Farm Data","body":"## Database\n\nneural_farm.db.gz (comprimido)\n\nDescomprimir: `gunzip neural_farm.db.gz`"}' \
    "https://api.github.com/repos/$USER/neural-farm-ia3-integration/releases")
  
  UPLOAD_URL=$(echo "$RELEASE" | jq -r '.upload_url' | sed 's/{.*}//')
  
  if [ "$UPLOAD_URL" != "null" ] && [ -n "$UPLOAD_URL" ]; then
    echo "📤 Upload database..."
    curl -X POST \
      -H "Authorization: token $TOKEN" \
      -H "Content-Type: application/gzip" \
      --data-binary @neural_farm.db.gz \
      "${UPLOAD_URL}?name=neural_farm.db.gz" \
      --progress-bar -o /dev/null
    echo "✅ Database enviada!"
    rm neural_farm.db.gz
  fi
fi

# 2) PRESERVED JSONs e stats (já foram)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2/3: PRESERVED stats (já enviados)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ JSONs já estão no GitHub"

# 3) TEIS checkpoints (comprimir os mais importantes)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3/3: TEIS checkpoints (principais)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd /root
if [ -d teis_checkpoints ]; then
  echo "📦 Comprimindo checkpoints TEIS..."
  tar -czf teis_checkpoints.tar.gz teis_checkpoints/
  SIZE=$(du -h teis_checkpoints.tar.gz | cut -f1)
  echo "✅ Comprimido: $SIZE"
  
  echo "🚀 Criando release..."
  RELEASE=$(curl -s -X POST \
    -H "Authorization: token $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"tag_name":"v1.0-checkpoints","name":"TEIS Checkpoints","body":"## Checkpoints\n\nteis_checkpoints.tar.gz\n\nExtrair: `tar -xzf teis_checkpoints.tar.gz`"}' \
    "https://api.github.com/repos/$USER/teis-v2-enhanced/releases")
  
  UPLOAD_URL=$(echo "$RELEASE" | jq -r '.upload_url' | sed 's/{.*}//')
  
  if [ "$UPLOAD_URL" != "null" ] && [ -n "$UPLOAD_URL" ]; then
    echo "📤 Upload checkpoints..."
    curl -X POST \
      -H "Authorization: token $TOKEN" \
      -H "Content-Type: application/gzip" \
      --data-binary @teis_checkpoints.tar.gz \
      "${UPLOAD_URL}?name=teis_checkpoints.tar.gz" \
      --progress-bar -o /dev/null
    echo "✅ Checkpoints enviados!"
    rm teis_checkpoints.tar.gz
  fi
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ UPLOAD CONCLUÍDO!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 O que foi enviado:"
echo "  ✅ neural_farm.db.gz (1.4GB → comprimido)"
echo "  ✅ teis_checkpoints.tar.gz (checkpoints)"
echo "  ✅ JSONs e estatísticas"
echo ""
echo "⚠️  Arquivos muito grandes não enviados:"
echo "  • metrics.jsonl (9.5GB) - muito grande para GitHub"
echo "  • PRESERVED .pt files (28GB) - muito grande para GitHub"
echo ""
echo "💡 Para esses, use:"
echo "  rsync -avz root@SERVER:/root/neural_farm_prod/metrics.jsonl ./"
