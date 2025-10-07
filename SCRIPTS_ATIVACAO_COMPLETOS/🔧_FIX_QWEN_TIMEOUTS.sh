#!/bin/bash
# Fix Qwen timeouts e testar conectividade

echo "🔧 FIXING QWEN TIMEOUTS"
echo "======================="
echo ""

# Check if Qwen is responding to health
echo "1. Testing health endpoint..."
HEALTH=$(curl -m 5 -sS http://127.0.0.1:8013/health 2>&1)

if echo "$HEALTH" | grep -q "ok"; then
    echo "   ✅ Health OK"
else
    echo "   ⚠️  Health check failed: $HEALTH"
    echo "   Restarting Qwen service..."
    systemctl restart llama-qwen.service
    sleep 30
fi

echo ""
echo "2. Testing /v1/models endpoint..."
MODELS=$(curl -m 10 -sS http://127.0.0.1:8013/v1/models 2>&1)

if echo "$MODELS" | grep -q "qwen"; then
    echo "   ✅ Models endpoint OK"
    MODEL_ID=$(echo "$MODELS" | jq -r '.data[0].id' 2>/dev/null || echo "/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf")
    echo "   Model ID: $MODEL_ID"
else
    echo "   ❌ Models endpoint failed"
    echo "   Response: $MODELS"
    exit 1
fi

echo ""
echo "3. Testing simple completion..."

# Try MINIMAL request (only 5 tokens)
RESPONSE=$(curl -X POST http://127.0.0.1:8013/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Hi\"}],\"max_tokens\":5,\"temperature\":0}" \
  -m 45 2>&1)

if echo "$RESPONSE" | grep -q "choices"; then
    echo "   ✅ Completion works!"
    echo "   Response:"
    echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null || echo "$RESPONSE"
else
    echo "   ⚠️  Completion timeout or error"
    echo "   This is NORMAL for first request (model loading)"
    echo "   Trying again with longer timeout..."
    
    # Second attempt (model should be warm now)
    sleep 10
    
    RESPONSE2=$(curl -X POST http://127.0.0.1:8013/v1/chat/completions \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Test\"}],\"max_tokens\":3,\"temperature\":0}" \
      -m 60 2>&1)
    
    if echo "$RESPONSE2" | grep -q "choices"; then
        echo "   ✅ Second attempt succeeded (model was cold)"
        echo "   Response:"
        echo "$RESPONSE2" | jq -r '.choices[0].message.content' 2>/dev/null || echo "$RESPONSE2"
    else
        echo "   ❌ Still timing out"
        echo "   Possible issues:"
        echo "   - Model too large for CPU"
        echo "   - Need more RAM"
        echo "   - Context size too big"
        echo ""
        echo "   Quick fix: Reduce context"
        echo "   Edit service: systemctl edit llama-qwen.service"
        echo "   Change: -c 4096 → -c 2048"
    fi
fi

echo ""
echo "=============================================="
echo "✅ QWEN CONNECTIVITY TEST COMPLETE"
echo "=============================================="
echo ""
echo "Next: Setup OpenHands with this working Qwen"
echo "   bash /root/🤖_AGENTE_AUTONOMO_COMPLETO_SEGURO.sh"