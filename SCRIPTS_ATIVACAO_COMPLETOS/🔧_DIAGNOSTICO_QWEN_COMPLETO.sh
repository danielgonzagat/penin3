#!/bin/bash
# Diagnóstico completo do Qwen - encontrar causa dos timeouts

echo "🔬 DIAGNÓSTICO COMPLETO: Qwen Timeouts"
echo "======================================"
echo ""

# 1. Check service
echo "1. SERVICE STATUS:"
systemctl status llama-qwen.service --no-pager | head -15
echo ""

# 2. Check if model is loaded
echo "2. CONTAINER LOGS (últimos 50):"
docker logs llama-qwen --tail 50 2>&1 | grep -E "server is listening|model loaded|error|fail" || echo "No significant logs"
echo ""

# 3. Check resource usage
echo "3. RESOURCE USAGE:"
QWEN_PID=$(ps aux | grep llama-server | grep 8080 | awk '{print $2}' | head -1)
if [ -n "$QWEN_PID" ]; then
    ps aux | grep $QWEN_PID | head -1
    echo "   CPU usage: $(ps aux | grep $QWEN_PID | awk '{print $3}')%"
    echo "   MEM usage: $(ps aux | grep $QWEN_PID | awk '{print $4}')%"
else
    echo "   ⚠️  PID not found"
fi
echo ""

# 4. Test with simpler endpoint
echo "4. TESTING /completion (simpler than chat):"
SIMPLE_TEST=$(curl -X POST http://127.0.0.1:8013/completion \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"test","n_predict":1}' \
  -m 15 2>&1)

if echo "$SIMPLE_TEST" | grep -q "content"; then
    echo "   ✅ Simple completion works!"
else
    echo "   ❌ Even simple endpoint times out"
    echo "   Response: $SIMPLE_TEST"
fi
echo ""

# 5. Diagnóstico
echo "5. DIAGNÓSTICO:"
echo ""

# Check model size vs RAM
MODEL_SIZE=$(ls -lh /srv/models/qwen/qwen2.5-coder-7b-instruct-q4_k_m.gguf | awk '{print $5}')
AVAILABLE_RAM=$(free -h | grep Mem | awk '{print $7}')

echo "   Model size: $MODEL_SIZE"
echo "   Available RAM: $AVAILABLE_RAM"
echo ""

# 6. Possíveis causas
echo "6. POSSÍVEIS CAUSAS DO TIMEOUT:"
echo ""
echo "   A) Modelo ainda carregando (primeira vez é lenta)"
echo "      Solução: Aguardar 5-10 minutos"
echo ""
echo "   B) Context size muito grande (-c 4096)"
echo "      Solução: Reduzir para -c 2048 ou -c 1024"
echo ""
echo "   C) Threads insuficientes (-t 48 pode ser muito)"
echo "      Solução: Reduzir para -t 24"
echo ""
echo "   D) Modelo muito pesado para CPU"
echo "      Solução: Usar modelo menor (Qwen2.5-Coder-3B ou 1.5B)"
echo ""

# 7. Recomendação
echo "7. RECOMENDAÇÃO IMEDIATA:"
echo ""
echo "   Opção A (RÁPIDA): Reduzir context size"
echo "   systemctl stop llama-qwen.service"
echo "   # Editar: /etc/systemd/system/llama-qwen.service"
echo "   # Mudar: -c 4096 → -c 2048"
echo "   # Mudar: -t 48 → -t 24"
echo "   systemctl daemon-reload"
echo "   systemctl start llama-qwen.service"
echo ""
echo "   Opção B (MELHOR): Aguardar 10 min e tentar de novo"
echo "   (Primeira carga é sempre lenta)"
echo ""
echo "=============================================="
echo ""