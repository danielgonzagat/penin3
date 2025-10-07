#!/bin/bash
# Otimizar Qwen para respostas rápidas em CPU

echo "⚡ OTIMIZANDO QWEN PARA CPU"
echo "==========================="
echo ""

echo "Parando service atual..."
systemctl stop llama-qwen.service
sleep 5

echo "Criando service otimizado..."

# Criar service OTIMIZADO (context menor, threads otimizados)
cat > /etc/systemd/system/llama-qwen-fast.service << 'UNIT'
[Unit]
Description=llama.cpp server - Qwen2.5-Coder-7B OTIMIZADO (menor context, CPU otimizado)
After=docker.service network.target
Requires=docker.service

[Service]
Type=simple
Restart=always
RestartSec=10

ExecStartPre=-/usr/bin/docker rm -f llama-qwen-fast 2>/dev/null || true

ExecStart=/usr/bin/docker run --rm --name llama-qwen-fast \
  -p 8014:8080 \
  -v /srv/models/qwen:/models \
  ghcr.io/ggerganov/llama.cpp:server \
    -m /models/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
    -c 2048 \
    -t 24 \
    -ngl 0 \
    -b 128 \
    --port 8080 \
    --host 0.0.0.0 \
    --parallel 2 \
    --cont-batching

ExecStop=/usr/bin/docker stop llama-qwen-fast

[Install]
WantedBy=multi-user.target
UNIT

systemctl daemon-reload
systemctl enable llama-qwen-fast.service
systemctl start llama-qwen-fast.service

echo "✅ Service otimizado criado: llama-qwen-fast"
echo "   Porta: 8014 (não 8013)"
echo "   Context: 2048 (reduzido de 4096)"
echo "   Threads: 24 (reduzido de 48)"
echo "   Batch: 128 (reduzido de 256)"
echo ""

echo "Aguardando 30s para carregar..."
sleep 30

echo "Testando..."
curl -m 20 -sS http://127.0.0.1:8014/health

echo ""
echo "Testando completion simples..."
curl -X POST http://127.0.0.1:8014/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf","messages":[{"role":"user","content":"Hi"}],"max_tokens":3,"temperature":0}' \
  -m 45 2>&1 | head -20

echo ""
echo "=============================================="
echo "Se funcionou: use porta 8014 (não 8013)"
echo "Se ainda timeout: modelo é muito pesado para CPU"
echo "   Alternativa: Usar Qwen2.5-Coder-1.5B (muito mais rápido)"
echo "=============================================="