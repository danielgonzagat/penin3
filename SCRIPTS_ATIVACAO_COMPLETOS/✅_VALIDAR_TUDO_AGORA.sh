#!/bin/bash
# Validação completa de TODOS os sistemas

clear
echo "╔═══════════════════════════════════════════════════════════════════════╗"
echo "║                                                                       ║"
echo "║              ✅ VALIDAÇÃO COMPLETA DO SISTEMA                        ║"
echo "║                                                                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════╝"
echo ""

echo "🔍 1. VERIFICANDO SERVICES..."
echo "─────────────────────────────────────────────────────────"
systemctl is-active llama-qwen.service && echo "✅ llama-qwen" || echo "❌ llama-qwen"
systemctl is-active intelligence-nexus.service && echo "✅ intelligence-nexus" || echo "❌ intelligence-nexus"
systemctl is-active v7-surprise-bridge.service && echo "✅ v7-surprise-bridge" || echo "❌ v7-surprise-bridge"
echo ""

echo "🧪 2. TESTANDO QWEN2.5..."
echo "─────────────────────────────────────────────────────────"
timeout 15 curl -sS http://127.0.0.1:8013/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"/models/qwen2.5-coder-7b-instruct-q4_k_m.gguf","messages":[{"role":"user","content":"2+2="}],"temperature":0.1,"max_tokens":5}' \
  | jq -r '.choices[0].message.content' && echo "✅ Qwen respondendo" || echo "❌ Qwen não responde"
echo ""

echo "📊 3. VERIFICANDO DATABASES..."
echo "─────────────────────────────────────────────────────────"
SURPRISES=$(sqlite3 /root/emergence_surprises.db "SELECT COUNT(*) FROM surprises;" 2>/dev/null)
echo "Surpresas totais: $SURPRISES"
RECENT=$(sqlite3 /root/emergence_surprises.db "SELECT COUNT(*) FROM surprises WHERE timestamp > datetime('now', '-1 hour');" 2>/dev/null)
echo "Surpresas última hora: $RECENT"
[ "$RECENT" -gt 0 ] && echo "✅ Sistema gerando surprises" || echo "⚠️  Sem surprises recentes"
echo ""

echo "🧬 4. VERIFICANDO DARWINACCI..."
echo "─────────────────────────────────────────────────────────"
tail -3 /root/darwinacci_daemon.log 2>/dev/null | grep "Gen" | tail -1 || echo "⚠️  Sem logs"
ps aux | grep -q "DARWINACCI_DAEMON" && echo "✅ DARWINACCI rodando" || echo "❌ DARWINACCI parado"
echo ""

echo "📈 5. EMERGENCE SCORE ATUAL..."
echo "─────────────────────────────────────────────────────────"
python3 -c "import json; s=json.load(open('/root/intelligence_nexus_state.json')); print(f'Emergence: {s[\"emergence_score\"]:.4f} ({s[\"emergence_score\"]*100:.1f}%)\nCycles: {s[\"cycle_count\"]}')" 2>/dev/null || echo "⚠️  Estado não disponível"
echo ""

echo "🤖 6. PROCESSOS PRINCIPAIS..."
echo "─────────────────────────────────────────────────────────"
ps aux | grep -E "V7_RUNNER|brain_daemon|DARWINACCI" | grep -v grep | awk '{printf "  %-40s %6s%%\n", $11, $3}' | head -5
echo ""

echo "═══════════════════════════════════════════════════════════════════════"
echo "✅ VALIDAÇÃO COMPLETA"
echo "═══════════════════════════════════════════════════════════════════════"
echo ""
uptime
echo ""
