#!/bin/bash
# CORREÇÕES URGENTES - EXECUTAR AGORA

echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║   🔧 CORREÇÕES URGENTES - EXECUTANDO                    ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# URGENTE #1: Matar Meta-Learner antigo
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Urgente #1: Matando Meta-Learner antigo..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

pkill -9 -f "META_LEARNER_REALTIME.py"
sleep 3

echo "Verificando..."
if pgrep -f "META_LEARNER"; then
    echo "⚠️  Ainda vivo! Força total..."
    pgrep -f "META_LEARNER" | xargs kill -9
    sleep 2
else
    echo "✅ Meta-Learner antigo morto"
fi

echo ""
sleep 1

# ============================================================================
# URGENTE #2: Iniciar Meta-Learner corrigido
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Urgente #2: Iniciando Meta-Learner corrigido..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd /root
nohup python3 -u META_LEARNER_REALTIME.py > meta_learner_CORRIGIDO_FINAL.log 2>&1 &
NEW_PID=$!

echo "✅ Novo Meta-Learner iniciado"
echo "   PID: $NEW_PID"
echo "   Log: /root/meta_learner_CORRIGIDO_FINAL.log"

sleep 5

echo "Primeiras linhas do log:"
head -15 meta_learner_CORRIGIDO_FINAL.log

echo ""
sleep 1

# ============================================================================
# URGENTE #3: Corrigir System Connector (remover ia3_score)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔧 Urgente #3: Corrigindo System Connector..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Backup
cp /root/EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py /root/EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py.backup_final

# Remove ia3_score (usa Python para edição precisa)
python3 << 'PYEOF'
from pathlib import Path

file_path = Path("/root/EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py")
content = file_path.read_text()

# Remove da query
content = content.replace(
    ", ia3_score",
    ""
)

# Remove do retorno
content = content.replace(
    '                "ia3_score": row[3],\n',
    ''
)

file_path.write_text(content)
print("✅ ia3_score removido")
PYEOF

# Reiniciar
echo "Reiniciando System Connector..."
pkill -f "EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR"
sleep 2

nohup python3 /root/EMERGENCE_CATALYST_4_SYSTEM_CONNECTOR.py 100 60 > /root/system_connector_CORRIGIDO.log 2>&1 &

echo "✅ System Connector reiniciado"

echo ""
sleep 1

# ============================================================================
# VALIDAÇÃO
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ VALIDAÇÃO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "1. Meta-Learner:"
if pgrep -f "META_LEARNER_REALTIME" > /dev/null; then
    PID=$(pgrep -f "META_LEARNER_REALTIME")
    echo "   ✅ Rodando (PID: $PID)"
else
    echo "   ❌ NÃO RODANDO!"
fi

echo ""
echo "2. System Connector:"
if pgrep -f "SYSTEM_CONNECTOR" > /dev/null; then
    PID=$(pgrep -f "SYSTEM_CONNECTOR")
    echo "   ✅ Rodando (PID: $PID)"
else
    echo "   ❌ NÃO RODANDO!"
fi

echo ""
echo "3. Darwin STORM:"
if pgrep -f "run_emergence_blocks_STORM" > /dev/null; then
    PID=$(pgrep -f "run_emergence_blocks_STORM")
    CPU=$(ps -p $PID -o %cpu --no-headers 2>/dev/null || echo "?")
    echo "   ✅ Rodando (PID: $PID, CPU: ${CPU}%)"
else
    echo "   ❌ NÃO RODANDO!"
fi

echo ""
echo "4. Diretórios:"
if [ -d "/root/intelligence_system/models/darwin_checkpoints" ]; then
    COUNT=$(ls /root/intelligence_system/models/darwin_checkpoints/*.pt 2>/dev/null | wc -l)
    echo "   ✅ Darwin checkpoints: $COUNT arquivos"
else
    echo "   ❌ Diretório não existe!"
fi

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "CORREÇÕES URGENTES APLICADAS!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Próximos passos:"
echo "   1. Aguardar 5 minutos"
echo "   2. Verificar logs:"
echo "      tail -f /root/meta_learner_CORRIGIDO_FINAL.log"
echo "   3. Meta-Learner NÃO deve mostrar 'RESTART_DARWIN'"
echo ""
