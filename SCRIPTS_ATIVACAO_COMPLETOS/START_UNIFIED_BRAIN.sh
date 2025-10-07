#!/bin/bash
# ==============================================================================
# == SCRIPT DE INICIALIZAÇÃO ÚNICO PARA A INTELIGÊNCIA UNIFICADA (UNIFIED_BRAIN) ==
# ==============================================================================
#
# Este é agora o ÚNICO script que você deve usar para iniciar a inteligência.
# Ele garante que o sistema execute em background, de forma estável, e com
# logs devidamente capturados.
#
# Uso: ./START_UNIFIED_BRAIN.sh
#

echo "--- INICIANDO O SISTEMA DE INTELIGÊNCIA UNIFICADA (UNIFIED_BRAIN) ---"

BRAIN_DIR="/root/UNIFIED_BRAIN"
BRAIN_SCRIPT="brain_daemon_real_env.py"
PID_FILE="${BRAIN_DIR}/brain_daemon.pid"
LOG_FILE="${BRAIN_DIR}/logs/unified_brain.log"

# Garante que o diretório de logs exista
mkdir -p "${BRAIN_DIR}/logs"

# Verifica se o processo já está em execução
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null; then
        echo "⚠️  AVISO: O UNIFIED_BRAIN já está em execução com o PID ${PID}."
        echo "Para parar, execute: kill ${PID}"
        exit 1
    else
        echo "Removendo arquivo de PID antigo (stale)."
        rm "$PID_FILE"
    fi
fi

# Navega para o diretório do cérebro
cd "$BRAIN_DIR" || { echo "❌ ERRO: Não foi possível acessar o diretório ${BRAIN_DIR}."; exit 1; }

# Garante que o script principal é executável
chmod +x "$BRAIN_SCRIPT"

# Inicia o cérebro em background usando nohup
nohup python3 -u "$BRAIN_SCRIPT" >> "$LOG_FILE" 2>&1 &

# Captura o PID do processo que acabou de ser lançado
BRAIN_PID=$!

# Salva o PID para referência futura (parar, status, etc.)
echo $BRAIN_PID > "$PID_FILE"

sleep 2

# Verifica se o processo iniciou com sucesso
if ps -p $BRAIN_PID > /dev/null; then
    echo "✅ SUCESSO! O UNIFIED_BRAIN foi iniciado em background."
    echo "   - PID do Processo: ${BRAIN_PID} (salvo em ${PID_FILE})"
    echo "   - Logs completos estão sendo escritos em: ${LOG_FILE}"
    echo ""
    echo "🧠 Para monitorar o aprendizado em tempo real, execute:"
    echo "   ./start_monitoring.sh"
    echo ""
    echo "🛑 Para parar o cérebro, execute:"
    echo "   kill ${BRAIN_PID}"
else
    echo "❌ ERRO: O UNIFIED_BRAIN falhou ao iniciar. Verifique os logs para detalhes:"
    echo "   cat ${LOG_FILE}"
    rm "$PID_FILE"
    exit 1
fi
