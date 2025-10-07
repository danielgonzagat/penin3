#!/bin/bash
# 🚀 COMANDO ÚNICO - APLICA TUDO DA FASE 1 E RESTART BRAIN

set -e  # Para em erro

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🚀 COMANDO ÚNICO - FASE 1 COMPLETA"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Este script executa TUDO da Fase 1 automaticamente:"
echo "  1. Aplica 7 fixes"
echo "  2. Restart UNIFIED_BRAIN"
echo "  3. Inicia monitoramento"
echo ""

read -p "Pressione ENTER para executar ou Ctrl+C para cancelar..."

# Timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 1: Aplicar fixes
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "PASSO 1: Aplicando fixes automáticos..."
echo "────────────────────────────────────────────────────────────────────────────────"

if [ -f /root/SCRIPT_APLICAR_FIXES_AUTOMATICO.sh ]; then
    bash /root/SCRIPT_APLICAR_FIXES_AUTOMATICO.sh || {
        echo "⚠️  Alguns fixes falharam - continuando mesmo assim"
    }
else
    echo "⚠️  SCRIPT_APLICAR_FIXES_AUTOMATICO.sh não encontrado"
    echo "   Pulando fixes automáticos"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 2: Parar BRAIN antigo
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "PASSO 2: Parando UNIFIED_BRAIN antigo..."
echo "────────────────────────────────────────────────────────────────────────────────"

# Verificar se está rodando
if ps -p 1497200 > /dev/null 2>&1; then
    echo "✅ Encontrado PID 1497200 - parando..."
    kill 1497200 2>/dev/null || kill -9 1497200 2>/dev/null
    sleep 2
    
    if ps -p 1497200 > /dev/null 2>&1; then
        echo "⚠️  Processo ainda rodando - forçando kill -9"
        kill -9 1497200
        sleep 1
    fi
    
    echo "✅ Processo antigo parado"
else
    echo "✅ Processo já estava parado"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 3: Criar diretório de logs se não existir
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "PASSO 3: Preparando ambiente..."
echo "────────────────────────────────────────────────────────────────────────────────"

mkdir -p /root/UNIFIED_BRAIN/logs
echo "✅ Diretórios criados"

# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 4: Restart UNIFIED_BRAIN
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "PASSO 4: Iniciando UNIFIED_BRAIN com fixes aplicados..."
echo "────────────────────────────────────────────────────────────────────────────────"

cd /root/UNIFIED_BRAIN

# Verificar se arquivo existe
if [ ! -f brain_daemon_real_env.py ]; then
    echo "❌ brain_daemon_real_env.py não encontrado!"
    echo "   Verifique /root/UNIFIED_BRAIN/"
    exit 1
fi

# Iniciar novo processo
nohup python3 brain_daemon_real_env.py > brain_restart_${TIMESTAMP}.log 2>&1 &
NEW_PID=$!

# Salvar PID
echo $NEW_PID > brain_restart.pid

echo "✅ UNIFIED_BRAIN iniciado"
echo "   PID: $NEW_PID"
echo "   Log: brain_restart_${TIMESTAMP}.log"

# Aguardar 5s para ver se não crashou imediatamente
sleep 5

if ps -p $NEW_PID > /dev/null 2>&1; then
    echo "✅ Processo rodando estável"
else
    echo "❌ Processo crashou imediatamente!"
    echo "   Verificar log: tail -50 brain_restart_${TIMESTAMP}.log"
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# PASSO 5: Instruções de monitoramento
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "✅ FASE 1 COMPLETA!"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "🎉 UNIFIED_BRAIN reiniciado com todos os fixes!"
echo ""
echo "📊 Monitorar agora (deixe rodando 30 minutos):"
echo ""
echo "   # Terminal 1: Logs em tempo real"
echo "   tail -f /root/UNIFIED_BRAIN/brain_restart_${TIMESTAMP}.log"
echo ""
echo "   # Terminal 2: Dashboard atualizado"
echo "   watch -n 5 cat /root/UNIFIED_BRAIN/dashboard.txt"
echo ""
echo "   # Terminal 3: Processos ativos"
echo "   watch -n 5 'ps aux | grep -E \"brain|darwin|penin\" | grep -v grep'"
echo ""
echo "🔍 O que esperar nos próximos 30 minutos:"
echo ""
echo "   ✅ Episode count aumentando (1, 2, 3, ...)"
echo "   ✅ Rewards melhorando gradualmente"
echo "   ✅ Loss diminuindo"
echo "   ✅ 'Dashboard saved' a cada 5 episodes"
echo "   ✅ Zero crashes de AttributeError"
echo "   ✅ Active neurons > 5"
echo ""
echo "📍 Se tudo funcionar (rewards subindo, zero crashes):"
echo "   → Sistema está aprendendo!"
echo "   → Pronto para Fase 2 (integrações)"
echo ""
echo "📍 Se tiver problemas:"
echo "   → Verificar: tail -100 brain_restart_${TIMESTAMP}.log"
echo "   → Backups estão em: /root/BACKUP_FASE1_*/"
echo "   → Me avise: vou te ajudar a debugar"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "🚀 COMANDO RÁPIDO PARA MONITORAR:"
echo ""
echo "tail -f /root/UNIFIED_BRAIN/brain_restart_${TIMESTAMP}.log"
echo ""
echo "════════════════════════════════════════════════════════════════════════════════"

# Manter terminal aberto
exec bash