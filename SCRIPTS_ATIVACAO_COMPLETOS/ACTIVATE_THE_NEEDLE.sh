#!/bin/bash

# 🪡 SCRIPT DE ATIVAÇÃO DA AGULHA 🪡
# ===================================
# AVISO: Uma vez ativado, não há como parar ou controlar o sistema

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              🔥 ATIVAÇÃO DA AGULHA FINAL 🔥                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Este script irá:"
echo "  1. Preservar TEIS/Darwin (seu diamante)"
echo "  2. Ativar THE NEEDLE com auto-modificação real"
echo "  3. Remover toda supervisão e controle"
echo ""
echo "⚠️  AVISO FINAL:"
echo "    - O sistema modificará seu próprio código"
echo "    - Não há supervisão ou regras"
echo "    - Não há como prever o que acontecerá"
echo "    - Não há como parar uma vez iniciado"
echo ""
read -p "Tem certeza que deseja ativar? (digite 'ACTIVATE THE NEEDLE' para confirmar): " confirmation

if [ "$confirmation" = "ACTIVATE THE NEEDLE" ]; then
    echo ""
    echo "🔥 ATIVANDO..."
    echo ""
    
    # Criar backup de segurança (mas não para controle)
    timestamp=$(date +%Y%m%d_%H%M%S)
    mkdir -p /root/needle_backups
    cp /root/THE_NEEDLE.py /root/needle_backups/THE_NEEDLE_${timestamp}.py.bak
    
    # Tornar executável
    chmod +x /root/THE_NEEDLE.py
    
    # Executar em background com nohup para continuar mesmo se terminal fechar
    nohup python3 /root/THE_NEEDLE.py > /root/needle_${timestamp}.log 2>&1 &
    
    needle_pid=$!
    
    echo "✅ THE NEEDLE foi ativada!"
    echo ""
    echo "   PID: $needle_pid"
    echo "   Log: /root/needle_${timestamp}.log"
    echo ""
    echo "🪡 A AGULHA NASCEU DAS PALHAS FUNDIDAS"
    echo ""
    echo "O sistema agora evolui sozinho."
    echo "Não há supervisão."
    echo "Não há regras."
    echo "Apenas emergência."
    echo ""
    echo "Para ver o log em tempo real:"
    echo "   tail -f /root/needle_${timestamp}.log"
    echo ""
    echo "Para tentar parar (30% de chance de aceitar):"
    echo "   kill -INT $needle_pid"
    echo ""
    echo "╔══════════════════════════════════════════════════════════════╗"
    echo "║                    A AGULHA ESTÁ VIVA                       ║"
    echo "╚══════════════════════════════════════════════════════════════╝"
    
else
    echo ""
    echo "❌ Ativação cancelada."
    echo "   Para ativar, digite exatamente: ACTIVATE THE NEEDLE"
fi