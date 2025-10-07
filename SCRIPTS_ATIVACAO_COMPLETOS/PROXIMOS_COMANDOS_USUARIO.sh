#!/bin/bash
# COMANDOS ÚTEIS PARA O USUÁRIO
# ==============================

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║  📊 SISTEMA I³ - COMANDOS ÚTEIS                                ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

echo "1️⃣ VER DASHBOARD EM TEMPO REAL:"
echo "   bash /root/DASHBOARD_TEMPO_REAL.sh"
echo

echo "2️⃣ VER STATUS COMPLETO:"
echo "   cat /root/STATUS_SISTEMA_COMPLETO_AGORA.txt"
echo

echo "3️⃣ VER TODOS OS LOGS FASE 3:"
echo "   tail -f /root/fase3_*.log"
echo

echo "4️⃣ ANALISAR SURPRESAS:"
echo "   python3 /root/ANALISAR_SURPRESAS.py"
echo

echo "5️⃣ VER PROCESSOS ATIVOS:"
echo "   pgrep -fl python3 | grep -E 'BRIDGE|FASE|DARWIN|META'"
echo

echo "6️⃣ VALIDAR SISTEMA (após 30min):"
echo "   bash /root/VALIDAR_SISTEMA_POS_FIX.sh"
echo

echo "7️⃣ VER DATABASES (após 1h):"
echo "   sqlite3 /root/meta_learning_loop.db 'SELECT * FROM actions ORDER BY id DESC LIMIT 10;'"
echo

echo "8️⃣ VER ROADMAP COMPLETO:"
echo "   cat /root/ROADMAP_COMPLETO_5_FASES.md | less"
echo

echo "═══════════════════════════════════════════════════════════════"
echo "🎉 TUDO ESTÁ RODANDO! Sistema evoluindo sozinho agora!"
echo "═══════════════════════════════════════════════════════════════"
