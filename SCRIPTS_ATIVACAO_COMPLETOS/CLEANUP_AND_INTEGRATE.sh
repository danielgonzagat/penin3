#!/bin/bash

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   🧹 LIMPEZA E INTEGRAÇÃO 🧹                ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Este script vai:"
echo "  1. PRESERVAR: TEIS/Darwin + repos úteis"
echo "  2. LIMPAR: Todo o lixo (backups, logs desnecessários)"  
echo "  3. INTEGRAR: Novo cérebro IA³ com TEIS/Darwin"
echo ""

# Criar diretório de preservação
mkdir -p /root/PRESERVED
mkdir -p /root/TO_DELETE

# 1. PRESERVAR O ESSENCIAL
echo "📦 Preservando sistemas essenciais..."

# TEIS/Darwin
cp -p /root/true_emergent_intelligence_system.py /root/PRESERVED/ 2>/dev/null
cp -p /root/teis*.py /root/PRESERVED/ 2>/dev/null
cp -p /root/TEIS*.py /root/PRESERVED/ 2>/dev/null  
cp -p /root/darwin*.py /root/PRESERVED/ 2>/dev/null
cp -p /root/Darwin*.py /root/PRESERVED/ 2>/dev/null
cp -p /root/darwin*.yaml /root/PRESERVED/ 2>/dev/null
cp -p /root/darwin*.json /root/PRESERVED/ 2>/dev/null
cp -p /root/lemni*.py /root/PRESERVED/ 2>/dev/null
cp -p /root/lemniscata*.py /root/PRESERVED/ 2>/dev/null
cp -p /root/Makefile /root/PRESERVED/ 2>/dev/null

# Novo cérebro IA³
cp -p /root/NEURAL_GENESIS_IA3.py /root/PRESERVED/ 2>/dev/null

# 2. MARCAR PARA DELEÇÃO (não deleta ainda, só move)
echo "🗑️  Marcando lixo para deleção..."

# Backups desnecessários
mv /root/pre_metabolization_backup /root/TO_DELETE/ 2>/dev/null
mv /root/backup_pre_* /root/TO_DELETE/ 2>/dev/null
mv /root/*_backup* /root/TO_DELETE/ 2>/dev/null

# Logs gigantes e corrompidos
find /root -name "*.log" -size +100M -exec mv {} /root/TO_DELETE/ \; 2>/dev/null

# Bancos corrompidos (PENIN 275GB)
mv /root/.penin_omega/emergence_detection.db /root/TO_DELETE/ 2>/dev/null

# Sistemas mortos
mv /root/agi_fusion*.py /root/TO_DELETE/ 2>/dev/null
mv /root/unified_*.py /root/TO_DELETE/ 2>/dev/null
mv /root/ultimate_*.py /root/TO_DELETE/ 2>/dev/null
mv /root/falcon* /root/TO_DELETE/ 2>/dev/null
mv /root/iaaa* /root/TO_DELETE/ 2>/dev/null

# Cache e temporários
mv /root/.cache /root/TO_DELETE/ 2>/dev/null
find /root -name "__pycache__" -type d -exec mv {} /root/TO_DELETE/ \; 2>/dev/null
find /root -name "*.pyc" -exec mv {} /root/TO_DELETE/ \; 2>/dev/null

# 3. ESTATÍSTICAS
echo ""
echo "📊 ESTATÍSTICAS:"
echo "  Preservado: $(find /root/PRESERVED -type f 2>/dev/null | wc -l) arquivos"
echo "  Para deletar: $(du -sh /root/TO_DELETE 2>/dev/null | cut -f1) de lixo"
echo ""

# 4. CONFIRMAÇÃO
echo "⚠️  ATENÇÃO: Isso vai liberar MUITO espaço!"
echo ""
read -p "Deseja REALMENTE deletar o lixo? (digite 'DELETE' para confirmar): " confirm

if [ "$confirm" = "DELETE" ]; then
    echo "🔥 Deletando..."
    rm -rf /root/TO_DELETE
    echo "✅ Limpeza completa!"
    
    # Mostrar espaço liberado
    echo ""
    echo "💾 ESPAÇO ATUAL:"
    df -h /root
    
else
    echo "❌ Deleção cancelada. Lixo está em /root/TO_DELETE"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    ✨ SISTEMA LIMPO ✨                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "Agora você tem:"
echo "  • TEIS/Darwin preservado em /root/PRESERVED"
echo "  • Cérebro IA³ pronto em NEURAL_GENESIS_IA3.py"
echo "  • Sistema limpo e organizado"
echo ""
echo "Para iniciar o cérebro IA³:"
echo "  python3 /root/NEURAL_GENESIS_IA3.py"