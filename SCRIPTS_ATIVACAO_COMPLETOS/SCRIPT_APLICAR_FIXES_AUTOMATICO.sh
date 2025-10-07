#!/bin/bash
# 🔧 SCRIPT AUTOMÁTICO - APLICAR TODOS OS FIXES DA FASE 1
# Execute: bash SCRIPT_APLICAR_FIXES_AUTOMATICO.sh

set -e  # Para em erro

echo "════════════════════════════════════════════════════════════════════════════════"
echo "🔧 APLICADOR AUTOMÁTICO DE FIXES - FASE 1"
echo "════════════════════════════════════════════════════════════════════════════════"
echo ""
echo "Este script aplicará as 7 correções triviais identificadas na auditoria."
echo "Backups serão criados automaticamente."
echo ""
echo "⚠️  ATENÇÃO: Requer Python 3 e permissões de escrita em /root/"
echo ""

read -p "Pressione ENTER para continuar ou Ctrl+C para cancelar..."

# Timestamp para backups
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/root/BACKUP_FASE1_${TIMESTAMP}"

echo ""
echo "📁 Criando diretório de backup: ${BACKUP_DIR}"
mkdir -p "${BACKUP_DIR}"

# ═══════════════════════════════════════════════════════════════════════════════
# FIX #1: Adicionar generation em NeuronMeta
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "FIX #1: Adicionar campo 'generation' em NeuronMeta"
echo "────────────────────────────────────────────────────────────────────────────────"

FILE="/root/UNIFIED_BRAIN/brain_spec.py"
if [ -f "$FILE" ]; then
    cp "$FILE" "${BACKUP_DIR}/brain_spec.py.backup"
    echo "✅ Backup criado"
    
    # Adicionar generation após tags
    if grep -q "tags: List\[str\] = None" "$FILE"; then
        sed -i '/tags: List\[str\] = None/a\    generation: int = 0  # ✅ FIX #1: Darwin generation tracking' "$FILE"
        echo "✅ Fix #1 aplicado: generation field adicionado"
    else
        echo "⚠️  Pattern não encontrado - pode já estar aplicado"
    fi
else
    echo "⚠️  Arquivo não encontrado: $FILE"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# FIX #2: Dashboard mkdir
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "FIX #2: Dashboard mkdir (criar diretório pai)"
echo "────────────────────────────────────────────────────────────────────────────────"

FILE="/root/UNIFIED_BRAIN/metrics_dashboard.py"
if [ -f "$FILE" ]; then
    cp "$FILE" "${BACKUP_DIR}/metrics_dashboard.py.backup"
    echo "✅ Backup criado"
    
    # Adicionar mkdir antes de write_text
    if grep -q "self.output_path.write_text(content)" "$FILE"; then
        # Usar Python para edição mais precisa
        python3 << 'PYEOF'
import sys
filepath = "/root/UNIFIED_BRAIN/metrics_dashboard.py"
with open(filepath, 'r') as f:
    content = f.read()

old = "        self.output_path.write_text(content)"
new = """        # ✅ FIX #2: Criar diretório pai se não existir
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(content)"""

if old in content and new not in content:
    content = content.replace(old, new)
    with open(filepath, 'w') as f:
        f.write(content)
    print("✅ Fix #2 aplicado: mkdir adicionado")
else:
    print("⚠️  Já aplicado ou pattern diferente")
PYEOF
    else
        echo "⚠️  Pattern não encontrado"
    fi
else
    echo "⚠️  Arquivo não encontrado: $FILE"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# FIX #3: Prometheus IPv4
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "FIX #3: Prometheus IPv4"
echo "────────────────────────────────────────────────────────────────────────────────"

FILE="/root/monitoring/prometheus.yml"
if [ -f "$FILE" ]; then
    cp "$FILE" "${BACKUP_DIR}/prometheus.yml.backup"
    echo "✅ Backup criado"
    
    sed -i 's/localhost:9109/127.0.0.1:9109/g' "$FILE"
    echo "✅ Fix #3 aplicado: localhost → 127.0.0.1"
    
    # Tentar reload Prometheus
    if docker ps | grep -q prometheus; then
        docker exec prometheus kill -HUP 1 2>/dev/null && echo "✅ Prometheus reloaded" || echo "⚠️  Prometheus reload falhou"
    else
        echo "⚠️  Prometheus não está rodando em Docker"
    fi
else
    echo "⚠️  Arquivo não encontrado: $FILE (Prometheus pode não estar configurado)"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# FIX #4: Router Learning (remover no_grad incorreto)
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "FIX #4: Router Learning (permitir gradientes)"
echo "────────────────────────────────────────────────────────────────────────────────"

FILE="/root/UNIFIED_BRAIN/brain_router.py"
if [ -f "$FILE" ]; then
    cp "$FILE" "${BACKUP_DIR}/brain_router.py.backup"
    echo "✅ Backup criado"
    
    python3 << 'PYEOF'
filepath = "/root/UNIFIED_BRAIN/brain_router.py"
with open(filepath, 'r') as f:
    lines = f.readlines()

modified = False
i = 0
while i < len(lines):
    if 'def update_competence' in lines[i]:
        # Procurar with torch.no_grad dentro deste método
        j = i + 1
        method_indent = len(lines[i]) - len(lines[i].lstrip())
        
        while j < len(lines):
            line_indent = len(lines[j]) - len(lines[j].lstrip())
            
            # Se voltou para indent da classe, saiu do método
            if line_indent <= method_indent and lines[j].strip() != '':
                break
            
            # Encontrou with torch.no_grad
            if 'with torch.no_grad():' in lines[j]:
                # Substituir por versão condicional
                indent = ' ' * line_indent
                new_code = [
                    f"{indent}# ✅ FIX #4: Permitir gradientes em training mode\n",
                    f"{indent}if self.training:\n",
                    f"{indent}    self.competence[neuron_idx] += lr * reward\n",
                    f"{indent}else:\n",
                    f"{indent}    with torch.no_grad():\n",
                    f"{indent}        self.competence[neuron_idx] += lr * reward\n",
                    f"{indent}self.competence.data.clamp_(min=0.0, max=10.0)\n",
                ]
                
                # Remover linhas antigas (with + conteúdo)
                lines[j:j+2] = new_code
                modified = True
                print("✅ Fix #4 aplicado: router agora pode aprender")
                break
            
            j += 1
        break
    i += 1

if modified:
    with open(filepath, 'w') as f:
        f.writelines(lines)
else:
    print("⚠️  Pattern não encontrado ou já aplicado")
PYEOF
else
    echo "⚠️  Arquivo não encontrado: $FILE"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# FIX #5 e #6: Requerem edição Python mais complexa - usar script Python
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "────────────────────────────────────────────────────────────────────────────────"
echo "FIX #5-7: Aplicando via script Python..."
echo "────────────────────────────────────────────────────────────────────────────────"

# Verificar se CODIGOS_PRONTOS existe
if [ -f "/root/CODIGOS_PRONTOS_FASE1_COMPLETA.py" ]; then
    echo "✅ Executando CODIGOS_PRONTOS_FASE1_COMPLETA.py"
    python3 /root/CODIGOS_PRONTOS_FASE1_COMPLETA.py --skip-prompts || echo "⚠️  Alguns fixes falharam"
else
    echo "⚠️  CODIGOS_PRONTOS_FASE1_COMPLETA.py não encontrado"
    echo "   Fixes #5-7 não aplicados"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICAÇÃO FINAL
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "🔍 VERIFICAÇÃO FINAL"
echo "════════════════════════════════════════════════════════════════════════════════"

PASSED=0
TOTAL=0

# Verificar Fix #1
if grep -q "generation: int = 0" /root/UNIFIED_BRAIN/brain_spec.py 2>/dev/null; then
    echo "✅ Fix #1 (generation): VERIFICADO"
    ((PASSED++))
else
    echo "❌ Fix #1 (generation): NÃO ENCONTRADO"
fi
((TOTAL++))

# Verificar Fix #2
if grep -q "mkdir(parents=True" /root/UNIFIED_BRAIN/metrics_dashboard.py 2>/dev/null; then
    echo "✅ Fix #2 (mkdir): VERIFICADO"
    ((PASSED++))
else
    echo "❌ Fix #2 (mkdir): NÃO ENCONTRADO"
fi
((TOTAL++))

# Verificar Fix #3
if grep -q "127.0.0.1:9109" /root/monitoring/prometheus.yml 2>/dev/null; then
    echo "✅ Fix #3 (IPv4): VERIFICADO"
    ((PASSED++))
elif [ ! -f /root/monitoring/prometheus.yml ]; then
    echo "⚠️  Fix #3: prometheus.yml não existe"
else
    echo "❌ Fix #3 (IPv4): NÃO ENCONTRADO"
fi
((TOTAL++))

# Verificar Fix #4
if grep -q "if self.training:" /root/UNIFIED_BRAIN/brain_router.py 2>/dev/null; then
    echo "✅ Fix #4 (router learning): VERIFICADO"
    ((PASSED++))
else
    echo "❌ Fix #4 (router learning): NÃO ENCONTRADO"
fi
((TOTAL++))

echo ""
echo "════════════════════════════════════════════════════════════════════════════════"
echo "📊 RESULTADO: ${PASSED}/${TOTAL} fixes verificados"
echo "════════════════════════════════════════════════════════════════════════════════"

if [ $PASSED -ge 3 ]; then
    echo ""
    echo "✅ SUCESSO! Fixes principais aplicados."
    echo ""
    echo "📍 PRÓXIMOS PASSOS:"
    echo ""
    echo "1. Restart UNIFIED_BRAIN:"
    echo "   cd /root/UNIFIED_BRAIN"
    echo "   kill 1497200 2>/dev/null || echo 'Já parado'"
    echo "   nohup python3 brain_daemon_real_env.py > brain_restart.log 2>&1 &"
    echo "   echo \$! > brain_restart.pid"
    echo ""
    echo "2. Monitorar logs (deixe rodando 30min):"
    echo "   tail -f /root/UNIFIED_BRAIN/brain_restart.log"
    echo ""
    echo "3. Verificar dashboard:"
    echo "   watch -n 5 cat /root/UNIFIED_BRAIN/dashboard.txt"
    echo ""
    echo "4. Esperar ver:"
    echo "   ✅ Rewards aumentando"
    echo "   ✅ Loss diminuindo"
    echo "   ✅ Zero crashes de AttributeError"
    echo "   ✅ 'Dashboard saved' a cada 5 episodes"
    echo ""
    echo "5. Se funcionou: Continuar para FASE 2 (Integrações)"
    echo ""
    exit 0
else
    echo ""
    echo "⚠️  ATENÇÃO: Apenas ${PASSED}/${TOTAL} fixes verificados."
    echo "   Alguns fixes podem precisar de edição manual."
    echo ""
    echo "📋 Verificar arquivos de backup em: ${BACKUP_DIR}/"
    echo ""
    exit 1
fi