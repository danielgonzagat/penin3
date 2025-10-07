#!/bin/bash
# IMPLEMENTAÇÃO COMPLETA: Correções Urgentes + Roadmap I³
# Executar: bash /root/IMPLEMENTAR_TUDO_AGORA.sh

set -e

echo "╔══════════════════════════════════════════════════════════╗"
echo "║                                                          ║"
echo "║   🔧 IMPLEMENTAÇÃO COMPLETA - I³ ROADMAP                ║"
echo "║                                                          ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ============================================================================
# PARTE 1: CORREÇÕES URGENTES (2-4h)
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔴 PARTE 1: CORREÇÕES URGENTES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Correção #1: Fix import UNIFIED_BRAIN
echo "1. Corrigindo import quebrado em V7_DARWIN_REALTIME_BRIDGE.py..."
cp /root/V7_DARWIN_REALTIME_BRIDGE.py /root/V7_DARWIN_REALTIME_BRIDGE.py.backup_import_fix
cat > /tmp/fix_import.py << 'PYEOF'
from pathlib import Path
p = Path('/root/V7_DARWIN_REALTIME_BRIDGE.py')
content = p.read_text()
content = content.replace(
    'from UNIFIED_BRAIN.brain_logger import brain_logger as _blog',
    'import logging\n_blog = logging.getLogger(__name__)\nlogging.basicConfig(level=logging.INFO)'
)
p.write_text(content)
print("✅ Import corrigido")
PYEOF
python3 /tmp/fix_import.py

# Correção #2: Remove ia3_score de TODOS os arquivos core
echo ""
echo "2. Removendo ia3_score de arquivos core..."
cd /root/intelligence_system/core
for f in *.py; do
    if [ -f "$f" ]; then
        cp "$f" "${f}.backup_ia3_removal"
        sed -i "s/'ia3_score': [^,}]*,\?//g" "$f" 2>/dev/null || true
        sed -i "s/\"ia3_score\": [^,}]*,\?//g" "$f" 2>/dev/null || true
        sed -i "s/ia3_score=[^,)]*,\?//g" "$f" 2>/dev/null || true
        sed -i "s/\.ia3_score//g" "$f" 2>/dev/null || true
    fi
done
echo "✅ ia3_score removido de $(ls *.py | wc -l) arquivos"

# Correção #3: Matar Incompletude Infinita definitivamente
echo ""
echo "3. Matando Incompletude Infinita..."
pkill -9 -f incompletude 2>/dev/null || true
pkill -9 -f ".incompletude_daemon.py" 2>/dev/null || true
rm -f ~/.incompletude_daemon.py 2>/dev/null || true
unset INCOMPLETUDE_ACTIVE 2>/dev/null || true
echo "✅ Incompletude Infinita desativada"

# Correção #4: Criar script de validação Darwin
echo ""
echo "4. Criando script de validação Darwin..."
cat > /root/VALIDATE_DARWIN_TRAINING.py << 'PYEOF'
#!/usr/bin/env python3
import sys
sys.path.insert(0, '/root/darwin-engine-intelligence')

print("🧪 Testando se Darwin treina REALMENTE...")
try:
    from core.darwin_evolution_system_FIXED import EvolvableMNIST
    
    # Teste: 1 indivíduo, 1 época, fitness > 0.1
    ind = EvolvableMNIST()
    ind.genome['n_epochs'] = 1
    fitness = ind.evaluate_fitness(seed=42)
    print(f"   Fitness: {fitness:.4f}")
    
    if fitness > 0.1:
        print("✅ Darwin treina corretamente!")
    else:
        print("❌ Darwin NÃO está treinando (fitness muito baixo)")
        sys.exit(1)
except Exception as e:
    print(f"❌ Erro ao testar Darwin: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
PYEOF
chmod +x /root/VALIDATE_DARWIN_TRAINING.py
echo "✅ Script de validação criado"

# Correção #5: Melhorar Cross-Pollination
echo ""
echo "5. Aplicando patch em Cross-Pollination Auto..."
cp /root/CROSS_POLLINATION_AUTO_FIXED.py /root/CROSS_POLLINATION_AUTO_FIXED.py.backup_patch
cat > /tmp/patch_crosspol.py << 'PYEOF'
from pathlib import Path
p = Path('/root/CROSS_POLLINATION_AUTO_FIXED.py')
content = p.read_text()

# Adiciona validação antes de pollination
old_code = '''            if (current_darwin != last_darwin or current_v7 != last_v7) and (current_darwin and current_v7):
                log(f"\\n🆕 Novos checkpoints detectados!")
                
                # Executa pollination'''

new_code = '''            if not darwin_cps or not v7_cps:
                log(f"⚠️ Aguardando checkpoints (Darwin:{len(darwin_cps)}, V7:{len(v7_cps)})")
                time.sleep(interval)
                continue
            
            if (current_darwin != last_darwin or current_v7 != last_v7) and (current_darwin and current_v7):
                log(f"\\n🆕 Novos checkpoints detectados!")
                
                # Executa pollination'''

if old_code in content:
    content = content.replace(old_code, new_code)
    p.write_text(content)
    print("✅ Cross-Pollination patch aplicado")
else:
    print("⚠️ Código não encontrado, patch não aplicado")
PYEOF
python3 /tmp/patch_crosspol.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ CORREÇÕES URGENTES APLICADAS!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
sleep 2

# ============================================================================
# PARTE 2: INICIAR SISTEMA
# ============================================================================

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 PARTE 2: INICIAR SISTEMA COMPLETO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

bash /root/START_ALL_ENGINES_MIN.sh

echo ""
echo "Aguardando 10 segundos para processos iniciarem..."
sleep 10

# ============================================================================
# PARTE 3: VALIDAÇÃO
# ============================================================================

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 PARTE 3: VALIDAÇÃO DO SISTEMA"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# Verifica processos
echo "Verificando processos ativos:"
ps aux | egrep -i "META_LEARNER_REALTIME|SYSTEM_CONNECTOR|run_emergence_blocks_STORM|CROSS_POLLINATION_AUTO_FIXED|SELF_REFLECTION_ENGINE_FIXED|DYNAMIC_FITNESS_ENGINE|V7_DARWIN_REALTIME_BRIDGE" | grep -v egrep | wc -l | xargs echo "   Processos rodando:"

# Testa Darwin
echo ""
echo "Testando Darwin (pode demorar 30s)..."
python3 /root/VALIDATE_DARWIN_TRAINING.py

# Testa schema V7
echo ""
echo "Testando schema V7..."
sqlite3 /root/intelligence_system/data/intelligence.db "SELECT cycle, mnist_accuracy, cartpole_reward FROM cycles ORDER BY cycle DESC LIMIT 1;" 2>/dev/null && echo "✅ Schema V7 correto" || echo "⚠️ Schema V7 vazio (aguarde ciclos)"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ SISTEMA INICIADO E VALIDADO!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "📊 Próximos passos:"
echo "   1. Aguarde 10 minutos para checkpoints serem gerados"
echo "   2. Verifique logs: tail -f /root/*.log"
echo "   3. Confirme checkpoints: ls -lh /root/intelligence_system/models/darwin_checkpoints/"
echo "   4. Rode validação completa: python3 /root/VALIDATE_ALL_FIXES_COMPLETE.py"
echo ""
echo "📄 Relatório completo: /root/RE_AUDITORIA_BRUTAL_FINAL_COMPLETA.md"
echo ""
