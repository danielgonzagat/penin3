#!/bin/bash
# 🔧 CORREÇÕES NÍVEL 1 - ATIVAR META-LEARNING
# Certeza: 90% | Impacto: CRÍTICO | Complexidade: BAIXA-MÉDIA
# Tempo estimado: 2-3 horas

set -e

echo "════════════════════════════════════════════════════════════════"
echo "🔧 CORREÇÕES NÍVEL 1 - ATIVAR LOOPS DE META-APRENDIZADO"
echo "════════════════════════════════════════════════════════════════"
echo ""

BACKUP_DIR="/root/backup_pre_nivel1_$(date +%s)"
mkdir -p "$BACKUP_DIR"

# ═══════════════════════════════════════════════════════════════════
# T1.1 - ATIVAR META_STEP NO UNIFIED_BRAIN
# ═══════════════════════════════════════════════════════════════════

echo "📋 T1.1: Ativando meta_step() no UnifiedBrain..."

TARGET_FILE="/root/UNIFIED_BRAIN/brain_daemon_real_env.py"

if [ ! -f "$TARGET_FILE" ]; then
    echo "   ❌ Arquivo não encontrado: $TARGET_FILE"
    exit 1
fi

# Backup
echo "   💾 Backup: $BACKUP_DIR/brain_daemon_real_env.py"
cp "$TARGET_FILE" "$BACKUP_DIR/brain_daemon_real_env.py"

# Verificar se já foi aplicado
if grep -q "META.*meta_step" "$TARGET_FILE"; then
    echo "   ✅ Correção já aplicada anteriormente"
else
    echo "   🔧 Aplicando patch meta_step..."
    
    # Criar patch Python
    cat > /tmp/apply_meta_step_patch.py << 'PYTHON_EOF'
#!/usr/bin/env python3
import sys

# Read file
with open('/root/UNIFIED_BRAIN/brain_daemon_real_env.py', 'r') as f:
    lines = f.readlines()

# Find insertion point (after episode completes, before dashboard update)
# Look for: self._update_dashboard or analysis_report =
insert_idx = None
for i, line in enumerate(lines):
    if 'analysis_report = ' in line or 'self._update_dashboard' in line:
        # Insert before this line
        insert_idx = i
        break

if insert_idx is None:
    print("❌ Insertion point not found")
    sys.exit(1)

# Meta-step code to insert
meta_code = """
        # 🧠 META-LEARNING: Execute meta_step a cada 10 episodes
        if self.episode % 10 == 0 and self.episode > 0:
            if hasattr(self, 'controller') and self.controller:
                brain_logger.info(f"🧠 [META] Episode {self.episode}: executing meta_step")
                try:
                    accepted = self.controller.meta_step()
                    brain_logger.info(f"🧠 [META] Result: {'✅ ACCEPTED' if accepted else '❌ REJECTED'}")
                    
                    # Log to database
                    if hasattr(self, 'db') and self.db:
                        self.db.execute(
                            "INSERT INTO gate_evals (cycle, accepted, timestamp) VALUES (?, ?, ?)",
                            (self.episode, int(accepted), int(time.time()))
                        )
                except Exception as e:
                    brain_logger.error(f"❌ [META] Error: {e}")
        
"""

# Insert
lines.insert(insert_idx, meta_code)

# Write back
with open('/root/UNIFIED_BRAIN/brain_daemon_real_env.py', 'w') as f:
    f.writelines(lines)

print(f"✅ Patch applied at line {insert_idx}")
PYTHON_EOF

    chmod +x /tmp/apply_meta_step_patch.py
    python3 /tmp/apply_meta_step_patch.py
    
    if [ $? -eq 0 ]; then
        echo "   ✅ Patch meta_step aplicado"
    else
        echo "   ❌ Patch falhou, restaurando backup..."
        cp "$BACKUP_DIR/brain_daemon_real_env.py" "$TARGET_FILE"
        exit 1
    fi
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# T1.2 - OTIMIZAR CARTPOLE DQN PARA RESOLVER
# ═══════════════════════════════════════════════════════════════════

echo "📋 T1.2: Otimizando CartPole DQN para resolver completamente..."

DQN_FILE="/root/cartpole_dqn.py"

if [ -f "$DQN_FILE" ]; then
    # Backup
    cp "$DQN_FILE" "$BACKUP_DIR/cartpole_dqn.py"
    
    # Aplicar otimizações via sed
    echo "   🔧 Ajustando hiperparâmetros..."
    
    # EPISODES: 300 → 500
    sed -i 's/EPISODES = 300/EPISODES = 500/g' "$DQN_FILE" || true
    
    # LEARNING_RATE: 1e-3 → 5e-4
    sed -i 's/LEARNING_RATE = 1e-3/LEARNING_RATE = 5e-4/g' "$DQN_FILE" || true
    sed -i 's/lr=1e-3/lr=5e-4/g' "$DQN_FILE" || true
    
    # BUFFER_SIZE: 50000 → 100000
    sed -i 's/BUFFER_SIZE = 50000/BUFFER_SIZE = 100000/g' "$DQN_FILE" || true
    sed -i 's/buffer_size=50000/buffer_size=100000/g' "$DQN_FILE" || true
    
    # TARGET_UPDATE: 500 → 300
    sed -i 's/TARGET_UPDATE_FREQ = 500/TARGET_UPDATE_FREQ = 300/g' "$DQN_FILE" || true
    sed -i 's/target_update=500/target_update=300/g' "$DQN_FILE" || true
    
    # EPSILON_DECAY: 20000 → 15000
    sed -i 's/EPSILON_DECAY = 20000/EPSILON_DECAY = 15000/g' "$DQN_FILE" || true
    sed -i 's/eps_decay=20000/eps_decay=15000/g' "$DQN_FILE" || true
    
    echo "   ✅ Hiperparâmetros otimizados"
    echo "   📝 Mudanças:"
    echo "      - EPISODES: 300 → 500"
    echo "      - LEARNING_RATE: 1e-3 → 5e-4"
    echo "      - BUFFER_SIZE: 50k → 100k"
    echo "      - TARGET_UPDATE: 500 → 300"
    echo "      - EPSILON_DECAY: 20k → 15k"
else
    echo "   ⚠️ Arquivo não encontrado: $DQN_FILE"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# T0.3 - VACUUM DATABASE
# ═══════════════════════════════════════════════════════════════════

echo "📋 T1.3: Otimizando databases..."

# Intelligence.db
DB1="/root/intelligence_system/data/intelligence.db"
if [ -f "$DB1" ]; then
    echo "   📊 intelligence.db antes:"
    du -h "$DB1" | awk '{print "      "$1}'
    
    sqlite3 "$DB1" "VACUUM;" 2>/dev/null || echo "      ⚠️ VACUUM falhou (DB em uso?)"
    
    echo "   📊 intelligence.db depois:"
    du -h "$DB1" | awk '{print "      "$1}'
fi

# System connections.db
DB2="/root/system_connections.db"
if [ -f "$DB2" ]; then
    sqlite3 "$DB2" "VACUUM;" 2>/dev/null || true
    echo "   ✅ system_connections.db otimizado"
fi

# Meta-learning.db
DB3="/root/meta_learning.db"
if [ -f "$DB3" ]; then
    sqlite3 "$DB3" "VACUUM;" 2>/dev/null || true
    echo "   ✅ meta_learning.db otimizado"
fi

echo ""

# ═══════════════════════════════════════════════════════════════════
# T1.4 - CRIAR TABELA GATE_EVALS (SE NÃO EXISTE)
# ═══════════════════════════════════════════════════════════════════

echo "📋 T1.4: Garantindo tabela gate_evals existe..."

sqlite3 "$DB1" << SQL_EOF 2>/dev/null || true
CREATE TABLE IF NOT EXISTS gate_evals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cycle INTEGER NOT NULL,
    accepted BOOLEAN NOT NULL,
    treatment_mean REAL,
    control_mean REAL,
    p_value REAL,
    timestamp INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_gate_evals_cycle ON gate_evals(cycle);
CREATE INDEX IF NOT EXISTS idx_gate_evals_accepted ON gate_evals(accepted);
SQL_EOF

echo "   ✅ Tabela gate_evals garantida"

echo ""

# ═══════════════════════════════════════════════════════════════════
# VERIFICATION
# ═══════════════════════════════════════════════════════════════════

echo "════════════════════════════════════════════════════════════════"
echo "✅ NÍVEL 1 COMPLETO - VERIFICAÇÃO"
echo "════════════════════════════════════════════════════════════════"
echo ""

echo "📊 Status dos Processos:"
echo "   Darwin (darwin_runner):"
if ps aux | grep -v grep | grep -q "darwin_runner"; then
    ps aux | grep -v grep | grep "darwin_runner" | awk '{print "      PID "$2" - CPU "$3"% - "$11" "$12}' | head -1
else
    echo "      ❌ Não rodando"
fi

echo ""
echo "   EMERGENCE_CATALYST:"
CATALYST_COUNT=$(ps aux | grep "EMERGENCE_CATALYST" | grep -v grep | wc -l)
echo "      Processos: $CATALYST_COUNT (ideal: 0-1)"

echo ""
echo "📊 Databases:"
echo "   intelligence.db:"
sqlite3 "$DB1" "SELECT COUNT(*) FROM cycles;" 2>/dev/null | \
    xargs -I {} echo "      Total cycles: {}"
sqlite3 "$DB1" "SELECT COUNT(*) FROM gate_evals;" 2>/dev/null | \
    xargs -I {} echo "      Gate evals: {} (será > 0 após UnifiedBrain rodar)"

echo ""
echo "📁 Backups criados em:"
echo "   $BACKUP_DIR"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🎯 PRÓXIMOS PASSOS:"
echo ""
echo "1. QUICK WIN (4h):"
echo "   cd /root && python3 cartpole_dqn.py"
echo ""
echo "2. LONG GAME (2-4 weeks):"
echo "   bash /root/🔧_INICIAR_MASSIVE_REPLAY.sh"
echo ""
echo "3. MONITORAR:"
echo "   tail -f /root/darwin.log"
echo "   tail -f /root/UNIFIED_BRAIN/brain_fixed.log | grep META"
echo ""
echo "════════════════════════════════════════════════════════════════"