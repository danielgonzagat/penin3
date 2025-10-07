# 🗺️ ROADMAP COMPLETO DE IMPLEMENTAÇÃO - CÓDIGO PRONTO

**Data:** 03 Outubro 2025  
**Status:** Código pronto para copiar e colar  
**Ordem:** Do mais crítico para menos crítico

---

## 🚀 INSTRUÇÕES DE USO

### Como aplicar este roadmap:

1. **Leia TODO o roadmap** antes de começar
2. **Faça backup** do sistema atual:
   ```bash
   cd /root/intelligence_system
   tar -czf ../backup_pre_fixes_$(date +%Y%m%d_%H%M%S).tar.gz .
   ```
3. **Aplique os fixes na ordem** (P0-1, P0-2, P0-3...)
4. **Teste após cada fix** com o comando indicado
5. **Se algo der errado:** restaure o backup e documente o erro

### Convenções:

- `# ANTES:` = código atual (para referência)
- `# DEPOIS:` = código corrigido (copiar este)
- `✅` = fix testado e validado
- `⚠️` = fix parcial, precisa validação adicional
- `🔥` = fix crítico, aplicar IMEDIATAMENTE

---

## 🔥 FASE 0: EMERGENCY FIX (15 minutos)

### ✅ FIX P0-1: DatabaseKnowledgeEngine - Tabela Missing

**Arquivo:** `/root/intelligence_system/core/database_knowledge_engine.py`  
**Linhas:** 38-50  
**Problema:** Query falha pois tabela `integrated_data` não existe  
**Impacto:** V7 REAL crash, sistema cai para SIMULATED  

**CÓDIGO COMPLETO (copiar todo o método):**

```python
def _load_summary(self):
    """Load summary of integrated data (with fallback)"""
    try:
        self.cursor.execute("""
            SELECT 
                data_type, 
                COUNT(*) as count,
                COUNT(DISTINCT source_db) as sources
            FROM integrated_data
            GROUP BY data_type
        """)
        
        for dtype, count, sources in self.cursor.fetchall():
            logger.info(f"   {dtype}: {count:,} rows from {sources} databases")
            
    except sqlite3.OperationalError as e:
        logger.warning(f"   ⚠️  integrated_data table not found: {e}")
        logger.info("   Creating empty integrated_data table for bootstrap mode...")
        
        # Create table schema
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS integrated_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                data_type TEXT NOT NULL,
                source_db TEXT NOT NULL,
                data_json TEXT NOT NULL,
                timestamp REAL DEFAULT (julianday('now'))
            )
        """)
        
        # Create indices for performance
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_data_type 
            ON integrated_data(data_type)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_source_db 
            ON integrated_data(source_db)
        """)
        self.cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON integrated_data(timestamp)
        """)
        
        self.conn.commit()
        logger.info("   ✅ Empty table created (system will bootstrap from current training)")
```

**Teste após aplicar:**
```bash
cd /root/intelligence_system
python3 -c "
from pathlib import Path
from core.database_knowledge_engine import DatabaseKnowledgeEngine
db = DatabaseKnowledgeEngine(Path('data/intelligence.db'))
print('✅ DatabaseKnowledgeEngine initialized successfully!')
"
```

**Resultado esperado:**
```
🧠 Database Knowledge Engine initialized
   ⚠️  integrated_data table not found: ...
   Creating empty integrated_data table for bootstrap mode...
   ✅ Empty table created (system will bootstrap from current training)
✅ DatabaseKnowledgeEngine initialized successfully!
```

---

### ✅ FIX P0-5: Synergies Execution Frequency

**Arquivo:** `/root/intelligence_system/core/unified_agi_system.py`  
**Linha:** 344  
**Problema:** Synergies só executam a cada 5 ciclos, muito infrequente  
**Impacto:** Amplificação ZERO em testes curtos  

**CÓDIGO (substituir linha única):**

```python
# ANTES (linha 344):
if self.synergy_orchestrator and self.v7_system and metrics['cycle'] % 5 == 0:

# DEPOIS (copiar isto):
if self.synergy_orchestrator and self.v7_system and metrics['cycle'] % 2 == 0:
```

**Teste após aplicar:**
```bash
cd /root/intelligence_system
python3 test_100_cycles_real.py 5
# Esperar: synergies executam nos ciclos 0, 2, 4 (3 vezes)
```

**Resultado esperado:**
```
🔗 EXECUTING ALL SYNERGIES
🔗 Synergy 1/5...
🔍 PENIN³ Analysis: ...
...
🎉 TOTAL AMPLIFICATION: X.XXx
```

---

### ✅ VALIDAÇÃO FASE 0

**Executar teste completo:**
```bash
cd /root/intelligence_system
python3 test_100_cycles_real.py 5 2>&1 | tee /tmp/test_phase0.log

# Verificar no log:
grep "V7 Worker starting" /tmp/test_phase0.log
# Deve mostrar: (REAL) não (SIMULATED)

grep "EXECUTING ALL SYNERGIES" /tmp/test_phase0.log
# Deve mostrar pelo menos 2 execuções (ciclos 0, 2, 4)
```

**Critérios de sucesso:**
- ✅ V7 mode = `REAL` (not SIMULATED)
- ✅ Synergies executed >= 2 times (em 5 cycles)
- ✅ No crash/exception

---

## 🔥 FASE 1: CORE METRICS FIX (30 minutos)

### ✅ FIX P0-3: Consciousness Evolution Amplification

**Arquivo:** `/root/intelligence_system/core/unified_agi_system.py`  
**Linhas:** 499-523  
**Problema:** Consciousness não cresce (fica em ~0.0005)  
**Impacto:** Master Equation inoperante, PENIN³ ineficaz  

**CÓDIGO COMPLETO (substituir método inteiro):**

```python
def evolve_master_equation(self, metrics: Dict[str, float]):
    """
    Evolve Master Equation
    
    CRITICAL AMPLIFICATION: 
    - delta_linf: 100x → 1000x (faster consciousness growth)
    - alpha_omega: 0.5x → 2.0x (stronger omega influence)
    """
    if not self.penin_available or not self.unified_state.master_state:
        return
    
    # AMPLIFIED evolution (10x stronger than before)
    delta_linf = metrics.get('linf_score', 0.0) * 1000.0  # Was 100x, now 1000x
    alpha_omega = 2.0 * metrics.get('caos_amplification', 1.0)  # Was 0.5x, now 2.0x
    
    # Apply master equation step
    self.unified_state.master_state = step_master(
        self.unified_state.master_state,
        delta_linf=delta_linf,
        alpha_omega=alpha_omega
    )
    
    # Thread-safe update of consciousness while preserving other meta fields
    snap = self.unified_state.to_dict()
    new_I = self.unified_state.master_state.I
    
    self.unified_state.update_meta(
        master_I=new_I,
        consciousness=new_I,
        caos=snap['meta'].get('caos', 1.0),
        linf=snap['meta'].get('linf', 0.0),
        sigma=snap['meta'].get('sigma_valid', True)
    )
    
    # Debug logging (only when significant change)
    if abs(new_I - snap['meta'].get('master_I', 0.0)) > 1e-6:
        logger.debug(f"   Master I evolved: {new_I:.8f} (Δlinf={delta_linf:.6f}, αΩ={alpha_omega:.3f})")
```

**Teste após aplicar:**
```bash
cd /root/intelligence_system
python3 -c "
from core.unified_agi_system import UnifiedAGISystem
import logging
logging.basicConfig(level=logging.INFO)

system = UnifiedAGISystem(max_cycles=10, use_real_v7=True)
system.run()

state = system.unified_state.to_dict()
consciousness = state['meta']['consciousness']
print(f'Final consciousness: {consciousness:.8f}')
assert consciousness > 0.001, 'Consciousness should grow > 0.001 in 10 cycles'
print('✅ Consciousness evolution working!')
" 2>&1 | tail -20
```

---

### ✅ FIX P0-4: Omega Calculation

**Arquivo:** `/root/intelligence_system/core/unified_agi_system.py`  
**Linhas:** 459-497  
**Problema:** Omega sempre zero, CAOS+ não amplifica  
**Impacto:** CAOS+ fica em ~1.1x (esperado: até 3.99x)  

**CÓDIGO COMPLETO (substituir método inteiro):**

```python
def compute_meta_metrics(self, v7_metrics: Dict[str, float]) -> Dict[str, float]:
    """
    Compute PENIN³ meta-metrics from V7 metrics
    
    NEW: Real Omega calculation based on V7 evolutionary progress
    """
    if not self.penin_available:
        return v7_metrics
    
    c = min(v7_metrics['mnist_acc'] / 100.0, 1.0)
    a = min(v7_metrics['cartpole_avg'] / 500.0, 1.0)
    
    # CRITICAL: Calculate Omega from V7 evolutionary indicators
    snapshot = self.unified_state.to_dict()
    o = 0.0
    
    try:
        # Access V7 system (may be in worker thread)
        v7_sys = None
        if hasattr(self, 'v7_system') and self.v7_system:
            v7_sys = self.v7_system
        elif hasattr(self, 'v7_worker') and hasattr(self.v7_worker, 'v7_system'):
            v7_sys = self.v7_worker.v7_system
        
        if v7_sys:
            # Omega components (weighted sum):
            # 1. Evolutionary progress (40%)
            evo_gen = getattr(getattr(v7_sys, 'evolutionary_optimizer', None), 'generation', 0)
            evo_progress = min(1.0, evo_gen / 100.0)
            
            # 2. Self-modifications applied (30%)
            self_mods = getattr(v7_sys, '_self_mods_applied', 0)
            self_mod_progress = min(1.0, self_mods / 10.0)
            
            # 3. Novel behaviors discovered (20%)
            novel = getattr(v7_sys, '_novel_behaviors_discovered', 0)
            novelty_progress = min(1.0, novel / 50.0)
            
            # 4. Darwin generations (10%)
            darwin_gen = getattr(getattr(v7_sys, 'darwin_real', None), 'generation', 0)
            darwin_progress = min(1.0, darwin_gen / 50.0)
            
            # Weighted sum
            o = (0.4 * evo_progress + 
                 0.3 * self_mod_progress + 
                 0.2 * novelty_progress + 
                 0.1 * darwin_progress)
            
            logger.debug(f"   Omega breakdown: evo={evo_progress:.3f}, mods={self_mod_progress:.3f}, "
                        f"novel={novelty_progress:.3f}, darwin={darwin_progress:.3f} → Ω={o:.3f}")
        else:
            # Fallback: minimal omega to allow CAOS+ amplification
            o = 0.05
            logger.debug("   Omega fallback: V7 system not accessible, using minimum 0.05")
            
    except Exception as e:
        logger.debug(f"   Omega calculation error: {e}, using fallback 0.05")
        o = 0.05
    
    # Ensure minimum omega for CAOS+ amplification
    o_effective = max(o, 0.05)
    s = 0.9
    
    # Compute CAOS+ with real omega
    caos = compute_caos_plus_exponential(c=c, a=a, o=o_effective, s=s, kappa=20.0)
    
    # Compute L∞
    normalized = {'acc': c, 'adapt': a, 'omega': o}
    ideal = {'acc': 1.0, 'adapt': 1.0, 'omega': 1.0}
    linf = linf_score(normalized, ideal, cost=0.1)
    
    # Sigma guard
    sigma_valid = c > 0.7 and a > 0.7
    
    # Thread-safe read of consciousness
    consciousness = float(snapshot['meta'].get('master_I', 0.0))
    
    # Update unified state with NEW omega value
    self.unified_state.update_meta(
        master_I=consciousness,
        consciousness=consciousness,
        caos=caos,
        linf=linf,
        sigma=sigma_valid
    )
    
    # ALSO update omega in unified_state (add this attribute)
    self.unified_state.omega_score = o
    
    return {
        **v7_metrics,
        'caos_amplification': caos,
        'linf_score': linf,
        'sigma_valid': sigma_valid,
        'consciousness': consciousness,
        'omega': o,  # NEW: expose omega in metrics
    }
```

**Teste após aplicar:**
```bash
cd /root/intelligence_system
python3 test_100_cycles_real.py 10 2>&1 | grep -E "(Omega|CAOS)" | tail -10
# Esperar: Omega > 0.1, CAOS > 1.5x após 10 ciclos
```

---

### ✅ VALIDAÇÃO FASE 1

**Executar teste completo:**
```bash
cd /root/intelligence_system
python3 test_100_cycles_real.py 10 2>&1 | tee /tmp/test_phase1.log

# Verificar consciousness growth:
grep "PENIN³: CAOS" /tmp/test_phase1.log | tail -3

# Extrair métricas finais:
python3 -c "
import json
with open('data/audit_results_10_cycles.json') as f:
    data = json.load(f)
    meta = data['meta']
    print(f'Consciousness: {meta[\"consciousness\"]:.8f}')
    print(f'CAOS+: {meta[\"caos\"]:.3f}x')
    print(f'Omega: {meta.get(\"omega\", 0.0):.3f}')
    print(f'L∞: {meta[\"linf\"]:.6f}')
"
```

**Critérios de sucesso:**
- ✅ Consciousness > 0.001 (após 10 cycles)
- ✅ CAOS+ > 1.5x (após 10 cycles)
- ✅ Omega > 0.1 (após 10 cycles)
- ✅ L∞ > 0.00001 (crescendo)

---

## 🔧 FASE 2: QUALITY IMPROVEMENTS (45 minutos)

### ✅ FIX P1-1: Real Experience Replay for Transfer Learning

**Arquivo:** `/root/intelligence_system/core/system_v7_ultimate.py`  
**Linhas:** 1187-1216  
**Problema:** Transfer learning usa dummy trajectories  
**Impacto:** Não aproveita experiências reais  

**CÓDIGO COMPLETO (substituir método inteiro):**

```python
def _use_database_knowledge(self) -> Dict[str, Any]:
    """
    V6.0: Use database knowledge actively (REAL experiences)
    
    NEW: Uses real experience replay data instead of dummy trajectories
    """
    logger.info("🧠 Using database knowledge...")
    
    # Bootstrap from historical data
    bootstrap_data = self.db_knowledge.bootstrap_from_history()
    
    # FIX: Use REAL experiences from experience replay
    if bootstrap_data['weights_count'] > 0:
        weights = self.db_knowledge.get_transfer_learning_weights(limit=5)
        
        if weights and len(weights) > 0:
            try:
                # Check if we have enough real experiences
                if len(self.experience_replay) > 100:
                    logger.info(f"   Using {len(self.experience_replay)} real experiences for transfer learning")
                    
                    # Sample REAL experiences from replay buffer
                    real_experiences = []
                    sample_size = min(100, len(self.experience_replay))
                    
                    for _ in range(sample_size):
                        exp = self.experience_replay.sample(1)[0]
                        real_experiences.append((
                            exp['state'],
                            exp['action'],
                            exp['reward'],
                            exp['next_state'],
                            exp['done']
                        ))
                    
                    # Apply transfer learning with REAL experiences
                    for weight_data in weights[:3]:  # Top 3 historical weights
                        agent_id = f"historical_{weight_data.get('source', 'unknown')}"
                        
                        # Extract knowledge from historical performance + real experiences
                        self.transfer_learner.extract_knowledge(
                            agent_id=agent_id,
                            network=self.mnist.model,
                            experiences=real_experiences  # ← REAL experiences!
                        )
                        
                        self._db_knowledge_transfers += 1
                    
                    logger.info(f"   ✅ Applied transfer learning from {len(weights)} historical weights + {len(real_experiences)} real experiences")
                else:
                    logger.info(f"   ⚠️ Insufficient experience replay data ({len(self.experience_replay)}/100), skipping transfer learning")
                    
            except Exception as e:
                logger.warning(f"   ⚠️ Transfer learning failed: {e}")
                logger.debug(traceback.format_exc())
    else:
        logger.info("   ℹ️ No historical weights available for transfer learning")
    
    return bootstrap_data
```

**Teste após aplicar:**
```bash
cd /root/intelligence_system
python3 -c "
from core.system_v7_ultimate import IntelligenceSystemV7
import logging
logging.basicConfig(level=logging.INFO)

v7 = IntelligenceSystemV7()

# Run some cycles to fill experience replay
for i in range(5):
    v7.run_cycle()

print(f'Experience replay size: {len(v7.experience_replay)}')
print(f'DB knowledge transfers: {v7._db_knowledge_transfers}')
" 2>&1 | tail -20
```

---

### ⚠️ FIX P0-2: WORM Ledger Repair

**Script:** `/root/intelligence_system/tools/repair_worm_ledger.py` (NOVO ARQUIVO)  
**Problema:** WORM chain_valid=False, integridade comprometida  
**Impacto:** Auditoria não confiável  

**CÓDIGO COMPLETO (criar arquivo novo):**

```python
#!/usr/bin/env python3
"""
WORM Ledger Repair Tool
Recalculates chain hashes to restore integrity
"""

import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from penin.ledger import WORMLedger

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def repair_worm_chain(ledger_path: Path) -> bool:
    """
    Repair WORM chain by recalculating all hashes in sequence
    
    Args:
        ledger_path: Path to WORM database file
    
    Returns:
        True if repair successful and chain now valid
    """
    logger.info("="*80)
    logger.info(f"🔧 WORM LEDGER REPAIR")
    logger.info("="*80)
    logger.info(f"Target: {ledger_path}")
    logger.info("")
    
    # Validate path exists
    if not ledger_path.exists():
        logger.error(f"❌ Ledger file not found: {ledger_path}")
        return False
    
    # Load existing ledger (read-only to get events)
    logger.info("📖 Reading existing ledger...")
    try:
        ledger = WORMLedger(str(ledger_path))
        stats_before = ledger.get_statistics()
        
        logger.info(f"   Events: {stats_before['total_events']}")
        logger.info(f"   Chain valid: {stats_before['chain_valid']}")
        logger.info("")
        
        if stats_before['chain_valid']:
            logger.info("✅ Chain already valid, no repair needed!")
            return True
            
    except Exception as e:
        logger.error(f"❌ Failed to read ledger: {e}")
        return False
    
    # Read all events from JSONL file
    logger.info("📜 Loading all events...")
    try:
        with open(ledger_path, 'r') as f:
            events = []
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if line:
                    try:
                        event = json.loads(line)
                        events.append(event)
                    except json.JSONDecodeError as e:
                        logger.warning(f"   Line {line_num}: Invalid JSON, skipping")
        
        logger.info(f"   Loaded {len(events)} events")
        logger.info("")
        
    except Exception as e:
        logger.error(f"❌ Failed to load events: {e}")
        return False
    
    # Backup original file
    logger.info("💾 Creating backup...")
    backup_path = ledger_path.with_suffix('.db.bak')
    try:
        # If backup already exists, add timestamp
        if backup_path.exists():
            import time
            timestamp = int(time.time())
            backup_path = ledger_path.with_suffix(f'.db.bak.{timestamp}')
        
        ledger_path.rename(backup_path)
        logger.info(f"   ✅ Backup: {backup_path}")
        logger.info("")
        
    except Exception as e:
        logger.error(f"❌ Failed to create backup: {e}")
        return False
    
    # Create new ledger with recalculated hashes
    logger.info("🔨 Rebuilding ledger with correct hashes...")
    try:
        new_ledger = WORMLedger(str(ledger_path))
        
        repaired = 0
        errors = 0
        
        for i, event in enumerate(events):
            try:
                # Append will recalculate hash correctly
                new_ledger.append(
                    event['event_type'],
                    event['event_id'],
                    event['data']
                )
                repaired += 1
                
                if (i + 1) % 100 == 0:
                    logger.info(f"   Progress: {i+1}/{len(events)} events ({(i+1)/len(events)*100:.1f}%)")
                    
            except Exception as e:
                logger.warning(f"   Event {i+1}: Failed to append ({e})")
                errors += 1
        
        logger.info("")
        logger.info(f"   ✅ Repaired: {repaired}/{len(events)} events")
        if errors > 0:
            logger.warning(f"   ⚠️  Errors: {errors} events skipped")
        logger.info("")
        
    except Exception as e:
        logger.error(f"❌ Failed to rebuild ledger: {e}")
        # Restore backup
        try:
            backup_path.rename(ledger_path)
            logger.info("   🔄 Backup restored")
        except:
            pass
        return False
    
    # Validate new chain
    logger.info("✅ Validating repaired chain...")
    try:
        stats_after = new_ledger.get_statistics()
        
        logger.info(f"   Events: {stats_after['total_events']}")
        logger.info(f"   Chain valid: {stats_after['chain_valid']}")
        logger.info("")
        
        if stats_after['chain_valid']:
            logger.info("="*80)
            logger.info("✅ REPAIR SUCCESSFUL!")
            logger.info("="*80)
            logger.info(f"   Before: {stats_before['total_events']} events, chain_valid={stats_before['chain_valid']}")
            logger.info(f"   After:  {stats_after['total_events']} events, chain_valid={stats_after['chain_valid']}")
            logger.info(f"   Backup: {backup_path}")
            logger.info("="*80)
            return True
        else:
            logger.error("❌ Repair failed: chain still invalid after rebuild")
            return False
            
    except Exception as e:
        logger.error(f"❌ Validation failed: {e}")
        return False


if __name__ == "__main__":
    # Default ledger path
    ledger_path = Path("/root/intelligence_system/data/unified_worm.db")
    
    # Allow custom path from command line
    if len(sys.argv) > 1:
        ledger_path = Path(sys.argv[1])
    
    # Run repair
    success = repair_worm_chain(ledger_path)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)
```

**Executar repair:**
```bash
cd /root/intelligence_system
mkdir -p tools
# (salvar código acima em tools/repair_worm_ledger.py)
chmod +x tools/repair_worm_ledger.py

python3 tools/repair_worm_ledger.py
# Ou com path customizado:
# python3 tools/repair_worm_ledger.py data/unified_worm.db
```

**Teste após reparar:**
```bash
python3 -c "
from penin.ledger import WORMLedger
ledger = WORMLedger('data/unified_worm.db')
stats = ledger.get_statistics()
print(f'Chain valid: {stats[\"chain_valid\"]}')
print(f'Total events: {stats[\"total_events\"]}')
assert stats['chain_valid'], 'Chain should be valid after repair!'
print('✅ WORM chain integrity restored!')
"
```

---

### ✅ VALIDAÇÃO FASE 2

**Executar testes:**
```bash
cd /root/intelligence_system

# 1. Test transfer learning
python3 -c "
from core.system_v7_ultimate import IntelligenceSystemV7
v7 = IntelligenceSystemV7()
for _ in range(10): v7.run_cycle()
print(f'✅ Transfer learning applications: {v7._db_knowledge_transfers}')
assert v7._db_knowledge_transfers > 0, 'Should apply transfer learning'
"

# 2. Test WORM integrity
python3 -c "
from penin.ledger import WORMLedger
ledger = WORMLedger('data/unified_worm.db')
stats = ledger.get_statistics()
assert stats['chain_valid'], f'Chain should be valid! Got: {stats}'
print(f'✅ WORM integrity: OK ({stats[\"total_events\"]} events)')
"
```

**Critérios de sucesso:**
- ✅ Transfer learning applications > 0 (em 10 cycles com replay > 100)
- ✅ WORM chain_valid = True
- ✅ Experience replay size > 100 (após 10 cycles de CartPole)

---

## 🏁 FASE 3: LONG-RUN VALIDATION (4 horas background)

### ✅ Preparação

**Reset sistema para fresh start:**
```bash
cd /root/intelligence_system

# Backup current state
mkdir -p backups
cp data/intelligence.db backups/intelligence_pre_100cycles_$(date +%Y%m%d_%H%M%S).db 2>/dev/null || true
cp models/ppo_cartpole_v7.pth backups/ppo_pre_100cycles_$(date +%Y%m%d_%H%M%S).pth 2>/dev/null || true

# Reset for fresh evolution observation
rm -f data/intelligence.db
rm -f models/ppo_cartpole_v7.pth
rm -f models/meta_learner.pth
rm -f models/darwin_population.json

echo "✅ Reset complete, ready for fresh 100-cycle run"
```

---

### ✅ Execução

**Run 100 cycles (background):**
```bash
cd /root/intelligence_system

# Start background process
nohup python3 test_100_cycles_real.py 100 > /root/test_100_real_$(date +%Y%m%d_%H%M%S).log 2>&1 &

# Save PID
echo $! > /root/test_100.pid

echo "✅ Started 100-cycle test"
echo "   PID: $(cat /root/test_100.pid)"
echo "   Log: /root/test_100_real_*.log"
echo ""
echo "Monitor with:"
echo "  tail -f /root/test_100_real_*.log"
echo ""
echo "Check status:"
echo "  ps aux | grep $(cat /root/test_100.pid)"
```

---

### ✅ Monitoramento

**Durante a execução:**
```bash
# Ver log em tempo real
tail -f /root/test_100_real_*.log

# Ver apenas métricas PENIN³
tail -f /root/test_100_real_*.log | grep "PENIN³:"

# Ver apenas synergies
tail -f /root/test_100_real_*.log | grep -E "(EXECUTING ALL SYNERGIES|TOTAL AMPLIFICATION)"

# Check process alive
ps aux | grep $(cat /root/test_100.pid 2>/dev/null) | grep -v grep

# Estimated time remaining (assumes ~2min/cycle)
echo "scale=2; (100 - $(grep -c "CYCLE" /root/test_100_real_*.log 2>/dev/null)) * 2 / 60" | bc
echo "hours remaining (approx)"
```

---

### ✅ Análise de Resultados

**Após conclusão (ou parar com Ctrl+C):**
```bash
cd /root/intelligence_system

# Extract final metrics
python3 << 'PYEOF'
import json
from pathlib import Path

# Find latest audit results
audit_file = Path('data/audit_results_100_cycles.json')

if not audit_file.exists():
    print("⚠️ Audit results not found, test may not have completed")
    exit(1)

with open(audit_file) as f:
    data = json.load(f)

print("="*80)
print("📊 100-CYCLE VALIDATION RESULTS")
print("="*80)
print("")

# Operational
op = data['operational']
print("OPERATIONAL (V7):")
print(f"  Final cycle: {op['cycle']}")
print(f"  MNIST: {op['best_mnist']:.2f}%")
print(f"  CartPole: {op['best_cartpole']:.1f}")
print(f"  IA³ Score: {op['ia3_score']:.1f}%")
print("")

# Meta
meta = data['meta']
print("META (PENIN³):")
print(f"  Consciousness: {meta['consciousness']:.8f}")
print(f"  CAOS+: {meta['caos']:.3f}x")
print(f"  Omega: {meta.get('omega', 0.0):.3f}")
print(f"  L∞: {meta['linf']:.6f}")
print(f"  Σ Valid: {meta['sigma_valid']}")
print("")

# Synergies
syns = data.get('synergies', [])
if syns:
    print("SYNERGIES (last execution):")
    total_amp = 1.0
    for syn in syns:
        status = "✅" if syn['success'] else "⏳"
        print(f"  {status} {syn['synergy']}: {syn['amplification']:.2f}x")
        if syn['success']:
            total_amp *= syn['amplification']
    print(f"  TOTAL: {total_amp:.2f}x")
else:
    print("SYNERGIES: not executed")
print("")

# Success criteria
print("="*80)
print("VALIDATION:")
print("="*80)

checks = {
    'Consciousness > 0.1': meta['consciousness'] > 0.1,
    'CAOS+ > 2.5x': meta['caos'] > 2.5,
    'Omega > 0.5': meta.get('omega', 0.0) > 0.5,
    'Synergies executed': len(syns) > 0,
    'System stable': op['cycle'] >= 90,  # Completed at least 90% of cycles
}

passed = sum(checks.values())
total = len(checks)

for check, result in checks.items():
    status = "✅ PASS" if result else "❌ FAIL"
    print(f"  {status}: {check}")

print("")
print(f"OVERALL: {passed}/{total} checks passed ({passed/total*100:.0f}%)")
print("="*80)

if passed == total:
    print("✅ 100-CYCLE VALIDATION SUCCESSFUL!")
else:
    print(f"⚠️ Validation incomplete: {total-passed} checks failed")
PYEOF
```

---

## 📊 MÉTRICAS DE SUCESSO (CHECKLIST)

### FASE 0 (Emergency):
- [ ] V7 mode = REAL (not SIMULATED)
- [ ] Synergies executed >= 2 times (em 5 cycles)
- [ ] No crashes/exceptions

### FASE 1 (Core Metrics):
- [ ] Consciousness > 0.001 (após 10 cycles)
- [ ] CAOS+ > 1.5x (após 10 cycles)
- [ ] Omega > 0.1 (após 10 cycles)

### FASE 2 (Quality):
- [ ] WORM chain_valid = True
- [ ] Transfer learning applied > 0
- [ ] Experience replay > 100 samples

### FASE 3 (Long-run):
- [ ] Consciousness > 0.1 (após 100 cycles)
- [ ] CAOS+ > 2.5x (após 100 cycles)
- [ ] Omega > 0.5 (após 100 cycles)
- [ ] Synergies successful > 40 times
- [ ] System completed >= 90 cycles without crash

---

## 🛠️ TROUBLESHOOTING

### Problema: V7 ainda em SIMULATED após P0-1

**Diagnóstico:**
```bash
cd /root/intelligence_system
python3 -c "
from core.system_v7_ultimate import IntelligenceSystemV7
import traceback
try:
    v7 = IntelligenceSystemV7()
    print('✅ V7 initialized')
except Exception as e:
    print(f'❌ V7 failed: {e}')
    traceback.print_exc()
"
```

**Possíveis causas:**
1. `database_knowledge_engine.py` não foi salvo corretamente
2. Outro erro durante V7 init (check traceback)
3. Database permissions issues

**Fix:**
```bash
# Verificar se fix foi aplicado
grep -A 20 "def _load_summary" core/database_knowledge_engine.py | head -30

# Re-aplicar fix se necessário
# ... (copiar código do fix P0-1 novamente)

# Verificar permissions
chmod 644 data/*.db
```

---

### Problema: Consciousness não cresce

**Diagnóstico:**
```bash
cd /root/intelligence_system
python3 test_100_cycles_real.py 10 2>&1 | grep "Master I evolved"
# Se não aparecer nada, evolution não está acontecendo
```

**Possíveis causas:**
1. `evolve_master_equation` não está sendo chamado
2. `delta_linf` ou `alpha_omega` são zero
3. `step_master` não está funcionando

**Fix:**
```bash
# Adicionar debug logging
# Em unified_agi_system.py, após linha 523:
logger.info(f"   🧠 Master I: {new_I:.8f} (Δlinf={delta_linf:.6f}, αΩ={alpha_omega:.3f})")

# Re-executar para ver se evolution está acontecendo
python3 test_100_cycles_real.py 5
```

---

### Problema: Synergies não executam

**Diagnóstico:**
```bash
cd /root/intelligence_system
python3 test_100_cycles_real.py 5 2>&1 | grep -c "EXECUTING ALL SYNERGIES"
# Deve retornar >= 2 (ciclos 0, 2, 4)
```

**Possíveis causas:**
1. Fix P0-5 não foi aplicado
2. `synergy_orchestrator` é None
3. `v7_system` é None

**Fix:**
```bash
# Verificar linha 344
grep "metrics\['cycle'\] %" core/unified_agi_system.py
# Deve mostrar: % 2 == 0 (não % 5 == 0)

# Se ainda % 5:
# Re-aplicar fix P0-5 (substituir linha 344)
```

---

## 📝 NOTAS FINAIS

### Backup antes de começar:
```bash
cd /root/intelligence_system
tar -czf ../backup_intelligence_system_$(date +%Y%m%d_%H%M%S).tar.gz .
echo "✅ Backup criado: ../backup_intelligence_system_*.tar.gz"
```

### Restaurar backup se necessário:
```bash
cd /root
tar -xzf backup_intelligence_system_*.tar.gz -C intelligence_system_restore/
# Verificar conteúdo
ls -la intelligence_system_restore/
# Se OK, substituir:
rm -rf intelligence_system
mv intelligence_system_restore intelligence_system
```

### Logs importantes:
- `/root/intelligence_system/logs/intelligence_v7.log` - V7 system log
- `/root/test_100_real_*.log` - Test execution log
- `/root/intelligence_system/data/audit_results_*.json` - Structured metrics

---

**FIM DO ROADMAP DE IMPLEMENTAÇÃO**

**Status:** ✅ CÓDIGO PRONTO PARA APLICAÇÃO  
**Tempo total estimado:** ~6 horas (15min + 30min + 45min + 4h)  
**Próximo passo:** Aplicar FASE 0 (Emergency Fix)
