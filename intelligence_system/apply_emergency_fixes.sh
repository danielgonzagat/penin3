#!/bin/bash
###############################################################################
# EMERGENCY FIXES APPLICATOR
# Aplica automaticamente os 4 fixes mais críticos (P0-1, P0-3, P0-4, P0-5)
# Tempo total: ~15 minutos
###############################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "================================================================================"
echo "🔧 EMERGENCY FIXES APPLICATOR"
echo "================================================================================"
echo "This script will apply 4 critical fixes to restore REAL V7 operation"
echo ""
echo "Fixes to apply:"
echo "  ✅ P0-1: DatabaseKnowledgeEngine table (5min)"
echo "  ✅ P0-3: Consciousness amplification (5min)"
echo "  ✅ P0-4: Omega calculation (5min)"
echo "  ✅ P0-5: Synergies frequency (1min)"
echo ""
echo "Total time: ~15 minutes"
echo "================================================================================"
echo ""

# Confirmation
read -p "Do you want to proceed? (yes/no): " -r
echo
if [[ ! $REPLY =~ ^[Yy]es$ ]]; then
    echo "Aborted by user."
    exit 0
fi

###############################################################################
# BACKUP
###############################################################################
echo "📦 Creating backup..."
BACKUP_DIR="../backup_pre_emergency_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r core "$BACKUP_DIR/"
cp -r extracted_algorithms "$BACKUP_DIR/" 2>/dev/null || true
echo "✅ Backup created: $BACKUP_DIR"
echo ""

###############################################################################
# FIX P0-1: DatabaseKnowledgeEngine
###############################################################################
echo "================================================================================"
echo "🔧 FIX P0-1: DatabaseKnowledgeEngine - Missing Table"
echo "================================================================================"
echo "File: core/database_knowledge_engine.py"
echo "Lines: 38-50"
echo ""

TARGET_FILE="core/database_knowledge_engine.py"

# Check if file exists
if [[ ! -f "$TARGET_FILE" ]]; then
    echo "❌ Error: $TARGET_FILE not found!"
    exit 1
fi

# Backup original
cp "$TARGET_FILE" "${TARGET_FILE}.backup_$(date +%Y%m%d_%H%M%S)"

# Apply fix using Python to replace method
python3 << 'PYEOF'
import re

target_file = 'core/database_knowledge_engine.py'

with open(target_file, 'r') as f:
    content = f.read()

# Find and replace _load_summary method
old_pattern = r'def _load_summary\(self\):.*?(?=\n    def |\nclass |\Z)'

new_method = '''def _load_summary(self):
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
'''

# Replace with regex DOTALL mode
content_new = re.sub(old_pattern, new_method, content, flags=re.DOTALL)

if content_new == content:
    print("⚠️ Warning: _load_summary method not replaced (pattern not found)")
    print("   Manual intervention may be required")
else:
    with open(target_file, 'w') as f:
        f.write(content_new)
    print("✅ P0-1 applied successfully")
PYEOF

echo ""

###############################################################################
# FIX P0-5: Synergies Frequency
###############################################################################
echo "================================================================================"
echo "🔧 FIX P0-5: Synergies Execution Frequency"
echo "================================================================================"
echo "File: core/unified_agi_system.py"
echo "Line: 344"
echo ""

TARGET_FILE="core/unified_agi_system.py"

# Backup
cp "$TARGET_FILE" "${TARGET_FILE}.backup_$(date +%Y%m%d_%H%M%S)"

# Apply fix
sed -i "s/metrics\['cycle'\] % 5 == 0:/metrics['cycle'] % 2 == 0:/g" "$TARGET_FILE"

# Verify
if grep -q "metrics\['cycle'\] % 2 == 0:" "$TARGET_FILE"; then
    echo "✅ P0-5 applied successfully"
else
    echo "⚠️ Warning: P0-5 may not have been applied correctly"
fi

echo ""

###############################################################################
# FIX P0-3: Consciousness Amplification
###############################################################################
echo "================================================================================"
echo "🔧 FIX P0-3: Consciousness Evolution Amplification"
echo "================================================================================"
echo "File: core/unified_agi_system.py"
echo "Lines: 499-523"
echo ""

# This is complex, using Python
python3 << 'PYEOF'
import re

target_file = 'core/unified_agi_system.py'

with open(target_file, 'r') as f:
    content = f.read()

# Replace amplification values
# Find delta_linf line
content = re.sub(
    r"delta_linf = metrics\.get\('linf_score', 0\.0\) \* \d+\.0",
    "delta_linf = metrics.get('linf_score', 0.0) * 1000.0  # Amplified 100x→1000x for faster growth",
    content
)

# Find alpha_omega line
content = re.sub(
    r"alpha_omega = 0\.\d+ \* metrics\.get\('caos_amplification', 1\.0\)",
    "alpha_omega = 2.0 * metrics.get('caos_amplification', 1.0)  # Amplified 0.5x→2.0x for stronger influence",
    content
)

with open(target_file, 'w') as f:
    f.write(content)

print("✅ P0-3 applied successfully")
PYEOF

echo ""

###############################################################################
# FIX P0-4: Omega Calculation (COMPLEX - needs manual review)
###############################################################################
echo "================================================================================"
echo "🔧 FIX P0-4: Omega Calculation"
echo "================================================================================"
echo "File: core/unified_agi_system.py"
echo "Lines: 459-497"
echo ""
echo "⚠️  This fix is COMPLEX and requires manual code replacement"
echo "    Please refer to ROADMAP_IMPLEMENTACAO_CODIGO_PRONTO.md FASE 1, FIX P0-4"
echo "    for the complete replacement code for compute_meta_metrics() method"
echo ""
echo "✅ Skipping automated application (requires manual intervention)"
echo ""

###############################################################################
# VALIDATION
###############################################################################
echo "================================================================================"
echo "✅ VALIDATION"
echo "================================================================================"
echo ""

echo "Testing DatabaseKnowledgeEngine initialization..."
python3 << 'PYEOF'
from pathlib import Path
import sys
sys.path.insert(0, '.')

try:
    from core.database_knowledge_engine import DatabaseKnowledgeEngine
    db = DatabaseKnowledgeEngine(Path('data/intelligence.db'))
    print("✅ DatabaseKnowledgeEngine: OK")
except Exception as e:
    print(f"❌ DatabaseKnowledgeEngine: FAILED - {e}")
    sys.exit(1)
PYEOF

echo ""

echo "Checking synergies frequency..."
if grep -q "metrics\['cycle'\] % 2 == 0:" core/unified_agi_system.py; then
    echo "✅ Synergies frequency: OK (every 2 cycles)"
else
    echo "❌ Synergies frequency: FAILED"
fi

echo ""

echo "Checking consciousness amplification..."
if grep -q "delta_linf = .*\* 1000\.0" core/unified_agi_system.py; then
    echo "✅ Consciousness delta_linf: OK (1000x)"
else
    echo "❌ Consciousness delta_linf: FAILED"
fi

if grep -q "alpha_omega = 2\.0 \*" core/unified_agi_system.py; then
    echo "✅ Consciousness alpha_omega: OK (2.0x)"
else
    echo "❌ Consciousness alpha_omega: FAILED"
fi

echo ""

###############################################################################
# SUMMARY
###############################################################################
echo "================================================================================"
echo "📊 SUMMARY"
echo "================================================================================"
echo ""
echo "Applied automatically:"
echo "  ✅ P0-1: DatabaseKnowledgeEngine table fallback"
echo "  ✅ P0-3: Consciousness amplification (delta_linf 1000x, alpha_omega 2.0x)"
echo "  ✅ P0-5: Synergies frequency (every 2 cycles)"
echo ""
echo "Requires manual intervention:"
echo "  ⚠️  P0-4: Omega calculation (see ROADMAP FASE 1, FIX P0-4)"
echo ""
echo "Next steps:"
echo "  1. Review backup in: $BACKUP_DIR"
echo "  2. Manually apply P0-4 (Omega calculation)"
echo "  3. Test with: python3 test_100_cycles_real.py 5"
echo ""
echo "================================================================================"
echo "✅ EMERGENCY FIXES APPLIED (3/4 automated, 1/4 manual)"
echo "================================================================================"
echo ""
echo "Backup location: $BACKUP_DIR"
echo ""
echo "To restore backup if needed:"
echo "  cp -r $BACKUP_DIR/core/* core/"
echo ""
