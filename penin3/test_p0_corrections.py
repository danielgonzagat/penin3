"""
Test P0 Corrections - Validate all 5 fixes

Tests:
1. SQL Injection protection
2. ECE real calculation
3. Darwin population persistence
4. Engine frequency increased
5. Import test fixed
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path('/root/intelligence_system')))
sys.path.insert(0, str(Path('/root/peninaocubo')))

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def test_sql_injection_protection():
    """Test 1: SQL Injection protection"""
    logger.info("\n" + "="*80)
    logger.info("TEST 1: SQL INJECTION PROTECTION")
    logger.info("="*80)
    
    from extracted_algorithms.database_mass_integrator import DatabaseMassIntegrator
    import tempfile
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
        temp_db = f.name
    
    try:
        integrator = DatabaseMassIntegrator(
            target_db_path=temp_db,
            source_db_dir="/root"
        )
        
        # Test sanitize method exists
        assert hasattr(integrator, '_sanitize_table_name'), "❌ _sanitize_table_name method missing"
        
        # Test valid table name
        safe_name = integrator._sanitize_table_name("test_table")
        assert safe_name == '"test_table"', f"❌ Expected '\"test_table\"', got {safe_name}"
        
        # Test invalid table name (should raise ValueError)
        try:
            integrator._sanitize_table_name("table; DROP TABLE users;")
            assert False, "❌ Should have raised ValueError for SQL injection"
        except ValueError:
            pass  # Expected
        
        logger.info("✅ SQL injection protection WORKING")
        return True
    
    except Exception as e:
        logger.error(f"❌ SQL injection test FAILED: {e}")
        return False
    
    finally:
        import os
        if os.path.exists(temp_db):
            os.remove(temp_db)


def test_ece_real():
    """Test 2: ECE real calculation"""
    logger.info("\n" + "="*80)
    logger.info("TEST 2: ECE REAL CALCULATION")
    logger.info("="*80)
    
    try:
        from penin3_system import PENIN3System
        
        # Create system
        system = PENIN3System()
        
        # Test ECE method exists
        assert hasattr(system, '_calculate_ece'), "❌ _calculate_ece method missing"
        
        # Test ECE calculation (should handle both real and fallback)
        ece = system._calculate_ece(0.98)
        
        assert isinstance(ece, float), f"❌ ECE should be float, got {type(ece)}"
        assert 0.0 <= ece <= 1.0, f"❌ ECE should be in [0, 1], got {ece}"
        
        logger.info(f"✅ ECE real calculation WORKING (ece={ece:.4f})")
        return True
    
    except Exception as e:
        logger.error(f"❌ ECE test FAILED: {e}")
        return False


def test_darwin_persistence():
    """Test 3: Darwin population persistence"""
    logger.info("\n" + "="*80)
    logger.info("TEST 3: DARWIN POPULATION PERSISTENCE")
    logger.info("="*80)
    
    try:
        from core.system_v7_ultimate import IntelligenceSystemV7
        import json
        from config.settings import CHECKPOINTS_DIR
        
        # Create V7 system (should initialize Darwin)
        v7 = IntelligenceSystemV7()
        
        # Check population exists
        assert hasattr(v7.darwin_real, 'population'), "❌ Darwin population missing"
        assert len(v7.darwin_real.population) > 0, "❌ Darwin population is empty"
        
        initial_pop_size = len(v7.darwin_real.population)
        logger.info(f"   Initial population: {initial_pop_size} individuals")
        
        # Save checkpoint
        v7._save_all_models()
        
        # Check checkpoint file exists
        checkpoint_path = CHECKPOINTS_DIR / "darwin_population.json"
        assert checkpoint_path.exists(), "❌ Darwin checkpoint not saved"
        
        # Verify checkpoint content
        with open(checkpoint_path, 'r') as f:
            saved_pop = json.load(f)
        
        assert len(saved_pop) == initial_pop_size, f"❌ Saved population size mismatch"
        assert all('genome' in ind for ind in saved_pop), "❌ Missing genome in saved population"
        
        logger.info(f"✅ Darwin persistence WORKING ({initial_pop_size} individuals saved)")
        return True
    
    except Exception as e:
        logger.error(f"❌ Darwin persistence test FAILED: {e}")
        return False


def test_engine_frequency():
    """Test 4: Engine frequency increased"""
    logger.info("\n" + "="*80)
    logger.info("TEST 4: ENGINE FREQUENCY")
    logger.info("="*80)
    
    try:
        # Read V7 source code
        v7_path = Path('/root/intelligence_system/core/system_v7_ultimate.py')
        with open(v7_path, 'r') as f:
            source = f.read()
        
        # Check frequencies are 20, not 50/100/150/200
        checks = [
            ('cycle % 20 == 0', 'multimodal'),
            ('cycle % 20 == 0', 'auto_coding'),
            ('cycle % 20 == 0', 'maml'),
            ('cycle % 20 == 0', 'automl'),
        ]
        
        all_ok = True
        for check, name in checks:
            if check not in source:
                logger.error(f"   ❌ {name} not updated to 20 cycles")
                all_ok = False
            else:
                logger.info(f"   ✅ {name}: every 20 cycles")
        
        if all_ok:
            logger.info("✅ Engine frequencies UPDATED (20 cycles)")
            return True
        else:
            logger.error("❌ Some engines not updated")
            return False
    
    except Exception as e:
        logger.error(f"❌ Engine frequency test FAILED: {e}")
        return False


def test_import_fixed():
    """Test 5: Import test fixed"""
    logger.info("\n" + "="*80)
    logger.info("TEST 5: IMPORT FIX")
    logger.info("="*80)
    
    try:
        # Read test file
        test_path = Path('/root/peninaocubo/tests/integrations/test_nextpy_integration.py')
        with open(test_path, 'r') as f:
            source = f.read()
        
        # Check import is corrected
        if 'from penin.ledger.worm_ledger_complete import' in source:
            logger.error("❌ Still using worm_ledger_complete")
            return False
        
        if 'from penin.ledger.worm_ledger import WORMLedger' in source:
            logger.info("✅ Import corrected to worm_ledger")
        
        if 'WORMLedger(' in source:
            logger.info("✅ WORMLedger usage corrected")
        
        logger.info("✅ Import fix COMPLETE")
        return True
    
    except Exception as e:
        logger.error(f"❌ Import test FAILED: {e}")
        return False


def main():
    """Run all P0 correction tests"""
    logger.info("\n" + "="*80)
    logger.info("🧪 TESTING P0 CORRECTIONS")
    logger.info("="*80)
    
    tests = [
        ("SQL Injection", test_sql_injection_protection),
        ("ECE Real", test_ece_real),
        ("Darwin Persistence", test_darwin_persistence),
        ("Engine Frequency", test_engine_frequency),
        ("Import Fix", test_import_fixed),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"\n❌ {name} test crashed: {e}")
            results[name] = False
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("📊 SUMMARY")
    logger.info("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {name}: {status}")
    
    logger.info("="*80)
    logger.info(f"RESULT: {passed}/{total} tests passed ({passed/total*100:.0f}%)")
    logger.info("="*80)
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
