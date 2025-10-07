#!/usr/bin/env python3
"""
TESTE DE INTEGRAÇÃO DARWINACCI + V7
Testa se o V7 consegue ativar e usar o Darwinacci corretamente
"""

import sys
import os
import logging
from datetime import datetime

# Add project root(s) to Python path
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/intelligence_system')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_darwinacci_import():
    """Testa import do Darwinacci"""
    logger.info("🔍 Testing Darwinacci import...")
    try:
        from extracted_algorithms.darwin_engine_darwinacci import DarwinacciOrchestrator
        logger.info("✅ Darwinacci import successful")
        return True, DarwinacciOrchestrator
    except ImportError as e:
        logger.error(f"❌ Darwinacci import failed: {e}")
        return False, None
    except Exception as e:
        logger.error(f"💥 Darwinacci critical error: {e}")
        return False, None

def test_darwinacci_instantiation(OrchestratorClass):
    """Testa instanciação do Darwinacci"""
    logger.info("🔧 Testing Darwinacci instantiation...")
    try:
        orch = OrchestratorClass(population_size=10, max_cycles=2, seed=42)
        logger.info("✅ Darwinacci instantiation successful")
        return True, orch
    except Exception as e:
        logger.error(f"❌ Darwinacci instantiation failed: {e}")
        return False, None

def test_darwinacci_activation(orch):
    """Testa ativação do Darwinacci"""
    logger.info("🚀 Testing Darwinacci activation...")
    try:
        success = orch.activate()
        if success:
            logger.info("✅ Darwinacci activation successful")
            return True
        else:
            logger.warning("⚠️ Darwinacci activation returned False")
            return False
    except Exception as e:
        logger.error(f"❌ Darwinacci activation failed: {e}")
        return False

def test_darwinacci_evolution(orch):
    """Testa uma geração de evolução"""
    logger.info("🧬 Testing Darwinacci evolution...")
    try:
        # Função de fitness simples
        def fitness_fn(ind):
            genome = ind.genome
            # Fitness baseado em parâmetros do genome
            score = 0
            if 'hidden_size' in genome:
                score += genome['hidden_size'] / 256.0
            if 'learning_rate' in genome:
                score += genome['learning_rate'] * 10
            return score

        stats = orch.evolve_generation(fitness_fn=fitness_fn)
        if stats and 'best_fitness' in stats:
            logger.info(f"✅ Darwinacci evolution successful - Best fitness: {stats['best_fitness']:.4f}")
            return True
        else:
            logger.warning("⚠️ Darwinacci evolution returned invalid stats")
            return False
    except Exception as e:
        logger.error(f"❌ Darwinacci evolution failed: {e}")
        return False

def test_v7_darwinacci_logic():
    """Testa a lógica de seleção Darwinacci no V7"""
    logger.info("🔄 Testing V7 Darwinacci selection logic...")

    # Simula a lógica do V7
    try:
        from extracted_algorithms.darwin_engine_darwinacci import DarwinacciOrchestrator
        _DARWINACCI_AVAILABLE = True
        logger.info("✅ V7 logic: Darwinacci available")
    except ImportError:
        _DARWINACCI_AVAILABLE = False
        logger.warning("⚠️ V7 logic: Darwinacci not available")
    except Exception as e:
        _DARWINACCI_AVAILABLE = False
        logger.error(f"❌ V7 logic: Darwinacci error: {e}")

    if _DARWINACCI_AVAILABLE:
        try:
            logger.info("🌟 Attempting V7-style Darwinacci activation...")
            darwin_real = DarwinacciOrchestrator(population_size=10, max_cycles=2, seed=42)
            success = darwin_real.activate()
            if success:
                logger.info("✅ V7-style activation successful")
                using_darwinacci = True
            else:
                logger.warning("⚠️ V7-style activation returned False")
                using_darwinacci = False
        except Exception as e:
            logger.error(f"❌ V7-style activation failed: {e}")
            using_darwinacci = False

        if not using_darwinacci:
            logger.info("🔥 V7 would fallback to original Darwin")
            try:
                from extracted_algorithms.darwin_engine_real import DarwinOrchestrator
                darwin_real = DarwinOrchestrator(population_size=10, survival_rate=0.4, sexual_rate=0.8)
                darwin_real.activate()
                logger.info("✅ V7 fallback successful")
            except Exception as e:
                logger.error(f"❌ V7 fallback failed: {e}")
                return False

    return True

def main():
    """Main test function"""
    logger.info("🚀 STARTING DARWINACCI + V7 INTEGRATION TESTS")
    logger.info("=" * 60)

    results = {}

    # Test 1: Import
    success, OrchestratorClass = test_darwinacci_import()
    results['import'] = success

    if success:
        # Test 2: Instantiation
        success, orch = test_darwinacci_instantiation(OrchestratorClass)
        results['instantiation'] = success

        if success:
            # Test 3: Activation
            success = test_darwinacci_activation(orch)
            results['activation'] = success

            if success:
                # Test 4: Evolution
                success = test_darwinacci_evolution(orch)
                results['evolution'] = success

    # Test 5: V7 Logic
    success = test_v7_darwinacci_logic()
    results['v7_logic'] = success

    # Summary
    logger.info("=" * 60)
    logger.info("📊 TEST RESULTS SUMMARY:")
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"   {test.upper()}: {status}")

    total_passed = sum(results.values())
    total_tests = len(results)

    logger.info(f"🎯 OVERALL: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        logger.info("🎉 ALL TESTS PASSED - DARWINACCI INTEGRATION READY!")
        return True
    else:
        logger.warning("⚠️ SOME TESTS FAILED - INVESTIGATION NEEDED")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)