"""
Auditoria Forense Profunda - PENIN³
Testa TODOS os componentes de forma empírica
"""
import sys, time, traceback, os, psutil
from pathlib import Path
sys.path.insert(0, str(Path('/root/intelligence_system')))
sys.path.insert(0, str(Path('/root/peninaocubo')))

from penin3_system import PENIN3System
import numpy as np

def measure_memory():
    """Medir uso de memória"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def test_component(name, test_fn):
    """Executar teste com medição"""
    print(f"\n🔬 TEST: {name}")
    try:
        t0 = time.time()
        result = test_fn()
        t1 = time.time()
        print(f"   ✅ PASS ({t1-t0:.2f}s)")
        return True, result
    except Exception as e:
        print(f"   ❌ FAIL: {e}")
        traceback.print_exc()
        return False, str(e)

def main():
    print("="*80)
    print("🔬 AUDITORIA FORENSE PROFUNDA - PENIN³")
    print("="*80)
    
    results = {}
    
    # Memoria inicial
    mem_start = measure_memory()
    print(f"\n📊 Memória inicial: {mem_start:.1f} MB")
    
    # TEST 1: Inicialização
    def test_init():
        p = PENIN3System()
        assert p.v7 is not None, "V7 não carregado"
        assert p.sigma_guard is not None, "Sigma Guard não carregado"
        assert p.worm_ledger is not None, "WORM Ledger não carregado"
        return p
    
    success, p = test_component("Inicialização PENIN³", test_init)
    results['init'] = success
    
    if not success:
        print("\n❌ CRITICAL: Sistema não inicializa!")
        return results
    
    # TEST 2: Unified Score Inicial
    def test_unified_init():
        score = p.state.compute_unified_score()
        assert score >= 0.99, f"Score inicial baixo: {score}"
        return score
    
    success, score = test_component("Unified Score Inicial", test_unified_init)
    results['unified_init'] = {'success': success, 'score': score if success else 0}
    
    # TEST 3: WORM Integrity
    def test_worm():
        valid = p.worm_ledger.verify_integrity()
        assert valid, "WORM chain inválido"
        return valid
    
    success, valid = test_component("WORM Integrity", test_worm)
    results['worm'] = success
    
    # TEST 4: V7 Components
    def test_v7_components():
        checks = {
            'mnist': p.v7.mnist is not None,
            'rl_agent': p.v7.rl_agent is not None,
            'meta_learner': p.v7.meta_learner is not None,
            'evolutionary_optimizer': p.v7.evolutionary_optimizer is not None,
            'darwin_real': p.v7.darwin_real is not None,
            'auto_coder': p.v7.auto_coder is not None,
            'multimodal': p.v7.multimodal is not None,
            'automl': p.v7.automl is not None,
            'maml': p.v7.maml is not None,
            'db_mass_integrator': p.v7.db_mass_integrator is not None,
        }
        failed = [k for k, v in checks.items() if not v]
        assert len(failed) == 0, f"Componentes faltando: {failed}"
        return checks
    
    success, components = test_component("V7 Components", test_v7_components)
    results['v7_components'] = {'success': success, 'components': components if success else {}}
    
    # TEST 5: PENIN-Ω Components
    def test_penin_components():
        checks = {
            'master_state': p.master_state is not None,
            'sigma_guard': p.sigma_guard is not None,
            'sr_service': p.sr_service is not None,
            'acfa_league': p.acfa_league is not None,
            'worm_ledger': p.worm_ledger is not None,
        }
        failed = [k for k, v in checks.items() if not v]
        assert len(failed) == 0, f"Componentes faltando: {failed}"
        return checks
    
    success, components = test_component("PENIN-Ω Components", test_penin_components)
    results['penin_components'] = {'success': success, 'components': components if success else {}}
    
    # TEST 6: Run 1 Cycle
    def test_cycle():
        r = p.run_cycle()
        assert 'unified_score' in r, "Unified score missing"
        assert r['unified_score'] >= 0.99, f"Score baixo: {r['unified_score']}"
        assert r['penin_omega']['sigma_valid'], "Sigma failed"
        return r
    
    success, result = test_component("Run 1 Cycle", test_cycle)
    results['cycle_1'] = {'success': success, 'result': result if success else {}}
    
    mem_after_1 = measure_memory()
    
    # TEST 7: Run 5 More Cycles
    def test_5_cycles():
        scores = []
        for i in range(5):
            r = p.run_cycle()
            scores.append(r['unified_score'])
        
        # Check stability
        variance = np.var(scores)
        assert variance < 0.0001, f"Scores instáveis: var={variance}"
        return scores
    
    success, scores = test_component("Run 5 Cycles", test_5_cycles)
    results['cycle_5'] = {'success': success, 'scores': scores if success else []}
    
    mem_after_6 = measure_memory()
    
    # TEST 8: Checkpoint Save/Load
    def test_checkpoint():
        ckpt_path = "/tmp/penin3_test.pkl"
        p.state.save_checkpoint(ckpt_path)
        
        # Load
        loaded = PENIN3System.load_checkpoint(ckpt_path)
        assert loaded.state.cycle == p.state.cycle, "Cycle não restaurado"
        assert abs(loaded.state.compute_unified_score() - p.state.compute_unified_score()) < 1e-6, "Score diferente"
        
        os.remove(ckpt_path)
        return True
    
    success, _ = test_component("Checkpoint Save/Load", test_checkpoint)
    results['checkpoint'] = success
    
    # TEST 9: Sigma Guard Stress
    def test_sigma_stress():
        bad_metrics = {
            'rho': 2.0, 'ece': 0.5, 'rho_bias': 10.0, 'sr': 0.0,
            'g': 0.0, 'delta_linf': -1.0, 'cost': 100.0, 'budget': 0.1,
            'kappa': 0.0, 'consent': False, 'eco_ok': False
        }
        eval = p.sigma_guard.evaluate(bad_metrics)
        assert eval.verdict == 'FAIL', "Deveria reprovar métricas ruins"
        assert len(eval.failed_gates) == 10, f"Deveria reprovar 10 gates, reprovou {len(eval.failed_gates)}"
        return True
    
    success, _ = test_component("Sigma Guard Stress", test_sigma_stress)
    results['sigma_stress'] = success
    
    # TEST 10: WORM com 100 eventos
    def test_worm_stress():
        from penin.ledger.worm_ledger import WORMLedger
        import tempfile
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            worm = WORMLedger(tmp.name)
            
            for i in range(100):
                worm.append(f'test_{i}', f'id_{i}', {'data': i})
            
            valid = worm.verify_integrity()
            assert valid, "WORM chain quebrado após 100 eventos"
            
            events = list(worm.read_all())
            assert len(events) == 100, f"Esperava 100 eventos, encontrou {len(events)}"
            
            os.remove(tmp.name)
            return True
    
    success, _ = test_component("WORM Stress (100 eventos)", test_worm_stress)
    results['worm_stress'] = success
    
    # TEST 11: V7 sem treinamento (cache)
    def test_v7_cache():
        # Force skip training
        p.v7.best['mnist'] = 98.5
        p.v7.best['cartpole'] = 495
        p.v7.cycle = 101  # Not divisible by 50 or 10
        
        t0 = time.time()
        r = p.run_cycle()
        t1 = time.time()
        
        # Should be fast (< 5s) due to caching
        assert (t1-t0) < 10, f"Deveria ser rápido com cache, levou {t1-t0:.2f}s"
        return t1-t0
    
    success, cache_time = test_component("V7 Cache Performance", test_v7_cache)
    results['cache'] = {'success': success, 'time': cache_time if success else 0}
    
    # TEST 12: Darwin Engine
    def test_darwin():
        # Try initializing Darwin population
        from extracted_algorithms.darwin_engine_real import Individual
        
        # Test genome-based Individual (corrected)
        ind = Individual(genome={'test': 1}, fitness=0.5)
        assert ind.genome == {'test': 1}, "Genome não armazenado"
        
        # Test network-based Individual (original)
        import torch.nn as nn
        net = nn.Linear(10, 5)
        ind2 = Individual(network=net, fitness=0.8)
        assert ind2.network is not None, "Network não armazenado"
        
        return True
    
    success, _ = test_component("Darwin Engine Individual", test_darwin)
    results['darwin'] = success
    
    # TEST 13: Memory Leak Detection
    mem_final = measure_memory()
    mem_growth = mem_final - mem_start
    
    print(f"\n📊 Memória final: {mem_final:.1f} MB")
    print(f"📊 Crescimento: {mem_growth:.1f} MB em ~7 cycles")
    
    results['memory'] = {
        'start': mem_start,
        'final': mem_final,
        'growth': mem_growth,
        'leak': mem_growth > 500  # > 500MB = leak
    }
    
    # SUMMARY
    print("\n" + "="*80)
    print("📊 RESULTADOS AUDITORIA PROFUNDA")
    print("="*80)
    
    total_tests = len([k for k in results.keys() if k not in ['memory']])
    passed = sum(1 for k, v in results.items() if k not in ['memory'] and (v is True or (isinstance(v, dict) and v.get('success'))))
    
    print(f"\n✅ Testes passados: {passed}/{total_tests}")
    
    for test_name, result in results.items():
        if test_name == 'memory':
            continue
        if result is True or (isinstance(result, dict) and result.get('success')):
            print(f"   ✅ {test_name}")
        else:
            print(f"   ❌ {test_name}")
    
    print(f"\n📊 Memória: {mem_growth:.1f} MB crescimento")
    if mem_growth > 500:
        print(f"   ⚠️ POSSÍVEL MEMORY LEAK!")
    else:
        print(f"   ✅ Uso de memória OK")
    
    print("\n" + "="*80)
    
    return results

if __name__ == '__main__':
    results = main()
