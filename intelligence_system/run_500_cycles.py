#!/usr/bin/env python3
"""
Executar 500 Ciclos - Sistema V7.0
Com checkpoints e relatórios a cada 50 ciclos
"""
import sys
sys.path.insert(0, '/root/intelligence_system')

from core.system_v7_ultimate import IntelligenceSystemV7
import json
import time
from pathlib import Path

print("╔════════════════════════════════════════════════════════════════╗")
print("║         🚀 EXECUÇÃO 500 CICLOS - SISTEMA V7.0                 ║")
print("╚════════════════════════════════════════════════════════════════╝")
print()

# Initialize
system = IntelligenceSystemV7()
start_time = time.time()

# Metrics tracking
all_scores = []
all_mnist = []
all_cartpole = []
api_success_count = 0

print(f"Ciclo inicial: {system.cycle}")
print(f"Target: {system.cycle + 500}")
print()
print("Executando... (checkpoint a cada 50 ciclos)")
print("=" * 70)
print()

for i in range(500):
    cycle_num = i + 1
    
    try:
        results = system.run_cycle()
        
        # Track metrics
        all_scores.append(results['ia3_score'])
        all_mnist.append(results['mnist']['test'])
        all_cartpole.append(results['cartpole']['avg_reward'])
        
        if 'apis' in results and results['apis'].get('successful', 0) > 0:
            api_success_count += 1
        
        # Checkpoint a cada 50 ciclos
        if cycle_num % 50 == 0:
            elapsed = time.time() - start_time
            avg_ia3 = sum(all_scores[-50:]) / 50
            
            print(f"Ciclo {cycle_num}/500:")
            print(f"  IA³: {avg_ia3:.2f}% (últimos 50)")
            print(f"  MNIST: {all_mnist[-1]:.2f}%")
            print(f"  CartPole: {all_cartpole[-1]:.1f}")
            print(f"  APIs: {api_success_count} sucessos")
            print(f"  Tempo: {elapsed/60:.1f} min")
            print()
    
    except KeyboardInterrupt:
        print("\n⚠️  Interrompido pelo usuário")
        break
    except Exception as e:
        print(f"\n❌ Erro no ciclo {cycle_num}: {e}")
        break

# Final report
print()
print("=" * 70)
print("📊 RELATÓRIO FINAL")
print("=" * 70)
print()

total_time = time.time() - start_time
cycles_completed = len(all_scores)

print(f"Ciclos completados: {cycles_completed}/500")
print(f"Tempo total: {total_time/60:.1f} min ({total_time/3600:.2f}h)")
if cycles_completed > 0:
    print(f"Tempo/ciclo: {total_time/cycles_completed:.1f}s")
else:
    print(f"Tempo/ciclo: N/A (0 ciclos)")
print()

print(f"IA³ Score:")
print(f"  Inicial: {all_scores[0]:.2f}%")
print(f"  Final: {all_scores[-1]:.2f}%")
print(f"  Evolução: {all_scores[-1] - all_scores[0]:+.2f}%")
print(f"  Máximo: {max(all_scores):.2f}%")
print()

print(f"MNIST:")
print(f"  Final: {all_mnist[-1]:.2f}%")
print(f"  Máximo: {max(all_mnist):.2f}%")
print()

print(f"CartPole:")
print(f"  Final: {all_cartpole[-1]:.1f}")
print(f"  Máximo: {max(all_cartpole):.1f}")
print()

print(f"APIs:")
print(f"  Chamadas com sucesso: {api_success_count}")
print()

# Save results
results_data = {
    "cycles_completed": cycles_completed,
    "time_seconds": total_time,
    "ia3_scores": all_scores,
    "mnist_scores": all_mnist,
    "cartpole_scores": all_cartpole,
    "api_success_count": api_success_count,
    "final_ia3": all_scores[-1],
    "ia3_evolution": all_scores[-1] - all_scores[0],
    "max_ia3": max(all_scores)
}

results_file = Path("/root/500_cycles_results.json")
with open(results_file, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"✅ Resultados salvos em: {results_file}")
print()
print("🎉 EXECUÇÃO COMPLETA!")
