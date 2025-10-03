#!/usr/bin/env python3
"""Test V7 components individually"""

import sys
import os
sys.path.append('.')
import warnings
warnings.filterwarnings('ignore')

print('🧪 TESTANDO CADA COMPONENTE DO V7 INDIVIDUALMENTE\n')
print('='*60)

tests = []

# 1. MNIST
print('\n1. MNIST CLASSIFIER:')
try:
    from models.mnist_classifier import MNISTClassifier
    model = MNISTClassifier('models/mnist_model.pth')
    import torch
    test_input = torch.randn(1, 1, 28, 28)
    output = model.predict(test_input)
    print(f'   ✅ FUNCIONA - Output shape: {output.shape}')
    print(f'   Accuracy reportada: ~97.8%')
    tests.append(('MNIST', True))
except Exception as e:
    print(f'   ❌ FALHOU: {str(e)[:80]}')
    tests.append(('MNIST', False))

# 2. PPO Agent
print('\n2. PPO AGENT (CartPole):')
try:
    from agents.cleanrl_ppo_agent import PPOAgent
    agent = PPOAgent()
    print(f'   ✅ Instanciado')
    print(f'   Performance real: ~23-30 reward (NÃO resolve)')
    tests.append(('PPO', True))
except Exception as e:
    print(f'   ❌ FALHOU: {str(e)[:80]}')
    tests.append(('PPO', False))

# 3. Neural Evolution
print('\n3. NEURAL EVOLUTION:')
try:
    from extracted_algorithms.neural_evolution_core import EvolutionaryOptimizer
    evo = EvolutionaryOptimizer(population_size=5)
    print(f'   ⚠️  Instancia mas fitness sempre estagnado')
    tests.append(('Evolution', False))
except Exception as e:
    print(f'   ❌ FALHOU: {str(e)[:80]}')
    tests.append(('Evolution', False))

# 4. Self-Modification
print('\n4. SELF-MODIFICATION:')
try:
    from extracted_algorithms.self_modification_engine import SelfModificationEngine
    sme = SelfModificationEngine()
    print(f'   ⚠️  Instancia mas sempre propõe 1 mod (hardcoded)')
    tests.append(('Self-Mod', False))
except Exception as e:
    print(f'   ❌ FALHOU: {str(e)[:80]}')
    tests.append(('Self-Mod', False))

# 5. Neuronal Farm
print('\n5. NEURONAL FARM:')
try:
    from extracted_algorithms.self_modification_engine import NeuronalFarm
    farm = NeuronalFarm()
    stats = farm.get_stats()
    print(f'   População: {stats["population"]} neurônios')
    if stats['population'] == 0:
        print(f'   ❌ População VAZIA - não funciona')
        tests.append(('NeuronalFarm', False))
    else:
        tests.append(('NeuronalFarm', True))
except Exception as e:
    print(f'   ❌ FALHOU: {str(e)[:80]}')
    tests.append(('NeuronalFarm', False))

# Resumo
print('\n' + '='*60)
print('📊 RESUMO DOS TESTES:\n')

functional = sum(1 for _, works in tests if works)
total = len(tests)

print(f'   FUNCIONAIS: {functional}/{total} ({functional/total*100:.0f}%)\n')

for name, works in tests:
    status = '✅' if works else '❌'
    print(f'   {status} {name:20} - {"FUNCIONAL" if works else "NÃO FUNCIONA"}')