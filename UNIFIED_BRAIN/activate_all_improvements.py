#!/usr/bin/env python3
"""
✅ ACTIVATE ALL IMPROVEMENTS
Script que adiciona TODAS melhorias ao brain daemon em execução
"""

import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')
sys.path.insert(0, '/root')

from pathlib import Path
import json
from datetime import datetime

def create_activation_config():
    """Cria arquivo de configuração para ativar melhorias"""
    
    config = {
        'timestamp': datetime.now().isoformat(),
        'activations': {
            'darwin_evolution': True,
            'module_synthesis': True,
            'maml_integration': True,
            'penin3_neurons': True,
            'recursive_improvement': True,
            'ia3_calculation': True
        },
        'parameters': {
            'darwin': {
                'survival_rate': 0.4,
                'elite_size': 5,
                'evolution_interval': 10  # Episodes
            },
            'synthesis': {
                'enabled': True,
                'max_neurons': 100
            },
            'maml': {
                'inner_lr': 0.01,
                'outer_lr': 0.001,
                'meta_train_interval': 50  # Episodes
            },
            'penin3': {
                'count': 3,  # Neurônios PENIN3 a adicionar
                'auto_evolve': True
            },
            'recursive': {
                'cycle_interval': 120,  # Seconds
                'meta_learn_interval': 10  # Cycles
            }
        }
    }
    
    config_path = Path('/root/UNIFIED_BRAIN/activation_config.json')
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"✅ Configuration created: {config_path}")
    print(f"   Darwin Evolution: ENABLED")
    print(f"   Module Synthesis: ENABLED")
    print(f"   MAML Integration: ENABLED")
    print(f"   PENIN3 Neurons: ENABLED (count=3)")
    print(f"   Recursive Improvement: ENABLED")
    
    return config_path

if __name__ == "__main__":
    print("🚀 Activating ALL improvements...")
    print()
    
    config_path = create_activation_config()
    
    print()
    print("════════════════════════════════════════════════════════════")
    print("✅ ALL IMPROVEMENTS CONFIGURED!")
    print("════════════════════════════════════════════════════════════")
    print()
    print("📋 Next step: Restart brain daemon to apply:")
    print()
    print("   cd /root/UNIFIED_BRAIN")
    print("   bash restart_with_improvements.sh")
    print()
    print("════════════════════════════════════════════════════════════")