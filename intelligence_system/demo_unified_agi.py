"""
Demo interativa do Unified AGI System
Mostra a comunicação V7 ↔ PENIN³ em tempo real
"""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from core.unified_agi_system import UnifiedAGISystem

# Setup logging colorido
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

if __name__ == "__main__":
    print("="*80)
    print("🎯 UNIFIED AGI SYSTEM - DEMO INTERATIVA")
    print("="*80)
    print("")
    print("Sistema: V7 (Operational) + PENIN³ (Meta)")
    print("Threads: 2 (V7Worker + PENIN3Orchestrator)")
    print("Comunicação: Bidirectional message queue")
    print("")
    print("="*80)
    print("")
    
    # Parse cycles
    cycles = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    
    print(f"Rodando {cycles} ciclos...")
    print("")
    
    # Create and run system
    system = UnifiedAGISystem(max_cycles=cycles)
    system.run()
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETA")
    print("="*80)
