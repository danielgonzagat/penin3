"""
Exemplo 2: Contaminação Viral
==============================

Demonstra como contaminar sistemas com Darwin Engine.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from contamination.darwin_viral_contamination import DarwinViralContamination

def main():
    print("="*80)
    print("EXEMPLO 2: Contaminação Viral")
    print("="*80)
    
    # Criar contaminador
    contaminator = DarwinViralContamination()
    
    # Dry run primeiro (teste)
    print("\n1. Executando DRY RUN (teste)...")
    results_dry = contaminator.contaminate_all_systems(
        dry_run=True,
        limit=1000
    )
    
    print(f"\nResultado DRY RUN:")
    print(f"  Evoluíveis: {results_dry['evolvable_files']}")
    print(f"  Taxa: {results_dry['evolvable_files']/results_dry['total_files']*100:.1f}%")
    
    # Executar de verdade (opcional)
    input("\nPressione ENTER para executar contaminação REAL (ou CTRL+C para cancelar)...")
    
    print("\n2. Executando CONTAMINAÇÃO REAL...")
    results = contaminator.contaminate_all_systems(
        dry_run=False,
        limit=5000
    )
    
    print(f"\nResultado REAL:")
    print(f"  Infectados: {results['infected']}")
    print(f"  Taxa sucesso: {results['infected']/(results['infected']+results['failed'])*100:.1f}%")

if __name__ == "__main__":
    main()
