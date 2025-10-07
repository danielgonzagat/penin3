"""
EXECUÇÃO DE CONTAMINAÇÃO VIRAL REAL
====================================

⚠️ ATENÇÃO: Este script modificará 22,000+ arquivos Python!

PROCESSO:
1. Escaneia /root recursivamente
2. Identifica classes ML/AI
3. Injeta decorator @make_evolvable
4. Salva arquivos _DARWIN_INFECTED.py

TEMPO ESTIMADO: 3 horas
"""

import sys
from pathlib import Path
sys.path.insert(0, '/root')

from darwin_viral_contamination import DarwinViralContamination
import time

print("="*80)
print("🦠 EXECUTANDO CONTAMINAÇÃO VIRAL REAL")
print("="*80)
print("\n⚠️  ATENÇÃO: Isso modificará 22,000+ arquivos!")
print("⚠️  Arquivos serão salvos como *_DARWIN_INFECTED.py")
print("\nPressione CTRL+C nos próximos 10 segundos para cancelar...\n")

for i in range(10, 0, -1):
    print(f"   {i}...", flush=True)
    time.sleep(1)

print("\n🚀 Iniciando contaminação REAL...")
print("="*80)

contaminator = DarwinViralContamination()

# EXECUTAR de verdade (não dry_run!)
results = contaminator.contaminate_all_systems(
    dry_run=False,  # ← EXECUTAR DE VERDADE
    limit=None      # ← TODOS OS ARQUIVOS
)

print("\n" + "="*80)
print("✅ CONTAMINAÇÃO COMPLETA!")
print("="*80)
print(f"\n📊 RESULTADO:")
print(f"   Total arquivos: {results['total_files']}")
print(f"   Evoluíveis: {results['evolvable_files']}")
print(f"   Infectados: {results['infected']}")
print(f"   Falhados: {results['failed']}")
print(f"   Taxa sucesso: {results['infected']/(results['infected']+results['failed'])*100:.1f}%")
print("\n🎉 TODOS OS SISTEMAS AGORA SÃO EVOLUÍVEIS!")
print("="*80)
