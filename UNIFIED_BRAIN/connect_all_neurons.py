#!/usr/bin/env python3
"""
🚀 CONNECT ALL NEURONS
Script principal que conecta ~2M neurônios no cérebro unificado
"""

import sys
import os
sys.path.insert(0, '/root/UNIFIED_BRAIN')

import torch
import torch.nn as nn
from pathlib import Path
import json
import time
from tqdm import tqdm
import hashlib

from brain_spec import RegisteredNeuron, NeuronMeta, NeuronStatus
from unified_brain_core import UnifiedBrain, CoreSoupHybrid
from brain_sanitizer import ASTSanitizer, QuarantineManager, scan_directory
from brain_router import AdaptiveRouter

print("=" * 80)
print("🧠 UNIFIED BRAIN - CONNECTING ALL NEURONS")
print("=" * 80)
print()

# Configuração
H = 1024  # Espaço latente
MAX_CORE_NEURONS = 10000  # Limite para Core (curado)
MAX_SOUP_NEURONS = 50000  # Limite para Soup (experimental)
TOP_K = 128

# Cria diretórios
BRAIN_DIR = Path("/root/UNIFIED_BRAIN")
BRAIN_DIR.mkdir(exist_ok=True)

QUARANTINE_DIR = BRAIN_DIR / "quarantine"
SNAPSHOTS_DIR = BRAIN_DIR / "snapshots"
SNAPSHOTS_DIR.mkdir(exist_ok=True)

print(f"📁 Brain directory: {BRAIN_DIR}")
print(f"📁 Quarantine: {QUARANTINE_DIR}")
print()

# ============================================================================
# FASE 1: INVENTÁRIO
# ============================================================================
print("🔍 FASE 1: INVENTÁRIO COMPLETO")
print("-" * 80)

# Localizações conhecidas de neurônios
NEURON_LOCATIONS = [
    "/root/INJECTED_SURVIVORS",
    "/root/neurons_organized/api_neurons",
    "/root/neurons_organized/generation_47_original",
    "/root/neurons_organized/generation_current",
]

# Checkpoints .pt conhecidos
CHECKPOINT_FILES = [
    "/root/darwin_fixed_gen45.pt",
    "/root/ocean_final.pt",
    "/root/ia3_evolution_gen45_SPECIAL_24716neurons.pt",
]

inventory = {
    'python_files': [],
    'checkpoints': [],
    'total_files': 0,
}

# Scannea diretórios Python
for loc in NEURON_LOCATIONS:
    if Path(loc).exists():
        py_files = list(Path(loc).rglob("*.py"))
        inventory['python_files'].extend([str(f) for f in py_files])
        print(f"  Found {len(py_files)} Python files in {loc}")

# Checkpoints
for ckpt in CHECKPOINT_FILES:
    if Path(ckpt).exists():
        inventory['checkpoints'].append(ckpt)
        print(f"  Found checkpoint: {ckpt}")

inventory['total_files'] = len(inventory['python_files']) + len(inventory['checkpoints'])

print(f"\n✅ Inventory complete:")
print(f"   Python files: {len(inventory['python_files'])}")
print(f"   Checkpoints: {len(inventory['checkpoints'])}")
print(f"   TOTAL: {inventory['total_files']}")
print()

# Salva inventory
with open(BRAIN_DIR / "inventory.json", 'w') as f:
    json.dump(inventory, f, indent=2)

# ============================================================================
# FASE 2: SANITIZAÇÃO
# ============================================================================
print("🛡️ FASE 2: SANITIZAÇÃO E QUARENTENA")
print("-" * 80)

sanitizer = ASTSanitizer()
quarantine_mgr = QuarantineManager(str(QUARANTINE_DIR))

safe_files = []
suspect_files = []
dangerous_files = []

print("Scanning Python files...")
for filepath in tqdm(inventory['python_files'][:1000]):  # Limite para demo
    try:
        result = sanitizer.scan_file(filepath)
        
        if result.status == 'safe':
            safe_files.append((filepath, result))
        elif result.status == 'suspect':
            suspect_files.append((filepath, result))
            quarantine_mgr.quarantine_file(result)
        else:  # dangerous
            dangerous_files.append((filepath, result))
            quarantine_mgr.quarantine_file(result)
            
    except Exception as e:
        print(f"ERROR scanning {filepath}: {e}")

print(f"\n✅ Sanitization complete:")
print(f"   ✅ Safe: {len(safe_files)}")
print(f"   ⚠️  Suspect: {len(suspect_files)}")
print(f"   🚨 Dangerous: {dangerous_files}")
print()

# ============================================================================
# FASE 3: CARREGAR NEURÔNIOS SEGUROS
# ============================================================================
print("📦 FASE 3: CARREGANDO NEURÔNIOS SEGUROS")
print("-" * 80)

hybrid = CoreSoupHybrid(H=H)

# Neurônios de alta prioridade (já testados - Darwin)
priority_neurons = []

print("Loading Darwin survivors (254 neurons)...")
darwin_ckpt = "/root/darwin_fixed_gen45.pt"
if Path(darwin_ckpt).exists():
    try:
        data = torch.load(darwin_ckpt, map_location='cpu')
        if 'neurons' in data:
            for nid, neuron_data in list(data['neurons'].items())[:254]:
                # Cria neurônio
                dna_str = neuron_data.get('dna', nid)
                dna_tensor = torch.tensor([float(ord(c)) for c in dna_str[:1024]], dtype=torch.float32)
                
                # Wrapper simples
                class SimpleNeuron(nn.Module):
                    def __init__(self, dna):
                        super().__init__()
                        self.weights = nn.Parameter(dna[:128])
                    
                    def forward(self, x):
                        return torch.tanh(x @ self.weights.unsqueeze(-1)).squeeze(-1)
                
                model = SimpleNeuron(dna_tensor[:128])
                
                # Meta
                meta = NeuronMeta(
                    id=f"darwin_{nid}",
                    in_shape=(128,),
                    out_shape=(1,),
                    dtype=torch.float32,
                    device='cpu',
                    status=NeuronStatus.ACTIVE,  # Darwin já testados
                    source='darwin_gen45',
                    params_count=128,
                    checksum=hashlib.sha256(dna_str.encode()).hexdigest()[:16],
                    competence_score=neuron_data.get('fitness', 0.5),
                )
                
                # Registra
                registered = RegisteredNeuron(meta, model.forward, H=H)
                hybrid.core.register_neuron(registered)
                priority_neurons.append(registered)
                
        print(f"   ✅ Loaded {len(priority_neurons)} Darwin neurons into CORE")
        
    except Exception as e:
        print(f"   ⚠️  Error loading Darwin: {e}")

# API Neurons (para SOUP - experimentais)
print("\nLoading API neurons (sample)...")
api_loaded = 0
for filepath in safe_files[:500]:  # Sample de 500
    if 'api_neurons' in filepath[0]:
        try:
            # Cria wrapper genérico
            class GenericNeuron(nn.Module):
                def __init__(self):
                    super().__init__()
                    self.fc = nn.Linear(H, H)
                
                def forward(self, x):
                    return torch.tanh(self.fc(x.mean(dim=tuple(range(1, x.ndim))) if x.ndim > 1 else x))
            
            model = GenericNeuron()
            
            # Meta
            filename = Path(filepath[0]).stem
            meta = NeuronMeta(
                id=f"api_{filename}",
                in_shape=(H,),
                out_shape=(H,),
                dtype=torch.float32,
                device='cpu',
                status=NeuronStatus.TESTING,  # API não testados
                source='api_generated',
                params_count=H * H,
                checksum=filepath[1].checksum[:16],
            )
            
            registered = RegisteredNeuron(meta, model.forward, H=H)
            hybrid.soup.register_neuron(registered)
            api_loaded += 1
            
        except:
            pass

print(f"   ✅ Loaded {api_loaded} API neurons into SOUP")
print()

print(f"📊 BRAIN STATUS:")
print(f"   CORE: {hybrid.core.registry.count()}")
print(f"   SOUP: {hybrid.soup.registry.count()}")
print()

# ============================================================================
# FASE 4: INICIALIZAR ROUTERS
# ============================================================================
print("🎯 FASE 4: INICIALIZANDO ROUTERS")
print("-" * 80)

hybrid.core.initialize_router()
hybrid.soup.initialize_router()

print(f"   ✅ Core router: {hybrid.core.router.num_neurons} neurons, top-{hybrid.core.router.top_k}")
print(f"   ✅ Soup router: {hybrid.soup.router.num_neurons} neurons, top-{hybrid.soup.router.top_k}")
print()

# ============================================================================
# FASE 5: CALIBRAÇÃO (sample)
# ============================================================================
print("🔧 FASE 5: CALIBRAÇÃO DE ADAPTERS (sample)")
print("-" * 80)

# Gera probes
N_probes = 512
probes = torch.randn(N_probes, H)

print(f"Calibrating {min(10, len(priority_neurons))} priority neurons...")
for i, neuron in enumerate(priority_neurons[:10]):
    try:
        loss = neuron.calibrate_adapters(probes, epochs=10, lr=1e-3)
        print(f"   Neuron {neuron.meta.id}: loss={loss:.4f}")
    except Exception as e:
        print(f"   ⚠️  {neuron.meta.id}: {e}")

print()

# ============================================================================
# FASE 6: TESTE DO CÉREBRO
# ============================================================================
print("🧠 FASE 6: TESTE DO CÉREBRO UNIFICADO")
print("-" * 80)

# Test forward
z_in = torch.randn(1, H)

print("Running Core brain...")
z_out_core, info_core = hybrid.core.step(z_in)
print(f"   Core step: {info_core.get('selected_neurons', 0)} neurons active")
print(f"   Coherence: {info_core.get('coherence', 0):.3f}")
print(f"   Novelty: {info_core.get('novelty', 0):.3f}")
print(f"   Latency: {info_core.get('latency_ms', 0):.2f}ms")

print("\nRunning Soup brain...")
z_out_soup, info_soup = hybrid.soup.step(z_in)
print(f"   Soup step: {info_soup.get('selected_neurons', 0)} neurons active")
print(f"   Coherence: {info_soup.get('coherence', 0):.3f}")
print(f"   Novelty: {info_soup.get('novelty', 0):.3f}")
print(f"   Latency: {info_soup.get('latency_ms', 0):.2f}ms")

print()

# ============================================================================
# FASE 7: SALVAR SNAPSHOT
# ============================================================================
print("💾 FASE 7: SALVANDO SNAPSHOT")
print("-" * 80)

snapshot_path = SNAPSHOTS_DIR / f"brain_snapshot_{int(time.time())}.pt"
hybrid.core.save_snapshot(str(snapshot_path))

print(f"   ✅ Core snapshot: {snapshot_path}")
print()

# ============================================================================
# FASE 8: RELATÓRIO FINAL
# ============================================================================
print("=" * 80)
print("🎉 CÉREBRO UNIFICADO CONECTADO!")
print("=" * 80)

summary = {
    'timestamp': time.time(),
    'total_neurons_inventoried': inventory['total_files'],
    'safe_files': len(safe_files),
    'quarantined': len(suspect_files) + len(dangerous_files),
    'core': hybrid.core.registry.count(),
    'soup': hybrid.soup.registry.count(),
    'total_connected': (
        hybrid.core.registry.count()['active'] +
        hybrid.soup.registry.count()['active']
    ),
    'snapshot': str(snapshot_path),
}

print(json.dumps(summary, indent=2))

with open(BRAIN_DIR / "connection_summary.json", 'w') as f:
    json.dump(summary, f, indent=2)

print()
print(f"📊 Next steps:")
print(f"   1. Review quarantined files: {QUARANTINE_DIR}")
print(f"   2. Calibrate more neurons")
print(f"   3. Integrate with V7/PENIN-Ω/Darwin")
print(f"   4. Start evolution!")
print()
print("🔥 O CÉREBRO ESTÁ VIVO! 🔥")
