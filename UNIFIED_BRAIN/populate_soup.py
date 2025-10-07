#!/usr/bin/env python3
"""
🧬 Populate soup with initial neurons
Fixes P4: Soup vazio
"""
import sys, torch, json, hashlib
sys.path.insert(0, '/root')

from UNIFIED_BRAIN.unified_brain_core import CoreSoupHybrid
from UNIFIED_BRAIN.brain_spec import RegisteredNeuron, NeuronMeta, NeuronStatus
from UNIFIED_BRAIN.brain_logger import brain_logger

brain_logger.info("="*60)
brain_logger.info("🧬 Populating soup with initial neurons...")
brain_logger.info("="*60)

hybrid = CoreSoupHybrid(H=512)

# Define diverse neuron types
neuron_configs = [
    ('identity', lambda x: x, 1.0),
    ('tanh', lambda x: torch.tanh(x), 0.9),
    ('relu', lambda x: torch.relu(x), 1.1),
    ('sigmoid', lambda x: torch.sigmoid(x), 0.8),
    ('leaky_relu', lambda x: torch.nn.functional.leaky_relu(x), 1.0),
]

created = 0
for i in range(20):  # Create 20 neurons (4 copies of each type)
    name, activation, base_scale = neuron_configs[i % len(neuron_configs)]
    scale = base_scale + (i // len(neuron_configs)) * 0.05
    
    def make_forward(act_fn, s):
        def fn(x):
            return act_fn(x * s)
        return fn
    
    neuron_id = f"synth_{name}_{i:02d}"
    checksum = hashlib.sha256(neuron_id.encode()).hexdigest()[:16]
    
    meta = NeuronMeta(
        id=neuron_id,
        in_shape=(512,),
        out_shape=(512,),
        dtype=torch.float32,
        device='cpu',
        status=NeuronStatus.ACTIVE,
        source='synthetic_bootstrap',
        params_count=0,
        checksum=checksum,
        competence_score=0.4 + (i * 0.01),  # Vary 0.4 → 0.6
        novelty_score=0.5,
    )
    
    neuron = RegisteredNeuron(meta, make_forward(activation, scale), H=512)
    success = hybrid.soup.register_neuron(neuron)
    
    if success:
        created += 1
        brain_logger.info(f"   ✅ {neuron_id} (comp={meta.competence_score:.2f})")

brain_logger.info(f"\n{'='*60}")
brain_logger.info(f"✅ Soup populated: {created} neurons created")
brain_logger.info(f"   Active in soup: {len(hybrid.soup.registry.get_active())}")
brain_logger.info(f"   Active in core: {len(hybrid.core.registry.get_active())}")
brain_logger.info(f"{'='*60}")

# Save manifest
manifest = {
    'neurons': [
        {
            'id': n.meta.id,
            'source': n.meta.source,
            'competence': float(n.meta.competence_score),
        }
        for n in hybrid.soup.registry.get_active()
    ],
    'total': len(hybrid.soup.registry.get_active()),
    'timestamp': __import__('datetime').datetime.now().isoformat(),
}

manifest_path = '/root/UNIFIED_BRAIN/soup_manifest.json'
with open(manifest_path, 'w') as f:
    json.dump(manifest, f, indent=2)

brain_logger.info(f"\n✅ Manifest saved to {manifest_path}")
