#!/usr/bin/env python3
"""Quick test to see why neurons aren't registering"""
import sys, torch
sys.path.insert(0, '/root')

from UNIFIED_BRAIN.unified_brain_core import CoreSoupHybrid
from UNIFIED_BRAIN.brain_spec import RegisteredNeuron, NeuronMeta, NeuronStatus

print("Creating hybrid...")
hybrid = CoreSoupHybrid(H=128)

print(f"Initial soup neurons: {len(hybrid.soup.registry.get_active())}")
print(f"Soup max_neurons: {hybrid.soup.max_neurons}")

# Try to register ONE neuron
def simple_fn(x):
    return torch.tanh(x)

meta = NeuronMeta(
    id="test_neuron_01",
    in_shape=(128,),
    out_shape=(128,),
    dtype=torch.float32,
    device='cpu',
    status=NeuronStatus.ACTIVE,
    source='test',
    params_count=0,
    checksum="test123",
    competence_score=0.5,
    novelty_score=0.5,
)

neuron = RegisteredNeuron(meta, simple_fn, H=128)
print(f"\nCreated neuron: {meta.id}")

result = hybrid.soup.register_neuron(neuron)
print(f"Register result: {result}")
print(f"After register, soup neurons: {len(hybrid.soup.registry.get_active())}")
print(f"Neurons in registry: {list(hybrid.soup.registry.neurons.keys())}")
print(f"\nDEBUG:")
print(f"  by_status keys: {list(hybrid.soup.registry.by_status.keys())}")
print(f"  by_status[ACTIVE]: {hybrid.soup.registry.by_status.get(NeuronStatus.ACTIVE, 'NOT FOUND')}")
print(f"  meta.status: {meta.status}")
print(f"  meta.status == NeuronStatus.ACTIVE: {meta.status == NeuronStatus.ACTIVE}")

# Test get_active directly
active = hybrid.soup.registry.get_active()
print(f"\nget_active() returned: {len(active)} neurons")
if len(active) > 0:
    print(f"  First: {active[0].meta.id}")
else:
    print(f"  Why zero? Let's check get_by_status...")
    by_stat = hybrid.soup.registry.get_by_status(NeuronStatus.ACTIVE)
    print(f"  get_by_status(ACTIVE): {len(by_stat)} neurons")
    print(f"  IDs in by_status list: {hybrid.soup.registry.by_status[NeuronStatus.ACTIVE]}")
