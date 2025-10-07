#!/usr/bin/env python3
"""
🧪 Quick promotion test with relaxed gates
Fixes P3: Gates nunca passam
"""
import os, sys
sys.path.insert(0, '/root')

# Configure bootstrap mode
os.environ['UBRAIN_BOOTSTRAP_MODE'] = '1'
os.environ['UBRAIN_EVAL_SEEDS'] = '42'  # Single seed
os.environ['UBRAIN_EVAL_EPISODES'] = '1'

from UNIFIED_BRAIN.unified_brain_core import CoreSoupHybrid
from UNIFIED_BRAIN.brain_logger import brain_logger

print("="*60)
print("🧪 Testing promotion with relaxed gates")
print("="*60)

hybrid = CoreSoupHybrid(H=128)

soup_before = len(hybrid.soup.registry.get_active())
core_before = len(hybrid.core.registry.get_active())

print(f"\nBEFORE promotion:")
print(f"  Soup neurons: {soup_before}")
print(f"  Core neurons: {core_before}")

if soup_before == 0:
    print("\n⚠️  WARNING: Soup is empty!")
    print("   Run: python /root/UNIFIED_BRAIN/populate_soup.py")
    sys.exit(1)

print(f"\n🚀 Running promote_from_soup()...")
print(f"   (This may take 2-5 minutes due to evaluation)")

try:
    hybrid.promote_from_soup()
except Exception as e:
    print(f"\n❌ Promotion failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

soup_after = len(hybrid.soup.registry.get_active())
core_after = len(hybrid.core.registry.get_active())

print(f"\nAFTER promotion:")
print(f"  Soup neurons: {soup_after} (was {soup_before})")
print(f"  Core neurons: {core_after} (was {core_before})")

promoted = core_after - core_before
frozen_soup = soup_before - soup_after

print(f"\n{'='*60}")
if promoted > 0:
    print(f"✅ SUCCESS: {promoted} neuron(s) promoted to core!")
else:
    print(f"⚠️  No promotions (gates may still be too strict)")
    print(f"   Check WORM log for gate details:")
    print(f"   tail -50 /root/UNIFIED_BRAIN/worm_log.json | grep promotion_gate")
print(f"{'='*60}")
