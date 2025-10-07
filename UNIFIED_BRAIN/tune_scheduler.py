#!/usr/bin/env python3
"""
Ajuste fino do scheduler para balancear melhor speed/exploration
"""
from pathlib import Path

phase2_hooks = Path('/root/UNIFIED_BRAIN/phase2_hooks.py')
content = phase2_hooks.read_text()

# Ajustes:
# 1. Reduzir threshold de "speed" para não ficar preso
# 2. Aumentar min_weight para mais exploration
# 3. Aumentar max_topk para mais capacity

adjustments = [
    ('self.min_weight = 0.0', 'self.min_weight = 0.03'),  # Mais exploration baseline
    ('self.max_weight = 0.2', 'self.max_weight = 0.25'),  # Mais range
    ('self.max_topk = 64', 'self.max_topk = 16'),  # Limite mais conservador (era 64, muito alto)
    ('if avg_time > 0.02:', 'if avg_time > 0.5:'),  # Só prioriza speed se MUITO lento
]

for old, new in adjustments:
    if old in content:
        content = content.replace(old, new)
        print(f"✅ {old} → {new}")
    else:
        print(f"⚠️  Not found: {old}")

phase2_hooks.write_text(content)
print("\n✅ Scheduler ajustado!")
print("Restart daemon para aplicar:")
print("  pkill -f brain_daemon_real_env")
print("  cd /root/UNIFIED_BRAIN && nohup python3 brain_daemon_real_env.py > ../brain_daemon_run.log 2>&1 &")