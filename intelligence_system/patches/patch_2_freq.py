#!/usr/bin/env python3
from pathlib import Path

target = Path("/root/intelligence_system/core/system_v7_ultimate.py")
code = target.read_text()

# MAML: 20 → 10
code = code.replace(
    "# FIX P1: MAML (every 20 cycles)\n        if self.cycle % 20 == 0:\n            results['maml'] = self._maml_few_shot()",
    "# FIX P1: MAML (every 10 cycles)\n        if self.cycle % 10 == 0:\n            results['maml'] = self._maml_few_shot()"
)

# AutoML: 20 → 10
code = code.replace(
    "# FIX P1: AutoML (every 20 cycles)\n        if self.cycle % 20 == 0:\n            results['automl'] = self._automl_search()",
    "# FIX P1: AutoML (every 10 cycles)\n        if self.cycle % 10 == 0:\n            results['automl'] = self._automl_search()"
)

target.write_text(code)
print("✅ Patch #2 aplicado com sucesso")
