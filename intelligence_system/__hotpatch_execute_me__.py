"""
Este arquivo executa automaticamente quando importado
"""
import sys
import time
import logging

logger = logging.getLogger("hotpatch_executor")

print("="*80)
print("🔥 HOTPATCH EXECUTOR RODANDO!")
print("="*80)

# Executa o patch
try:
    exec(open('/tmp/patch_optimizer_v3.py').read())
    print("✅ Patch optimizer v3 executado")
    logger.info("✅ HOTPATCH: Optimizer patch v3 active")
except Exception as e:
    print(f"❌ Erro ao executar patch: {e}")
    logger.error(f"❌ HOTPATCH failed: {e}")

print("="*80)

# Auto-delete após 1 execução
import os
try:
    time.sleep(0.5)
    os.remove(__file__)
    print("🗑️  Self-deleted")
except:
    pass
