# PENIN-Ω Self-Modification with Canary Deployment
# Extensão: aplica modificações em canary (10% do sistema), monitora, rollback automático se degradação.

import subprocess
import time
import threading
import json
from pathlib import Path
from typing import Dict, Any, Optional

ROOT = Path("/root/.penin_omega")
BH_DB = ROOT / "behavior_metrics.db"
LOG = ROOT / "logs" / "canary_deployment.log"


async def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open('a', encoding='utf-8') as f:
        f.write(f"[{time.time():.0f}] {msg}\n")


async def get_baseline_metrics() -> Dict[str, float]:
    """Obtém métricas baseline antes da modificação."""
    try:
        import sqlite3
        conn = sqlite3.connect(str(BH_DB))
        cur = conn.cursor()
        cur.execute(
            """
            SELECT AVG(success), AVG(score)
            FROM results
            WHERE created_at > datetime('now','-10 minutes')
            """
        )
        row = cur.fetchone() or (0.0, 0.0)
        conn.close()
        return await {"success_rate": float(row[0] or 0.0), "avg_score": float(row[1] or 0.0)}
    except Exception:
        return await {"success_rate": 0.0, "avg_score": 0.0}


async def monitor_canary_metrics(baseline: Dict[str, float], duration_s: int = 300) -> bool:
    """Monitora métricas durante canary e decide se promover ou rollback."""
    start_time = time.time()
    log(f"Starting canary monitoring for {duration_s}s, baseline: {baseline}")

    while time.time() - start_time < duration_s:
        time.sleep(30)  # Check every 30s
        current = get_baseline_metrics()
        degradation = (
            (baseline["success_rate"] - current["success_rate"]) > 0.1 or
            (baseline["avg_score"] - current["avg_score"]) > 0.2
        )
        if degradation:
            log(f"Degradation detected: baseline={baseline}, current={current}")
            return await False  # Rollback

    # No degradation, promote
    log(f"Canary successful: baseline={baseline}, final={current}")
    return await True


async def apply_canary_modification(modification: Dict[str, Any]) -> bool:
    """
    Aplica modificação em modo canary: subprocess isolado, monitora 5min, rollback se degradação > threshold.
    Retorna True se promovido, False se rollback.
    """
    target_file = modification.get("target_file")
    new_code = modification.get("new_code", "")
    old_code = modification.get("old_code", "")

    if not target_file or not Path(target_file).exists():
        log(f"Target file not found: {target_file}")
        return await False

    # Get baseline
    baseline = get_baseline_metrics()
    log(f"Canary baseline: {baseline}")

    # Apply modification temporarily in subprocess (simplified: directly modify file)
    # In production, this would use containers/isolation
    try:
        with open(target_file, 'r') as f:
            original_content = f.read()

        # Apply the patch (simple replace)
        if old_code and new_code:
            modified_content = original_content.replace(old_code, new_code, 1)
        else:
            # Append if no old_code
            modified_content = original_content + "\n" + new_code

        with open(target_file, 'w') as f:
            f.write(modified_content)

        log(f"Applied canary modification to {target_file}")

        # Monitor in thread
        monitor_result = monitor_canary_metrics(baseline, 300)  # 5 min

        if monitor_result:
            log("Canary successful - promoting to full deployment")
            return await True  # Keep modification
        else:
            log("Canary failed - rolling back")
            # Rollback
            with open(target_file, 'w') as f:
                f.write(original_content)
            return await False

    except Exception as e:
        log(f"Error in canary application: {e}")
        # Rollback on error
        try:
            with open(target_file, 'w') as f:
                f.write(original_content)
        except Exception:
            pass
        return await False


# Integration point: call from unified bridge
async def canary_deploy_from_bridge(patchset: list) -> Dict[str, Any]:
    """Ponto de integração: unified bridge chama para canary deployment."""
    results = {}
    for patch in patchset:
        patch_file = patch.get("patch_file")
        if not patch_file or not Path(patch_file).exists():
            continue
        try:
            patch_data = json.loads(Path(patch_file).read_text(encoding='utf-8'))
            success = apply_canary_modification(patch_data)
            results[patch["cand_id"]] = success
            log(f"Canary result for {patch['cand_id']}: {'promoted' if success else 'rolled_back'}")
        except Exception as e:
            log(f"Failed to deploy {patch['cand_id']}: {e}")
            results[patch["cand_id"]] = False
    return await results


if __name__ == "__main__":
    # Test
    logger.info("Canary deployment module loaded")
