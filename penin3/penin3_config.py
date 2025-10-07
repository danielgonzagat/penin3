"""
PENIN³ Configuration
====================

Configuração centralizada para o sistema unificado.
"""

from pathlib import Path
from typing import Dict, Any

# Paths
PENIN3_ROOT = Path("/root/penin3")
V7_ROOT = Path("/root/intelligence_system")
PENIN_OMEGA_ROOT = Path("/root/peninaocubo")

# Database
PENIN3_DB = PENIN3_ROOT / "data" / "penin3.db"
V7_DB = V7_ROOT / "data" / "intelligence.db"

# Logs
LOGS_DIR = PENIN3_ROOT / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# WORM Ledger
WORM_LEDGER_PATH = PENIN3_ROOT / "data" / "worm_audit.db"
WORM_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)

# Checkpoints
CHECKPOINTS_DIR = PENIN3_ROOT / "checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# PENIN³ Configuration
PENIN3_CONFIG = {
    # Master Equation
    "master_equation": {
        "initial_I": 0.0,
        "alpha_base": 0.1,
        "enable": True
    },
    
    # CAOS+ Amplification
    "caos": {
        "c": 0.8,  # Consciousness
        "a": 0.5,  # Autonomy
        "o": 0.7,  # Optimization
        "s": 0.9,  # Self-modification
        "kappa": 20.0,
        "enable": True,
        "stagnation_threshold": 5  # cycles without improvement
    },
    
    # L∞ Aggregation
    "linf": {
        "weights": {
            "mnist": 1.0,
            "cartpole": 1.0,
            "ia3": 0.5  # If available
        },
        "cost_weight": 0.1,
        "enable": True
    },
    
    # Sigma Guard
    "sigma_guard": {
        "enable": True,
        "fail_closed": True,
        "thresholds": {
            "accuracy": 0.80,
            "robustness": 0.70,
            "fairness": 0.70,
        }
    },
    
    # SR-Ω∞ Self-Reflection
    "sr_omega": {
        "enable": True,
        "dimensions": ["accuracy", "robustness", "fairness", "cost"]
    },
    
    # ACFA League
    "acfa_league": {
        "enable": True,
        "promotion_threshold": 0.02,  # 2% improvement in L∞
        "rollback_threshold": -0.05,  # 5% degradation
        "eval_window": 10  # cycles
    },
    
    # WORM Ledger
    "worm_ledger": {
        "enable": True,
        "log_every_cycle": True,
        "verify_chain": True
    },
    
    # Router Multi-LLM
    "router": {
        "enable": False,  # Optional for now
        "daily_budget_usd": 5.0
    },
    
    # V7 Integration
    "v7": {
        "cycles_per_penin3": 1,  # How many V7 cycles per PENIN³ cycle
        "enable_mnist": True,
        "enable_cartpole": True,
        "enable_apis": False  # Optional
    },
    
    # Monitoring
    "monitoring": {
        "log_level": "INFO",
        "save_checkpoints": True,
        "checkpoint_interval": 10,
        "print_summary": True
    }
}


def get_config() -> Dict[str, Any]:
    """Get PENIN³ configuration"""
    return PENIN3_CONFIG.copy()


def update_config(updates: Dict[str, Any]) -> None:
    """Update PENIN³ configuration"""
    global PENIN3_CONFIG
    
    for key, value in updates.items():
        if key in PENIN3_CONFIG:
            if isinstance(value, dict):
                PENIN3_CONFIG[key].update(value)
            else:
                PENIN3_CONFIG[key] = value
        else:
            PENIN3_CONFIG[key] = value
