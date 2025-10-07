"""
Professional Configuration System
Clean, modular, production-ready
"""
import os
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"

# Ensure directories exist
for d in [DATA_DIR, MODELS_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_PATH = DATA_DIR / "intelligence.db"

# Model Paths
MNIST_MODEL_PATH = MODELS_DIR / "mnist_model.pth"
DQN_MODEL_PATH = MODELS_DIR / "dqn_cartpole.pth"

# Training Config
MNIST_CONFIG = {
    "hidden_size": 128,
    "lr": 0.001,
    "epochs": 1,
    "batch_size": 64,
}

# FIX P2-4: Expose MNIST training frequency via environment (default 50)
MNIST_TRAIN_FREQ = int(os.getenv("MNIST_TRAIN_FREQ", "50"))

DQN_CONFIG = {
    "hidden_size": 128,
    "lr": 0.001,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.995,
    "memory_size": 10000,
    "batch_size": 64,
}

PPO_CONFIG = {
    "hidden_size": 128,
    "lr": 5e-4,  # V6 PATCH: Increased from 3e-4 for faster convergence
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_coef": 0.2,
    "entropy_coef": 0.05,  # V6 PATCH: Increased from 0.01 for MORE exploration
    "value_coef": 0.5,
    "batch_size": 64,
    "n_steps": 256,  # V6 PATCH: Increased from 128 for better gradient estimates
    "n_epochs": 6,  # V6 PATCH: Increased from 4 for more learning per update
}

# API Configuration - production: read from environment only (no defaults)
API_KEYS = {
    "openai": os.getenv("OPENAI_API_KEY", ""),
    "mistral": os.getenv("MISTRAL_API_KEY", ""),
    "gemini": os.getenv("GEMINI_API_KEY", ""),
    "deepseek": os.getenv("DEEPSEEK_API_KEY", ""),
    "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
    "grok": os.getenv("GROK_API_KEY", ""),
}

API_MODELS = {
    "openai": "gpt-5",                     # OpenAI Responses API
    "mistral": "mistral/codestral-2508",   # Codestral 2508
    "gemini": "gemini/gemini-2.5-pro",     # Gemini 2.5 Pro
    "deepseek": "deepseek/deepseek-chat",  # DeepSeek-V3.1 (non-thinking)
    "anthropic": "claude-opus-4-1-20250805",
    "grok": "xai/grok-4",
}

# System Config
CYCLE_INTERVAL = 60  # seconds between cycles
API_CALL_INTERVAL = 20  # call APIs every N cycles
CHECKPOINT_INTERVAL = 10  # save checkpoint every N cycles

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# ===== Additional runtime validation and logging configuration =====

def validate_api_keys():
    """Validate which API keys are actually configured and warn for missing keys."""
    available = {}
    for provider, key in API_KEYS.items():
        is_set = bool(key and key.strip())
        available[provider] = is_set
        if not is_set:
            import logging as _logging
            _logging.getLogger(__name__).warning(
                f"⚠️ API key for '{provider}' not configured (set {provider.upper()}_API_KEY env var)"
            )
    return available

AVAILABLE_APIS = validate_api_keys()

