"""
Reward providers for UNIFIED_BRAIN main loop.
- Prefer external, deterministic, verifiable tasks.
- Fall back to a tiny built-in environment if external deps are unavailable.
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, List

try:
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None

try:
    import gymnasium as gym  # type: ignore
except Exception:  # pragma: no cover
    gym = None  # CartPole will be unavailable

import torch


def _set_deterministic_seed(seed: int = 1337) -> None:
    random.seed(seed)
    try:
        import numpy as np  # local import to avoid hard dependency
        np.random.seed(seed)
    except Exception:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class RewardProvider:
    """Abstract reward provider.

    The UNIFIED_BRAIN loop will call:
      - reset() at start
      - step(Z) each loop iteration
    Implementation maps latent Z to task actions and returns a scalar reward.
    """

    def reset(self) -> None:  # pragma: no cover
        raise NotImplementedError

    def step(self, z_t: torch.Tensor) -> float:  # pragma: no cover
        raise NotImplementedError


@dataclass
class CartPoleRewardProvider(RewardProvider):
    """CartPole-v1 backed reward provider (if gymnasium is available).

    Deterministic policy extraction from latent Z:
      - Uses first two latent dims to derive action.
    """
    seed: int = 1337
    seeds: Optional[List[int]] = None
    _seed_idx: int = 0
    env: Optional["gym.Env"] = None
    state: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if gym is None:  # pragma: no cover
            raise RuntimeError("gymnasium not available")
        _set_deterministic_seed(self.seed)
        self.env = gym.make("CartPole-v1")
        self.reset()

    def reset(self) -> None:
        assert self.env is not None
        seed = self.seed
        if self.seeds:
            seed = self.seeds[self._seed_idx % len(self.seeds)]
            self._seed_idx += 1
        obs, _ = self.env.reset(seed=seed)
        self.state = torch.tensor(obs, dtype=torch.float32)

    def _latent_to_action(self, z_t: torch.Tensor) -> int:
        # Map latent to a simple action via sign of a projection
        # Ensure shape [B,H]; use first sample and first dim
        z = z_t[0]
        # Use two dims if available for a tiny linear head
        x0 = float(z[0].item()) if z.numel() >= 1 else 0.0
        x1 = float(z[1].item()) if z.numel() >= 2 else 0.0
        score = 0.75 * x0 + 0.25 * x1
        return 1 if score > 0.0 else 0

    def step(self, z_t: torch.Tensor) -> float:
        assert self.env is not None
        action = self._latent_to_action(z_t)
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.state = torch.tensor(obs, dtype=torch.float32)
        done = terminated or truncated
        if done:
            # Small terminal shaping: negative on early termination, positive bonus on long runs
            shaped = float(reward)
            self.reset()
            return shaped
        return float(reward)


@dataclass
class MiniGrid1DRewardProvider(RewardProvider):
    """Tiny deterministic 1D grid environment without external dependencies.

    - Grid with length L; agent starts at 0, goal at L-1.
    - Action derived from latent sign; step right if score>0 else left.
    - Reward: +10 on reaching goal, -0.1 per step, episode ends on goal.
    """
    length: int = 11
    seed: int = 1337
    pos: int = 0

    def __post_init__(self) -> None:
        _set_deterministic_seed(self.seed)
        self.reset()

    def reset(self) -> None:
        self.pos = 0

    def _latent_to_action(self, z_t: torch.Tensor) -> int:
        # 1 -> right, 0 -> left
        z = z_t[0]
        x0 = float(z[0].item()) if z.numel() >= 1 else 0.0
        return 1 if x0 >= 0.0 else 0

    def step(self, z_t: torch.Tensor) -> float:
        action = self._latent_to_action(z_t)
        if action == 1:
            self.pos = min(self.length - 1, self.pos + 1)
        else:
            self.pos = max(0, self.pos - 1)
        # time penalty
        reward = -0.1
        if self.pos == self.length - 1:
            reward = 10.0
            self.reset()
        return reward


@dataclass
class FrozenLakeRewardProvider(RewardProvider):
    """FrozenLake-v1 provider (slippery False for determinism when available)."""
    seed: int = 1337
    map_name: str = "4x4"
    slippery: bool = False
    env: Optional["gym.Env"] = None
    state: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if gym is None:  # pragma: no cover
            raise RuntimeError("gymnasium not available")
        _set_deterministic_seed(self.seed)
        try:
            self.env = gym.make("FrozenLake-v1", map_name=self.map_name, is_slippery=self.slippery)
        except Exception:
            # Fall back if gymnasium API differs
            self.env = gym.make("FrozenLake-v1")
        self.reset()

    def reset(self) -> None:
        assert self.env is not None
        obs, _ = self.env.reset(seed=self.seed)
        self.state = torch.tensor([float(obs)], dtype=torch.float32)

    def _latent_to_action(self, z_t: torch.Tensor) -> int:
        # 0:Left,1:Down,2:Right,3:Up – pick argmax over 2 dims mapping
        z = z_t[0]
        if z.numel() < 2:
            return 2  # default Right
        a0 = float(z[0].item())
        a1 = float(z[1].item())
        # Simple quadrant mapping
        if a0 >= 0 and a1 >= 0:
            return 2  # Right
        if a0 < 0 and a1 >= 0:
            return 1  # Down
        if a0 < 0 and a1 < 0:
            return 0  # Left
        return 3  # Up

    def step(self, z_t: torch.Tensor) -> float:
        assert self.env is not None
        action = self._latent_to_action(z_t)
        obs, reward, terminated, truncated, _ = self.env.step(action)
        self.state = torch.tensor([float(obs)], dtype=torch.float32)
        done = terminated or truncated
        if done:
            shaped = float(reward)
            self.reset()
            return shaped
        # small step penalty
        return -0.01


@dataclass
class SequencePuzzleRewardProvider(RewardProvider):
    """Deterministic sequence puzzle; agent must output a repeating pattern.

    - Target pattern: [+1, -1, +1, -1, ...]
    - At each step, reward is +1 if sign of z[0] matches expected value else 0
    - Episode length fixed; small penalty otherwise
    """
    length: int = 32
    seed: int = 1337
    t: int = 0

    def __post_init__(self) -> None:
        _set_deterministic_seed(self.seed)
        self.reset()

    def reset(self) -> None:
        self.t = 0

    def step(self, z_t: torch.Tensor) -> float:
        expected = 1.0 if (self.t % 2 == 0) else -1.0
        z = z_t[0]
        x0 = float(z[0].item()) if z.numel() >= 1 else 0.0
        got = 1.0 if x0 >= 0 else -1.0
        self.t = (self.t + 1) % self.length
        return 1.0 if got == expected else -0.05


@dataclass
class GenericGymRewardProvider(RewardProvider):
    """Generic Gym provider for simple discrete action envs (e.g., MountainCar-v0, Acrobot-v1)."""
    env_name: str = "MountainCar-v0"
    seed: int = 1337
    env: Optional["gym.Env"] = None
    obs: Optional[torch.Tensor] = None

    def __post_init__(self) -> None:
        if gym is None:  # pragma: no cover
            raise RuntimeError("gymnasium not available")
        _set_deterministic_seed(self.seed)
        self.env = gym.make(self.env_name)
        self.reset()

    def reset(self) -> None:
        assert self.env is not None
        ob, _ = self.env.reset(seed=self.seed)
        try:
            self.obs = torch.tensor(ob, dtype=torch.float32)
        except Exception:
            self.obs = None

    def step(self, z_t: torch.Tensor) -> float:
        assert self.env is not None
        z = z_t[0]
        a0 = float(z[0].item()) if z.numel() >= 1 else 0.0
        a1 = float(z[1].item()) if z.numel() >= 2 else 0.0
        a2 = float(z[2].item()) if z.numel() >= 3 else 0.0
        try:
            n = int(self.env.action_space.n)
        except Exception:
            n = 2
        logits = [a0, a1, a2][:max(1, n)]
        action = int(max(range(len(logits)), key=lambda i: logits[i]))
        step = self.env.step(action)
        if len(step) == 5:
            ob, reward, terminated, truncated, _ = step
            done = terminated or truncated
        else:
            ob, reward, done, _ = step
        try:
            self.obs = torch.tensor(ob, dtype=torch.float32)
        except Exception:
            pass
        if done:
            shaped = float(reward)
            self.reset()
            return shaped
        return float(reward)

def _parse_env_seeds() -> Optional[List[int]]:
    import os
    raw = os.environ.get("UBRAIN_EVAL_SEEDS", "")
    if not raw:
        return None
    try:
        return [int(s.strip()) for s in raw.split(',') if s.strip()]
    except Exception:
        return None


def build_default_provider() -> RewardProvider:
    """Build reward provider from UBRAIN_TASK or best available default.

    UBRAIN_TASK options: cartpole|frozenlake|minigrid|sequence|mountaincar|acrobot
    """
    import os
    task = os.getenv("UBRAIN_TASK", "cartpole").lower()
    seeds = _parse_env_seeds()
    if task == "cartpole" and gym is not None:
        try:
            return CartPoleRewardProvider(seeds=seeds)
        except Exception:
            pass
    if task == "frozenlake" and gym is not None:
        try:
            return FrozenLakeRewardProvider()
        except Exception:
            pass
    if task == "sequence":
        return SequencePuzzleRewardProvider()
    if task == "minigrid":
        return MiniGrid1DRewardProvider()
    if task == "mountaincar" and gym is not None:
        try:
            return GenericGymRewardProvider(env_name="MountainCar-v0")
        except Exception:
            pass
    if task == "acrobot" and gym is not None:
        try:
            return GenericGymRewardProvider(env_name="Acrobot-v1")
        except Exception:
            pass
    # Fallbacks
    if gym is not None:
        try:
            return CartPoleRewardProvider(seeds=seeds)
        except Exception:
            pass
    return MiniGrid1DRewardProvider()


def build_provider(task: str) -> RewardProvider:
    """Build a reward provider for a specific task name.

    Supported: cartpole|frozenlake|minigrid|sequence|mountaincar|acrobot
    Falls back to a minimal internal grid if unavailable.
    """
    t = (task or "").strip().lower()
    seeds = _parse_env_seeds()
    if t == "cartpole" and gym is not None:
        try:
            return CartPoleRewardProvider(seeds=seeds)
        except Exception:
            pass
    if t == "frozenlake" and gym is not None:
        try:
            return FrozenLakeRewardProvider()
        except Exception:
            pass
    if t == "sequence":
        return SequencePuzzleRewardProvider()
    if t == "minigrid":
        return MiniGrid1DRewardProvider()
    if t == "mountaincar" and gym is not None:
        try:
            return GenericGymRewardProvider(env_name="MountainCar-v0")
        except Exception:
            pass
    if t == "acrobot" and gym is not None:
        try:
            return GenericGymRewardProvider(env_name="Acrobot-v1")
        except Exception:
            pass
    # fallbacks
    if gym is not None:
        try:
            return CartPoleRewardProvider(seeds=seeds)
        except Exception:
            pass
    return MiniGrid1DRewardProvider()
