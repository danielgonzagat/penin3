from __future__ import annotations

import random
from typing import Dict, Any, Callable, List

Genome = Dict[str, float]
EvalFn = Callable[[Genome, random.Random], Dict[str, Any]]


class EnvPluginRegistry:
    def __init__(self):
        self._plugins: Dict[str, EvalFn] = {}

    def register(self, name: str, fn: EvalFn) -> None:
        self._plugins[name] = fn

    def get(self, name: str) -> EvalFn:
        return self._plugins[name]

    def list(self) -> List[str]:
        return sorted(self._plugins.keys())


REGISTRY = EnvPluginRegistry()


def dummy_symbolic(genome: Genome, rng: random.Random) -> Dict[str, Any]:
    # Simple symbolic-like task: parity and simple arithmetic signal from genome
    a = float(genome.get('a', 0.0))
    b = float(genome.get('b', 0.0))
    s = a + 2 * b + rng.uniform(-0.01, 0.01)
    parity = 1.0 if (int(abs(a)) + int(abs(b))) % 2 == 0 else 0.0
    objective = 0.5 * (1.0 - min(1.0, abs(s) / 10.0)) + 0.5 * parity
    return {
        'objective': objective,
        'behavior': [a % 1.0, b % 1.0],
        'linf': min(1.0, objective * 1.05),
        'robustness': 0.95,
        'caos_plus': 1.0,
        'cost_penalty': 1.0,
        'ece': 0.05,
        'rho_bias': 1.0,
        'rho': min(0.98, objective),
        'eco_ok': True,
        'consent': True,
    }


# Auto-register dummy task
REGISTRY.register('dummy_symbolic', dummy_symbolic)


# --------- Gym-based plugin helpers (optional deps) ---------
def _try_import_gym():
    try:
        from common.gymx import gym  # type: ignore
        return gym
    except Exception:
        return None


def register_gym_env(env_id: str, episodes: int = 3):
    """Register a gym-like environment as plugin if gym is available."""
    gym = _try_import_gym()
    if gym is None:
        return False

    def eval_fn(genome: Genome, rng: random.Random) -> Dict[str, Any]:
        env = gym.make(env_id)
        total = 0.0
        for ep in range(episodes):
            seed = rng.randint(1, 10_000_000)
            try:
                obs, _ = env.reset(seed=seed)
            except Exception:
                obs = env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
            done = False
            steps = 0
            # Simple policy derived from genome
            def act_from_obs(_obs):
                try:
                    import numpy as np  # type: ignore
                    arr = np.asarray(_obs, dtype='float32').flatten()
                    w = float(genome.get('w', 0.0))
                    b = float(genome.get('b', 0.0))
                    y = float(arr.sum() * (1.0 + w) + b)
                    # assume binary action when possible
                    return 1 if y > 0.0 else 0
                except Exception:
                    return 0
            while not done and steps < 1000:
                a = act_from_obs(obs)
                step = env.step(a)
                if len(step) >= 5:
                    obs, reward, terminated, truncated, _ = step
                    done = bool(terminated or truncated)
                else:
                    obs, reward, done, _ = step
                total += float(reward)
                steps += 1
        try:
            env.close()
        except Exception:
            pass
        mean_reward = total / max(1, episodes)
        # Safe normalization to (0,1)
        objective = 1.0 / (1.0 + pow(2.71828, -mean_reward / 100.0))
        return {
            'objective': float(objective),
            'behavior': [float(genome.get('w', 0.0)) % 1.0, float(genome.get('b', 0.0)) % 1.0],
            'linf': min(1.0, float(objective) * 1.05),
            'robustness': 0.95,
            'caos_plus': 1.0,
            'cost_penalty': 1.0,
            'ece': 0.05,
            'rho_bias': 1.0,
            'rho': min(0.98, float(objective)),
            'eco_ok': True,
            'consent': True,
        }

    REGISTRY.register(f'gym::{env_id}', eval_fn)
    return True


def load_portfolio_preset(name: str) -> (list, list):
    """Return (eval_fns, names) for a given preset name."""
    name = (name or '').lower()
    fns = []
    names = []
    if name in ('default', 'symbolic'):
        fns = [REGISTRY.get('dummy_symbolic')]
        names = ['symbolic']
    elif name in ('cartpole', 'gym'):
        ok = register_gym_env('CartPole-v1')
        if ok:
            fns = [REGISTRY.get('gym::CartPole-v1')]
            names = ['cartpole']
        else:
            fns = [REGISTRY.get('dummy_symbolic')]
            names = ['symbolic']
    elif name in ('procgen', 'procgen-coinrun'):
        # Safe stub: try Procgen via gym ID; fallback to symbolic
        ok = register_gym_env('procgen:procgen-coinrun-v0')
        if ok:
            fns = [REGISTRY.get('gym::procgen:procgen-coinrun-v0')]
            names = ['procgen-coinrun']
        else:
            fns = [REGISTRY.get('dummy_symbolic')]
            names = ['symbolic']
    elif name in ('atari', 'pong'):
        # Try Gym Atari Pong; fallback
        ok = register_gym_env('ALE/Pong-v5') or register_gym_env('PongNoFrameskip-v4')
        if ok:
            # Prefer the first that registered
            if 'gym::ALE/Pong-v5' in REGISTRY.list():
                fns = [REGISTRY.get('gym::ALE/Pong-v5')]
                names = ['pong']
            else:
                fns = [REGISTRY.get('gym::PongNoFrameskip-v4')]
                names = ['pong']
        else:
            fns = [REGISTRY.get('dummy_symbolic')]
            names = ['symbolic']
    elif name in ('mujoco', 'hopper'):
        ok = register_gym_env('Hopper-v4') or register_gym_env('Hopper-v3')
        if ok:
            if 'gym::Hopper-v4' in REGISTRY.list():
                fns = [REGISTRY.get('gym::Hopper-v4')]
            else:
                fns = [REGISTRY.get('gym::Hopper-v3')]
            names = ['hopper']
        else:
            fns = [REGISTRY.get('dummy_symbolic')]
            names = ['symbolic']
    elif name in ('minigrid', 'empty'):
        ok = register_gym_env('MiniGrid-Empty-5x5-v0')
        if ok:
            fns = [REGISTRY.get('gym::MiniGrid-Empty-5x5-v0')]
            names = ['minigrid-empty']
        else:
            fns = [REGISTRY.get('dummy_symbolic')]
            names = ['symbolic']
    else:
        fns = [REGISTRY.get('dummy_symbolic')]
        names = ['symbolic']
    return fns, names

