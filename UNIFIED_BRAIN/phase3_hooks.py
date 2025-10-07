#!/usr/bin/env python3
"""
Phase 3 Hooks
- EpisodicMemory: stores compressed state sketches and computes novelty
- NoveltyReward: converts novelty to intrinsic reward suggestions
"""
from __future__ import annotations

from typing import List, Tuple, Dict, Any
from collections import deque
import torch
import torch.nn.functional as F


class EpisodicMemory:
    def __init__(self, max_episodes: int = 1000, max_per_episode: int = 512, sketch_dim: int = 32):
        self.max_episodes = max_episodes
        self.max_per_episode = max_per_episode
        self.sketch_dim = sketch_dim
        # Memory: list of deques, each stores tensors of size [sketch_dim]
        self.episodes: deque = deque(maxlen=max_episodes)
        self.current: deque = deque(maxlen=max_per_episode)

    def start_episode(self) -> None:
        if len(self.current) > 0:
            self.episodes.append(self.current)
        self.current = deque(maxlen=self.max_per_episode)

    def end_episode(self) -> None:
        if len(self.current) > 0:
            self.episodes.append(self.current)
        self.current = deque(maxlen=self.max_per_episode)

    def _sketch(self, obs: torch.Tensor) -> torch.Tensor:
        """P2.2 UPGRADE: Random Fourier Features para preservar estrutura"""
        flat = obs.flatten()
        if flat.numel() == 0:
            return torch.zeros(self.sketch_dim)
        
        # Random Fourier Features (determinístico por seed)
        if not hasattr(self, '_rff_W'):
            # Inicializar matriz de projeção uma vez
            seed = hash(str(flat.shape)) % (2**32)
            torch.manual_seed(seed)
            input_dim = flat.numel()
            self._rff_W = torch.randn(input_dim, self.sketch_dim // 2)
        
        # Ajustar se dimensão mudou
        if self._rff_W.shape[0] != flat.numel():
            seed = hash(str(flat.shape)) % (2**32)
            torch.manual_seed(seed)
            self._rff_W = torch.randn(flat.numel(), self.sketch_dim // 2)
        
        # Projeção não-linear (cos + sin preserva distâncias)
        z = torch.matmul(flat.unsqueeze(0), self._rff_W)
        sketch = torch.cat([torch.cos(z), torch.sin(z)], dim=1).squeeze(0)
        
        # Normalizar
        sketch = (sketch - sketch.mean()) / (sketch.std() + 1e-6)
        return sketch

    def record(self, obs: torch.Tensor) -> None:
        try:
            sketch = self._sketch(obs.detach().cpu())
            self.current.append(sketch)
        except Exception:
            pass

    def novelty(self, obs: torch.Tensor) -> float:
        try:
            q = self._sketch(obs.detach().cpu())
            # compute min L2 distance to any sketch in memory
            min_dist = None
            for ep in self.episodes:
                for s in ep:
                    d = torch.dist(q, s, p=2).item()
                    min_dist = d if min_dist is None else min(min_dist, d)
            if min_dist is None:
                return 1.0  # no memory yet → highly novel
            # normalize novelty to [0,1] using 1 - exp(-d)
            return float(1.0 - torch.exp(torch.tensor([-min_dist]))[0].item())
        except Exception:
            return 0.0


class NoveltyReward:
    def __init__(self, scale: float = 0.02):
        self.scale = scale

    def intrinsic(self, novelty: float) -> float:
        return float(self.scale * novelty)
