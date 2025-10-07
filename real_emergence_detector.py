#!/usr/bin/env python3
"""
RealEmergenceDetector
- Thin wrapper around emergence_detector.EmergenceDetector
- Adapts heterogeneous action logs into info-theoretic agent state tuples
- Returns a compact dict with metrics useful for orchestration and reporting
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

# Uses the local information-theoretic detector
from emergence_detector import EmergenceDetector


class RealEmergenceDetector:
    """Adapter for detecting emergence from generic multi-agent action records.

    Expected input (per action dict):
      {
        'agent_id': 'agent_001',
        'action': 'explore' | 'exploit' | ...,
        'reward': float,
        'success': bool,
        'generation': int,            # optional
        'timestamp': <str/float>,     # optional
        'effects': { ... }            # optional
      }
    """

    def __init__(self, min_agents: int = 3, threshold: float = 0.7) -> None:
        self._detector = EmergenceDetector(min_agents=min_agents, threshold=threshold)

    @staticmethod
    def _hash_pos(agent_id: str, generation: int) -> Tuple[float, float]:
        """Derive a stable pseudo-position in 2D from agent_id and generation.
        This is a harmless embedding used only for clustering metrics when no real positions exist.
        """
        h = int(hashlib.md5(f"{agent_id}:{generation}".encode()).hexdigest(), 16)
        # Map to a 20x20 toroidal grid and normalize to [0,1]
        x = (h % 20) / 19.0
        y = ((h // 20) % 20) / 19.0
        return float(x), float(y)

    def detect_emergence(self, actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Build agent_states for EmergenceDetector
        agent_states: List[Dict[str, Any]] = []
        for a in actions:
            agent_id = str(a.get('agent_id', 'agent'))
            action = str(a.get('action', 'unknown'))
            generation = int(a.get('generation', 0) or 0)

            # Synthetic target from effects or coarse context
            effects = a.get('effects', {}) or {}
            rt = effects.get('real_task', {}) if isinstance(effects, dict) else {}
            target = str(rt.get('type', 'none'))

            # Position proxy for clustering
            pos = self._hash_pos(agent_id, generation)

            agent_states.append({
                'agent_id': agent_id,
                'action': action,
                'target': target,
                'position': pos,
                'cycle': generation,
            })

        is_emergent, metrics = self._detector.detect_emergence(agent_states)
        metrics = dict(metrics or {})
        metrics['emergent'] = bool(is_emergent)
        # Add a simple qualitative interpretation
        score = float(metrics.get('emergence_score', 0.0))
        mi = float(metrics.get('mutual_info', 0.0))
        ent = float(metrics.get('action_entropy', 0.0))
        coordination = float(metrics.get('coordination_score', 0.0))
        label = 'none'
        if is_emergent and mi > 0.7 and coordination > 0.5 and ent < 1.5:
            label = 'strong'
        elif score > 0.5 or (mi > 0.3 and ent < 2.0):
            label = 'weak'
        metrics['emergence_label'] = label
        return metrics
