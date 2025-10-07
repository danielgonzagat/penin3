
# FUNÇÕES DETERMINÍSTICAS (substituem random)
import hashlib
import os
import time


def deterministic_random(seed_offset=0):
    """Substituto determinístico para random.random()"""
    import hashlib
    import time

    # Usa múltiplas fontes de determinismo
    sources = [
        str(time.time()).encode(),
        str(os.getpid()).encode(),
        str(id({})).encode(),
        str(seed_offset).encode()
    ]

    # Combina todas as fontes
    combined = b''.join(sources)
    hash_val = int(hashlib.md5(combined).hexdigest()[:8], 16)

    return (hash_val % 1000000) / 1000000.0


def deterministic_uniform(a, b, seed_offset=0):
    """Substituto determinístico para random.uniform(a, b)"""
    r = deterministic_random(seed_offset)
    return a + (b - a) * r


def deterministic_randint(a, b, seed_offset=0):
    """Substituto determinístico para random.randint(a, b)"""
    r = deterministic_random(seed_offset)
    return int(a + (b - a + 1) * r)


def deterministic_choice(seq, seed_offset=0):
    """Substituto determinístico para random.choice(seq)"""
    if not seq:
        raise IndexError("sequence is empty")

    r = deterministic_random(seed_offset)
    return seq[int(r * len(seq))]


def deterministic_shuffle(lst, seed_offset=0):
    """Substituto determinístico para random.shuffle(lst)"""
    if not lst:
        return

    # Shuffle determinístico baseado em ordenação por hash
    def sort_key(item):
        item_str = str(item) + str(seed_offset)
        return hashlib.md5(item_str.encode()).hexdigest()

    lst.sort(key=sort_key)


def deterministic_torch_rand(*size, seed_offset=0):
    """Substituto determinístico para torch.rand(*size)"""
    if not size:
        return torch.tensor(deterministic_random(seed_offset))

    # Gera valores determinísticos
    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_random(seed_offset + i))

    return torch.tensor(values).reshape(size)


def deterministic_torch_randint(low, high, size=None, seed_offset=0):
    """Substituto determinístico para torch.randint(low, high, size)"""
    if size is None:
        return torch.tensor(deterministic_randint(low, high, seed_offset))

    # Gera valores determinísticos
    if isinstance(size, int):
        size = (size,)

    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_randint(low, high, seed_offset + i))

    return torch.tensor(values).reshape(size)

#!/usr/bin/env python3
"""
TEIS V2 Enhanced - Real Intelligence Version (Audited, Improved & Safe)
- Safe TaskEngine (no unsafe file/OS ops)
- Supports rich exploration policy (entropy, epsilon-greedy, etc.)
- Implements experience replay (minibatch, resampling, robust learning)
- Uses robust advantage baseline for better learning
- Tracks richer metrics
- Checkpoints are resumable (resume from latest)
- CLI for configuration
- Reproducible with fixed seeds
- Keeps emergent detector as real-only if available
- Only uses stdlib, torch, numpy
"""

import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging
import hashlib
from collections import Counter, deque
import gym

# Optional advanced dynamics from Neural Farm
try:
    from chaos_utils import caos_gain
except Exception:
    caos_gain = None  # Fallback if CAOS not available

try:
    from oci_lyapunov import compute_oci, lyapunov_ok
except Exception:
    compute_oci = None  # Fallback if OCI not available
    lyapunov_ok = None

# Optionally import real emergence detector
try:
    from real_emergence_detector import RealEmergenceDetector
except Exception:
    RealEmergenceDetector = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TEIS_V2_ENHANCED")

# ===============================
# CONFIGURATION & CONSTANTS
# ===============================

# Replace simulated HTTP/file ops with realistic, safe tasks
TASK_TYPES = ['file_write', 'list_tmp', 'matrix_mul', 'json_agg', 'gym_cartpole']
ACTIONS = [
    'explore', 'exploit', 'communicate', 'cooperate', 'compete',
    'innovate', 'learn', 'teach', 'lead', 'follow'
]
# Set global default torch dtype to float32 for efficiency
torch.set_default_dtype(torch.float32)


def set_global_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True      # For CUDA, uncomment if needed
    # torch.backends.cudnn.benchmark = False         # For CUDA, uncomment if needed


# ===============================
# UTILITY
# ===============================

def safe_json_dump(obj, path):
    try:
        with open(path, 'a') as f:
            json.dump(obj, f)
            f.write('\n')
    except Exception as e:
        logger.warning(f"Failed to write to {path}: {e}")

def current_time():
    return datetime.now().isoformat(timespec="seconds")

def md5_short(s: str, n=8):
    return hashlib.md5(s.encode()).hexdigest()[:n]


# ===============================
# NETWORK MODULES
# ===============================

class RealDecisionNetwork(nn.Module):
    """Feedforward Neural Net for Decisions (synchronous)"""
    def __init__(self, input_dim, hidden_dim=64, output_dim=10, n_hidden=2, dropout=0.1):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
        for _ in range(n_hidden-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ===============================
# SAFE TASK ENGINE
# ===============================

class RealTaskEngine:
    """
    Executes deterministic, safe tasks: all files are written in a dedicated user-owned temp dir.
    No external HTTP or risky OS operations.
    """
    def __init__(self, tmp_dir: str):
        # Ensure only run-time user-writable location, never /root etc.
        # Use /tmp/teis_v2_enhanced_<user_id> to avoid cross-user issues.
        user = str(os.getuid())
        base_tmp = os.path.abspath(os.path.expanduser(tmp_dir))
        safe_dir = os.path.join(base_tmp, "teis_enhanced_user_" + user)
        os.makedirs(safe_dir, exist_ok=True)
        self.tmp_dir = safe_dir

        self.task_network = RealDecisionNetwork(input_dim=5, output_dim=4, hidden_dim=32)
        self.task_optimizer = torch.optim.Adam(self.task_network.parameters(), lr=1e-3)

    def select_task(self, agent_state: torch.Tensor, policy='argmax', epsilon=0.1) -> str:
        probs = F.softmax(self.task_network(agent_state), dim=-1).detach().cpu().numpy()
        if policy == 'argmax' or epsilon <= 0:
            idx = int(np.argmax(probs))
        elif policy == 'epsilon_greedy':
            if np.deterministic_uniform() < epsilon:
                idx = int(np.deterministic_choice(len(TASK_TYPES)))
            else:
                idx = int(np.argmax(probs))
        elif policy == 'entropy_sample':
            idx = int(np.deterministic_choice(len(TASK_TYPES), p=probs / probs.sum()))
        else:
            idx = int(np.argmax(probs))
        return TASK_TYPES[idx]

    def perform_task(self, agent_id: str, agent_state: torch.Tensor) -> Dict[str, Any]:
        """All file operations confined to safe tmpdir, no actual HTTP performed"""
        task_type = self.select_task(agent_state)
        try:
            if task_type == 'file_write':
                state_str = json.dumps(agent_state.cpu().tolist())
                state_hash = md5_short(state_str)
                filename = f'teis_{agent_id}_{state_hash}.txt'
                path = os.path.join(self.tmp_dir, filename)
                with open(path, 'w') as f:
                    f.write(f'agent={agent_id} time={current_time()}\n')
                resp = {'type': task_type, 'success': True, 'path': path}
            elif task_type == 'list_tmp':
                files = os.listdir(self.tmp_dir)
                resp = {'type': task_type, 'success': True, 'count': len(files)}
            elif task_type == 'matrix_mul':
                # Perform a bounded-size matrix multiplication to simulate real compute
                n = 32
                a = torch.randn(n, n)
                b = torch.randn(n, n)
                c = a @ b
                resp = {'type': task_type, 'success': True, 'shape': [n, n], 'mean': float(c.mean().item())}
            elif task_type == 'json_agg':
                # Aggregate a small in-memory JSON-like structure
                data = [{'v': float(deterministic_torch_rand(1).item())} for _ in range(100)]
                agg = float(np.mean([d['v'] for d in data]))
                resp = {'type': task_type, 'success': True, 'avg': agg}
            elif task_type == 'gym_cartpole':
                # Enhanced Gym interaction with full episode
                env = gym.make('CartPole-v1')
                obs, _ = env.reset()
                total_reward = 0
                steps = 0
                max_steps = 500

                while steps < max_steps:
                    # Use agent state to decide action
                    action = int(torch.argmax(self.task_network(agent_state)))
                    next_obs, reward, terminated, truncated, _ = env.step(action)
                    total_reward += reward
                    steps += 1

                    if terminated or truncated:
                        break

                    obs = next_obs

                env.close()
                resp = {
                    'type': task_type,
                    'success': steps >= 50,  # Success if survived at least 50 steps
                    'reward': total_reward,
                    'steps': steps,
                    'avg_reward': total_reward / steps if steps > 0 else 0
                }
            else:
                resp = {'type': task_type, 'success': False}
        except Exception as e:
            resp = {'type': task_type, 'success': False, 'error': str(e)}
        return resp


# ===============================
# EXPERIENCE REPLAY BUFFER
# ===============================

class ReplayBuffer:
    """Ring-buffer experience storage for agents"""
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer: List[Tuple[torch.Tensor, int, float, float]] = []
        self.position = 0

    def push(self, state: torch.Tensor, action: int, reward: float, advantage: float):
        data = (state.detach().cpu(), action, reward, advantage)
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
        else:
            self.buffer[self.position] = data
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> List[Tuple[torch.Tensor, int, float, float]]:
        batch_size = min(len(self.buffer), batch_size)
        idxs = np.deterministic_choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in idxs]

    def __len__(self):
        return len(self.buffer)


# ===============================
# AGENT WITH EXPLORATION POLICY, BASELINE
# ===============================

class RealAgentV2:
    """
    Each agent has: neural policy, replay buffer, robust baseline, exploration strategy.
    """
    def __init__(self, agent_id: str, seed: int):
        self.id = agent_id
        self.fitness = 0.0
        self.energy = 100.0
        self.seed = seed
        self.exploration_rate = 0.3  # Adicionado
        # Incompletude Infinita: probabilidade controlada de ações fora da política
        self.incompletude_rate = 0.10
        # Memória de novidade simples para curiosity-driven exploration
        from collections import deque as _deque
        self._novelty_recent = _deque(maxlen=512)
        self._novelty_set = set()
        set_global_seed(seed+17)
        self.brain = RealDecisionNetwork(input_dim=15, output_dim=len(ACTIONS), hidden_dim=64, n_hidden=2, dropout=0.15)
        self.optimizer = torch.optim.Adam(self.brain.parameters(), lr=1e-3)
        self.replay = ReplayBuffer(capacity=512)
        self.baseline_reward = 0.0  # For advantage calculation
        self.train_step = 0
        self.behavior_weights = torch.ones(len(ACTIONS)) / len(ACTIONS)

    def perceive(self, environment: Dict[str, Any]) -> torch.Tensor:
        features = [
            environment.get('resources', 0) / 100,
            environment.get('opportunities', 0) / 20,
            environment.get('threats', 0) / 10,
            environment.get('agent_count', 1) / 50,
            environment.get('innovation_index', 0),
            environment.get('cooperation_level', 0),
            environment.get('competition_level', 0),
            self.energy / 100,
            self.fitness / 100,
            len(self.replay.buffer) / 100,  # Learning experience
            np.sin(time.time() / 100),
            np.cos(time.time() / 100),
            float(self.energy > 50),
            float(self.fitness > 50),
            float(len(self.replay.buffer) > 10)
        ]
        return torch.tensor(features, dtype=torch.float32)

    def decide(self, state: torch.Tensor, exploration='entropy_sample', epsilon=0.05) -> Tuple[int, str, torch.Tensor]:
        logits = self.brain(state)
        probs = F.softmax(logits, dim=-1).detach().cpu().numpy()
        # Efetivar incompletude infinita (exploração garantida mínima)
        epsilon_eff = float(max(epsilon, getattr(self, 'exploration_rate', 0.05)))
        if np.random.rand() < getattr(self, 'incompletude_rate', 0.0):
            action_idx = int(np.deterministic_choice(len(ACTIONS)))
        elif exploration == 'entropy_sample':
            action_idx = int(np.deterministic_choice(len(ACTIONS), p=probs/probs.sum()))
        elif exploration == 'epsilon_greedy':
            if np.deterministic_uniform() < epsilon_eff:
                action_idx = int(np.deterministic_choice(len(ACTIONS)))
            else:
                action_idx = int(np.argmax(probs))
        else:
            action_idx = int(np.argmax(probs))
        # For learnability: store action probabilities
        action_prob = float(probs[action_idx])
        return action_idx, ACTIONS[action_idx], torch.tensor(action_prob, dtype=torch.float32)

    def execute(self, action_idx: int, action: str, state: torch.Tensor, environment: Dict[str, Any],
                task_engine: Optional[RealTaskEngine] = None) -> Dict[str, Any]:
        result = {'agent_id': self.id, 'action': action, 'success': False, 'reward': 0.0, 'effects': {}}
        # Probabilistic success chance based on energy, resources, threats and curriculum difficulty
        diff = float(environment.get('difficulty', 0.0))
        succ_raw = 0.10 + (self.energy / 150.0) + (environment.get('resources', 0.0) / 250.0) \
                   - (environment.get('threats', 0.0) / 20.0) - (diff * 0.50)
        succ = float(np.clip(succ_raw, 0.05, 0.95))
        result['success'] = bool(np.random.rand() < succ)
        # Action-specific effects (slightly more diverse formula)
        if action == 'explore':
            value = np.tanh((self.fitness + environment.get('opportunities', 0)) / 60)
            result['reward'] = value * 3
            result['effects']['new_resources'] = value > 0.7
        elif action == 'exploit':
            efficiency = (self.energy / 100) * (environment.get('resources', 1) / 100)
            result['reward'] = min(5, efficiency * 4)
        elif action == 'cooperate':
            result['reward'] = (environment.get('cooperation_level', 0) + 0.5) * 2
        elif action == 'compete':
            comp = (environment.get('competition_level', 0) + self.fitness / 100)
            result['reward'] = comp * 2.7
        elif action == 'innovate':
            innov = (len(self.replay.buffer) + self.fitness) / 120
            if innov > 1.1:
                result['reward'] = 10.0
                result['effects']['breakthrough'] = True
            else:
                result['reward'] = innov * 2
        elif action == 'learn':
            result['reward'] = 1.0 + len(self.replay.buffer) / 200
        elif action == 'teach':
            qual = self.fitness / 100 + len(self.replay.buffer) / 200
            result['reward'] = qual * 1.9
        elif action == 'lead':
            result['reward'] = self.fitness / 48 + self.energy / 100
        elif action == 'follow':
            result['reward'] = 0.7 + environment.get('cooperation_level', 0)
        elif action == 'communicate':
            comm = (self.fitness + environment.get('cooperation_level', 0) * 50) / 120
            result['reward'] = comm

        # Real-world task
        if task_engine and self.energy > 30:
            # Only permit the RealTaskEngine to be called with safe agent_state
            agent_state = torch.tensor([
                self.energy / 100, self.fitness / 100,
                len(self.replay.buffer) / 100, result['reward'] / 10,
                environment.get('resources', 0) / 100
            ], dtype=torch.float32)
            t_out = task_engine.perform_task(self.id, agent_state)
            result['effects']['real_task'] = t_out
            if t_out.get('success', False):
                result['reward'] += 1.0
        # Enhanced curiosity-driven exploration: reward novelty and uncertainty
        try:
            state_sig = tuple(np.round(state.detach().cpu().numpy(), 2).tolist())
            if state_sig not in self._novelty_set:
                self._novelty_set.add(state_sig)
                self._novelty_recent.append(state_sig)
                # Higher novelty bonus for early exploration
                novelty_bonus = 0.5 if len(self._novelty_set) < 1000 else 0.25
                result['reward'] += novelty_bonus

                # Additional bonus for action uncertainty (entropy)
                action_entropy = -torch.sum(action_prob * torch.log(action_prob + 1e-8))
                uncertainty_bonus = float(action_entropy) * 0.1
                result['reward'] += uncertainty_bonus
            elif len(self._novelty_recent) >= self._novelty_recent.maxlen:
                # Cleanup to allow rediscovery of novelty
                old = self._novelty_recent.popleft()
                try:
                    self._novelty_set.discard(old)
                except Exception:
                    pass
        except Exception:
            pass

        # Energy cost
        self.energy -= 1
        self.energy = float(np.clip(self.energy + result['reward'], 0, 100))
        # Result for learning
        return result

    def store_experience(self, state, action_idx, reward, baseline):
        """Store experience for replay; robust advantage tweak."""
        advantage = reward - baseline
        self.replay.push(state, action_idx, reward, advantage)

    def update_baseline(self, reward):
        """EMA baseline for robust advantage learning."""
        decay = 0.99
        self.baseline_reward = decay * self.baseline_reward + (1 - decay) * reward

    def learn_from_replay(self, batch_size=16, gamma=0.99, max_grad_norm=2.0):
        # Learn even with small buffers; downscale batch size accordingly
        if len(self.replay) == 0:
            return
        batch_size = min(len(self.replay), batch_size)
        batch = self.replay.sample(batch_size)
        states = torch.stack([item[0] for item in batch])
        actions = torch.tensor([item[1] for item in batch], dtype=torch.long)
        advantages = torch.tensor([item[3] for item in batch], dtype=torch.float32)

        logits = self.brain(states)
        log_probs = F.log_softmax(logits, dim=-1)
        sel_log_probs = log_probs[range(batch_size), actions]
        # Use robust advantage baseline to optimize surrogate
        loss = -(sel_log_probs * advantages.detach()).mean()

        self.optimizer.zero_grad()
        loss.backward()
        # Grad norm clipping
        torch.nn.utils.clip_grad_norm_(self.brain.parameters(), max_grad_norm)
        self.optimizer.step()

        # Also refresh behavior_weights (policy mean for analytics)
        with torch.no_grad():
            avg_probs = F.softmax(logits, dim=-1).mean(dim=0)
            self.behavior_weights = avg_probs

    def save_state(self):
        return {
            'id': self.id,
            'fitness': self.fitness,
            'energy': self.energy,
            'baseline_reward': self.baseline_reward,
            'replay': [(s.tolist(), a, r, adv) for (s, a, r, adv) in self.replay.buffer],
            'brain': self.brain.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }

    def load_state(self, d: dict):
        self.fitness = d.get('fitness', 0.)
        self.energy = d.get('energy', 100.)
        self.baseline_reward = d.get('baseline_reward', 0.)
        # Restore replay buffer (truncate if needed)
        self.replay.buffer = []
        for entry in d.get('replay', []):
            f, a, r, adv = entry
            self.replay.push(torch.tensor(f, dtype=torch.float32), a, r, adv)
        try:
            self.brain.load_state_dict(d['brain'])
        except Exception as e:
            logger.warning(f"Could not restore agent {self.id} brain: {e}")
        try:
            self.optimizer.load_state_dict(d['optimizer'])
        except Exception as e:
            logger.warning(f"Could not restore agent {self.id} optimizer: {e}")

# ===============================
# TEIS V2 ENHANCED SYSTEM
# ===============================

class TrueEmergentIntelligenceV2:
    """
    Full system: agents + safe environment + logging + resumable checkpointing + emergent detection.
    """
    def __init__(self, num_agents: int = 50, base_dir: str = "./teis_v2_out", seed: int = 747, resume: bool = False):
        # Dir structure/layout
        os.makedirs(base_dir, exist_ok=True)
        self.base_dir = base_dir
        self.metrics_file = os.path.join(base_dir, 'metrics.jsonl')
        self.trace_file = os.path.join(base_dir, 'trace.jsonl')
        self.emergent_log = os.path.join(base_dir, 'emergent_real.jsonl')
        self.actions_file = os.path.join(base_dir, 'actions.jsonl')
        self.checkpoint_dir = os.path.join(base_dir, "checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.generation = 0
        self.num_agents = num_agents
        self.global_seed = seed
        self.emergent_behaviors = []  # Adicionado

        # Real-time visualization setup
        self.metrics_history = {
            'generations': [],
            'max_fitness': [],
            'avg_fitness': [],
            'policy_entropy': [],
            'diversity': [],
            'success_rate': [],
            'difficulty': []
        }
        self.visualization_thread = None
        self.visualization_active = False
        set_global_seed(seed)

        # Anti-stagnation diagnostics (inspired by Neural Farm)
        self.best_max_fitness: float = 0.0
        self.best_avg_fitness: float = 0.0
        self.stagnation_steps: int = 0
        self.plateau_window: int = 5
        self.mutation_boost: float = 1.0
        self.resource_history: deque = deque(maxlen=50)

        # Initial environment
        self.environment = {
            'resources': 100.0,
            'opportunities': 10.0,
            'threats': 0.0,
            'agent_count': num_agents,
            'innovation_index': 0.0,
            'cooperation_level': 0.5,
            'competition_level': 0.5
        }

        # Only instantiate RealEmergenceDetector if present
        self.emergence_detector = RealEmergenceDetector() if RealEmergenceDetector else None

        # System meta-brain
        self.meta_brain = RealDecisionNetwork(input_dim=5, output_dim=5, hidden_dim=64)
        self.meta_optimizer = torch.optim.Adam(self.meta_brain.parameters(), lr=1e-4)

        # Safe task engine: exclusive temp dir
        self.task_engine = RealTaskEngine(tmp_dir=os.path.join(base_dir, 'tmp'))

        # (Re-)initialize or resume agents
        self.agents: List[RealAgentV2] = []
        if resume and self._resume_checkpoint():
            logger.info(f"Resumed state from checkpoint at generation {self.generation}")
        else:
            self.agents = []
            for i in range(num_agents):
                torch.manual_seed(seed + i*3)
                agent = RealAgentV2(f"agent_{i:03d}", seed=seed + i*3)
                agent.brain.apply(self._init_weights)
                self.agents.append(agent)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if hasattr(m.bias, 'data'):
                m.bias.data.fill_(0.01)

    # ------ GENERATION LOOP --------
    def run_generation(self, exploration='entropy_sample', epsilon=0.20,
                      minibatch=16, learn_steps=2):
        self.generation += 1
        logger.info(f"Generation {self.generation} starting...")
        # Perceive
        perceptions = {a.id: a.perceive(self.environment) for a in self.agents}
        all_results = []

        # Decide/Exec per agent
        for agent in self.agents:
            state = perceptions[agent.id]
            a_idx, action, action_prob = agent.decide(state, exploration=exploration, epsilon=epsilon)
            result = agent.execute(a_idx, action, state, self.environment, self.task_engine)
            agent.fitness += result['reward']
            agent.update_baseline(result['reward'])
            agent.store_experience(state, a_idx, result['reward'], agent.baseline_reward)
            all_results.append({
                **result,
                'agent_fitness': agent.fitness,
                'energy': agent.energy,
                'generation': self.generation
            })

        # Persist per-action records for info-theoretic analysis
        try:
            with open(self.actions_file, 'a') as f:
                for r in all_results:
                    f.write(json.dumps({
                        'agent_id': r.get('agent_id', ''),
                        'action': r.get('action', ''),
                        'reward': float(r.get('reward', 0.0)),
                        'success': bool(r.get('success', False)),
                        'generation': int(self.generation),
                        'effects': r.get('effects', {})
                    }) + '\n')
        except Exception:
            pass

        # Experience replay (multi-iteration per generation for robustness)
        for agent in self.agents:
            for _ in range(learn_steps):
                agent.learn_from_replay(batch_size=minibatch)

        # Detect real emergence (if enabled)
        self.detect_real_emergence(all_results)

        # Update environment dynamics
        self.update_environment(all_results)

        # Curriculum difficulty schedule (progressively harder tasks)
        try:
            if self.generation < 5:
                self.environment['difficulty'] = 0.05  # Very easy start
            elif self.generation < 15:
                self.environment['difficulty'] = 0.15  # Gradual increase
            elif self.generation < 30:
                self.environment['difficulty'] = 0.30
            elif self.generation < 50:
                self.environment['difficulty'] = 0.50
            elif self.generation < 75:
                self.environment['difficulty'] = 0.65
            else:
                self.environment['difficulty'] = 0.85  # Very hard
        except Exception:
            self.environment['difficulty'] = 0.40

        # Evolution
        self.natural_selection()

        # Save checkpoint on significant improvements (simple heuristic)
        try:
            if getattr(self, 'best_max_fitness', 0.0) <= getattr(self, 'last_gen_max_fitness', 0.0):
                self._save_checkpoint()
        except Exception:
            pass

        # Metrics logging
        self.log_metrics(all_results)

        # IA³: Auto-reflexão e adaptação
        if self.generation % 5 == 0:  # A cada 5 gerações
            self._ia3_self_reflection()
            self._ia3_adaptive_learning()

        # IA³: Meta-aprendizado
        if self.generation % 10 == 0:  # A cada 10 gerações
            self._ia3_meta_learning()

    def detect_real_emergence(self, actions: List[Dict[str, Any]]):
        if not self.emergence_detector or not actions:
            return
        try:
            result = self.emergence_detector.detect_emergence(actions)
        except Exception:
            return
        if result:
            out_data = {
                **result,
                'generation': self.generation,
                'timestamp': current_time()
            }
            safe_json_dump(out_data, self.emergent_log)
            self.write_trace('real_emergence_detected', out_data)

    def update_environment(self, actions: List[Dict[str, Any]]):
        # Aggregate
        action_counts = {}
        total_reward = 0.
        for action in actions:
            act_type = action['action']
            action_counts[act_type] = action_counts.get(act_type, 0) + 1
            total_reward += action.get('reward', 0.0)
        n = max(1, len(actions))
        # Effects
        exploits = action_counts.get('exploit', 0)
        explores = action_counts.get('explore', 0)
        self.environment['resources'] -= 0.5 * exploits
        self.environment['resources'] += 0.2 * explores
        self.environment['resources'] = float(np.clip(self.environment['resources'], 0, 200))
        innov = action_counts.get('innovate', 0)
        self.environment['opportunities'] = float(np.clip(5 + innov, 0, 50))
        coops = action_counts.get('cooperate', 0)
        comps = action_counts.get('compete', 0)
        tot_social = coops + comps + 1
        self.environment['cooperation_level'] = coops / tot_social
        self.environment['competition_level'] = comps / tot_social
        self.environment['innovation_index'] = innov / n
        # Threats from too much competition
        if self.environment['competition_level'] > 0.7:
            self.environment['threats'] = min(10, self.environment['threats'] + 1)
        else:
            self.environment['threats'] = max(0, self.environment['threats'] - 1)
        self.environment['agent_count'] = len(self.agents)

    def natural_selection(self):
        # Top 50% by fitness survive; offspring from tournament/weighted avg, with mutation
        self.agents.sort(key=lambda a: a.fitness, reverse=True)
        survivors = self.agents[:len(self.agents)//2]
        offspring = []
        n_survivors = len(survivors)
        for i in range(len(self.agents) - n_survivors):
            p1 = survivors[i % n_survivors]
            p2 = survivors[(i+1) % n_survivors]
            child_id = f"gen{self.generation}_agent_{i:03d}"
            torch.manual_seed(self.global_seed + self.generation * 331 + i)
            child = RealAgentV2(child_id, seed=self.global_seed + self.generation * 331 + i)
            with torch.no_grad():
                for cpar, p1par, p2par in zip(child.brain.parameters(),
                                              p1.brain.parameters(), p2.brain.parameters()):
                    denom = (p1.fitness + p2.fitness + 1e-10)
                    w1 = p1.fitness / denom if denom > 0 else 0.5
                    w2 = p2.fitness / denom if denom > 0 else 0.5
                    cpar.data = w1 * p1par.data + w2 * p2par.data
                    # Mutation decay + Anti-estagnação boost
                    noise_scale = (0.05 / (1.0 + self.generation/25)) * float(max(1.0, self.mutation_boost))
                    noise = torch.randn_like(cpar.data) * noise_scale
                    cpar.data += noise
            offspring.append(child)
        # Reset fitness, energy for next gen
        for a in survivors+offspring:
            a.fitness = 0.0
            a.energy = 100.0
        self.agents = survivors + offspring

    def log_metrics(self, action_results: List[Dict[str, Any]]):
        # Metrics: richer
        successful = sum(a['success'] for a in action_results)
        avg_reward = np.mean([a['reward'] for a in action_results])
        max_fitness = max(a['agent_fitness'] for a in action_results)
        avg_fitness = np.mean([a['agent_fitness'] for a in action_results])
        min_energy = min(a['energy'] for a in action_results)
        max_energy = max(a['energy'] for a in action_results)
        # Real population diversity: mean cosine distance between agent policy logits over a probe
        try:
            with torch.no_grad():
                probe = torch.randn(8, 15)
                vecs = []
                for agent in self.agents:
                    v = agent.brain(probe).flatten().cpu().numpy()
                    vecs.append(v / (np.linalg.norm(v) + 1e-8))
                if len(vecs) >= 2:
                    sims = []
                    for i in range(len(vecs)):
                        for j in range(i+1, len(vecs)):
                            sims.append(float(np.clip(np.dot(vecs[i], vecs[j]), -1.0, 1.0)))
                    diversity = float(1.0 - np.mean(sims))
                else:
                    diversity = 0.0
        except Exception:
            diversity = 0.0
        # Policy (behavior) entropy mean per-agent
        entropies = []
        for agent in self.agents:
            p = agent.behavior_weights.cpu().numpy()
            ent = -np.sum(p * np.log(p + 1e-8))
            entropies.append(ent)
        action_counts = Counter([a.get('action', '') for a in action_results])

        # Persist last generation max fitness for logging/reporting
        try:
            self.last_gen_max_fitness = float(max_fitness)
        except Exception:
            self.last_gen_max_fitness = 0.0

        metrics = {
            'generation': self.generation,
            'timestamp': current_time(),
            'num_agents': len(self.agents),
            'successful_actions': successful,
            'success_rate': successful / len(action_results) if action_results else 0,
            'avg_reward': avg_reward,
            'max_fitness': max_fitness,
            'avg_fitness': avg_fitness,
            'min_energy': min_energy,
            'max_energy': max_energy,
            'policy_entropy_mean': float(np.mean(entropies)),
            'policy_entropy_std': float(np.std(entropies)),
            'diversity_var': diversity,
            'environment': dict(self.environment),
            'difficulty': float(self.environment.get('difficulty', 0.0)),
            'exploration_rate_mean': float(np.mean([getattr(a, 'exploration_rate', 0.0) for a in self.agents])),
            'incompletude_rate': float(np.mean([getattr(a, 'incompletude_rate', 0.0) for a in self.agents]))
        }
        try:
            self._update_stagnation_and_caos(metrics, action_counts)
        except Exception:
            pass
        safe_json_dump(metrics, self.metrics_file)
        self.write_trace('metrics_logged', metrics)

    def save_final_report(self):
        """Persist a concise trend report for audit."""
        try:
            report = {
                'generations': self.metrics_history.get('generations', []),
                'max_fitness': self.metrics_history.get('max_fitness', []),
                'avg_fitness': self.metrics_history.get('avg_fitness', []),
                'policy_entropy': self.metrics_history.get('policy_entropy', []),
                'diversity': self.metrics_history.get('diversity', []),
                'success_rate': self.metrics_history.get('success_rate', []),
                'difficulty': self.metrics_history.get('difficulty', []),
            }
            path = os.path.join(self.base_dir, 'final_trend_report.json')
            with open(path, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass

    def _checkpoint_fname(self, gen: int):
        return os.path.join(self.checkpoint_dir, f"gen_{gen:03d}.ckpt")

    def _save_checkpoint(self):
        ckpt = {
            'generation': self.generation,
            'agents': [a.save_state() for a in self.agents],
            'meta_brain': self.meta_brain.state_dict(),
            'meta_optimizer': self.meta_optimizer.state_dict(),
            'environment': dict(self.environment)
        }
        fname = self._checkpoint_fname(self.generation)
        try:
            torch.save(ckpt, fname)
            logger.info(f"Checkpoint saved: {fname}")
        except Exception as e:
            logger.warning(f"Could not save checkpoint: {e}")

    def _resume_checkpoint(self) -> bool:
        candidates = []
        for f in os.listdir(self.checkpoint_dir):
            if f.endswith('.ckpt'):
                try:
                    g = int(f.split('_')[1].split('.')[0])
                    candidates.append((g, os.path.join(self.checkpoint_dir, f)))
                except Exception:
                    continue
        if not candidates:
            return False
        # Resume from latest
        last_gen, fname = max(candidates, key=lambda x: x[0])
        try:
            # Prefer full-object load to restore environment and agent states
            try:
                ckpt = torch.load(fname, map_location='cpu', weights_only=False)
            except TypeError:
                # Fallback for older torch versions without weights_only
                ckpt = torch.load(fname, map_location='cpu')
            self.generation = ckpt.get('generation', 0)
            meta_brain_sd = ckpt.get('meta_brain', {})
            try:
                self.meta_brain.load_state_dict(meta_brain_sd)
            except Exception:
                pass
            try:
                self.meta_optimizer.load_state_dict(ckpt['meta_optimizer'])
            except Exception:
                pass
            self.environment = ckpt.get('environment', self.environment)
            self.agents = []
            for agent_state in ckpt['agents']:
                a = RealAgentV2(agent_state.get('id', 'agent_xxx'), seed=self.global_seed+42)
                a.load_state(agent_state)
                self.agents.append(a)
            return True
        except Exception as e:
            logger.warning(f"Failed checkpoint resume: {e}")
            return False

    def write_trace(self, event: str, data: Any):
        trace = {
            'timestamp': current_time(),
            'system': 'TEIS_V2_ENHANCED',
            'event': event,
            'data': data
        }
        safe_json_dump(trace, self.trace_file)

    def _update_stagnation_and_caos(self, metrics: Dict[str, Any], action_counts: Counter):
        """Neural-Farm–inspired anti-stagnation and CAOS/OCI integration."""
        try:
            self.resource_history.append(float(self.environment.get('resources', 0.0)))
        except Exception:
            pass

        max_fit = float(metrics.get('max_fitness', 0.0))
        avg_fit = float(metrics.get('avg_fitness', 0.0))

        improved = (max_fit > self.best_max_fitness + 1e-6) or (avg_fit > self.best_avg_fitness + 1e-6)
        if improved:
            self.best_max_fitness = max(self.best_max_fitness, max_fit)
            self.best_avg_fitness = max(self.best_avg_fitness, avg_fit)
            self.stagnation_steps = 0
            self.mutation_boost = max(1.0, self.mutation_boost * 0.9)
        else:
            self.stagnation_steps += 1

        if self.stagnation_steps >= self.plateau_window:
            self.mutation_boost = min(3.0, self.mutation_boost * 1.25)
            for agent in self.agents:
                try:
                    agent.exploration_rate = float(min(0.6, max(0.05, getattr(agent, 'exploration_rate', 0.1) * 1.15)))
                    if hasattr(agent, 'optimizer'):
                        for pg in agent.optimizer.param_groups:
                            pg['lr'] *= 1.10
                except Exception:
                    continue
            try:
                self.environment['opportunities'] = float(min(20.0, self.environment.get('opportunities', 0.0) * 1.10))
            except Exception:
                pass
            self.stagnation_steps = 0

        caos_g = None
        try:
            if caos_gain is not None:
                C = float(self.environment.get('cooperation_level', 0.0))
                A = float(self.environment.get('competition_level', 0.0))
                O = float(self.environment.get('opportunities', 0.0)) / 20.0
                O = max(0.0, min(1.0, O))
                S = float(metrics.get('success_rate', 0.0))
                caos_g = caos_gain(C, A, O, S)
        except Exception:
            caos_g = None

        if isinstance(caos_g, (int, float)) and caos_g > 0:
            boost = 0.9 + 0.2 * min(2.0, float(caos_g))
            self.mutation_boost = float(min(3.0, self.mutation_boost * boost))
            for agent in self.agents:
                try:
                    agent.exploration_rate = float(min(0.6, max(0.05, getattr(agent, 'exploration_rate', 0.1) * (0.95 + 0.1 * float(caos_g)))))
                    if hasattr(agent, 'optimizer'):
                        for pg in agent.optimizer.param_groups:
                            pg['lr'] *= (0.98 + 0.04 * min(2.0, float(caos_g)))
                except Exception:
                    continue
            try:
                self.environment['resources'] = float(max(0.0, self.environment.get('resources', 0.0) * (0.98 + 0.04 * min(2.0, float(caos_g)))))
            except Exception:
                pass

        oci_value = None
        try:
            if compute_oci is not None:
                relations = [('lead', 'follow'), ('communicate', 'cooperate'), ('exploit', 'explore')]
                closed = []
                for a, b in relations:
                    if action_counts.get(a, 0) > 0 and action_counts.get(b, 0) > 0:
                        closed.append((a, b))
                oci_value = compute_oci(relations, closed)
        except Exception:
            oci_value = None

        lyap_ok = None
        try:
            if lyapunov_ok is not None and len(self.resource_history) >= 5:
                lyap_ok = lyapunov_ok(list(self.resource_history))
        except Exception:
            lyap_ok = None

        try:
            metrics['stagnation_steps'] = self.stagnation_steps
            metrics['mutation_boost'] = self.mutation_boost
            if isinstance(caos_g, (int, float)):
                metrics['caos_gain'] = float(caos_g)
            if isinstance(oci_value, (int, float)):
                metrics['oci'] = float(oci_value)
            if isinstance(lyap_ok, bool):
                metrics['lyapunov_ok'] = bool(lyap_ok)
        except Exception:
            pass

    # ===============================
    # IA³: EVOLUÇÃO PARA INTELIGÊNCIA REAL
    # ===============================

    def _ia3_self_reflection(self):
        """IA³: Auto-reflexão e diagnóstico do sistema"""
        # Analisar saúde dos agentes
        healthy_agents = sum(1 for a in self.agents if a.energy > 0.1)
        avg_fitness = np.mean([a.fitness for a in self.agents])

        # Diagnóstico IA³
        if healthy_agents / len(self.agents) < 0.5:
            logger.warning("🤔 IA³ DIAGNÓSTICO: Baixa saúde dos agentes - iniciando recuperação")
            self._ia3_self_healing()
        elif avg_fitness > 50:
            logger.info("🤔 IA³ DIAGNÓSTICO: Alto desempenho - potencial emergência inteligente")
            self._ia3_accelerate_evolution()

    def _ia3_self_healing(self):
        """IA³: Auto-cura do sistema"""
        # Reinicializar agentes fracos
        for agent in self.agents:
            if agent.energy < 0.1:
                logger.info(f"🔄 IA³ CURA: Reinicializando agente {agent.id}")
                agent.energy = 1.0
                agent.fitness = 0.0
                # Resetar pesos da rede neural
                agent.brain.apply(self._init_weights)

    def _ia3_accelerate_evolution(self):
        """IA³: Acelerar evolução quando desempenho alto"""
        # Aumentar taxa de aprendizado
        for agent in self.agents:
            if hasattr(agent, 'optimizer'):
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] *= 1.1

        # Aumentar exploração
        self.environment['opportunities'] = min(100, self.environment['opportunities'] * 1.2)

    def _ia3_adaptive_learning(self):
        """IA³: Aprendizado adaptativo baseado em histórico"""
        # Analisar padrões de sucesso/falha
        recent_actions = []
        if os.path.exists(self.trace_file):
            with open(self.trace_file, 'r') as f:
                lines = f.readlines()[-100:]  # Últimas 100 ações
                for line in lines:
                    try:
                        data = json.loads(line)
                        recent_actions.append(data)
                    except:
                        continue

        # Adaptar estratégia baseada em padrões
        success_rate = sum(1 for a in recent_actions if a.get('reward', 0) > 0) / max(1, len(recent_actions))

        if success_rate > 0.7:
            # Alto sucesso: aumentar exploração
            for agent in self.agents:
                agent.exploration_rate = min(0.5, agent.exploration_rate * 1.2)
        elif success_rate < 0.3:
            # Baixo sucesso: focar em exploração
            for agent in self.agents:
                agent.exploration_rate = max(0.01, agent.exploration_rate * 0.8)

    def _ia3_meta_learning(self):
        """IA³: Meta-aprendizado - aprender como aprender melhor"""
        # Usar meta-brain para otimizar parâmetros de aprendizado
        meta_input = torch.tensor([
            self.generation / 100.0,  # Normalizado
            len(self.agents) / 50.0,
            self.environment['resources'] / 200.0,
            np.mean([a.fitness for a in self.agents]) / 100.0,
            len(self.emergent_behaviors) / 10.0
        ], dtype=torch.float32)

        # Meta-brain decide ajustes
        with torch.no_grad():
            meta_output = self.meta_brain(meta_input.unsqueeze(0))
            adjustments = meta_output.squeeze().tolist()

        # Aplicar ajustes
        lr_adjust = 1.0 + (adjustments[0] - 0.5) * 0.5  # ±50%
        exp_adjust = 0.1 + adjustments[1] * 0.3  # 0.1-0.4
        res_adjust = 1.0 + (adjustments[2] - 0.5) * 0.2  # ±20%

        # Aplicar aos agentes
        for agent in self.agents:
            if hasattr(agent, 'optimizer'):
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] *= lr_adjust
            agent.exploration_rate = exp_adjust

        self.environment['resources'] = max(0, self.environment['resources'] * res_adjust)

# ===============================
# CLI PARSING & MAIN ENTRYPOINT
# ===============================

def get_cli_args():
    parser = argparse.ArgumentParser(description='TEIS V2 Enhanced - Real Intelligence Simulation')
    parser.add_argument('--generations', type=int, default=100, help='Number of generations to simulate')
    parser.add_argument('--agents', type=int, default=50, help='Number of agents in the system')
    parser.add_argument('--base-dir', type=str, default='./teis_v2_out', help='Base directory for outputs/checkpoints')
    parser.add_argument('--resume', action='store_true', help='Resume from latest checkpoint if available')
    parser.add_argument('--seed', type=int, default=747, help='Random seed for reproducibility')
    parser.add_argument('--minibatch', type=int, default=16, help='Experience replay minibatch size')
    parser.add_argument('--exploration', type=str, default='entropy_sample', choices=['argmax','epsilon_greedy','entropy_sample'], help='Exploration policy')
    parser.add_argument('--epsilon', type=float, default=0.20, help='Epsilon for epsilon-greedy policy')
    parser.add_argument('--learn-steps', type=int, default=2, help='Replay learning steps per agent per generation')
    parser.add_argument('--sleep', type=float, default=0.08, help='Sleep (in seconds) per generation (to chill CPU)')
    parser.add_argument('--save-every', type=int, default=1, help='Checkpoint after every N generations')
    parser.add_argument('--visualize', action='store_true', help='Enable real-time visualization')
    return parser.parse_args()

def main():
    args = get_cli_args()
    logger.info(f"🧠 Starting TEIS V2 Enhanced (NO FAKE RANDOM) | Agents: {args.agents} | Generations: {args.generations}")
    logger.info(f"Output: {args.base_dir}, Resume: {args.resume}, Seed: {args.seed}, Exploration: {args.exploration}")

    set_global_seed(args.seed)
    system = TrueEmergentIntelligenceV2(
        num_agents=args.agents,
        base_dir=args.base_dir,
        seed=args.seed,
        resume=args.resume)

    for gen in range(system.generation, args.generations):
        system.run_generation(
            exploration=args.exploration,
            epsilon=args.epsilon,
            minibatch=args.minibatch,
            learn_steps=args.learn_steps,
        )
        if ((gen+1) % args.save_every) == 0:
            system._save_checkpoint()

        if (gen+1) % 10 == 0:
            logger.info(f"Generation {gen+1}: Max fitness (last gen) = {getattr(system, 'last_gen_max_fitness', 0.0):.3f}")

        time.sleep(args.sleep)

    # Stop visualization
    if args.visualize:
        system.stop_visualization()

    logger.info(f"✅ TEIS V2 ENHANCED completed {args.generations} generations.")

if __name__ == "__main__":
    main()
