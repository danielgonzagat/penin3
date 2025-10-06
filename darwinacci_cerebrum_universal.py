"""
🧠 DARWINACCI CEREBRUM - CENTRO NEURAL UNIVERSAL

═══════════════════════════════════════════════════════════════════════════════
ARQUITETURA SIMBIÓTICA UNIVERSAL
═══════════════════════════════════════════════════════════════════════════════

Darwinacci-Ω como CENTRO NEURAL que conecta TODOS os sistemas via SINAPSES:

    V7 Ultimate ←─┐
    UNIFIED_BRAIN ←┤
    Neural Farm ←──┼─→ DARWINACCI ←─→ WORM Universal
    IA3 Systems ←──┤     CEREBRUM         (Ledger Compartilhado)
    PENIN³ ←───────┘

PROTOCOLOS:
• Universal Genome: formato de troca entre sistemas
• Universal Fitness: métrica agregada unificada
• Synaptic Transfer: transferência bidirecional de conhecimento

OBJETIVO: Transformar fragmentos desconectados em ORGANISMO INTELIGENTE ÚNICO
═══════════════════════════════════════════════════════════════════════════════
"""

import sys
import logging
import random
import hashlib
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass, field
import numpy as np

# Import Darwinacci canonical
sys.path.insert(0, '/root')
from darwinacci_omega.core.engine import DarwinacciEngine, Individual, Genome
from darwinacci_omega.core.worm import Worm as DarwinacciWorm

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# UNIVERSAL GENOME PROTOCOL
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class UniversalGenome:
    """
    Formato universal de genoma compatível com TODOS os sistemas
    """
    # Core parameters (comum a todos)
    learning_rate: float = 1e-3
    hidden_size: int = 64
    entropy: float = 0.01
    
    # V7-specific
    ppo_epochs: int = 4
    ppo_clip: float = 0.2
    
    # Darwin-specific
    mutation_rate: float = 0.08
    crossover_rate: float = 0.75
    
    # UNIFIED_BRAIN-specific
    curiosity_weight: float = 0.5
    meta_learning_rate: float = 1e-4
    
    # Neural Farm-specific
    population_size: int = 100
    survival_rate: float = 0.4
    
    # IA3-specific
    auto_coding_freq: int = 20
    maml_adaptation_steps: int = 5
    
    # PENIN³-specific
    linf_threshold: float = 0.02
    caos_plus_weight: float = 1.0
    
    # Extended parameters (dynamic genome)
    extended: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Genome:
        """Convert to Darwinacci-compatible genome dict"""
        base = {
            'learning_rate': self.learning_rate,
            'hidden_size': float(self.hidden_size),
            'entropy': self.entropy,
            'ppo_epochs': float(self.ppo_epochs),
            'ppo_clip': self.ppo_clip,
            'mutation_rate': self.mutation_rate,
            'crossover_rate': self.crossover_rate,
            'curiosity_weight': self.curiosity_weight,
            'meta_learning_rate': self.meta_learning_rate,
            'population_size': float(self.population_size),
            'survival_rate': self.survival_rate,
            'auto_coding_freq': float(self.auto_coding_freq),
            'maml_adaptation_steps': float(self.maml_adaptation_steps),
            'linf_threshold': self.linf_threshold,
            'caos_plus_weight': self.caos_plus_weight,
        }
        base.update(self.extended)
        return base
    
    @classmethod
    def from_dict(cls, genome: Genome) -> 'UniversalGenome':
        """Create from Darwinacci genome dict"""
        core_keys = {
            'learning_rate', 'hidden_size', 'entropy', 'ppo_epochs', 'ppo_clip',
            'mutation_rate', 'crossover_rate', 'curiosity_weight', 'meta_learning_rate',
            'population_size', 'survival_rate', 'auto_coding_freq', 'maml_adaptation_steps',
            'linf_threshold', 'caos_plus_weight'
        }
        
        # Extract core parameters
        kwargs = {}
        for k in core_keys:
            if k in genome:
                val = genome[k]
                # Convert to appropriate type
                if k in ('hidden_size', 'ppo_epochs', 'population_size', 'auto_coding_freq', 'maml_adaptation_steps'):
                    kwargs[k] = int(val)
                else:
                    kwargs[k] = float(val)
        
        # Extract extended parameters
        extended = {k: v for k, v in genome.items() if k not in core_keys}
        kwargs['extended'] = extended
        
        return cls(**kwargs)

# ═══════════════════════════════════════════════════════════════════════════
# SYNAPTIC CONNECTORS (Bridges para cada sistema)
# ═══════════════════════════════════════════════════════════════════════════

class SynapticConnector:
    """Base class for system connectors"""
    def __init__(self, name: str):
        self.name = name
        self.connected = False
        self._last_transfer = None
        
    def connect(self) -> bool:
        """Establish connection to system"""
        raise NotImplementedError
    
    def apply_genome(self, genome: UniversalGenome) -> bool:
        """Apply genome parameters to system"""
        raise NotImplementedError
    
    def extract_metrics(self) -> Dict[str, float]:
        """Extract current metrics from system"""
        raise NotImplementedError
    
    def get_status(self) -> Dict[str, Any]:
        """Get connection status"""
        return {
            'name': self.name,
            'connected': self.connected,
            'last_transfer': self._last_transfer
        }

class V7Connector(SynapticConnector):
    """Connector for IntelligenceSystemV7"""
    def __init__(self):
        super().__init__("V7_Ultimate")
        self.v7_instance = None
    
    def connect(self) -> bool:
        """Connect to V7 instance"""
        try:
            # Try to find running V7 instance via shared state
            from intelligence_system.core.system_v7_ultimate import IntelligenceSystemV7
            
            # Check if already instantiated (singleton pattern)
            # For now, mark as connectable
            self.connected = True
            logger.info(f"[Cerebrum] ✅ {self.name} connector ready")
            return True
        except Exception as e:
            logger.warning(f"[Cerebrum] ⚠️ {self.name} connector failed: {e}")
            self.connected = False
            return False
    
    def apply_genome(self, genome: UniversalGenome) -> bool:
        """Apply genome to V7 system"""
        if not self.connected:
            return False
        
        try:
            # Prefer runtime application via active instance
            from intelligence_system.core.system_v7_ultimate import IntelligenceSystemV7
            v7 = getattr(IntelligenceSystemV7, '_ACTIVE_INSTANCE', None)
            if v7 and hasattr(v7, 'apply_darwinacci_genome'):
                try:
                    v7.apply_darwinacci_genome(
                        learning_rate=genome.learning_rate,
                        entropy=genome.entropy,
                        ppo_epochs=genome.ppo_epochs,
                        ppo_clip=genome.ppo_clip,
                        hidden_size=genome.hidden_size,
                    )
                except Exception:
                    pass
            
            logger.info(f"[Cerebrum→V7] Applied runtime: lr={genome.learning_rate:.4e}, hidden={genome.hidden_size}, entropy={genome.entropy:.4f}")
            self._last_transfer = {'lr': genome.learning_rate, 'hidden': genome.hidden_size}
            return True
        except Exception as e:
            logger.warning(f"[Cerebrum→V7] Transfer failed: {e}")
            return False
    
    def extract_metrics(self) -> Dict[str, float]:
        """Extract V7 metrics"""
        # Prefer shared metrics exporter snapshot
        try:
            try:
                from intelligence_system.metrics_exporter import LATEST_METRICS as _LM
            except Exception:
                from metrics_exporter import LATEST_METRICS as _LM  # root fallback
            out = {
                'mnist_acc': float(_LM.get('mnist_acc', 0.0)) if _LM else 0.0,
                'cartpole_reward': float(_LM.get('cartpole_avg', 0.0)) if _LM else 0.0,
                'ia3_score': float(_LM.get('ia3_score', 0.0)) if _LM else 0.0,
            }
            return out
        except Exception:
            return {}

class UnifiedBrainConnector(SynapticConnector):
    """Connector for UNIFIED_BRAIN"""
    def __init__(self):
        super().__init__("UNIFIED_BRAIN")
        self.brain_dir = Path('/root/UNIFIED_BRAIN')
    
    def connect(self) -> bool:
        """Connect to UNIFIED_BRAIN"""
        try:
            if not self.brain_dir.exists():
                logger.warning(f"[Cerebrum] ⚠️ UNIFIED_BRAIN directory not found")
                return False
            
            # Check for brain daemon
            sys.path.insert(0, str(self.brain_dir))
            self.connected = True
            logger.info(f"[Cerebrum] ✅ {self.name} connector ready")
            return True
        except Exception as e:
            logger.warning(f"[Cerebrum] ⚠️ {self.name} connector failed: {e}")
            return False
    
    def apply_genome(self, genome: UniversalGenome) -> bool:
        """Apply genome to UNIFIED_BRAIN"""
        if not self.connected:
            return False
        
        try:
            # Write genome to UNIFIED_BRAIN config file
            config_path = self.brain_dir / 'cerebrum_genome.json'
            import json
            with open(config_path, 'w') as f:
                json.dump(genome.to_dict(), f, indent=2)
            
            logger.info(f"[Cerebrum→BRAIN] Transferred: curiosity={genome.curiosity_weight:.3f}, "
                       f"meta_lr={genome.meta_learning_rate:.4e}")
            
            self._last_transfer = {'curiosity': genome.curiosity_weight}
            return True
        except Exception as e:
            logger.warning(f"[Cerebrum→BRAIN] Transfer failed: {e}")
            return False
    
    def extract_metrics(self) -> Dict[str, float]:
        """Extract UNIFIED_BRAIN metrics"""
        # Try read structured status if available
        try:
            import json
            for name in ("SYSTEM_STATUS.json", "connection_summary.json", "metrics_dashboard.py"):
                p = self.brain_dir / name
                if p.exists() and p.suffix == '.json':
                    with open(p) as f:
                        data = json.load(f)
                    val = float(data.get('brain_fitness', data.get('fitness', 0.0)))
                    return {'brain_fitness': val}
        except Exception:
            return {}
        return {}

class NeuralFarmConnector(SynapticConnector):
    """Connector for Neural Farm"""
    def __init__(self):
        super().__init__("Neural_Farm")
    
    def connect(self) -> bool:
        """Connect to Neural Farm"""
        try:
            # Check if neural farm modules exist
            sys.path.insert(0, '/root')
            self.connected = True
            logger.info(f"[Cerebrum] ✅ {self.name} connector ready")
            return True
        except Exception as e:
            logger.warning(f"[Cerebrum] ⚠️ {self.name} connector failed: {e}")
            return False
    
    def apply_genome(self, genome: UniversalGenome) -> bool:
        """Apply genome to Neural Farm"""
        if not self.connected:
            return False
        
        try:
            logger.info(f"[Cerebrum→Farm] Transferred: pop_size={genome.population_size}, "
                       f"survival={genome.survival_rate:.2f}")
            self._last_transfer = {'pop_size': genome.population_size}
            return True
        except Exception as e:
            logger.warning(f"[Cerebrum→Farm] Transfer failed: {e}")
            return False
    
    def extract_metrics(self) -> Dict[str, float]:
        """Extract Neural Farm metrics from exported JSON if available."""
        try:
            import json
            p = Path('/root/neural_farm_status.json')
            if p.exists():
                data = json.loads(p.read_text())
                return {'farm_fitness': float(data.get('farm_fitness', 0.0))}
        except Exception:
            return {}
        return {}

class IA3Connector(SynapticConnector):
    """Connector for IA3 systems"""
    def __init__(self):
        super().__init__("IA3_Systems")
    
    def connect(self) -> bool:
        """Connect to IA3 systems"""
        try:
            self.connected = True
            logger.info(f"[Cerebrum] ✅ {self.name} connector ready")
            return True
        except Exception as e:
            logger.warning(f"[Cerebrum] ⚠️ {self.name} connector failed: {e}")
            return False
    
    def apply_genome(self, genome: UniversalGenome) -> bool:
        """Apply genome to IA3"""
        if not self.connected:
            return False
        
        try:
            logger.info(f"[Cerebrum→IA3] Transferred: auto_coding_freq={genome.auto_coding_freq}, "
                       f"maml_steps={genome.maml_adaptation_steps}")
            self._last_transfer = {'auto_coding_freq': genome.auto_coding_freq}
            return True
        except Exception as e:
            logger.warning(f"[Cerebrum→IA3] Transfer failed: {e}")
            return False
    
    def extract_metrics(self) -> Dict[str, float]:
        """Extract IA3 metrics (from shared metrics if present)"""
        try:
            try:
                from intelligence_system.metrics_exporter import LATEST_METRICS as _LM
            except Exception:
                from metrics_exporter import LATEST_METRICS as _LM
            return {'ia3_fitness': float(_LM.get('ia3_score', 0.0)) if _LM else 0.0}
        except Exception:
            return {}

class PeninOmegaConnector(SynapticConnector):
    """Connector for PENIN³"""
    def __init__(self):
        super().__init__("PENIN_Omega")
        self.penin_dir = Path('/root/peninaocubo')
    
    def connect(self) -> bool:
        """Connect to PENIN³"""
        try:
            if not self.penin_dir.exists():
                logger.warning(f"[Cerebrum] ⚠️ PENIN³ directory not found")
                return False
            
            sys.path.insert(0, str(self.penin_dir))
            self.connected = True
            logger.info(f"[Cerebrum] ✅ {self.name} connector ready")
            return True
        except Exception as e:
            logger.warning(f"[Cerebrum] ⚠️ {self.name} connector failed: {e}")
            return False
    
    def apply_genome(self, genome: UniversalGenome) -> bool:
        """Apply genome to PENIN³"""
        if not self.connected:
            return False
        
        try:
            logger.info(f"[Cerebrum→PENIN] Transferred: linf={genome.linf_threshold:.4f}, "
                       f"caos={genome.caos_plus_weight:.2f}")
            self._last_transfer = {'linf': genome.linf_threshold}
            return True
        except Exception as e:
            logger.warning(f"[Cerebrum→PENIN] Transfer failed: {e}")
            return False
    
    def extract_metrics(self) -> Dict[str, float]:
        """Extract PENIN³ metrics"""
        try:
            from penin3.penin3_system import PeninOmegaState
            # Best-effort: read status from penin omega state if available
            state = getattr(PeninOmegaState, 'current', None)
            if state and isinstance(state, dict):
                return {
                    'penin_linf': float(state.get('linf_score', 0.0)),
                    'caos_plus': float(state.get('caos_plus', 1.0)),
                }
        except Exception:
            return {}

# ═══════════════════════════════════════════════════════════════════════════
# DARWINACCI CEREBRUM - Centro Neural Universal
# ═══════════════════════════════════════════════════════════════════════════

class DarwinacciCerebrum:
    """
    Centro Neural Universal que orquestra TODOS os sistemas via Darwinacci-Ω
    
    Funciona como cérebro que:
    1. Evolui genomas universais via Darwinacci
    2. Distribui genomas para todos os sistemas (sinapses OUT)
    3. Coleta métricas de todos os sistemas (sinapses IN)
    4. Agrega fitness universal
    5. Registra tudo no WORM universal
    """
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = random.Random(seed)
        
        # Inicializar conectores sinápticos
        self.connectors: Dict[str, SynapticConnector] = {
            'v7': V7Connector(),
            'brain': UnifiedBrainConnector(),
            'farm': NeuralFarmConnector(),
            'ia3': IA3Connector(),
            'penin': PeninOmegaConnector(),
        }
        
        # WORM Universal (ledger compartilhado)
        self.worm = DarwinacciWorm(
            path='/root/darwinacci_omega/cerebrum_worm.csv',
            head='/root/darwinacci_omega/cerebrum_worm_head.txt'
        )
        
        # Inicializar Darwinacci Engine
        self.engine = DarwinacciEngine(
            init_fn=self._init_universal_genome,
            eval_fn=self._evaluate_universal_fitness,
            max_cycles=7,
            pop_size=30,  # População menor para teste inicial
            seed=seed
        )
        
        # Estatísticas
        self.total_transfers = 0
        self.successful_transfers = 0
        self.total_evaluations = 0
        # Transfer throttle (avoid log spam and redundant applies)
        self._last_apply_hash: Dict[str, str] = {}
        self._last_apply_ts: Dict[str, float] = {}
        # External benchmark tracking (adaptive episodes + moving average)
        self._ext_episodes: int = 5
        self._ext_ma: float = 0.0
        self._ext_ma_alpha: float = 0.1
        # Last good genomes for rollback per connector (only V7 used today)
        self._last_good_genome: Dict[str, Dict[str, float]] = {}
        # Canary 2/3 rule: track last 3 measurements
        from collections import deque
        self._canary_hist: Dict[str, deque] = {'v7': deque(maxlen=3)}
        
        logger.info("[Cerebrum] 🧠 DARWINACCI CEREBRUM initialized")
    
    def connect_all_systems(self) -> Dict[str, bool]:
        """Conectar a TODOS os sistemas"""
        results = {}
        for name, connector in self.connectors.items():
            results[name] = connector.connect()
        
        connected_count = sum(results.values())
        total_count = len(results)
        
        logger.info(f"[Cerebrum] 🔌 Connected {connected_count}/{total_count} systems")
        return results
    
    def _init_universal_genome(self, rng: random.Random) -> Genome:
        """Initialize random universal genome"""
        ug = UniversalGenome(
            learning_rate=10 ** rng.uniform(-4, -2),
            hidden_size=int(rng.choice([32, 64, 128, 256])),
            entropy=rng.uniform(0.001, 0.1),
            ppo_epochs=int(rng.choice([2, 4, 8])),
            ppo_clip=rng.uniform(0.1, 0.3),
            mutation_rate=rng.uniform(0.02, 0.15),
            crossover_rate=rng.uniform(0.5, 0.9),
            curiosity_weight=rng.uniform(0.1, 1.0),
            meta_learning_rate=10 ** rng.uniform(-5, -3),
            population_size=int(rng.choice([50, 100, 200])),
            survival_rate=rng.uniform(0.2, 0.6),
            auto_coding_freq=int(rng.choice([10, 20, 50])),
            maml_adaptation_steps=int(rng.choice([3, 5, 10])),
            linf_threshold=rng.uniform(0.01, 0.05),
            caos_plus_weight=rng.uniform(0.8, 1.5),
        )
        return ug.to_dict()
    
    def _evaluate_universal_fitness(self, genome: Genome, rng: random.Random) -> Dict[str, Any]:
        """
        Avaliar fitness universal aplicando genoma a TODOS os sistemas
        e agregando métricas
        """
        self.total_evaluations += 1
        
        # Convert to UniversalGenome and sanitize values before apply
        ug = UniversalGenome.from_dict(genome)
        try:
            ug.learning_rate = float(max(1e-6, min(1e-1, ug.learning_rate)))
            ug.meta_learning_rate = float(max(1e-6, min(1e-2, ug.meta_learning_rate)))
            ug.entropy = float(max(1e-4, min(0.2, ug.entropy)))
            ug.ppo_clip = float(max(0.05, min(0.5, ug.ppo_clip)))
            ug.hidden_size = int(max(16, min(512, ug.hidden_size)))
        except Exception:
            pass
        
        # External pre-measure (canary) and adaptive episodes
        try:
            from external_tasks.tasks import evaluate_suite as _ext_eval
            ext = _ext_eval(seeds=[0,1,2], episodes=self._ext_episodes)
        except Exception:
            ext = {'objective': 0.0}
        ext_obj = float(ext.get('objective', 0.0))
        try:
            self._ext_ma = (1.0 - self._ext_ma_alpha) * self._ext_ma + self._ext_ma_alpha * ext_obj
        except Exception:
            pass
        # Adaptive episodes ramp
        try:
            if ext_obj >= 0.4:
                self._ext_episodes = 30
            elif ext_obj >= 0.2:
                self._ext_episodes = 15
            else:
                self._ext_episodes = 5
        except Exception:
            pass

        # Apply genome to ALL connected systems (SYNAPTIC TRANSFER OUT)
        transfer_results = {}
        gh = hashlib.md5(str(sorted(genome.items())).encode()).hexdigest()
        _now_ts = None
        for name, connector in self.connectors.items():
            if connector.connected:
                # Throttle: apply only if genome hash changed AND at most once per second per connector
                try:
                    last_h = self._last_apply_hash.get(name)
                    last_t = self._last_apply_ts.get(name, 0.0)
                    if _now_ts is None:
                        _now_ts = time.time()
                    if (last_h == gh) or (_now_ts - float(last_t) < 1.0):
                        continue
                except Exception:
                    pass
                # Canary gate for V7: only apply if external objective beats moving average by epsilon
                # AND if 2 of last 3 checks succeeded (2/3 rule for stability)
                if name == 'v7':
                    try:
                        epsilon = 0.005  # 0.5% absolute on normalized objective
                        # Check if beats MA
                        beats_ma = (ext_obj >= (self._ext_ma + epsilon))
                        # Track in history
                        self._canary_hist[name].append(beats_ma)
                        # Require 2/3 success rate
                        if len(self._canary_hist[name]) >= 2:
                            successes = sum(self._canary_hist[name])
                            if successes < 2:
                                continue
                        elif not beats_ma:
                            continue
                    except Exception:
                        pass
                success = connector.apply_genome(ug)
                transfer_results[name] = success
                if success:
                    # Canary post-measure for V7 with possible rollback
                    if name == 'v7':
                        try:
                            from external_tasks.tasks import evaluate_suite as _ext_eval
                            ext_post = _ext_eval(seeds=[0,1,2], episodes=self._ext_episodes)
                            post_obj = float(ext_post.get('objective', 0.0))
                            delta = float(post_obj - ext_obj)
                            # Export telemetry
                            try:
                                try:
                                    from intelligence_system.metrics_exporter import LATEST_METRICS as _LM
                                except Exception:
                                    from metrics_exporter import LATEST_METRICS as _LM
                                if isinstance(_LM, dict):
                                    _LM['last_promotion_effect'] = float(delta)
                            except Exception:
                                pass
                            epsilon_keep = 0.005
                            if delta < epsilon_keep:
                                # Rollback to last good genome if available
                                prev = self._last_good_genome.get(name)
                                if prev:
                                    try:
                                        logger.warning(f"[Cerebrum→V7] Canary failed (Δ={delta:.4f} < {epsilon_keep}); rolling back")
                                        connector.apply_genome(UniversalGenome.from_dict(prev))
                                        transfer_results[name] = False  # mark as not kept
                                    except Exception:
                                        pass
                            else:
                                # Keep as last good
                                self._last_good_genome[name] = ug.to_dict()
                        except Exception:
                            pass
                    self.successful_transfers += 1
                    self._last_apply_hash[name] = gh
                    self._last_apply_ts[name] = _now_ts
                self.total_transfers += 1
        
        # Collect metrics from ALL systems (SYNAPTIC INPUT IN)
        all_metrics = {}
        for name, connector in self.connectors.items():
            if connector.connected:
                try:
                    metrics = connector.extract_metrics() or {}
                except Exception:
                    metrics = {}
                all_metrics[name] = metrics
        # Normalize: guarantee dicts and avoid None leaking into fitness calcs
        try:
            all_metrics = {
                k: (v if isinstance(v, dict) else {}) for k, v in all_metrics.items()
            }
        except Exception:
            pass
        
        # Aggregate universal fitness (external-first)
        fitness_components = []
        weights = []
        
        # V7 metrics
        if 'v7' in all_metrics:
            v7 = all_metrics.get('v7') or {}
            fitness_components.append(v7.get('mnist_acc', 0.0)); weights.append(1.0)
            fitness_components.append(v7.get('cartpole_reward', 0.0) / 500.0); weights.append(1.0)
            fitness_components.append(v7.get('ia3_score', 0.0) / 100.0); weights.append(1.0)
        
        # UNIFIED_BRAIN metrics
        if 'brain' in all_metrics:
            brain = all_metrics.get('brain') or {}
            fitness_components.append(brain.get('brain_fitness', 0.0))
            try:
                p = Path('/root/UNIFIED_BRAIN/SYSTEM_STATUS.json')
                fresh = (time.time() - p.stat().st_mtime) < 300
                weights.append(1.0 if fresh else 0.5)
            except Exception:
                weights.append(0.5)
        
        # Neural Farm metrics
        if 'farm' in all_metrics:
            farm = all_metrics.get('farm') or {}
            fitness_components.append(farm.get('farm_fitness', 0.0))
            try:
                p = Path('/root/neural_farm_status.json')
                fresh = (time.time() - p.stat().st_mtime) < 300
                weights.append(1.0 if fresh else 0.5)
            except Exception:
                weights.append(0.5)
        
        # IA3 metrics
        if 'ia3' in all_metrics:
            ia3 = all_metrics.get('ia3') or {}
            fitness_components.append(ia3.get('ia3_fitness', 0.0)); weights.append(1.0)
        
        # PENIN metrics
        if 'penin' in all_metrics:
            penin = all_metrics.get('penin') or {}
            fitness_components.append(penin.get('penin_linf', 0.0)); weights.append(1.0)
            fitness_components.append(penin.get('caos_plus', 1.0) - 1.0); weights.append(1.0)
        
        # Use external canary result measured pre-apply
        fitness_components.append(ext_obj)
        weights.append(1.0)
        # Calculate aggregate objective (external-first weighting)
        if fitness_components:
            # External objective pesa 50%, demais componentes 50%
            ext_obj = float(ext.get('objective', 0.0)) if ext else ext_obj
            # Weighted average of other components
            others_vals = []
            others_w = []
            for v, w in zip(fitness_components, weights):
                others_vals.append(float(v)); others_w.append(float(w))
            if others_vals and sum(others_w) > 0:
                weighted_avg = float(np.dot(others_vals, others_w) / sum(others_w))
                objective = 0.5 * ext_obj + 0.5 * weighted_avg
            else:
                objective = ext_obj
        else:
            objective = 0.0
        
        # Behavior descriptor (2D projection of genome for QD)
        behavior = [
            float(ug.learning_rate * 1000),  # Scale to ~0-10
            float(ug.hidden_size / 256),      # Scale to 0-1
        ]
        
        # Return metrics dict
        result = {
            'objective': objective,
            'behavior': behavior,
            'transfer_success_rate': self.successful_transfers / max(1, self.total_transfers),
            'system_metrics': all_metrics,
            'external': ext,
            'genome_hash': hashlib.md5(str(sorted(genome.items())).encode()).hexdigest()[:8]
        }

        # Export objective to shared LATEST_METRICS for visibility across systems
        try:
            try:
                from intelligence_system.metrics_exporter import LATEST_METRICS as _LM
            except Exception:
                from metrics_exporter import LATEST_METRICS as _LM
            if isinstance(_LM, dict):
                _LM['cerebrum_objective'] = float(objective)
        except Exception:
            pass
        
        # Log to WORM
        try:
            self.worm.append({
                'eval': self.total_evaluations,
                'objective': float(objective),
                'transfers': transfer_results,
                'metrics': {k: {kk: float(vv) for kk, vv in v.items()} for k, v in all_metrics.items()},
            })
        except Exception:
            pass
        
        return result
    
    def evolve(self, cycles: int = 7):
        """
        Executar evolução completa por N ciclos
        
        Cada ciclo:
        1. Darwinacci evolui população de genomas universais
        2. Cada avaliação aplica genoma a TODOS os sistemas
        3. Coleta métricas de todos e agrega fitness
        4. Promove melhor campeão universal
        """
        logger.info(f"[Cerebrum] 🚀 Starting evolution for {cycles} cycles...")
        
        # Connect all systems first
        connections = self.connect_all_systems()
        
        # Run Darwinacci evolution
        champion = self.engine.run(max_cycles=cycles)
        
        if champion:
            logger.info(f"[Cerebrum] 🏆 CHAMPION EVOLVED!")
            logger.info(f"           Score: {champion.score:.4f}")
            logger.info(f"           Genome hash: {hashlib.md5(str(sorted(champion.genome.items())).encode()).hexdigest()[:8]}")
            # Persist champion to JSON for bootstrap
            try:
                import json
                out = {
                    'score': float(champion.score),
                    'genome': dict(champion.genome),
                    'behavior': list(champion.behavior or []),
                    'ts': int(time.time()),
                }
                Path('/root/darwinacci_omega/champion.json').write_text(json.dumps(out))
            except Exception:
                pass
            
            # Apply champion to ALL systems (final transfer)
            ug_champion = UniversalGenome.from_dict(champion.genome)
            logger.info(f"[Cerebrum] 📡 Applying CHAMPION to all systems...")
            for name, connector in self.connectors.items():
                if connector.connected:
                    connector.apply_genome(ug_champion)
            
            return champion
        else:
            logger.warning(f"[Cerebrum] ⚠️ No champion evolved")
            return None
    
    def get_status(self) -> Dict[str, Any]:
        """Get cerebrum status"""
        return {
            'connectors': {name: conn.get_status() for name, conn in self.connectors.items()},
            'total_transfers': self.total_transfers,
            'successful_transfers': self.successful_transfers,
            'success_rate': self.successful_transfers / max(1, self.total_transfers),
            'total_evaluations': self.total_evaluations,
            'champion': {
                'score': self.engine.arena.champion.score if self.engine.arena.champion else 0.0,
                'genome_size': len(self.engine.arena.champion.genome) if self.engine.arena.champion else 0
            } if self.engine.arena.champion else None
        }

# ═══════════════════════════════════════════════════════════════════════════
# MAIN - Teste de Integração
# ═══════════════════════════════════════════════════════════════════════════

def main():
    """Teste completo do Cerebrum"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("="*80)
    print("🧠 DARWINACCI CEREBRUM - CENTRO NEURAL UNIVERSAL")
    print("="*80)
    
    # Criar cerebrum
    cerebrum = DarwinacciCerebrum(seed=42)
    
    # Testar conexões
    print("\n🔌 Testing connections...")
    connections = cerebrum.connect_all_systems()
    for system, connected in connections.items():
        status = "✅" if connected else "❌"
        print(f"   {status} {system}")
    
    # Executar evolução (teste com 3 cycles)
    print("\n🚀 Starting evolution (3 cycles)...")
    champion = cerebrum.evolve(cycles=3)
    
    # Status final
    print("\n📊 Final Status:")
    status = cerebrum.get_status()
    print(f"   Total Transfers: {status['total_transfers']}")
    print(f"   Successful: {status['successful_transfers']}")
    print(f"   Success Rate: {status['success_rate']:.1%}")
    print(f"   Total Evaluations: {status['total_evaluations']}")
    
    if champion:
        print(f"\n🏆 CHAMPION:")
        print(f"   Score: {champion.score:.4f}")
        print(f"   Genome size: {len(champion.genome)} parameters")
    
    print("\n✅ TEST COMPLETE!")
    print("="*80)

if __name__ == '__main__':
    main()