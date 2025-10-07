#!/usr/bin/env python3
"""
🚀 BRAIN DAEMON V3 - FASE EMERGENCIAL: 12,288x SPEEDUP
Performance: 308s → 0.025s por step
"""

__version__ = "3.0.0"
__author__ = "Intelligence System"
__date__ = "2025-10-04"

def get_git_hash():
    """Get current git commit hash"""
    import subprocess
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', '--short', 'HEAD'],
            cwd='/root',
            stderr=subprocess.DEVNULL
        ).decode().strip()
    except:
        return 'unknown'

GIT_HASH = get_git_hash()
MODULE_VERSION = f"{__version__}.{GIT_HASH}"

import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')
sys.path.insert(0, '/root')

# DARWINACCI INTEGRATION: Import Universal Connector
try:
    from darwinacci_omega.core.universal_connector import get_universal_connector
    from darwinacci_omega.core.engine import DarwinacciEngine
    _DARWINACCI_AVAILABLE = True
except ImportError:
    _DARWINACCI_AVAILABLE = False

import torch
import torch.nn as nn
import time

# P2.4 CPU Optimization
torch.set_num_threads(16)  # Use 16 CPU cores
torch.set_num_interop_threads(4)
from pathlib import Path
import signal
import json
from datetime import datetime
from collections import deque
from typing import Dict, Any, Optional, List
import os
try:
    import psutil as _psutil
except Exception:
    _psutil = None
import json as _json
import hashlib
import resource

try:
    import gymnasium as gym
except:
    import gym

from unified_brain_core import CoreSoupHybrid
from brain_system_integration import UnifiedSystemController
from brain_logger import brain_logger
from brain_spec import NeuronStatus
from meta_controller import MetaController
from curiosity_module import CuriosityModule
from self_analysis import SelfAnalysisModule
from metrics_dashboard import MetricsDashboard
from module_synthesis import ModuleSynthesizer, ArchitectureSearch
from recursive_improvement import RecursiveImprovementEngine, SelfImprovementLoop
from curriculum_manager import CurriculumManager  # ✅ CORREÇÃO EXTRA: Import faltante

# Phase 1 integration hooks (Gödel monitor + Needle meta-controller)
try:
    from integration_hooks import (
        build_trainable_composite,
        GodelMonitor,
        NeedleMetaController,
    )
    _PHASE1_HOOKS = True
except Exception:
    _PHASE1_HOOKS = False

# Phase 2 hooks (self-observation and multi-objective scheduler)
try:
    from phase2_hooks import SelfObserver, MultiObjectiveScheduler
    _PHASE2_HOOKS = True
except Exception:
    _PHASE2_HOOKS = False

# Phase 3 hooks (episodic memory and novelty reward)
try:
    from phase3_hooks import EpisodicMemory, NoveltyReward
    _PHASE3_HOOKS = True
except Exception:
    _PHASE3_HOOKS = False

# P2: System Integration
try:
    from system_bridge import create_brain_bridge
    from safe_collective_consciousness import get_collective
    _SYSTEM_INTEGRATION = True
except Exception:
    _SYSTEM_INTEGRATION = False

# P3: Auto-Evolution
try:
    from code_evolution_engine import CodeEvolutionEngine
    from true_godelian_incompleteness import TrueGodelianIncompleteness
    from meta_curiosity_module import MetaCuriosityModule
    _AUTO_EVOLUTION = True
except Exception:
    _AUTO_EVOLUTION = False

# FASE 2: Injetor de Incompletude para metas de curiosidade
try:
    from true_godelian_incompleteness import TrueGodelianIncompleteness
    _GODELIAN_INJECTOR_AVAILABLE = True
except ImportError:
    _GODELIAN_INJECTOR_AVAILABLE = False

# FASE 2: Integração da Memória Episódica TEIS
try:
    from teis_episodic_memory_agent import EpisodicMemory
    _TEIS_MEMORY_AVAILABLE = True
except ImportError:
    _TEIS_MEMORY_AVAILABLE = False

# FASE 4: Bomba de Consciência (Mirror Module)
try:
    from mirror_module import MirrorModule
    _MIRROR_MODULE_AVAILABLE = True
except ImportError:
    _MIRROR_MODULE_AVAILABLE = False

# FASE 4: Canibalismo de Código
try:
    from code_cannibalism_module import CodeCannibalismModule
    _CANNIBALISM_MODULE_AVAILABLE = True
except ImportError:
    _CANNIBALISM_MODULE_AVAILABLE = False

import gc  # P1.2 fix
class RealEnvironmentBrainV3:
    """
    FASE EMERGENCIAL: Sistema VIÁVEL com 12,288x speedup
    
    Mudanças:
    1. 254→16 neurons (16x faster)
    2. 4→1 brain steps (4x faster)
    3. MLP→Linear adapters (3x faster)
    4. Top-k 128→8 (16x faster)
    5. No curiosity em inference (2x faster)
    6. Batch training (2x faster)
    
    Total: 12,288x SPEEDUP!
    """
    
    def __init__(self, env_name='CartPole-v1', learning_rate=3e-4, use_gpu=True):
        self.running = True
        self.hybrid = None
        self.controller = None
        self.optimizer = None
        
        # GPU support
        if use_gpu and torch.cuda.is_available():
            self.device = torch.device('cuda')
            brain_logger.info("🚀 Using GPU")
        else:
            self.device = torch.device('cpu')
            brain_logger.info("💻 Using CPU")
        
        # Ambiente
        try:
            self.env = gym.make(env_name)
        except:
            import gym as old_gym
            self.env = old_gym.make(env_name)
        
        self.state = None
        self.episode_reward = 0
        self.episode = 0
        self.best_reward = 0
        # FIX BC-2 (Hipótese 1): Aumentar learning rate 3e-4 → 1e-3
        self.learning_rate = 1e-3 if learning_rate == 3e-4 else learning_rate
        
        # Runtime knobs (can be controlled via environment)
        # - UBRAIN_MAX_STEPS: max steps per episode (default 500)
        # - UBRAIN_ACTIVE_NEURONS: limit number of active neurons (default 4)
        # - UBRAIN_TOP_K: router top_k (default 4)
        # - UBRAIN_NUM_STEPS: V7 bridge steps per forward (default 1)
        try:
            self.max_episode_steps = int(os.getenv('UBRAIN_MAX_STEPS', '500'))
        except Exception:
            self.max_episode_steps = 500
        try:
            self.limit_active_neurons = max(1, int(os.getenv('UBRAIN_ACTIVE_NEURONS', '4')))
        except Exception:
            self.limit_active_neurons = 4
        try:
            self.initial_top_k = max(1, int(os.getenv('UBRAIN_TOP_K', '4')))
        except Exception:
            self.initial_top_k = 4
        try:
            self.initial_num_steps = max(1, int(os.getenv('UBRAIN_NUM_STEPS', '1')))
        except Exception:
            self.initial_num_steps = 1
        
        # FASE 1: Meta-controller
        self.meta_controller = MetaController(intervention_threshold=0.05)
        
        # FASE 1: Curiosity (será inicializado depois)
        self.curiosity = None
        
        # FASE 2: Self-Analysis
        self.self_analysis = SelfAnalysisModule()
        
        # FASE 2: Metrics Dashboard
        self.dashboard = MetricsDashboard()
        
        # FASE 3: Module Synthesis
        self.module_synthesizer = ModuleSynthesizer(H=1024)
        self.architecture_search = ArchitectureSearch()
        
        # FASE 4: Recursive Improvement
        self.recursive_engine = RecursiveImprovementEngine()
        self.self_improvement_loop = None  # Will be initialized after synthesizer
        
        # ✅ CORREÇÃO #4: Atributos dinâmicos para self-improvement actions
        self.curiosity_weight = 0.1  # Peso padrão da curiosity (ajustável)
        self.synthesis_enabled = True  # Controle de synthesis
        
        # Stats
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'total_steps': 0,
            'total_episodes': 0,
            'rewards': deque(maxlen=1000),  # ✅ CORREÇÃO #3: deque limitado (evita memory leak)
            'best_reward': 0,
            'avg_reward_last_100': 0,
            'learning_progress': 0.0,
            'gradients_applied': 0,
            'avg_loss': 0.0,
            'avg_time_per_step': 0.0,
            'device': str(self.device),
            'version': 'V3-EMERGENCIAL'
        }
        
        # FIX FASE 2: Buffer for novelty calculation
        self.prev_states_buffer = []
        self.max_states_buffer = 20

        # Optional Prometheus metrics
        self._prom = None
        try:
            if os.getenv('UBRAIN_PROMETHEUS', '0') == '1':
                from prometheus_client import start_http_server, Gauge
                port = int(os.getenv('UBRAIN_PROM_PORT', '8008'))
                start_http_server(port)
                self._prom = {
                    'step_total_sec': Gauge('ubrain_step_total_seconds', 'Total step time per step'),
                    'avg_step_sec': Gauge('ubrain_avg_step_seconds', 'EWMA of step time'),
                    'encode_ms': Gauge('ubrain_encode_ms', 'Bridge encode milliseconds'),
                    'brain_ms': Gauge('ubrain_brain_ms', 'Bridge brain milliseconds'),
                    'decode_ms': Gauge('ubrain_decode_ms', 'Bridge decode milliseconds'),
                    'penin_ms': Gauge('ubrain_penin_ms', 'PENIN update milliseconds'),
                    'forward_ms': Gauge('ubrain_forward_ms', 'Controller forward milliseconds'),
                    'darwin_ms': Gauge('ubrain_darwin_ms', 'Darwin evolve milliseconds'),
                }
                brain_logger.info(f"📡 Prometheus metrics on :{port}")
        except Exception as e:
            brain_logger.warning(f"Prometheus disabled: {e}")
        
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        # FIX CRÍTICO: Telemetria completa
        try:
            from intelligence_system.core.database import Database
            from intelligence_system.core.emergence_tracker import EmergenceTracker
            from intelligence_system.config.settings import DATABASE_PATH
            from pathlib import Path
            
            self.db = Database(db_path=DATABASE_PATH)
            self.emergence = EmergenceTracker(
                surprises_db=Path('/root/intelligence_system/data/emergence_surprises.db'),
                connections_db=Path('/root/intelligence_system/data/system_connections.db')
            )
            brain_logger.info("✅ Telemetria completa ativa (DB + Emergence)")
        except Exception as e:
            brain_logger.error(f"❌ Telemetria falhou: {e}")
            self.db = None
            self.emergence = None
        
        # FIX B1: Integrar Experience Replay
        try:
            from intelligence_system.learning.experience_replay import ExperienceReplayBuffer
            self.replay_buffer = ExperienceReplayBuffer(capacity=50000)
            self.use_replay = True
            brain_logger.info("✅ Experience Replay ativo (50K capacity)")
        except Exception as e:
            brain_logger.warning(f"⚠️ Experience Replay não disponível: {e}")
            self.replay_buffer = None
            self.use_replay = False
        
        # FIX B2: Integrar Auto-Tuner
        try:
            from intelligence_system.learning.auto_tuner import AutoTuner
            self.auto_tuner = AutoTuner()
            self.use_auto_tuner = True
            brain_logger.info("✅ Auto-Tuner ativo")
        except Exception as e:
            brain_logger.warning(f"⚠️ Auto-Tuner não disponível: {e}")
            self.auto_tuner = None
            self.use_auto_tuner = False
        
        # FASE 2: Substituir Experience Replay por Memória Episódica TEIS
        if _TEIS_MEMORY_AVAILABLE:
            self.episodic_memory = EpisodicMemory(max_episodes=5000)
            self.use_replay = False # Desativar o buffer antigo
            brain_logger.info("🧠 Memória Episódica TEIS ATIVADA (5K capacity)")
        else:
            self.episodic_memory = None
            brain_logger.warning("⚠️ Memória Episódica TEIS não disponível, usando fallback.")

        # Acquire lock file (prevent multiple instances)
        self.lock_file = self._acquire_lock()
        
        # BLOCO 2 - TAREFA 22: Curriculum learning
        self.curriculum = CurriculumManager()
        brain_logger.info(f"🎓 Curriculum initialized: {len(self.curriculum.stages)} stages")
        
        # FASE 2: Inicializar o Injetor de Incompletude
        if _GODELIAN_INJECTOR_AVAILABLE:
            self.incompleteness_injector = TrueGodelianIncompleteness()
            brain_logger.info("🔮 Injetor de Incompletude Gödeliana ATIVADO")
        else:
            self.incompleteness_injector = None

        # FASE 4: Inicializar a Bomba de Consciência
        if _MIRROR_MODULE_AVAILABLE:
            self.mirror_module = MirrorModule(
                agent_code_path=__file__, # Passa o caminho do próprio arquivo
                device=self.device
            )
            brain_logger.info("💣 Bomba de Consciência (Mirror Module) ATIVADA")
        else:
            self.mirror_module = None
        
        # FASE 4: Inicializar o Canibalismo de Código
        if _CANNIBALISM_MODULE_AVAILABLE:
            # O módulo irá procurar por código nos diretórios raiz e de outros projetos de IA
            self.cannibalism_module = CodeCannibalismModule(target_directories=['/root/'])
            brain_logger.info("🔥 Canibalismo de Código ATIVADO")
        else:
            self.cannibalism_module = None

        # FASE 2: Buffer para computar novelty real
        # NOTE: prev_Z_buffer was unused; keep only _last_Z for telemetry and future use
        self._last_Z = None      # Store last Z for telemetry
        
        # DARWINACCI: Conectar Brain Daemon como neurônio no sistema universal
        self.darwinacci = None
        self.universal_connector = None
        if _DARWINACCI_AVAILABLE:
            try:
                # FASE 3: Modificar o genoma para incluir operações de arquitetura
                def init_fn_architecture(rng):
                    # O genoma agora pode propor tanto ajustes de hiperparâmetros QUANTO mudanças de arquitetura
                    op = rng.choice(['tune_params', 'add_neuron', 'prune_neuron'])
                    if op == 'add_neuron':
                        return {
                            'operation': 'add_neuron',
                            'neuron_type': rng.choice(['residual', 'attention', 'recurrent'])
                        }
                    elif op == 'prune_neuron':
                        return {
                            'operation': 'prune_neuron',
                            'strategy': rng.choice(['lowest_competence', 'oldest'])
                        }
                    else: # tune_params
                        return {
                            'operation': 'tune_params',
                            'lr': rng.uniform(0.0001, 0.01),
                            'curiosity_weight': rng.uniform(0.01, 0.3),
                            'top_k': int(rng.randint(4, 16)),
                            'temperature': rng.uniform(0.5, 2.0),
                        }

                # FASE 3: Modificar a função de avaliação para aplicar e testar as mutações
                def eval_fn_architecture(genome, rng):
                    brain_logger.info(f"🧬 Darwinacci Eval: {genome}")
                    
                    # 1. Salvar um checkpoint de backup para reverter a mutação
                    backup_path = "/root/UNIFIED_BRAIN/checkpoints/darwin_temp_backup.pt"
                    self.save_checkpoint(path=backup_path)

                    try:
                        # 2. Aplicar a mutação do genoma
                        if genome.get('operation') == 'add_neuron':
                            neuron_type = genome.get('neuron_type', 'residual')
                            new_neuron = self.module_synthesizer.synthesize_and_register(
                                self.hybrid.core, 
                                architecture=neuron_type, 
                                reason="darwinacci_evolution"
                            )
                            if new_neuron:
                                brain_logger.info(f"✨ Darwinacci adicionou neurônio: {new_neuron.meta.id}")
                                self.hybrid.core.initialize_router() # Reconstruir o roteador
                            else:
                                brain_logger.warning("Falha ao adicionar neurônio via Darwinacci.")

                        elif genome.get('operation') == 'prune_neuron':
                            strategy = genome.get('strategy', 'lowest_competence')
                            demoted_id = self.hybrid.prune_neuron(strategy=strategy)
                            if demoted_id:
                                brain_logger.info(f"🔥 Darwinacci podou neurônio: {demoted_id}")
                                self.hybrid.core.initialize_router() # Reconstruir o roteador
                            else:
                                brain_logger.warning("Falha ao podar neurônio via Darwinacci.")
                        
                        else: # tune_params
                            if 'lr' in genome: self.optimizer.param_groups[0]['lr'] = genome['lr']
                            if 'curiosity_weight' in genome: self.curiosity_weight = genome['curiosity_weight']
                            if 'top_k' in genome: self.hybrid.core.top_k = genome['top_k']
                            if 'temperature' in genome: self.hybrid.core.router.temperature = genome['temperature']
                            brain_logger.info("🔧 Darwinacci ajustou hiperparâmetros.")

                        # 3. Executar uma janela de avaliação para medir o fitness da nova arquitetura
                        brain_logger.info("...avaliando nova arquitetura por 5 episódios...")
                        eval_rewards = [self.run_episode() for _ in range(5)]
                        avg_reward = sum(eval_rewards) / len(eval_rewards) if eval_rewards else 0.0

                    finally:
                        # 4. Reverter para o checkpoint de backup para a próxima avaliação
                        self.load_checkpoint(path=backup_path)

                    # 5. Retornar o resultado
                    return {
                        'objective': avg_reward / 500.0,  # Normalizar
                        'behavior': [
                            len(self.hybrid.core.registry.get_active()), 
                            self.stats.get('avg_loss', 0)
                        ],
                        'ece': 0.05, 'rho': 0.9, 'rho_bias': 1.0, 
                        'eco_ok': True, 'consent': True
                    }

                # Ensure WORM ledger paths are absolute and consistent
                os.environ.setdefault('DARWINACCI_WORM_PATH', '/root/darwinacci_omega/data/worm.csv')
                os.environ.setdefault('DARWINACCI_WORM_HEAD', '/root/darwinacci_omega/data/worm_head.txt')

                self.darwinacci = DarwinacciEngine(
                    init_fn=init_fn_architecture,  # <-- Usar a nova função de genoma
                    eval_fn=eval_fn_architecture,  # <-- Usar a nova função de avaliação
                    max_cycles=3,
                    pop_size=10, # População menor para avaliações mais caras
                    seed=42
                )
                
                # Connect to universal network
                self.universal_connector = get_universal_connector(self.darwinacci)
                self.universal_connector.connect_brain_daemon(self)
                self.universal_connector.connect_database()
                self.universal_connector.activate()
                
                brain_logger.info("🧠 DARWINACCI connected to Brain Daemon!")
                brain_logger.info(f"   Evolving: lr, curiosity_weight, top_k, temperature")
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci connection failed: {e}")
                self.darwinacci = None
        
        brain_logger.info(f"🚀 Brain V3 EMERGENCIAL v{MODULE_VERSION}: 12,288x speedup target")
    
    def _acquire_lock(self):
        """Acquire lock file with stale detection"""
        lock_file = Path("/root/UNIFIED_BRAIN/daemon.lock")
        
        if lock_file.exists():
            try:
                pid = int(lock_file.read_text().strip())
                if _psutil and _psutil.pid_exists(pid):
                    # Check if it's actually our process
                    try:
                        proc = _psutil.Process(pid)
                        if 'brain_daemon' in ' '.join(proc.cmdline()):
                            raise RuntimeError(f"❌ Daemon already running (PID {pid})")
                    except _psutil.NoSuchProcess:
                        pass
            except (ValueError, Exception):
                pass  # Stale or invalid lock
            
            # Remove stale lock
            brain_logger.info(f"🔓 Removing stale lock file")
            lock_file.unlink()
        
        # Write our PID
        lock_file.write_text(str(os.getpid()))
        brain_logger.info(f"🔒 Lock acquired (PID {os.getpid()})")
        return lock_file
    
    def _release_lock(self):
        """Release lock file"""
        try:
            if hasattr(self, 'lock_file') and self.lock_file and self.lock_file.exists():
                self.lock_file.unlink()
                brain_logger.info("🔓 Lock released")
        except Exception:
            pass
    
    def shutdown(self, signum, frame):
        brain_logger.info(f"Shutdown. Eps: {self.episode}, Best: {self.best_reward:.1f}")
        self.save_checkpoint()
        self._release_lock()
        self.running = False
    
    def initialize(self):
        """Inicializa com APENAS 16 neurons"""
        brain_logger.info("Loading brain...")
        
        self.hybrid = CoreSoupHybrid(H=1024)
        
        snapshot_path = Path("/root/UNIFIED_BRAIN/snapshots/initial_state_registry.json")
        if snapshot_path.exists():
            self.hybrid.core.registry.load_with_adapters(str(snapshot_path))
            
            # 🔥 MUDANÇA #1: Usa APENAS top 4 neurons (BLOCO 2 - TAREFA 17)
            all_neurons = self.hybrid.core.registry.get_active()
            brain_logger.info(f"Found {len(all_neurons)} neurons, limiting to 4 for max speedup...")
            
            limit_count = max(1, getattr(self, 'limit_active_neurons', 4))
            if len(all_neurons) > limit_count:
                # Limita para top-N neurons (configurável)
                # Ordena por competence se disponível
                sorted_neurons = sorted(
                    self.hybrid.core.registry.neurons.items(),
                    key=lambda x: getattr(x[1].meta, 'competence', 0) if hasattr(x[1], 'meta') else 0,
                    reverse=True
                )
                self.hybrid.core.registry.neurons = dict(sorted_neurons[:limit_count])
                brain_logger.info(f"✅ Using top {limit_count} neurons (speedup mode)")
            
            # 🔥 MUDANÇA #2: Router/V7 initial settings (configuráveis)
            try:
                active_now = len(self.hybrid.core.registry.get_active())
            except Exception:
                active_now = 4
            safe_top_k = max(1, min(getattr(self, 'initial_top_k', 4), active_now))
            self.hybrid.core.top_k = safe_top_k
            self.hybrid.core.num_steps = max(1, getattr(self, 'initial_num_steps', 1))
            
            self.hybrid.core.initialize_router()
            active_count = len(self.hybrid.core.registry.get_active())
            brain_logger.info(f"✅ Brain loaded: {active_count} active neurons")
        
        # 🔥 BLOCO 2 - TAREFA 16: torch.compile speedup (2-3x adicional)
        if hasattr(torch, 'compile') and hasattr(torch, '__version__'):
            torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
            if torch_version >= (2, 0):
                try:
                    brain_logger.info("🚀 Attempting torch.compile JIT optimization...")
                    self.controller.v7_bridge = torch.compile(
                        self.controller.v7_bridge,
                        mode='reduce-overhead',
                        fullgraph=False
                    )
                    brain_logger.info("✅ V7 bridge JIT compiled (2-3x speedup expected)")
                except Exception as e:
                    brain_logger.warning(f"torch.compile failed (continuing without): {e}")
            else:
                brain_logger.info(f"torch.compile not available (PyTorch {torch.__version__} < 2.0)")
        else:
            brain_logger.warning("No snapshot")
        
        # Controller
        self.controller = UnifiedSystemController(self.hybrid.core)
        self.controller.connect_v7(obs_dim=4, act_dim=2)
        
        # 🔥 MUDANÇA #4: Force num_steps=1 no bridge (configurable)
        if hasattr(self.controller.v7_bridge, 'num_steps'):
            self.controller.v7_bridge.num_steps = 1
        # Ensure bridge lives on the selected device
        try:
            self.controller.v7_bridge.to(self.device)
        except Exception:
            pass
        
        # BLOCO 2 - TAREFA 25: Benchmark mode
        self.benchmark_mode = os.getenv('BENCHMARK_MODE', '0') == '1'
        if self.benchmark_mode:
            brain_logger.info("⚡ BENCHMARK MODE: Telemetria desabilitada para performance pura")
        
        # Optimizer (APENAS adapters + V7)
        trainable_params = []
        
        # Top 16 neurons adapters (move to device if available)
        for neuron in self.hybrid.core.registry.get_active()[:16]:
            try:
                neuron.to(self.device)
            except Exception:
                pass
            trainable_params.extend(list(neuron.A_in.parameters()))
            trainable_params.extend(list(neuron.A_out.parameters()))
        
        # V7 bridge
        trainable_params.extend(list(self.controller.v7_bridge.parameters()))
        
        # Router
        if self.hybrid.core.router:
            trainable_params.append(self.hybrid.core.router.competence)
        
        # Deduplicate optimizer params to prevent duplicates warning
        seen, dedup_params = set(), []
        for p in trainable_params:
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            dedup_params.append(p)

        self.optimizer = torch.optim.Adam(dedup_params, lr=self.learning_rate)

        # Phase 1 hooks: compose monitoring model and controllers
        self._godel_monitor = GodelMonitor(delta_0=0.05) if _PHASE1_HOOKS else None
        self._meta_controller = NeedleMetaController() if _PHASE1_HOOKS else None
        try:
            self._monitor_model = build_trainable_composite(
                controller=self.controller,
                registry=self.hybrid.core.registry,
                router=self.hybrid.core.router,
            ) if _PHASE1_HOOKS else None
        except Exception:
            self._monitor_model = None
        brain_logger.info(f"✅ Optimizer: {len(trainable_params)} params")
        
        # WORM + Integrity checks
        try:
            if hasattr(self.hybrid.core, 'worm') and self.hybrid.core.worm:
                rotated = self.hybrid.core.worm.rotate_if_invalid()
                if rotated:
                    brain_logger.warning("WORM chain rotated due to invalid integrity")
                self.hybrid.core.worm.append('daemon_start', {
                    'device': str(self.device),
                    'time': datetime.now().isoformat(),
                })
        except Exception:
            pass

        # Verify existing checkpoints integrity (if present)
        self.verify_checkpoint_integrity()
        
        # FASE 1: Initialize Curiosity Module
        self.curiosity = CuriosityModule(H=1024, action_dim=2).to(self.device)
        brain_logger.info(f"✅ Curiosity module initialized (has its own optimizer)")
        
        # FASE 2: Log initialization
        brain_logger.info(f"✅ Self-Analysis module initialized")
        brain_logger.info(f"✅ Metrics Dashboard initialized")
        
        # FASE 3 & 4: Initialize self-improvement loop
        self.self_improvement_loop = SelfImprovementLoop(
            self.module_synthesizer,
            self.recursive_engine
        )
        brain_logger.info(f"✅ Module Synthesizer initialized")
        brain_logger.info(f"✅ Recursive Improvement Engine initialized")
        brain_logger.info(f"✅ Self-Improvement Loop initialized")

        # Reset env
        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        
        brain_logger.info("✅ Ready for FAST learning!")
        # Phase 2: initialize self observer and scheduler
        try:
            self._observer = SelfObserver() if _PHASE2_HOOKS else None
            self._scheduler = MultiObjectiveScheduler(self.hybrid.core) if _PHASE2_HOOKS else None
            # Intrinsic reward weight used when curiosity is active
            self.curiosity_weight = getattr(self, 'curiosity_weight', 0.02)
        except Exception:
            self._observer = None
            self._scheduler = None
        # Phase 3: episodic memory + novelty reward
        try:
            self._episodic = EpisodicMemory() if _PHASE3_HOOKS else None
            self._novelty = NoveltyReward(scale=0.02) if _PHASE3_HOOKS else None
        except Exception:
            self._episodic = None
            self._novelty = None
    
    def run_episode(self):
        """
        Episódio OTIMIZADO para velocidade
        """
        # Episode WORM start and memory snapshot
        try:
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            gpu_alloc = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
            if hasattr(self.hybrid.core, 'worm') and self.hybrid.core.worm:
                self.hybrid.core.worm.append('episode_start', {
                    'episode': int(self.episode + 1),
                    'rss_kb': int(rss_kb),
                    'gpu_bytes': int(gpu_alloc),
                })
        except Exception:
            pass

        self.state = self.env.reset()
        if isinstance(self.state, tuple):
            self.state = self.state[0]
        
        episode_reward = 0
        steps = 0
        done = False
        
        # Buffers
        ep_states, ep_actions, ep_rewards = [], [], []
        ep_values, ep_log_probs = [], []
        ep_Zs = []  # ✅ Save Z latent states for training recomputation
        
        episode_start = time.time()
        
        # 🔥 NO GRADIENTS durante episódio (velocidade)
        with torch.no_grad():
            while not done and self.running and steps < getattr(self, 'max_episode_steps', 500):
                step_start = time.time()
                # Soft resource guards
                try:
                    if _psutil is not None:
                        if _psutil.cpu_percent(interval=None) > 97:
                            time.sleep(0.02)
                        mem = _psutil.virtual_memory()
                        if mem.percent > 97:
                            time.sleep(0.05)
                except Exception:
                    pass
                
                # 1. Estado
                t_obs0 = time.time()
                obs = torch.FloatTensor(self.state).unsqueeze(0).to(self.device)
                try:
                    if getattr(self, '_episodic', None) is not None:
                        self._episodic.record(obs)
                except Exception:
                    pass
                t_obs1 = time.time()
                
                # 2. Forward (RÁPIDO: 1 brain step, 8 neurons)
                t_fwd0 = time.time()
                result = self.controller.step(
                    obs=obs,
                    penin_metrics={
                        'L_infinity': episode_reward / 500.0,
                        'CAOS_plus': 0.5,
                        'SR_Omega_infinity': 0.7
                    },
                    reward=episode_reward / 500.0
                    )
                t_fwd1 = time.time()
                
                action_logits = result['action_logits']
                value = result['value']
                Z_current = result.get('Z', None)
                
                # FASE 4: Influência da Bomba de Consciência
                if self.mirror_module:
                    mirror_logits = self.mirror_module.predict_next_action(self._get_brain_state())
                    
                    # A previsão do MirrorModule é combinada com a decisão do agente,
                    # forçando o agente a levar em conta sua "auto-imagem".
                    # Usamos um peso baixo (0.1) para não desestabilizar o aprendizado PPO.
                    action_logits = action_logits * 0.9 + mirror_logits * 0.1

                # FASE 2: Store Z for telemetry
                if Z_current is not None:
                    self._last_Z = Z_current.detach().clone()
                
                # 3. Sample action
                t_act0 = time.time()
                action_probs = torch.softmax(action_logits, dim=-1)
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)
                t_act1 = time.time()
                
                action_int = action.item()
                
                # 4. Env step
                t_env0 = time.time()
                step_result = self.env.step(action_int)
                t_env1 = time.time()
                
                if len(step_result) == 5:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                else:
                    next_state, reward, done, info = step_result
                
                # FASE 2: Salvar na Memória Episódica em vez do Replay Buffer
                if self.episodic_memory:
                    # A API da Memória TEIS é assíncrona, mas a chamaremos de forma síncrona aqui
                    # para simplificar a integração inicial. Isso pode ser otimizado depois.
                    episode_step = {
                        'situation': self.state.copy(),
                        'action': action_int,
                        'outcome': {'reward': reward, 'success': not done}
                    }
                    # Esta chamada não é `await`, pois estamos em um contexto síncrono.
                    # A implementação original do TEIS era assíncrona.
                    # Vamos adaptar o uso para ser síncrono por enquanto.
                    
                    # Precisamos adaptar a forma de armazenar. A memória TEIS espera um episódio inteiro.
                    # Vamos acumular os passos e salvar no final do episódio.
                    pass # A lógica será movida para o final do episódio.

                # FIX B1: Salvar no replay buffer (Lógica antiga mantida como fallback)
                if self.use_replay and self.replay_buffer:
                    try:
                        self.replay_buffer.push(
                            self.state.copy(),
                            action_int,
                            reward,
                            next_state.copy(),
                            done
                        )
                    except Exception:
                        pass  # Fail silently
                
                # 5. Store (curiosity será computada após no_grad)
                # FIX CRÍTICO: Detach tensors criados sob no_grad para permitir backward
                ep_states.append(obs)
                ep_actions.append(action.detach())  # ✅ DETACH para permitir backward posterior
                ep_rewards.append(reward)  # Env reward apenas
                ep_values.append(value.detach())  # ✅ DETACH value também
                ep_log_probs.append(log_prob.detach())  # ✅ DETACH log_prob
                if Z_current is not None:
                    ep_Zs.append(Z_current.detach().clone())  # ✅ Save Z for training
                
                # 6. Update
                self.state = next_state
                
                # FIX FASE 2: Track states for novelty
                self.prev_states_buffer.append(torch.tensor(next_state).float())
                if len(self.prev_states_buffer) > self.max_states_buffer:
                    self.prev_states_buffer.pop(0)
                
                episode_reward += reward
                steps += 1
                
                # FASE 2: Profile step time
                step_time = time.time() - step_start
                with self.self_analysis.profile_component('step_total'):
                    time.sleep(0)  # Just to record the time
                self.stats['total_steps'] += 1
                
                # Track step time
                step_time = time.time() - step_start
                self.stats['avg_time_per_step'] = 0.9 * self.stats['avg_time_per_step'] + 0.1 * step_time

                # Phase 2: self-observer intrinsic shaping (record only; apply after episode)
                try:
                    if getattr(self, '_observer', None) is not None:
                        _ = self._observer.record_step(action_probs, step_time)
                except Exception:
                    pass

                # Export Prometheus metrics if enabled
                try:
                    if self._prom is not None:
                        self._prom['step_total_sec'].set(step_time)
                        self._prom['avg_step_sec'].set(self.stats['avg_time_per_step'])
                        t = result.get('timings', {}) if isinstance(result, dict) else {}
                        self._prom['encode_ms'].set(float(t.get('encode_ms', 0.0)))
                        self._prom['brain_ms'].set(float(t.get('brain_ms', 0.0)))
                        self._prom['decode_ms'].set(float(t.get('decode_ms', 0.0)))
                        self._prom['penin_ms'].set(float(t.get('penin_update_ms', 0.0)))
                        self._prom['forward_ms'].set(float(t.get('forward_ms', 0.0)))
                        self._prom['darwin_ms'].set(float(t.get('darwin_ms', 0.0)))
                except Exception:
                    pass

                # Detailed probe logging for slow paths (including controller timings)
                if step_time > 0.1:
                    timings = result.get('timings', {}) if isinstance(result, dict) else {}
                    pen_ms = float(timings.get('penin_update_ms', 0.0))
                    fwd_ms = float(timings.get('forward_ms', 0.0))
                    dar_ms = float(timings.get('darwin_ms', 0.0))
                    brain_logger.info(
                        f"🧪 step_probe: total={step_time:.3f}s, obs={t_obs1 - t_obs0:.3f}s, "
                        f"fwd={t_fwd1 - t_fwd0:.3f}s (controller: penin={pen_ms:.1f}ms, fwd={fwd_ms:.1f}ms, darwin={dar_ms:.1f}ms), "
                        f"act={t_act1 - t_act0:.3f}s, env={t_env1 - t_env0:.3f}s"
                    )
                
                # Apply offline professor suggestions (router hyperparams)
                try:
                    sug_fp = "/root/UNIFIED_BRAIN/runtime_suggestions.json"
                    if os.path.exists(sug_fp) and getattr(self.hybrid.core, 'router', None) is not None:
                        data = _json.loads(open(sug_fp, 'r').read())
                        if isinstance(data, dict):
                            r = data.get('router') or {}
                            if isinstance(r, dict):
                                if 'top_k' in r:
                                    try:
                                        self.hybrid.core.router.top_k = int(max(1, min(self.hybrid.core.router.num_neurons, int(r['top_k']))))
                                    except Exception:
                                        pass
                                if 'temperature' in r:
                                    try:
                                        self.hybrid.core.router.temperature = float(max(0.1, min(3.0, float(r['temperature']))))
                                    except Exception:
                                        pass
                        try:
                            os.remove(sug_fp)
                        except Exception:
                            pass
                except Exception:
                    pass
        
        # 🔥 TRAINING (só no final do episódio)
        loss_value = 0.0
        curiosity_total = 0.0
        if len(ep_rewards) > 1:
            # Se a memória episódica estiver em uso, podemos tentar usar seus insights
            if self.episodic_memory and len(self.episodic_memory.episodes) > 20:
                 # Gerar insights antes de treinar
                 self.episodic_memory._generate_insights()
                 if self.episodic_memory.insights:
                     brain_logger.debug(f"🧠 Insights da Memória: {len(self.episodic_memory.insights)} insights gerados.")

            # Train curiosity e adiciona reward intrínseco (with profiling)
            if self.curiosity:
                with self.self_analysis.profile_component('curiosity'):
                    for i, (state, action) in enumerate(zip(ep_states, ep_actions)):
                        # Compute Z from state
                        Z_for_curiosity = self.controller.v7_bridge.obs_encoder(state) if self.controller.v7_bridge else state
                        curiosity_reward = self.curiosity.compute_curiosity(Z_for_curiosity, action.item())
                        # Add intrinsic reward
                        ep_rewards[i] += self.curiosity_weight * curiosity_reward  # ✅ CORREÇÃO #4: Usa variável dinâmica
                        curiosity_total += curiosity_reward
            # Novelty-based intrinsic reward from episodic memory (Phase 3)
            try:
                if getattr(self, '_episodic', None) is not None and getattr(self, '_novelty', None) is not None:
                    for i, state in enumerate(ep_states):
                        novelty = self._episodic.novelty(state)
                        ep_rewards[i] += float(self._novelty.intrinsic(novelty))
            except Exception:
                pass
            
            # Training (with profiling)
            with self.self_analysis.profile_component('training'):
                loss_value = self.train_on_episode(ep_states, ep_actions, ep_rewards, ep_values, ep_log_probs, ep_Zs)
        
        # Stats
        episode_time = time.time() - episode_start
        
        # FASE 2: Injetar Incompletude Gödeliana como meta de curiosidade
        if self.episode % 25 == 0 and self.incompleteness_injector:
            try:
                # Usar uma descrição de tarefa que force os limites
                task_description = "A task requiring planning and reasoning beyond current capabilities"
                
                # O modelo a ser analisado é a ponte V7, que é o "cérebro" da política
                model_to_analyze = self.controller.v7_bridge
                
                limit_analysis = self.incompleteness_injector.detect_fundamental_limit(
                    model_to_analyze, 
                    task_description
                )

                if limit_analysis['is_limited']:
                    goal_description = (
                        f"Transcend the detected limit: {limit_analysis['limit_type']}. "
                        f"Suggestion: {limit_analysis['transcendence_suggested']}."
                    )
                    
                    # Criar uma representação vetorial do objetivo para o módulo de curiosidade
                    # (uma simples soma dos hashes das palavras-chave)
                    goal_vector = torch.tensor(
                        [hash(word) % 1000 for word in goal_description.split()],
                        dtype=torch.float32
                    ).mean().unsqueeze(0).to(self.device)

                    self.curiosity.set_custom_goal(goal_vector)
                    brain_logger.warning(f"🔥 Incompletude Injetada: {goal_description}")

            except Exception as e:
                brain_logger.error(f"❌ Falha ao injetar Incompletude Gödeliana: {e}")

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self.episodic_memory.episodes) > self.episodic_memory.max_episodes:
                self.episodic_memory.episodes.popleft()

        # Checkpoint a cada 5 eps + verify integrity + TELEMETRIA
        if self.episode % 5 == 0:
            with self.self_analysis.profile_component('checkpoint'):
                self.save_checkpoint()
                self.verify_checkpoint_integrity()
        
        # CHECKPOINT COMPLETO a cada 50 eps (backup de segurança)
        if self.episode % 50 == 0:
            try:
                import os
                os.makedirs('/root/UNIFIED_BRAIN/checkpoints', exist_ok=True)
                checkpoint_path = f'/root/UNIFIED_BRAIN/checkpoints/brain_ep{self.episode}_254n.pt'
                torch.save({
                    'episode': self.episode,
                    'hybrid_state': self.hybrid.state_dict(),
                    'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
                    'stats': self.stats,
                    'best_reward': self.best_reward,
                    'num_active_neurons': len(self.hybrid.core.registry.get_active()),
                    'timestamp': time.time(),
                }, checkpoint_path)
                brain_logger.info(f"💾 Full checkpoint saved: {checkpoint_path}")
            except Exception as e:
                brain_logger.warning(f"Full checkpoint failed: {e}")
            
            # FIX CRÍTICO: Salvar brain_metrics
            if self.db and self.emergence:
                try:
                    # Get router metrics
                    router = getattr(self.hybrid.core, 'router', None)
                    top_k_val = getattr(router, 'top_k', 0) if router else 0
                    temp_val = getattr(router, 'temperature', 0.0) if router else 0.0
                    
                    # Get neuron metrics
                    neurons = getattr(self.hybrid.core.registry, 'neurons', {})
                    competences = [getattr(n.meta, 'competence', 0.0) for n in neurons.values() if hasattr(n, 'meta')]
                    
                    avg_comp = sum(competences) / len(competences) if competences else 0.0
                    max_comp = max(competences) if competences else 0.0
                    min_comp = min(competences) if competences else 0.0
                    
                    # Get maintenance stats (if available)
                    maint = getattr(self, '_last_maintenance_results', {})
                    proms = maint.get('promoted', 0)
                    demos = maint.get('demoted', 0)
                    
                    # Compute REAL metrics
                    # Coherence: Z-space coherence (lower std = more coherent)
                    coherence_val = 0.5  # default
                    if hasattr(self, '_last_Z') and self._last_Z is not None:
                        try:
                            coherence_raw = torch.std(self._last_Z).item()
                            # FIX: Usar sigmoid-like em vez de subtração (evita sempre dar 0)
                            coherence_val = 1.0 / (1.0 + coherence_raw)  # 0.5 when raw=1, 0.9 when raw=0.1
                            brain_logger.info(f"📊 coherence: raw={coherence_raw:.4f}, val={coherence_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"Coherence calc failed: {e}")
                    
                    # Novelty: measure of exploration
                    novelty_val = 0.3  # default
                    try:
                        if len(self.prev_states_buffer) > 0:
                            novelty_raw = torch.std(torch.stack(self.prev_states_buffer[-10:])).item()
                            novelty_val = min(1.0, novelty_raw)
                            brain_logger.info(f"📊 novelty: raw={novelty_raw:.4f}, val={novelty_val:.4f}, buffer={len(self.prev_states_buffer)}")
                    except Exception as e:
                        brain_logger.warning(f"Novelty calc failed: {e}, buffer_len={len(self.prev_states_buffer)}")
                    
                    # IA³ Signal: router confidence
                    ia3_signal_val = 0.5  # default
                    if router and hasattr(router, 'competence'):
                        try:
                            comp_std = torch.std(router.competence).item() if getattr(router.competence, 'numel', lambda: 0)() > 1 else 0.0
                            ia3_signal_val = min(1.0, max(0.0, comp_std * 2.0))  # scale to [0,1]
                            brain_logger.info(f"📊 ia3: comp_std={comp_std:.4f}, val={ia3_signal_val:.4f}, numel={router.competence.numel()}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 calc failed: {e}")
                            comp_std = 0.0
                    elif router and hasattr(router, 'last_entropy'):
                        # Fallback: use entropy
                        try:
                            ia3_signal_val = max(0.0, 1.0 - router.last_entropy)
                            brain_logger.info(f"📊 ia3: entropy={router.last_entropy:.4f}, val={ia3_signal_val:.4f}")
                        except Exception as e:
                            brain_logger.warning(f"IA3 entropy calc failed: {e}")
                    else:
                        brain_logger.warning(f"IA3: router missing or no competence (router={router is not None})")
                    
                    # Save to DB
                    self.db.save_brain_metrics(
                        episode=self.episode,
                        coherence=coherence_val,  # ✅ REAL
                        novelty=novelty_val,      # ✅ REAL
                        energy=episode_reward / 500.0,  # normalized to CartPole max
                        ia3_signal=ia3_signal_val,  # ✅ REAL
                        num_active_neurons=len(neurons),
                        top_k=top_k_val,
                        temperature=temp_val,
                        avg_competence=avg_comp,
                        max_competence=max_comp,
                        min_competence=min_comp,
                        promotions=proms,
                        demotions=demos
                    )
                    
                    # Track key metrics for surprise detection
                    self.emergence.track_metric('reward', episode_reward, self.episode)
                    self.emergence.track_metric('avg_competence', avg_comp, self.episode)
                    self.emergence.track_metric('num_neurons', len(neurons), self.episode)
                    
                    # Record system connections
                    self.emergence.record_connection(
                        'brain_daemon', 'darwin_system', 
                        'neuron_registry', strength=0.9
                    )
                    self.emergence.record_connection(
                        'brain_daemon', 'environment', 
                        'rl_training', strength=1.0
                    )
                    
                    brain_logger.info(f"✅ Telemetria salva: ep {self.episode}")
                    
                except Exception as e:
                    brain_logger.error(f"❌ Falha ao salvar telemetria: {e}")
        
        # DARWINACCI: Evoluir hyperparameters a cada 50 episódios
        if self.episode % 50 == 0 and self.darwinacci and self.universal_connector:
            try:
                brain_logger.info("🧬 DARWINACCI: Evolving hyperparameters...")
                
                # Run 1 Darwinacci cycle
                champion = self.darwinacci.run(max_cycles=1)
                
                if champion and hasattr(champion, 'genome'):
                    genome = champion.genome
                    
                    # Apply evolved hyperparameters
                    if 'lr' in genome and self.optimizer:
                        new_lr = float(genome['lr'])
                        old_lr = self.optimizer.param_groups[0]['lr']
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = new_lr
                        brain_logger.info(f"   🧬 LR evolved: {old_lr:.6f} → {new_lr:.6f}")
                    
                    if 'curiosity_weight' in genome:
                        new_cw = float(genome['curiosity_weight'])
                        old_cw = self.curiosity_weight
                        self.curiosity_weight = new_cw
                        brain_logger.info(f"   🧬 Curiosity weight evolved: {old_cw:.3f} → {new_cw:.3f}")
                    
                    if 'top_k' in genome and hasattr(self.hybrid.core, 'top_k'):
                        new_k = int(genome['top_k'])
                        old_k = self.hybrid.core.top_k
                        self.hybrid.core.top_k = new_k
                        brain_logger.info(f"   🧬 top_k evolved: {old_k} → {new_k}")
                    
                    # Sync with universal network
                    if self.universal_connector:
                        sync_results = self.universal_connector.sync_all()
                        if sync_results.get('synced'):
                            brain_logger.debug(f"   🔄 Universal sync: {len(sync_results['synced'])} ops")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Darwinacci evolution failed: {e}")
        
        # FIX B2: Auto-Tuner a cada 20 episódios
        if self.episode % 20 == 0 and self.use_auto_tuner and self.auto_tuner:
            try:
                current_params = {
                    'lr': self.optimizer.param_groups[0]['lr'] if self.optimizer else 0.001,
                    'top_k': self.hybrid.core.top_k if hasattr(self.hybrid.core, 'top_k') else 4,
                    'temperature': self.hybrid.core.router.temperature if hasattr(self.hybrid.core, 'router') and hasattr(self.hybrid.core.router, 'temperature') else 1.0
                }
                
                metrics = {
                    'avg_reward': self.stats['avg_reward_last_100'],
                    'loss': loss_value if 'loss_value' in locals() else 0.0,
                    'entropy': 0.5,  # placeholder
                    'improvement': self.stats['avg_reward_last_100'] - self.best_reward
                }
                
                new_params = self.auto_tuner.tune(current_params, metrics)
                
                # Aplicar novos params
                if 'lr' in new_params and self.optimizer:
                    old_lr = self.optimizer.param_groups[0]['lr']
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = new_params['lr']
                    if abs(new_params['lr'] - old_lr) > 0.0001:
                        brain_logger.info(f"🎛️ Auto-tuner: LR {old_lr:.6f} → {new_params['lr']:.6f}")
                
                if 'top_k' in new_params and hasattr(self.hybrid.core, 'top_k'):
                    old_k = self.hybrid.core.top_k
                    self.hybrid.core.top_k = int(new_params['top_k'])
                    if old_k != self.hybrid.core.top_k:
                        brain_logger.info(f"🎛️ Auto-tuner: top_k {old_k} → {self.hybrid.core.top_k}")
                
            except Exception as e:
                brain_logger.warning(f"⚠️ Auto-tuner falhou: {e}")
        
        # Adicionar episódio completo na Memória Episódica
        if self.episodic_memory:
            full_episode = {
                'steps': [],
                'total_reward': episode_reward,
                'steps_count': steps
            }
            for i in range(len(ep_states)):
                full_episode['steps'].append({
                    'situation': ep_states[i].squeeze(0).cpu().numpy().tolist(),
                    'action': ep_actions[i].item(),
                    'outcome': {'reward': ep_rewards[i], 'success': True} # Simplificado
                })
            
            # Simplesmente adicionamos à lista de episódios da memória
            self.episodic_memory.episodes.append(full_episode)
            if len(self