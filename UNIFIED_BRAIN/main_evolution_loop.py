#!/usr/bin/env python3
"""
🔄 Main Evolution Loop
Fixes P7: Nenhum loop de auto-evolução contínua
"""
import os, sys, time, signal
sys.path.insert(0, '/root')

# Configure environment + deterministic seeds (allow caller overrides)
os.environ.setdefault('UBRAIN_METRICS_PORT', '9109')
os.environ.setdefault('UBRAIN_BOOTSTRAP_MODE', '1')  # Relaxed gates
os.environ.setdefault('UBRAIN_EVAL_SEEDS', '42,43')  # Default; can be overridden by env
os.environ.setdefault('UBRAIN_EVAL_EPISODES', '1')
import torch
import random
random.seed(1337)
import numpy as _np
try:
    _np.random.seed(1337)
except Exception:
    pass
torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(1337)

from UNIFIED_BRAIN.unified_brain_core import CoreSoupHybrid
from UNIFIED_BRAIN.brain_logger import brain_logger
from UNIFIED_BRAIN.metrics_bridge import record_reward
try:
    from UNIFIED_BRAIN.metrics_worm import append_record as worm_append
except Exception:
    worm_append = None  # type: ignore

# Graceful shutdown
running = True
def signal_handler(sig, frame):
    global running
    running = False
    brain_logger.info("🛑 Shutdown signal received")

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

brain_logger.info("="*60)
brain_logger.info("🧠 Starting Main Evolution Loop")
brain_logger.info("="*60)

# Initialize brain
hybrid = CoreSoupHybrid(H=512)

# Check soup population and auto-populate if empty
soup_count = len(hybrid.soup.registry.get_active())
if soup_count == 0:
    brain_logger.warning("⚠️  Soup is empty! Auto-populating...")
    
    # Auto-populate soup inline
    neuron_configs = [
        ('identity', lambda x: x, 1.0),
        ('tanh', lambda x: torch.tanh(x), 0.9),
        ('relu', lambda x: torch.relu(x), 1.1),
        ('sigmoid', lambda x: torch.sigmoid(x), 0.8),
    ]
    
    from UNIFIED_BRAIN.brain_spec import RegisteredNeuron, NeuronMeta, NeuronStatus
    import hashlib
    
    for i in range(20):
        name, activation, base_scale = neuron_configs[i % len(neuron_configs)]
        scale = base_scale + (i // len(neuron_configs)) * 0.05
        
        def make_forward(act_fn, s):
            def fn(x):
                return act_fn(x * s)
            return fn
        
        neuron_id = f"synth_{name}_{i:02d}"
        checksum = hashlib.sha256(neuron_id.encode()).hexdigest()[:16]
        
        meta = NeuronMeta(
            id=neuron_id,
            in_shape=(512,),
            out_shape=(512,),
            dtype=torch.float32,
            device='cpu',
            status=NeuronStatus.ACTIVE,
            source='synthetic_bootstrap',
            params_count=0,
            checksum=checksum,
            competence_score=0.4 + (i * 0.01),
            novelty_score=0.5,
        )
        
        neuron = RegisteredNeuron(meta, make_forward(activation, scale), H=512)
        result = hybrid.soup.register_neuron(neuron)
        if result is not False:  # None or True is OK
            pass
    
    soup_count = len(hybrid.soup.registry.get_active())
    brain_logger.info(f"   ✅ Auto-populated with {soup_count} neurons")
    
    # Initialize router after populating
    if soup_count > 0:
        hybrid.soup.initialize_router()
        brain_logger.info(f"   ✅ Soup router initialized")

brain_logger.info(f"✅ Soup neurons: {soup_count}")
brain_logger.info(f"✅ Core neurons (before bootstrap): {len(hybrid.core.registry.get_active())}")

# CRITICAL FIX: Bootstrap core if empty
if len(hybrid.core.registry.get_active()) == 0 and len(hybrid.soup.registry.get_active()) > 0:
    brain_logger.warning("🚨 Core is EMPTY! Bootstrapping with top neurons from soup...")
    
    # Promote top 5 neurons by competence directly (NO GATES for bootstrap)
    soup_neurons = sorted(
        hybrid.soup.registry.get_active(), 
        key=lambda n: n.meta.competence_score, 
        reverse=True
    )[:5]
    
    for neuron in soup_neurons:
        # Register in core (COPY, don't move)
        hybrid.core.register_neuron(neuron)
        # Mark as FROZEN in soup (keep it there but inactive)
        hybrid.soup.registry.promote(neuron.meta.id, NeuronStatus.FROZEN)
        brain_logger.info(f"   ✅ Bootstrapped {neuron.meta.id} (comp={neuron.meta.competence_score:.2f})")
    
    brain_logger.info(f"✅ Core bootstrapped with {len(hybrid.core.registry.get_active())} neurons")

brain_logger.info(f"✅ Core neurons (after bootstrap): {len(hybrid.core.registry.get_active())}")

# Initialize core router if not already
if hybrid.core.router is None and len(hybrid.core.registry.get_active()) > 0:
    hybrid.core.initialize_router()
    brain_logger.info(f"✅ Core router initialized with {len(hybrid.core.registry.get_active())} neurons")

# Initialize state
Z = torch.randn(1, 512)
step = 0
last_maintenance = 0
# Every N steps (configurable)
try:
    maintenance_interval = int(os.getenv('UBRAIN_MAINT_INTERVAL', '500'))
except Exception:
    maintenance_interval = 500
last_report = time.time()
_worm_seed = 1337
try:
    _seeds_raw = os.environ.get('UBRAIN_EVAL_SEEDS', '')
    if _seeds_raw:
        _first = _seeds_raw.split(',')[0].strip()
        if _first and _first.lstrip('-').isdigit():
            _worm_seed = int(_first)
except Exception:
    pass

# Optional: stop after fixed number of steps for evaluation runs
_max_steps_env = os.environ.get('UBRAIN_MAX_STEPS')
_max_steps = None
try:
    if _max_steps_env is not None and _max_steps_env.strip():
        _max_steps = int(_max_steps_env)
except Exception:
    _max_steps = None

# --- External, deterministic reward provider(s) with optional rotation ---
providers = []
try:
    from UNIFIED_BRAIN.reward_providers import build_default_provider, build_provider
    raw_tasks = os.getenv('UBRAIN_TASK', 'cartpole')
    task_list = [t.strip() for t in raw_tasks.split(',') if t.strip()]
    for t in task_list:
        try:
            providers.append(build_provider(t))
        except Exception:
            providers.append(build_default_provider())
    if not providers:
        providers.append(build_default_provider())
    _prov_idx = 0
    _reward_provider = providers[_prov_idx]
    brain_logger.info(f"✅ External reward provider(s) initialized: {len(providers)}")
except Exception as e:
    _reward_provider = None
    providers = []
    _prov_idx = 0
    brain_logger.warning(f"⚠️ Reward provider unavailable, falling back to random: {e}")

brain_logger.info(f"\n🚀 Starting evolution loop (maintenance every {maintenance_interval} steps)")
brain_logger.info("   Press Ctrl+C to stop gracefully\n")

try:
    while running:
        try:
            # 1. Brain step with external reward (fallback: small random)
            if _reward_provider is not None:
                try:
                    reward = _reward_provider.step(Z)
                except Exception as e:
                    brain_logger.error(f"Reward provider error: {e}")
                    reward = torch.randn(1).item() * 0.1
            else:
                reward = torch.randn(1).item() * 0.1
            Z, info = hybrid.core.step(Z, reward=reward)
            try:
                record_reward(reward)
            except Exception:
                pass
            try:
                if worm_append is not None:
                    worm_append(
                        os.getenv('UBRAIN_TASK', 'cartpole'),
                        _worm_seed,
                        float(reward),
                        {'H': 512}
                    )
            except Exception:
                pass
            step += 1
            if _max_steps is not None and step >= _max_steps:
                running = False
            
            # 2. Periodic maintenance (promotion/demotion) with rollback guard
            if step - last_maintenance >= maintenance_interval:
                brain_logger.info(f"\n{'='*60}")
                brain_logger.info(f"🔧 MAINTENANCE at step {step}")
                brain_logger.info(f"{'='*60}")
                
                try:
                    # Prefer rollback-protected maintenance if available
                    if hasattr(hybrid, 'run_maintenance_with_rollback'):
                        result = hybrid.run_maintenance_with_rollback()
                    else:
                        result = hybrid.tick_maintenance()
                    brain_logger.info(f"   Promoted: {result.get('promoted', 0)}")
                    brain_logger.info(f"   Demoted:  {result.get('demoted', 0)}")
                    brain_logger.info(f"   Unfrozen: {result.get('unfrozen', 0)}")
                    brain_logger.info(f"   Errors:   {len(result.get('errors', []))}")
                    if result.get('rollback') is True:
                        brain_logger.warning("   🔁 Rollback applied: no multi-seed improvement detected")
                    elif result.get('rollback') is False:
                        brain_logger.info("   ✅ Change kept: multi-seed improvement confirmed")
                    
                    if result.get('errors'):
                        for err in result['errors']:
                            brain_logger.error(f"      - {err}")
                    
                    brain_logger.info(f"   Active neurons: {result.get('active_after', 0)}")
                
                except Exception as e:
                    brain_logger.error(f"❌ Maintenance failed: {e}", exc_info=True)
                
                last_maintenance = step
                brain_logger.info(f"{'='*60}\n")
            
            # 3. Periodic status report (every 30s)
            now = time.time()
            if now - last_report >= 30:
                m = hybrid.core.get_metrics_summary() or {}
                brain_logger.info(
                    f"📊 Step {step:6d}: "
                    f"coh={m.get('avg_coherence', 0):.3f}, "
                    f"nov={m.get('avg_novelty', 0):.3f}, "
                    f"lat={m.get('avg_latency_ms', 0):.1f}ms, "
                    f"active={len(hybrid.core.registry.get_active())}/{len(hybrid.soup.registry.get_active())}"
                )
                last_report = now

            # 4. Optional task rotation (every maintenance interval)
            if providers and maintenance_interval > 0 and step > 0 and step % maintenance_interval == 0:
                try:
                    _prov_idx = (_prov_idx + 1) % len(providers)
                    _reward_provider = providers[_prov_idx]
                    brain_logger.info(f"🔄 Switched reward provider index to #{_prov_idx}")
                except Exception:
                    pass
            
            # 5. Throttle to ~20 Hz
            time.sleep(0.05)
        
        except KeyboardInterrupt:
            brain_logger.info("\n🛑 Keyboard interrupt")
            break
        
        except Exception as e:
            brain_logger.error(f"❌ Loop error at step {step}: {e}", exc_info=False)
            time.sleep(1.0)

except Exception as e:
    brain_logger.critical(f"💥 Fatal error: {e}", exc_info=True)

finally:
    brain_logger.info("\n" + "="*60)
    brain_logger.info("🛑 Evolution loop stopped")
    brain_logger.info(f"   Total steps: {step}")
    m = hybrid.core.get_metrics_summary() or {}
    brain_logger.info(f"   Final active neurons: {len(hybrid.core.registry.get_active())}")
    brain_logger.info(f"   Avg coherence: {m.get('avg_coherence', 0):.3f}")
    brain_logger.info(f"   Avg novelty: {m.get('avg_novelty', 0):.3f}")
    brain_logger.info("="*60)
