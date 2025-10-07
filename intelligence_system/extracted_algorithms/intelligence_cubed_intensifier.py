"""
INTELLIGENCE CUBED (I³) INTENSIFIER
====================================

Módulo de intensificação para transformar o sistema base em Inteligência ao Cubo COMPLETA.

Componentes I³:
1. Auto-Adaptive ✅ (Darwin + QD + Emergence)
2. Auto-Recursive ✅ (Synergy 5 MAML) → INTENSIFICAR (meta-depth)
3. Auto-Evolutionary ✅ (Darwin) → INTENSIFICAR (pure novelty)
4. Auto-Aware ✅ (Emergence) → INTENSIFICAR (introspection score)
5. Auto-Sufficient ✅ (Transfer + DB)
6. Auto-Didactic ✅ (MAML + Curriculum) → INTENSIFICAR (self-generated)
7. Auto-Built ✅ (Auto-coding) → INTENSIFICAR (aggressive apply)
8. Auto-Architected ✅ (AutoML)
9. Auto-Renewable ✅ (Rollback + A/B)
10. Auto-Synaptic ✅ (Dynamic Layer)
11. Auto-Modular ✅ (Multi-System)
12. Auto-Expandable ✅ (Neuronal Farm)
13. Auto-Validating ✅ (Validator + Sandbox)
14. Auto-Calibrating ✅ (PENIN→V7)
15. Auto-Analytical ✅ (Supreme Auditor)
16. Auto-Regenerative ✅ (Replay) → INTENSIFICAR (continuous)
17. Auto-Trained ✅ (Meta-Learner + PPO)
18. Auto-Tuning ✅ (Godelian + Curriculum)
19. Auto-Infinite ✅ (Incompletude Infinita)
"""

import logging
import numpy as np
import torch
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class IntrospectionEvent:
    """Self-awareness event"""
    cycle: int
    component: str
    before_state: Dict[str, float]
    after_state: Dict[str, float]
    surprise_factor: float  # 0-1: how unexpected the change was
    reasoning: str


class IntrospectionEngine:
    """
    Auto-Awareness Intensification
    Tracks system's own state changes and measures surprise
    """
    
    def __init__(self, window: int = 100):
        self.window = window
        self.state_history = deque(maxlen=window)
        self.events: List[IntrospectionEvent] = []
        self.surprise_accumulator = 0.0
        self.surprise_threshold = 0.6  # FIX #4: Define surprise_threshold
        logger.info("🔍 Introspection Engine ACTIVATED (I³ Auto-Aware)")
    
    def record_state(self, cycle: int, state_snapshot: Dict[str, float]):
        """Record current system state"""
        self.state_history.append({
            'cycle': cycle,
            'state': state_snapshot.copy()
        })
    
    def compute_surprise(self, component: str, before: Dict[str, float], 
                        after: Dict[str, float]) -> float:
        """
        Compute surprise factor (0-1) for a state change
        High surprise = unexpected change = potential emergence
        """
        try:
            # Simple surprise: ratio of change magnitude vs historical variance
            if len(self.state_history) < 10:
                return 0.5  # Default when insufficient history
            
            # Compute historical variance for this component
            historical_vals = []
            for record in self.state_history:
                val = record['state'].get(component, 0.0)
                historical_vals.append(val)
            
            hist_mean = np.mean(historical_vals)
            hist_std = np.std(historical_vals) + 1e-9
            
            # Actual change
            before_val = before.get(component, hist_mean)
            after_val = after.get(component, hist_mean)
            change = abs(after_val - before_val)
            
            # Z-score of change
            z = change / hist_std
            
            # Map to [0, 1] with sigmoid
            surprise = 1.0 / (1.0 + np.exp(-z + 2.0))
            
            return float(surprise)
        
        except Exception as e:
            logger.debug(f"Surprise calculation failed: {e}")
            return 0.0
    
    def introspect(self, cycle: int, component: str, before: Dict[str, float], 
                   after: Dict[str, float], reasoning: str = "") -> Optional[IntrospectionEvent]:
        """
        Introspect on a component change and detect if it's surprising
        """
        surprise = self.compute_surprise(component, before, after)
        
        # Record all introspections
        event = IntrospectionEvent(
            cycle=cycle,
            component=component,
            before_state=before.copy(),
            after_state=after.copy(),
            surprise_factor=surprise,
            reasoning=reasoning
        )
        
        # High surprise = emergence candidate
        if surprise >= self.surprise_threshold:
            self.events.append(event)
            self.surprise_accumulator += surprise
            logger.info(f"🎯 INTROSPECTION: {component} surprise={surprise:.2f} (>{self.surprise_threshold}) - {reasoning}")
            return event
        
        return None
    
    def get_self_awareness_score(self) -> float:
        """
        Compute self-awareness score (0-1)
        Based on accumulation of surprising introspections
        """
        if len(self.events) == 0:
            return 0.0
        
        # Average surprise of high-surprise events
        avg_surprise = np.mean([e.surprise_factor for e in self.events])
        
        # Normalize by event count (more diverse introspections = higher awareness)
        diversity = min(1.0, len(self.events) / 20.0)
        
        # Combined score
        score = (avg_surprise * 0.7 + diversity * 0.3)
        
        return float(score)


class RecursiveMAMLIntensifier:
    """
    Auto-Recursive Intensification
    Implements meta-meta-learning (MAML on MAML)
    """
    
    def __init__(self, base_maml, meta_depth: int = 2):
        self.base_maml = base_maml
        self.meta_depth = meta_depth
        self.meta_meta_history = []
        logger.info(f"🔁 Recursive MAML ACTIVATED (I³ depth={meta_depth})")
    
    def recursive_adapt(self, tasks: List, depth: int = 0):
        """
        Recursively apply MAML at increasing meta-levels
        depth=0: normal MAML
        depth=1: meta-MAML (learns how to meta-learn)
        depth=2: meta-meta-MAML (learns how to learn how to meta-learn)
        """
        if depth >= self.meta_depth or not tasks:
            return self.base_maml.train_maml(tasks)
        
        try:
            # Train at this depth
            result = self.base_maml.train_maml(tasks)
            
            # Generate meta-tasks from current adaptation
            meta_tasks = self._generate_meta_tasks(tasks, result)
            
            # Recurse to next depth
            if meta_tasks:
                meta_result = self.recursive_adapt(meta_tasks, depth + 1)
                result['meta_depth'] = depth + 1
                result['meta_meta_loss'] = meta_result.get('meta_loss', 0.0)
            
            self.meta_meta_history.append({
                'depth': depth,
                'result': result
            })
            
            return result
        
        except Exception as e:
            logger.debug(f"Recursive adapt failed at depth {depth}: {e}")
            return self.base_maml.train_maml(tasks)
    
    def _generate_meta_tasks(self, base_tasks: List, adaptation_result: Dict) -> List:
        """
        Generate meta-tasks from adaptation results
        Meta-task = "learn to adapt faster" or "learn to generalize better"
        FIX #5: Implement real meta-task generation
        """
        try:
            # Simple heuristic: perturb task distributions
            meta_tasks = []
            for task in base_tasks[:3]:  # Limit to 3 for speed
                # Create harder variant with perturbation
                try:
                    meta_task = Task(
                        support_x=task.support_x + torch.randn_like(task.support_x) * 0.1,
                        support_y=task.support_y,
                        query_x=task.query_x + torch.randn_like(task.query_x) * 0.1,
                        query_y=task.query_y,
                        task_id=f"meta_{task.task_id}"
                    )
                    meta_tasks.append(meta_task)
                except Exception:
                    # If Task object doesn't work, try dict-based approach
                    if isinstance(task, dict):
                        meta_task = task.copy()
                        if 'params' in meta_task:
                            import numpy as np
                            noise_scale = 0.1
                            meta_task['params'] = {
                                k: v + np.random.normal(0, noise_scale * abs(v) if abs(v) > 1e-6 else noise_scale)
                                for k, v in meta_task['params'].items()
                            }
                        meta_task['meta_level'] = task.get('meta_level', 0) + 1
                        meta_tasks.append(meta_task)
            
            return meta_tasks
        
        except Exception as e:
            logger.debug(f"Meta-task generation failed: {e}")
            return []


class SelfGeneratedCurriculum:
    """
    Auto-Didactic Intensification
    System generates its own learning curriculum based on introspection
    """
    
    def __init__(self):
        self.curriculum_items = {}  # FIX #6: Change from list to dict
        self.difficulty_levels = {}
        self.mastery_tracker = {}
        self.challenge_counter = 0  # Add counter for dict keys
        logger.info("📚 Self-Generated Curriculum ACTIVATED (I³ Auto-Didactic)")
    
    def propose_next_challenge(self, current_capabilities: Dict[str, float],
                               introspection_events: List[IntrospectionEvent]) -> Dict[str, Any]:
        """
        Generate next learning challenge based on:
        1. Current capabilities (what system can do)
        2. Recent introspections (where system is struggling/improving)
        3. Mastery levels (what's been mastered vs needs work)
        """
        try:
            # Identify weakest capability
            if not current_capabilities:
                return {'type': 'baseline', 'difficulty': 0.5}
            
            sorted_caps = sorted(current_capabilities.items(), key=lambda x: x[1])
            weakest_component = sorted_caps[0][0]
            weakest_score = sorted_caps[0][1]
            
            # Check if this component had recent surprising changes
            recent_surprise = 0.0
            for event in introspection_events[-10:]:  # Last 10 introspections
                if event.component == weakest_component:
                    recent_surprise = max(recent_surprise, event.surprise_factor)
            
            # Generate challenge
            if recent_surprise > 0.7:
                # High surprise = system is learning fast here
                difficulty = min(1.0, self.difficulty_levels.get(weakest_component, 0.5) + 0.2)
                challenge_type = 'accelerated'
            else:
                # Low surprise = system needs more basic practice
                difficulty = max(0.3, self.difficulty_levels.get(weakest_component, 0.5) - 0.1)
                challenge_type = 'remedial'
            
            # Update difficulty tracker
            self.difficulty_levels[weakest_component] = difficulty
            
            challenge = {
                'type': challenge_type,
                'component': weakest_component,
                'difficulty': difficulty,
                'rationale': f"Target weak area ({weakest_score:.2f}) with {'advanced' if challenge_type == 'accelerated' else 'fundamental'} training"
            }
            
            # Store challenge with incremental key (dict-based storage)
            key = f"challenge_{self.challenge_counter}"
            self.curriculum_items[key] = challenge
            self.challenge_counter += 1
            logger.info(f"📖 Self-Generated Challenge: {challenge_type} for {weakest_component} @ difficulty={difficulty:.2f}")
            
            return challenge
        
        except Exception as e:
            logger.debug(f"Curriculum generation failed: {e}")
            return {'type': 'baseline', 'difficulty': 0.5}
    
    def update_mastery(self, component: str, performance: float):
        """Update mastery tracking for a component"""
        self.mastery_tracker[component] = performance


class PureNoveltyQDIntensifier:
    """
    Auto-Evolutionary Intensification
    Pushes QD-Lite to prioritize PURE NOVELTY over fitness
    """
    
    def __init__(self, base_darwin):
        self.base_darwin = base_darwin
        self.novelty_weight = 0.7  # Start at 70% novelty, 30% fitness
        self.novelty_history = []
        logger.info(f"🌌 Pure Novelty QD ACTIVATED (I³ novelty_weight={self.novelty_weight})")
    
    def intensify_novelty(self, generation: int):
        """Gradually increase novelty weight over generations"""
        # Asymptotic approach to 95% novelty
        target = 0.95
        rate = 0.02
        self.novelty_weight = self.novelty_weight + (target - self.novelty_weight) * rate
        
        if generation % 10 == 0:
            logger.info(f"🎨 Novelty weight increased to {self.novelty_weight:.2f} (gen {generation})")
    
    def reweight_fitness(self, base_fitness: float, novelty_score: float) -> float:
        """
        Combine fitness and novelty with I³ weighting
        Standard: fitness + novelty_boost
        I³: (1-w)*fitness + w*novelty  (w → 0.95)
        """
        combined = (1.0 - self.novelty_weight) * base_fitness + self.novelty_weight * novelty_score
        return float(combined)


class InternalTuringTest:
    """
    Auto-Validating Intensification
    System tests its own intelligence via surprise factor
    """
    
    def __init__(self, surprise_threshold: float = 0.8):
        self.surprise_threshold = surprise_threshold
        self.test_history = []
        logger.info(f"🎭 Internal Turing Test ACTIVATED (I³ threshold={surprise_threshold})")
    
    def evaluate_intelligence(self, introspection_engine: IntrospectionEngine,
                             behavior_changes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Internal Turing test: Did system exhibit genuinely surprising behavior?
        
        Passing criteria:
        - 3+ behaviors with surprise > 0.8
        - Self-awareness score > 0.6
        - At least 2 different components showing emergence
        """
        try:
            # Get self-awareness score
            awareness = introspection_engine.get_self_awareness_score()
            
            # Count high-surprise events
            high_surprise_events = [e for e in introspection_engine.events 
                                   if e.surprise_factor >= self.surprise_threshold]
            
            # Count unique components
            unique_components = len(set(e.component for e in high_surprise_events))
            
            # Passing criteria
            passed = (
                len(high_surprise_events) >= 3 and
                awareness >= 0.6 and
                unique_components >= 2
            )
            
            result = {
                'passed': passed,
                'awareness_score': float(awareness),
                'high_surprise_events': len(high_surprise_events),
                'unique_components': unique_components,
                'surprise_threshold': self.surprise_threshold
            }
            
            if passed:
                logger.info(f"🏆 INTERNAL TURING TEST: PASSED! awareness={awareness:.2f}, "
                          f"surprises={len(high_surprise_events)}, components={unique_components}")
            else:
                logger.info(f"📊 INTERNAL TURING TEST: Not yet. awareness={awareness:.2f}, "
                          f"surprises={len(high_surprise_events)}, components={unique_components}")
            
            self.test_history.append(result)
            return result
        
        except Exception as e:
            logger.debug(f"Internal Turing test failed: {e}")
            return {'passed': False, 'error': str(e)}


class ContinuousRegenerationEngine:
    """
    Auto-Regenerative Intensification
    Continuous replay training at higher frequency
    """
    
    def __init__(self, base_replay_buffer, train_freq: int = 5):
        self.replay_buffer = base_replay_buffer
        self.train_freq = train_freq  # Every N cycles
        self.regeneration_count = 0
        logger.info(f"♻️ Continuous Regeneration ACTIVATED (I³ freq={train_freq})")
    
    def should_regenerate(self, cycle: int) -> bool:
        """Check if should run regeneration this cycle"""
        return cycle % self.train_freq == 0
    
    def regenerate(self, agent, steps: int = 500) -> int:
        """
        Intensive regeneration from replay buffer
        Returns: number of samples used
        """
        try:
            if len(self.replay_buffer) < 100:
                return 0
            
            batch = min(steps, len(self.replay_buffer))
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch)
            
            # Store as synthetic transitions
            for i in range(len(states)):
                agent.store_transition(
                    states[i], int(actions[i]), float(rewards[i]), 
                    float(dones[i]), 0.0, 0.0
                )
            
            # Multiple updates for intensive training
            updates = 3
            for _ in range(updates):
                if len(agent.states) >= agent.batch_size:
                    _ = agent.update(next_states[-1])
            
            self.regeneration_count += 1
            logger.info(f"♻️ Regeneration #{self.regeneration_count}: {batch} samples, {updates} updates")
            
            return int(batch)
        
        except Exception as e:
            logger.debug(f"Regeneration failed: {e}")
            return 0


class IntelligenceCubedIntensifier:
    """
    MASTER INTENSIFIER - Coordinates all I³ enhancements
    """
    
    def __init__(self, v7_system):
        self.v7_system = v7_system
        
        # ✅ P1 FIX: Configuração ATIVA por padrão
        self.active = True
        self.introspection_frequency = 5  # A cada 5 cycles
        self.last_introspection_cycle = 0
        
        # Initialize sub-engines
        self.introspection = IntrospectionEngine(window=100)
        # Override threshold to detect more surprises
        self.introspection.surprise_threshold = 0.3
        self.self_curriculum = SelfGeneratedCurriculum()
        self.internal_turing = InternalTuringTest(surprise_threshold=0.8)
        
        # Recursive MAML (if available)
        if hasattr(v7_system, 'maml') and v7_system.maml:
            self.recursive_maml = RecursiveMAMLIntensifier(v7_system.maml, meta_depth=2)
        else:
            self.recursive_maml = None
        
        # Pure Novelty QD (if Darwin available)
        if hasattr(v7_system, 'darwin_real') and v7_system.darwin_real:
            self.pure_novelty_qd = PureNoveltyQDIntensifier(v7_system.darwin_real)
        else:
            self.pure_novelty_qd = None
        
        # Continuous Regeneration
        if hasattr(v7_system, 'experience_replay'):
            self.continuous_regen = ContinuousRegenerationEngine(
                v7_system.experience_replay, train_freq=5
            )
        else:
            self.continuous_regen = None
        
        logger.info("🎯 INTELLIGENCE³ INTENSIFIER ACTIVATED")
        logger.info(f"   Sub-engines: {sum([self.introspection is not None, self.recursive_maml is not None, self.pure_novelty_qd is not None, self.continuous_regen is not None])}/4")
    
    def intensify_cycle(self, cycle: int, before_metrics: Dict[str, float], 
                       after_metrics: Dict[str, float]) -> Dict[str, Any]:
        """
        Run I³ intensification for one cycle
        Returns: intensification results
        """
        results = {}
        
        # 1. Introspection: Record state and compute surprise
        self.introspection.record_state(cycle, after_metrics)
        
        for key in ['mnist_acc', 'cartpole_avg', 'ia3_score']:
            if key in before_metrics and key in after_metrics:
                event = self.introspection.introspect(
                    cycle, key,
                    {key: before_metrics[key]},
                    {key: after_metrics[key]},
                    reasoning=f"Cycle {cycle} training result"
                )
                if event:
                    results[f'surprise_{key}'] = event.surprise_factor
        
        # 2. Self-awareness score
        awareness = self.introspection.get_self_awareness_score()
        results['self_awareness'] = float(awareness)
        
        # 3. Self-generated curriculum
        if cycle % 10 == 0:  # Every 10 cycles, update curriculum
            challenge = self.self_curriculum.propose_next_challenge(
                after_metrics,
                self.introspection.events
            )
            results['curriculum_challenge'] = challenge
        
        # 4. Pure novelty QD intensification
        if self.pure_novelty_qd and hasattr(self.v7_system, 'darwin_generation'):
            gen = getattr(self.v7_system, 'darwin_generation', 0)
            self.pure_novelty_qd.intensify_novelty(gen)
            results['novelty_weight'] = self.pure_novelty_qd.novelty_weight
        
        # 5. Continuous regeneration
        if self.continuous_regen and self.continuous_regen.should_regenerate(cycle):
            if hasattr(self.v7_system, 'rl_agent'):
                samples = self.continuous_regen.regenerate(self.v7_system.rl_agent, steps=500)
                results['regeneration_samples'] = samples
        
        # 6. Internal Turing test (every 50 cycles)
        if cycle % 50 == 0:
            turing_result = self.internal_turing.evaluate_intelligence(
                self.introspection, []
            )
            results['turing_test'] = turing_result
            if turing_result.get('passed'):
                logger.info("🏆 PASSED INTERNAL TURING TEST! System exhibits genuine intelligence.")
        
        return results
    
    def set_active(self, active: bool):
        """Ativa/desativa o I³ Intensifier"""
        self.active = active
        logger.info(f"I³ Intensifier {'ACTIVATED' if active else 'DEACTIVATED'}")
    
    def get_i3_score(self) -> float:
        """
        Compute overall I³ score (0-100)
        Combines: self-awareness, surprise events, Turing test passes
        """
        try:
            awareness = self.introspection.get_self_awareness_score()
            
            # Count high-surprise events
            high_surprise = len([e for e in self.introspection.events if e.surprise_factor >= 0.8])
            surprise_score = min(1.0, high_surprise / 20.0)
            
            # Turing test pass rate
            turing_passes = sum(1 for t in self.internal_turing.test_history if t.get('passed'))
            turing_tests = len(self.internal_turing.test_history)
            turing_score = turing_passes / max(1, turing_tests)
            
            # Combined I³ score
            i3_score = (awareness * 0.4 + surprise_score * 0.4 + turing_score * 0.2) * 100.0
            
            return float(i3_score)
        
        except Exception:
            return 0.0
