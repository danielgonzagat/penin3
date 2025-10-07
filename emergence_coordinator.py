#!/usr/bin/env python3
"""
Emergence Coordinator - Sistema de Emergência Não-Planejada
Coordena comportamentos emergentes entre conglomerados para criar inteligência coletiva
"""

import os
import json
import torch
import torch.nn as nn
import numpy as np
import random
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import argparse

# Import integrated systems
from meta_learning_integration import AdaptiveMetaSystem
from teis_gym_integration import TEISGymIntegration
from neural_farm_ia3_integration import EvolutionaryArchitectureSearch

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger("EMERGENCE_COORDINATOR")

class EmergenceDetector(nn.Module):
    """Neural network to detect emergent patterns across systems"""

    def __init__(self, input_dim: int = 50, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4)
        )

        self.emergence_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()  # Emergence probability
        )

        self.pattern_head = nn.Sequential(
            nn.Linear(hidden_dim // 4, 32),
            nn.ReLU(),
            nn.Linear(32, 10)  # Emergence pattern type
        )

    def forward(self, x):
        encoded = self.encoder(x)
        emergence_prob = self.emergence_head(encoded)
        pattern_type = torch.softmax(self.pattern_head(encoded), dim=-1)
        return emergence_prob, pattern_type, encoded

class EmergentBehaviorPromoter:
    """Promotes and amplifies detected emergent behaviors"""

    def __init__(self):
        self.emergent_patterns = []
        self.promotion_history = []
        self.successful_emergences = []

    def promote_emergence(self, emergence_type: str, systems_state: Dict[str, Any],
                         emergence_strength: float) -> Dict[str, Any]:
        """Promote a detected emergent behavior across systems"""

        promotion_actions = {
            'coordinated_excellence': self._promote_coordinated_excellence,
            'system_diversification': self._promote_system_diversification,
            'adaptive_resonance': self._promote_adaptive_resonance,
            'collective_learning': self._promote_collective_learning,
            'creative_synthesis': self._promote_creative_synthesis,
            'self_organization': self._promote_self_organization,
            'meta_adaptation': self._promote_meta_adaptation,
            'cross_domain_transfer': self._promote_cross_domain_transfer,
            'emergent_optimization': self._promote_emergent_optimization,
            'intelligence_amplification': self._promote_intelligence_amplification
        }

        promoter = promotion_actions.get(emergence_type, self._promote_generic)
        return promoter(systems_state, emergence_strength)

    def _promote_coordinated_excellence(self, systems_state, strength):
        """Promote simultaneous high performance across all systems"""
        return {
            'actions': [
                {'system': 'all', 'action': 'increase_resources', 'magnitude': strength * 2.0},
                {'system': 'teis', 'action': 'boost_exploration', 'magnitude': strength * 1.5},
                {'system': 'neural_farm', 'action': 'accelerate_evolution', 'magnitude': strength * 1.5},
                {'system': 'meta_learner', 'action': 'optimize_parameters', 'magnitude': strength}
            ],
            'description': 'Coordinated excellence promotion',
            'expected_impact': 'Synchronized high performance'
        }

    def _promote_system_diversification(self, systems_state, strength):
        """Promote specialization and diversity among systems"""
        return {
            'actions': [
                {'system': 'teis', 'action': 'increase_task_variety', 'magnitude': strength * 1.8},
                {'system': 'neural_farm', 'action': 'diversify_population', 'magnitude': strength * 1.8},
                {'system': 'meta_learner', 'action': 'encourage_specialization', 'magnitude': strength * 1.2}
            ],
            'description': 'System diversification promotion',
            'expected_impact': 'Specialized system roles'
        }

    def _promote_adaptive_resonance(self, systems_state, strength):
        """Promote resonant adaptation patterns"""
        return {
            'actions': [
                {'system': 'meta_learner', 'action': 'amplify_resonance', 'magnitude': strength * 2.5},
                {'system': 'all', 'action': 'synchronize_learning_rates', 'magnitude': strength * 1.5},
                {'system': 'emergence_detector', 'action': 'focus_on_patterns', 'magnitude': strength}
            ],
            'description': 'Adaptive resonance promotion',
            'expected_impact': 'Harmonic system adaptation'
        }

    def _promote_collective_learning(self, systems_state, strength):
        """Promote learning that benefits all systems"""
        return {
            'actions': [
                {'system': 'all', 'action': 'share_knowledge', 'magnitude': strength * 2.2},
                {'system': 'meta_learner', 'action': 'optimize_collective', 'magnitude': strength * 1.8},
                {'system': 'neural_farm', 'action': 'collaborative_evolution', 'magnitude': strength * 1.5}
            ],
            'description': 'Collective learning promotion',
            'expected_impact': 'Mutual system improvement'
        }

    def _promote_creative_synthesis(self, systems_state, strength):
        """Promote synthesis of novel solutions"""
        return {
            'actions': [
                {'system': 'neural_farm', 'action': 'creative_mutation', 'magnitude': strength * 2.8},
                {'system': 'teis', 'action': 'innovative_exploration', 'magnitude': strength * 2.2},
                {'system': 'meta_learner', 'action': 'encourage_creativity', 'magnitude': strength * 1.8}
            ],
            'description': 'Creative synthesis promotion',
            'expected_impact': 'Novel solution generation'
        }

    def _promote_self_organization(self, systems_state, strength):
        """Promote self-organizing behaviors"""
        return {
            'actions': [
                {'system': 'all', 'action': 'reduce_external_control', 'magnitude': strength * 2.0},
                {'system': 'emergence_detector', 'action': 'detect_self_org', 'magnitude': strength * 1.5},
                {'system': 'meta_learner', 'action': 'facilitate_emergence', 'magnitude': strength * 1.2}
            ],
            'description': 'Self-organization promotion',
            'expected_impact': 'Autonomous system behavior'
        }

    def _promote_meta_adaptation(self, systems_state, strength):
        """Promote meta-level adaptation"""
        return {
            'actions': [
                {'system': 'meta_learner', 'action': 'self_modify', 'magnitude': strength * 3.0},
                {'system': 'all', 'action': 'adaptive_meta_control', 'magnitude': strength * 2.2},
                {'system': 'emergence_detector', 'action': 'meta_pattern_recognition', 'magnitude': strength * 1.8}
            ],
            'description': 'Meta-adaptation promotion',
            'expected_impact': 'Higher-order learning'
        }

    def _promote_cross_domain_transfer(self, systems_state, strength):
        """Promote knowledge transfer across domains"""
        return {
            'actions': [
                {'system': 'all', 'action': 'enable_knowledge_transfer', 'magnitude': strength * 2.5},
                {'system': 'teis', 'action': 'gym_to_arch_transfer', 'magnitude': strength * 2.0},
                {'system': 'neural_farm', 'action': 'arch_to_meta_transfer', 'magnitude': strength * 2.0}
            ],
            'description': 'Cross-domain transfer promotion',
            'expected_impact': 'Inter-system knowledge flow'
        }

    def _promote_emergent_optimization(self, systems_state, strength):
        """Promote optimization through emergence"""
        return {
            'actions': [
                {'system': 'all', 'action': 'emergent_optimization', 'magnitude': strength * 2.8},
                {'system': 'meta_learner', 'action': 'optimize_emergence', 'magnitude': strength * 2.2},
                {'system': 'emergence_detector', 'action': 'guide_optimization', 'magnitude': strength * 1.8}
            ],
            'description': 'Emergent optimization promotion',
            'expected_impact': 'Optimization through emergence'
        }

    def _promote_intelligence_amplification(self, systems_state, strength):
        """Promote intelligence amplification effects"""
        return {
            'actions': [
                {'system': 'all', 'action': 'intelligence_amplification', 'magnitude': strength * 3.2},
                {'system': 'meta_learner', 'action': 'amplify_intelligence', 'magnitude': strength * 2.8},
                {'system': 'emergence_detector', 'action': 'intelligence_focus', 'magnitude': strength * 2.2},
                {'system': 'neural_farm', 'action': 'cognitive_enhancement', 'magnitude': strength * 2.0}
            ],
            'description': 'Intelligence amplification promotion',
            'expected_impact': 'Super-linear intelligence growth'
        }

    def _promote_generic(self, systems_state, strength):
        """Generic emergence promotion"""
        return {
            'actions': [
                {'system': 'all', 'action': 'general_emergence_boost', 'magnitude': strength * 1.5}
            ],
            'description': 'Generic emergence promotion',
            'expected_impact': 'General system enhancement'
        }

class EmergenceCoordinator:
    """Coordinates emergence detection and promotion across all systems"""

    def __init__(self, base_dir: str = './emergence_coordination'):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        # Initialize emergence detection
        self.emergence_detector = EmergenceDetector()
        self.emergence_optimizer = torch.optim.Adam(self.emergence_detector.parameters(), lr=0.001)

        # Initialize emergence promoter
        self.promoter = EmergentBehaviorPromoter()

        # State tracking
        self.emergence_history = deque(maxlen=100)
        self.system_interactions = []
        self.emergent_events = []

        # Integrated meta-system
        self.meta_system = None

        logger.info("Emergence Coordinator initialized")

    def initialize_meta_system(self):
        """Initialize the meta-learning system"""
        self.meta_system = AdaptiveMetaSystem(
            base_dir=os.path.join(self.base_dir, 'meta_system')
        )
        self.meta_system.initialize_subsystems()
        logger.info("Meta-system initialized for emergence coordination")

    def detect_and_promote_emergence(self) -> Dict[str, Any]:
        """Main emergence detection and promotion cycle"""

        if not self.meta_system:
            self.initialize_meta_system()

        # Run meta-learning iteration to get system states
        meta_results = self.meta_system.run_meta_iteration()

        # Extract system states for emergence detection
        system_states = self._extract_system_states(meta_results)

        # Detect emergence
        emergence_info = self._detect_emergence(system_states)

        # Promote detected emergence
        if emergence_info['detected']:
            promotion_results = self._promote_emergence(emergence_info, system_states)
        else:
            promotion_results = {'promoted': False, 'reason': 'No emergence detected'}

        # Record emergence event
        emergence_event = {
            'timestamp': datetime.now().isoformat(),
            'system_states': system_states,
            'emergence_detection': emergence_info,
            'promotion_results': promotion_results,
            'meta_results': meta_results
        }

        self.emergent_events.append(emergence_event)
        self.emergence_history.append(emergence_info)

        # Save emergence data
        emergence_file = os.path.join(self.base_dir, 'emergence_events.jsonl')
        with open(emergence_file, 'a') as f:
            json.dump(emergence_event, f)
            f.write('\n')

        logger.info(f"Emergence cycle completed. Detected: {emergence_info['detected']}")
        if emergence_info['detected']:
            logger.info(f"Promoted emergence type: {emergence_info['type']} (strength: {emergence_info['strength']:.3f})")

        return emergence_event

    def _extract_system_states(self, meta_results: Dict) -> Dict[str, Any]:
        """Extract comprehensive system states for emergence detection"""
        system_state = meta_results.get('system_state', {})

        # Create rich feature vector for emergence detection
        features = []

        # Basic performance metrics
        features.extend([
            system_state.get('teis_fitness', 0.0) / 100.0,
            system_state.get('gym_performance', 0.0) / 500.0,
            system_state.get('architecture_fitness', 0.0) / 100.0,
            system_state.get('learning_efficiency', 0.0),
            system_state.get('adaptation_rate', 0.0),
            system_state.get('system_diversity', 0.0),
            system_state.get('stagnation_level', 0.0),
            len(system_state.get('emergent_behaviors', [])) / 10.0
        ])

        # Learning parameters from meta-learner
        learning_params = meta_results.get('learning_params', {})
        features.extend([
            learning_params.get('learning_rate', 0.001) * 1000,  # Scale up
            learning_params.get('exploration_rate', 0.1) * 10,
            learning_params.get('mutation_rate', 0.01) * 100,
            learning_params.get('selection_pressure', 0.1) * 10,
            learning_params.get('crossover_rate', 0.5) * 2,
            learning_params.get('architecture_complexity', 1.0),
            learning_params.get('task_difficulty', 0.1) * 10,
            learning_params.get('reward_shaping', 0.5) * 2
        ])

        # Historical trends
        if len(self.emergence_history) >= 3:
            recent = list(self.emergence_history)[-3:]
            features.extend([
                np.mean([h.get('strength', 0) for h in recent]),  # Recent emergence strength
                np.std([h.get('strength', 0) for h in recent]),   # Emergence variability
                sum(1 for h in recent if h.get('detected', False)) / 3.0  # Emergence frequency
            ])
        else:
            features.extend([0.0, 0.0, 0.0])

        # Interaction patterns
        features.extend([
            system_state.get('teis_fitness', 0.0) * system_state.get('gym_performance', 0.0) / 10000.0,  # TEIS-Gym interaction
            system_state.get('architecture_fitness', 0.0) * system_state.get('learning_efficiency', 0.0),  # Arch-Learning interaction
            system_state.get('system_diversity', 0.0) * system_state.get('adaptation_rate', 0.0),  # Diversity-Adaptation interaction
            len(system_state.get('emergent_behaviors', [])) * system_state.get('creativity_score', 1.0)  # Emergence-Creativity interaction
        ])

        # Ensure we have exactly 50 features
        while len(features) < 50:
            features.append(0.0)
        features = features[:50]

        return {
            'features': features,
            'performance_metrics': {
                'teis_fitness': system_state.get('teis_fitness', 0.0),
                'gym_performance': system_state.get('gym_performance', 0.0),
                'architecture_fitness': system_state.get('architecture_fitness', 0.0),
                'overall_performance': meta_results.get('overall_performance', 0.0)
            },
            'learning_params': learning_params,
            'system_state': system_state
        }

    def _detect_emergence(self, system_states: Dict[str, Any]) -> Dict[str, Any]:
        """Detect emergence using neural network"""
        features = torch.tensor(system_states['features'], dtype=torch.float32)

        with torch.no_grad():
            emergence_prob, pattern_logits, encoded_features = self.emergence_detector(features.unsqueeze(0))

            emergence_prob_val = emergence_prob.item()
            pattern_probs = torch.softmax(pattern_logits, dim=-1)[0]

            # Determine if emergence is detected
            detected = emergence_prob_val > 0.7  # High confidence threshold

            if detected:
                # Get pattern type
                pattern_idx = torch.argmax(pattern_probs).item()
                pattern_types = [
                    'coordinated_excellence', 'system_diversification', 'adaptive_resonance',
                    'collective_learning', 'creative_synthesis', 'self_organization',
                    'meta_adaptation', 'cross_domain_transfer', 'emergent_optimization',
                    'intelligence_amplification'
                ]
                pattern_type = pattern_types[pattern_idx]

                emergence_info = {
                    'detected': True,
                    'type': pattern_type,
                    'strength': emergence_prob_val,
                    'pattern_confidence': pattern_probs[pattern_idx].item(),
                    'features': encoded_features.squeeze().tolist()
                }
            else:
                emergence_info = {
                    'detected': False,
                    'strength': emergence_prob_val,
                    'reason': 'Low emergence probability'
                }

        return emergence_info

    def _promote_emergence(self, emergence_info: Dict, system_states: Dict) -> Dict[str, Any]:
        """Promote detected emergence"""
        emergence_type = emergence_info['type']
        emergence_strength = emergence_info['strength']

        # Get promotion actions
        promotion_plan = self.promoter.promote_emergence(
            emergence_type, system_states['system_state'], emergence_strength
        )

        # Apply promotion actions (in a real system, this would modify the running systems)
        logger.info(f"Applying emergence promotion: {promotion_plan['description']}")
        logger.info(f"Expected impact: {promotion_plan['expected_impact']}")

        # Record promotion
        self.promoter.promotion_history.append({
            'emergence_type': emergence_type,
            'emergence_strength': emergence_strength,
            'promotion_plan': promotion_plan,
            'timestamp': datetime.now().isoformat()
        })

        return {
            'promoted': True,
            'emergence_type': emergence_type,
            'promotion_plan': promotion_plan,
            'expected_impact': promotion_plan['expected_impact']
        }

    def get_emergence_statistics(self) -> Dict[str, Any]:
        """Get statistics about emergence detection and promotion"""
        total_events = len(self.emergent_events)
        detected_events = sum(1 for e in self.emergent_events if e['emergence_detection']['detected'])

        if total_events > 0:
            detection_rate = detected_events / total_events
        else:
            detection_rate = 0.0

        # Emergence type distribution
        emergence_types = {}
        for event in self.emergent_events:
            if event['emergence_detection']['detected']:
                etype = event['emergence_detection']['type']
                emergence_types[etype] = emergence_types.get(etype, 0) + 1

        # Performance correlation with emergence
        emergence_periods = []
        non_emergence_periods = []

        for event in self.emergent_events:
            perf = event['meta_results'].get('overall_performance', 0)
            if event['emergence_detection']['detected']:
                emergence_periods.append(perf)
            else:
                non_emergence_periods.append(perf)

        return {
            'total_events': total_events,
            'detected_events': detected_events,
            'detection_rate': detection_rate,
            'emergence_types': emergence_types,
            'avg_performance_with_emergence': np.mean(emergence_periods) if emergence_periods else 0.0,
            'avg_performance_without_emergence': np.mean(non_emergence_periods) if non_emergence_periods else 0.0,
            'emergence_advantage': (np.mean(emergence_periods) - np.mean(non_emergence_periods)) if emergence_periods and non_emergence_periods else 0.0
        }

def main():
    parser = argparse.ArgumentParser(description="Emergence Coordinator - Unplanned Emergence System")
    parser.add_argument('--cycles', type=int, default=5, help='Number of emergence cycles')
    parser.add_argument('--base-dir', type=str, default='./emergence_coordination', help='Base directory')

    args = parser.parse_args()

    logger.info(f"Starting Emergence Coordinator: {args.cycles} cycles")

    coordinator = EmergenceCoordinator(base_dir=args.base_dir)

    for cycle in range(args.cycles):
        try:
            logger.info(f"=== Emergence Cycle {cycle + 1}/{args.cycles} ===")
            emergence_event = coordinator.detect_and_promote_emergence()
            logger.info(f"Cycle {cycle + 1} completed")
        except KeyboardInterrupt:
            logger.info("Emergence coordination interrupted by user")
            break
        except Exception as e:
            logger.error(f"Error in emergence cycle {cycle + 1}: {e}")
            break

    # Final statistics
    stats = coordinator.get_emergence_statistics()
    logger.info("Emergence coordination completed!")
    logger.info(f"Detection rate: {stats['detection_rate']:.2%}")
    logger.info(f"Emergence advantage: {stats['emergence_advantage']:.3f}")
    logger.info(f"Most common emergence: {max(stats['emergence_types'].items(), key=lambda x: x[1], default=('none', 0))}")

if __name__ == '__main__':
    main()
