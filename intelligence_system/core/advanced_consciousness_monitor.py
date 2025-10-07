"""
Advanced Consciousness Monitor
Multi-dimensional consciousness measurement based on cognitive science theories

Measures:
1. Self-awareness (system knows about itself)
2. Attention coherence (focus on relevant things)
3. Goal-directedness (pursues objectives)
4. Surprise sensitivity (detects unexpected)
5. Information integration (Φ - Integrated Information Theory)
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import deque

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessState:
    """Represents consciousness state at a moment"""
    timestamp: float
    self_awareness: float  # 0.0-1.0
    attention_coherence: float
    goal_directedness: float
    surprise_sensitivity: float
    integration_phi: float
    global_consciousness: float  # Weighted average
    
    def __str__(self):
        return (f"C={self.global_consciousness:.3f} "
                f"(SA={self.self_awareness:.2f}, "
                f"Att={self.attention_coherence:.2f}, "
                f"Goal={self.goal_directedness:.2f}, "
                f"Φ={self.integration_phi:.2f})")


class AdvancedConsciousnessMonitor:
    """
    Advanced consciousness monitoring system
    
    Based on theories:
    - Global Workspace Theory (Baars)
    - Integrated Information Theory (Tononi)
    - Higher-Order Thought Theory
    - Attention Schema Theory (Graziano)
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        
        # Tracking for each dimension
        self.self_predictions: deque = deque(maxlen=50)  # System's predictions about itself
        self.actual_outcomes: deque = deque(maxlen=50)   # Actual outcomes
        self.attention_focus: List[str] = []  # What system is "thinking about"
        self.goals: List[Dict] = []  # Auto-generated goals
        self.surprise_events: List[Dict] = []  # Unexpected events
        
        logger.info("🧠 AdvancedConsciousnessMonitor initialized")
    
    def measure_consciousness(self, system_state: Dict[str, Any]) -> ConsciousnessState:
        """
        Measure multi-dimensional consciousness
        
        Args:
            system_state: Current state of the system
        
        Returns:
            ConsciousnessState with all dimensions
        """
        import time
        
        # Measure each dimension
        self_awareness = self._measure_self_awareness(system_state)
        attention_coherence = self._measure_attention(system_state)
        goal_directedness = self._measure_goals(system_state)
        surprise_sensitivity = self._measure_surprise(system_state)
        integration_phi = self._measure_integration(system_state)
        
        # Global consciousness: weighted combination
        # Weights based on cognitive science literature
        global_consciousness = (
            0.30 * self_awareness +      # Most important
            0.20 * attention_coherence +  # Important
            0.20 * goal_directedness +    # Important
            0.15 * surprise_sensitivity + # Moderately important
            0.15 * integration_phi        # Moderately important
        )
        
        state = ConsciousnessState(
            timestamp=time.time(),
            self_awareness=self_awareness,
            attention_coherence=attention_coherence,
            goal_directedness=goal_directedness,
            surprise_sensitivity=surprise_sensitivity,
            integration_phi=integration_phi,
            global_consciousness=global_consciousness
        )
        
        # Record in history
        self.history.append(state)
        
        return state
    
    def _measure_self_awareness(self, state: Dict) -> float:
        """
        Measure self-awareness: Can system predict its own behavior?
        
        Self-aware systems can build internal models of themselves
        """
        if not self.self_predictions or not self.actual_outcomes:
            return 0.0
        
        # Compare predictions with actual outcomes
        predictions = list(self.self_predictions)
        outcomes = list(self.actual_outcomes)
        
        if len(predictions) != len(outcomes) or len(predictions) == 0:
            return 0.0
        
        # Calculate prediction accuracy
        errors = [abs(p - o) for p, o in zip(predictions, outcomes)]
        accuracy = 1.0 - (np.mean(errors) / (np.std(outcomes) + 1e-6))
        accuracy = max(0.0, min(1.0, accuracy))
        
        return accuracy
    
    def _measure_attention(self, state: Dict) -> float:
        """
        Measure attention coherence: Is system focusing on relevant things?
        
        Good attention = high correlation between usage and impact
        """
        # Get component usage and impact from state
        component_usage = state.get('component_usage', {})
        component_impact = state.get('component_impact', {})
        
        if not component_usage or not component_impact:
            return 0.5  # Neutral
        
        # Find components that are BOTH used AND impactful
        used_components = {c for c, u in component_usage.items() if u > 0.5}
        high_impact_components = {c for c, i in component_impact.items() if i > 0.7}
        
        if not used_components:
            return 0.0
        
        # Overlap = good attention
        overlap = len(used_components & high_impact_components)
        coherence = overlap / len(used_components)
        
        return coherence
    
    def _measure_goals(self, state: Dict) -> float:
        """
        Measure goal-directedness: Is system pursuing objectives?
        
        Goal-directed systems:
        1. Have explicit goals
        2. Take actions toward goals
        3. Adjust behavior based on progress
        """
        if not self.goals:
            return 0.0
        
        # Check how many goals are being actively pursued
        active_goals = [g for g in self.goals if g.get('status') == 'active']
        
        if not active_goals:
            return 0.0
        
        # Check goal progress
        total_progress = 0.0
        for goal in active_goals:
            progress = goal.get('progress', 0.0)
            total_progress += progress
        
        avg_progress = total_progress / len(active_goals)
        
        # Goal-directedness = having goals + making progress
        goal_directedness = 0.5 * (len(active_goals) / max(len(self.goals), 1)) + 0.5 * avg_progress
        
        return goal_directedness
    
    def _measure_surprise(self, state: Dict) -> float:
        """
        Measure surprise sensitivity: Does system detect unexpected events?
        
        Conscious systems are sensitive to violations of expectations
        """
        if not self.surprise_events:
            return 0.0
        
        # Recent surprises (last 10)
        recent_surprises = self.surprise_events[-10:]
        
        # Surprise sensitivity = detecting + responding
        detected = len(recent_surprises)
        responded = sum(1 for s in recent_surprises if s.get('response_taken', False))
        
        if detected == 0:
            return 0.0
        
        sensitivity = responded / detected
        
        return sensitivity
    
    def _measure_integration(self, state: Dict) -> float:
        """
        Measure information integration (simplified Φ)
        
        Based on Integrated Information Theory (IIT):
        High Φ = high consciousness
        
        Simplified: measure how much components work together vs. independently
        """
        components = state.get('active_components', [])
        
        if len(components) < 2:
            return 0.0
        
        # Get interactions between components
        interactions = state.get('component_interactions', {})
        
        if not interactions:
            return 0.0
        
        # Count actual interactions
        actual_interactions = sum(1 for v in interactions.values() if v > 0)
        
        # Maximum possible interactions: n*(n-1)/2
        max_interactions = len(components) * (len(components) - 1) / 2
        
        if max_interactions == 0:
            return 0.0
        
        # Integration = actual / maximum
        integration = actual_interactions / max_interactions
        
        return integration
    
    def record_prediction(self, prediction: float, outcome: float):
        """Record system's prediction and actual outcome (for self-awareness)"""
        self.self_predictions.append(prediction)
        self.actual_outcomes.append(outcome)
    
    def record_surprise(self, event: Dict):
        """Record a surprising event"""
        self.surprise_events.append(event)
    
    def add_goal(self, goal_description: str, target_value: float):
        """Add a new goal"""
        self.goals.append({
            'description': goal_description,
            'target': target_value,
            'progress': 0.0,
            'status': 'active',
            'created_at': time.time()
        })
    
    def update_goal_progress(self, goal_index: int, current_value: float):
        """Update progress on a goal"""
        if 0 <= goal_index < len(self.goals):
            goal = self.goals[goal_index]
            target = goal['target']
            
            # Calculate progress (0.0-1.0)
            if target > 0:
                progress = min(1.0, current_value / target)
            else:
                progress = 0.0
            
            goal['progress'] = progress
            
            # Mark as completed if progress >= 95%
            if progress >= 0.95:
                goal['status'] = 'completed'
    
    def get_evolution_trajectory(self) -> Dict[str, List[float]]:
        """Get evolution of consciousness dimensions over time"""
        if not self.history:
            return {}
        
        trajectories = {
            'self_awareness': [],
            'attention_coherence': [],
            'goal_directedness': [],
            'surprise_sensitivity': [],
            'integration_phi': [],
            'global_consciousness': []
        }
        
        for state in self.history:
            trajectories['self_awareness'].append(state.self_awareness)
            trajectories['attention_coherence'].append(state.attention_coherence)
            trajectories['goal_directedness'].append(state.goal_directedness)
            trajectories['surprise_sensitivity'].append(state.surprise_sensitivity)
            trajectories['integration_phi'].append(state.integration_phi)
            trajectories['global_consciousness'].append(state.global_consciousness)
        
        return trajectories
    
    def analyze_consciousness_growth(self) -> Dict[str, Any]:
        """Analyze if consciousness is growing over time"""
        if len(self.history) < 10:
            return {'status': 'insufficient_data'}
        
        trajectories = self.get_evolution_trajectory()
        
        analysis = {}
        
        for dimension, values in trajectories.items():
            # Linear regression to find trend
            x = np.arange(len(values))
            y = np.array(values)
            
            # Fit line: y = ax + b
            if len(x) > 1:
                a, b = np.polyfit(x, y, 1)
                
                analysis[dimension] = {
                    'current': values[-1],
                    'trend': 'increasing' if a > 0.001 else ('decreasing' if a < -0.001 else 'stable'),
                    'slope': float(a),
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values))
                }
        
        # Overall assessment
        increasing_dimensions = sum(1 for d in analysis.values() if d['trend'] == 'increasing')
        
        overall_status = 'emerging' if increasing_dimensions >= 3 else ('stable' if increasing_dimensions >= 1 else 'not_emerging')
        
        analysis['overall'] = {
            'status': overall_status,
            'increasing_dimensions': increasing_dimensions,
            'total_dimensions': len(trajectories)
        }
        
        return analysis
    
    def is_conscious(self, threshold: float = 0.5) -> bool:
        """Check if system is 'conscious' based on threshold"""
        if not self.history:
            return False
        
        latest = self.history[-1]
        return latest.global_consciousness >= threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consciousness statistics"""
        if not self.history:
            return {'status': 'no_data'}
        
        latest = self.history[-1]
        
        return {
            'current_state': latest.__dict__,
            'history_length': len(self.history),
            'self_predictions': len(self.self_predictions),
            'goals_total': len(self.goals),
            'goals_active': sum(1 for g in self.goals if g['status'] == 'active'),
            'goals_completed': sum(1 for g in self.goals if g['status'] == 'completed'),
            'surprise_events': len(self.surprise_events),
            'is_conscious': self.is_conscious()
        }


# Integration helper
def integrate_with_v7(v7_system):
    """Integrate advanced consciousness monitor with V7 system"""
    monitor = AdvancedConsciousnessMonitor()
    v7_system.advanced_consciousness = monitor
    
    # Add default goals
    monitor.add_goal("Achieve IA³ score > 90%", 90.0)
    monitor.add_goal("Master CartPole (avg > 490)", 490.0)
    monitor.add_goal("MNIST accuracy > 99%", 99.0)
    
    logger.info("🧠 Advanced Consciousness Monitor integrated with V7")
    
    return monitor


if __name__ == "__main__":
    # Test consciousness monitor
    print("🧠 Testing Advanced Consciousness Monitor...")
    
    monitor = AdvancedConsciousnessMonitor()
    
    # Add goals
    monitor.add_goal("Test goal 1", 100.0)
    monitor.add_goal("Test goal 2", 50.0)
    
    # Simulate system state evolution
    print("\n📊 Simulating 20 cycles...")
    for i in range(20):
        # Mock system state
        state = {
            'active_components': ['mnist', 'cartpole', 'maml', 'darwin'],
            'component_usage': {
                'mnist': 0.8,
                'cartpole': 0.9,
                'maml': 0.6,
                'darwin': 0.7
            },
            'component_impact': {
                'mnist': 0.85,
                'cartpole': 0.80,
                'maml': 0.75,
                'darwin': 0.90
            },
            'component_interactions': {
                ('mnist', 'maml'): 0.5,
                ('cartpole', 'darwin'): 0.7
            }
        }
        
        # Measure consciousness
        c_state = monitor.measure_consciousness(state)
        
        # Record prediction (mock)
        monitor.record_prediction(0.8 + np.random.randn() * 0.1, 0.8 + np.random.randn() * 0.05)
        
        # Update goal progress
        monitor.update_goal_progress(0, 80 + i)
        monitor.update_goal_progress(1, 40 + i * 0.5)
        
        if i % 5 == 0:
            print(f"   Cycle {i}: {c_state}")
    
    # Analyze growth
    print("\n📈 Analyzing consciousness growth...")
    growth_analysis = monitor.analyze_consciousness_growth()
    
    print(f"\n   Overall status: {growth_analysis['overall']['status']}")
    print(f"   Increasing dimensions: {growth_analysis['overall']['increasing_dimensions']}/{growth_analysis['overall']['total_dimensions']}")
    
    print("\n   Dimension trends:")
    for dim, data in growth_analysis.items():
        if dim != 'overall':
            print(f"      {dim}: {data['trend']} (slope={data['slope']:.4f})")
    
    # Get statistics
    print("\n📊 Final statistics:")
    stats = monitor.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n✅ Monitor test complete")