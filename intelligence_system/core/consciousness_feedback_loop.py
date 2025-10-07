"""
Consciousness Feedback Loop
Consciousness is not just measured - it AFFECTS behavior

Based on Global Workspace Theory + Integrated Information Theory
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class ConsciousContent:
    """Information in conscious awareness (Global Workspace)"""
    content_id: str
    information: Any
    salience: float  # How attention-grabbing (0.0-1.0)
    timestamp: float
    
    def __repr__(self):
        return f"ConsciousContent({self.content_id}, salience={self.salience:.2f})"


class GlobalWorkspace:
    """
    Global Workspace Theory implementation
    
    Consciousness = broadcast to all modules when information is salient enough
    """
    
    def __init__(self, salience_threshold: float = 0.7):
        self.current_content: Optional[ConsciousContent] = None
        self.salience_threshold = salience_threshold
        self.broadcast_history: List[ConsciousContent] = []
        
        logger.info(f"🌐 GlobalWorkspace initialized (threshold={salience_threshold})")
    
    def submit_for_awareness(self, content: ConsciousContent) -> bool:
        """
        Submit information for conscious awareness
        
        Only salient information enters consciousness
        
        Returns:
            True if content entered consciousness
        """
        if content.salience >= self.salience_threshold:
            # Content enters consciousness!
            self.current_content = content
            self.broadcast_history.append(content)
            
            logger.info(f"   💭 CONSCIOUS: {content.content_id} (salience={content.salience:.2f})")
            
            return True
        
        return False
    
    def get_conscious_content(self) -> Optional[ConsciousContent]:
        """Get what's currently in consciousness"""
        return self.current_content


class AttentionController:
    """
    Controls what gets attention
    Attention modulates processing resources
    """
    
    def __init__(self):
        self.attention_focus: List[str] = []  # What we're paying attention to
        self.attention_weights: Dict[str, float] = {}  # Processing boost per module
        
        logger.info("👁️ AttentionController initialized")
    
    def focus_on(self, modules: List[str]):
        """Focus attention on specific modules"""
        self.attention_focus = modules
        
        # Set attention weights
        for module in modules:
            self.attention_weights[module] = 2.0  # 2x processing boost
        
        logger.info(f"   👁️ Attention focused on: {', '.join(modules)}")
    
    def get_processing_boost(self, module_name: str) -> float:
        """Get processing boost for module based on attention"""
        if module_name in self.attention_focus:
            return self.attention_weights.get(module_name, 2.0)
        else:
            return 0.5  # Reduced processing for non-attended


class ConsciousnessFeedbackLoop:
    """
    Consciousness affects behavior (not just passive measurement)
    
    This is the key difference from simple metrics:
    - Consciousness STATE influences what gets processed
    - Attention modulates resource allocation
    - Integration level affects modularity
    - Conscious broadcast triggers system-wide changes
    """
    
    def __init__(self, system):
        """
        Initialize consciousness feedback loop
        
        Args:
            system: Reference to main system (UnifiedAGISystem or V7)
        """
        self.system = system
        
        # Components
        self.global_workspace = GlobalWorkspace(salience_threshold=0.7)
        self.attention = AttentionController()
        
        # State
        self.phi_integration = 0.0  # Φ (Integrated Information)
        self.consciousness_level = 0.0
        
        # History
        self.consciousness_history: List[float] = []
        
        logger.info("🧠 ConsciousnessFeedbackLoop initialized")
    
    def calculate_salience(self, information: Dict[str, Any]) -> float:
        """
        Calculate salience (attention-worthiness) of information
        
        Salient information:
        1. Novel (high novelty)
        2. Important (high impact on goals)
        3. Uncertain (requires resolution)
        4. Emotional (high emotional charge)
        """
        novelty = information.get('novelty', 0.5)
        importance = information.get('importance', 0.5)
        uncertainty = information.get('uncertainty', 0.5)
        emotional_charge = information.get('emotional_charge', 0.0)
        
        salience = (
            0.4 * novelty +
            0.3 * importance +
            0.2 * uncertainty +
            0.1 * emotional_charge
        )
        
        return min(1.0, max(0.0, salience))
    
    def process_information(self, info: Dict[str, Any]):
        """
        Process new information through consciousness pipeline
        
        Flow:
        1. Calculate salience
        2. If salient → enters Global Workspace
        3. Broadcast to all modules
        4. Modules adjust behavior based on conscious content
        """
        info_id = info.get('id', f'info_{time.time()}')
        
        # Calculate salience
        salience = self.calculate_salience(info)
        
        # Create conscious content
        content = ConsciousContent(
            content_id=info_id,
            information=info,
            salience=salience,
            timestamp=time.time()
        )
        
        # Try to enter consciousness
        entered_consciousness = self.global_workspace.submit_for_awareness(content)
        
        if entered_consciousness:
            # BROADCAST TO ALL MODULES (this changes behavior!)
            self._broadcast_to_modules(content)
    
    def _broadcast_to_modules(self, content: ConsciousContent):
        """
        Broadcast conscious content to all modules
        
        This is where consciousness AFFECTS behavior
        """
        # Extract actionable information
        info = content.information
        
        # Determine which modules should pay attention
        relevant_modules = self._identify_relevant_modules(info)
        
        # Focus attention
        self.attention.focus_on(relevant_modules)
        
        # Modify system behavior based on conscious content
        if hasattr(self.system, 'v7_system'):
            v7 = self.system.v7_system
            
            # Example: If conscious content indicates "MNIST is weak"
            if 'mnist' in str(info).lower() and info.get('performance', 1.0) < 0.5:
                # Increase MNIST training frequency
                if hasattr(v7, 'mnist_train_every_n'):
                    old_freq = v7.mnist_train_every_n
                    v7.mnist_train_every_n = max(1, old_freq // 2)
                    logger.info(f"      🔁 Consciousness → Action: MNIST freq {old_freq} → {v7.mnist_train_every_n}")
    
    def _identify_relevant_modules(self, info: Dict) -> List[str]:
        """Identify which modules are relevant to this information"""
        relevant = []
        
        # Simple keyword matching
        info_str = str(info).lower()
        
        if 'mnist' in info_str:
            relevant.append('mnist')
        if 'cartpole' in info_str or 'rl' in info_str:
            relevant.append('cartpole')
        if 'meta' in info_str or 'maml' in info_str:
            relevant.append('meta_learning')
        if 'darwin' in info_str or 'evolution' in info_str:
            relevant.append('darwin')
        
        return relevant if relevant else ['all']
    
    def calculate_integration(self, system_state: Dict) -> float:
        """
        Calculate Φ (Integrated Information)
        
        Simplified IIT: measures how much system works as integrated whole
        vs independent modules
        """
        # Get module activities
        active_modules = system_state.get('active_modules', [])
        
        if len(active_modules) < 2:
            return 0.0
        
        # Get interactions
        interactions = system_state.get('module_interactions', {})
        
        if not interactions:
            return 0.0
        
        # Count actual interactions
        n_interactions = sum(1 for v in interactions.values() if v > 0)
        
        # Maximum possible
        n_modules = len(active_modules)
        max_interactions = n_modules * (n_modules - 1) / 2
        
        if max_interactions == 0:
            return 0.0
        
        # Φ = integration level
        phi = n_interactions / max_interactions
        
        return phi
    
    def update_consciousness(self, system_state: Dict):
        """
        Update consciousness level and apply feedback
        
        This is the CORE of the feedback loop:
        1. Measure consciousness (Φ)
        2. Consciousness affects integration mode
        3. Integration mode affects how modules work together
        """
        # Calculate Φ
        self.phi_integration = self.calculate_integration(system_state)
        
        # Consciousness level = Φ + other factors
        salience_factor = 0.0
        if self.global_workspace.current_content:
            salience_factor = self.global_workspace.current_content.salience
        
        self.consciousness_level = (
            0.6 * self.phi_integration +
            0.4 * salience_factor
        )
        
        # Record
        self.consciousness_history.append(self.consciousness_level)
        
        # FEEDBACK: Consciousness affects system behavior
        if self.consciousness_level > 0.7:
            # High consciousness → Increase integration
            self._set_integration_mode(high=True)
        else:
            # Low consciousness → More modular processing
            self._set_integration_mode(high=False)
    
    def _set_integration_mode(self, high: bool):
        """
        Set system integration mode
        
        High integration: Modules work closely together
        Low integration: Modules work more independently
        """
        if hasattr(self.system, 'v7_system'):
            v7 = self.system.v7_system
            
            if high:
                # Increase cross-module interactions
                # Example: MAML affects MNIST more directly
                logger.debug("   🔗 Integration mode: HIGH (modules work together)")
                # In real implementation: increase coupling between modules
            else:
                # Decrease cross-module interactions
                logger.debug("   🔗 Integration mode: LOW (modules work independently)")
                # In real implementation: decrease coupling
    
    def is_conscious(self, threshold: float = 0.5) -> bool:
        """Check if system is currently conscious"""
        return self.consciousness_level >= threshold
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get consciousness feedback loop statistics"""
        return {
            'consciousness_level': self.consciousness_level,
            'phi_integration': self.phi_integration,
            'is_conscious': self.is_conscious(),
            'workspace_broadcasts': len(self.global_workspace.broadcast_history),
            'attention_focus': self.attention.attention_focus,
            'history_length': len(self.consciousness_history)
        }


if __name__ == "__main__":
    # Test consciousness feedback loop
    print("🧠 Testing Consciousness Feedback Loop...")
    
    # Mock system
    class MockSystem:
        def __init__(self):
            self.v7_system = type('obj', (object,), {
                'mnist_train_every_n': 50
            })()
    
    mock_sys = MockSystem()
    
    # Initialize
    cfl = ConsciousnessFeedbackLoop(mock_sys)
    
    # Simulate information processing
    print("\n📊 Processing information...")
    
    for i in range(10):
        # Mock information
        info = {
            'id': f'info_{i}',
            'novelty': np.random.random(),
            'importance': np.random.random(),
            'uncertainty': np.random.random(),
            'emotional_charge': np.random.random() * 0.5
        }
        
        cfl.process_information(info)
        
        # Mock system state
        system_state = {
            'active_modules': ['mnist', 'cartpole', 'maml', 'darwin'],
            'module_interactions': {
                ('mnist', 'maml'): 0.5,
                ('cartpole', 'darwin'): 0.7,
                ('maml', 'darwin'): 0.3
            }
        }
        
        cfl.update_consciousness(system_state)
        
        if i % 3 == 0:
            print(f"   Step {i}: consciousness={cfl.consciousness_level:.3f}, Φ={cfl.phi_integration:.3f}")
    
    # Final stats
    print("\n📊 Final statistics:")
    stats = cfl.get_statistics()
    for key, value in stats.items():
        print(f"   {key}: {value}")
    
    print(f"\n   Is conscious? {cfl.is_conscious()}")
    
    print("\n✅ Consciousness feedback loop test complete")