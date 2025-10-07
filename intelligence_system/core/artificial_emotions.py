"""
Artificial Emotions System
Uses "emotions" as heuristic signals for decision-making

Inspired by:
- Damasio's Somatic Marker Hypothesis
- Picard's Affective Computing
- Emotion as fast heuristics (System 1 vs System 2)
"""

import logging
import numpy as np
from typing import Dict, List, Any
from dataclasses import dataclass
from collections import deque
import time

logger = logging.getLogger(__name__)


@dataclass
class EmotionalState:
    """Represents emotional state at a moment"""
    timestamp: float
    curiosity: float      # 0.0-1.0 (exploration drive)
    fear: float          # 0.0-1.0 (risk aversion)
    joy: float           # 0.0-1.0 (reward satisfaction)
    frustration: float   # 0.0-1.0 (stagnation indicator)
    confidence: float    # 0.0-1.0 (self-efficacy)
    surprise: float      # 0.0-1.0 (expectation violation)
    
    def __str__(self):
        return (f"Emotions(curiosity={self.curiosity:.2f}, joy={self.joy:.2f}, "
                f"confidence={self.confidence:.2f}, frustration={self.frustration:.2f})")


class ArtificialEmotions:
    """
    Emotional system for AI
    
    Emotions serve as:
    1. Fast heuristics for decision-making
    2. Motivation signals (what to do next)
    3. Learning rate modulators
    4. Exploration bias
    5. Social signals (in multi-agent systems)
    """
    
    def __init__(self):
        # Current emotional state
        self.emotions = {
            'curiosity': 0.5,      # Moderate initial curiosity
            'fear': 0.1,           # Low initial fear
            'joy': 0.5,            # Neutral joy
            'frustration': 0.0,    # No frustration initially
            'confidence': 0.5,     # Moderate confidence
            'surprise': 0.0        # No surprise initially
        }
        
        # History
        self.history: deque = deque(maxlen=1000)
        
        # Event counts
        self.event_counts = {
            'successes': 0,
            'failures': 0,
            'surprises': 0,
            'discoveries': 0
        }
        
        logger.info("❤️ ArtificialEmotions initialized")
    
    def update_emotions(self, events: List[Dict[str, Any]]):
        """
        Update emotional state based on events
        
        Args:
            events: List of events that occurred
        """
        for event in events:
            event_type = event.get('type')
            
            if event_type == 'new_discovery':
                # Exciting! Increase curiosity and joy
                self.emotions['curiosity'] += 0.1
                self.emotions['joy'] += 0.15
                self.emotions['confidence'] += 0.05
                self.event_counts['discoveries'] += 1
                logger.debug("   🎉 New discovery! Emotions: curiosity↑, joy↑")
            
            elif event_type == 'repeated_failure':
                # Frustrating! Decrease confidence, increase frustration
                self.emotions['frustration'] += 0.15
                self.emotions['confidence'] -= 0.1
                self.emotions['fear'] += 0.05
                self.event_counts['failures'] += 1
                logger.debug("   😤 Repeated failure. Emotions: frustration↑, confidence↓")
            
            elif event_type == 'unexpected_success':
                # Surprising and joyful!
                self.emotions['joy'] += 0.2
                self.emotions['surprise'] += 0.3
                self.emotions['confidence'] += 0.15
                self.event_counts['successes'] += 1
                logger.debug("   🎊 Unexpected success! Emotions: joy↑↑, surprise↑")
            
            elif event_type == 'near_miss':
                # Almost there! Increase curiosity and slight frustration
                self.emotions['curiosity'] += 0.15
                self.emotions['frustration'] += 0.05
                logger.debug("   🎯 Near miss. Emotions: curiosity↑")
            
            elif event_type == 'catastrophic_failure':
                # Scary! High fear, low confidence
                self.emotions['fear'] += 0.3
                self.emotions['confidence'] -= 0.2
                self.emotions['frustration'] += 0.2
                self.event_counts['failures'] += 1
                logger.debug("   ⚠️ Catastrophic failure. Emotions: fear↑↑")
            
            elif event_type == 'stagnation':
                # Boring! High frustration, low joy
                self.emotions['frustration'] += 0.2
                self.emotions['joy'] -= 0.1
                self.emotions['curiosity'] += 0.1  # Try something new
                logger.debug("   😐 Stagnation. Emotions: frustration↑, curiosity↑")
            
            elif event_type == 'breakthrough':
                # Amazing! All positive emotions up
                self.emotions['joy'] += 0.3
                self.emotions['confidence'] += 0.2
                self.emotions['curiosity'] += 0.1
                self.emotions['frustration'] = max(0, self.emotions['frustration'] - 0.2)
                self.event_counts['discoveries'] += 1
                logger.debug("   🌟 Breakthrough! Emotions: joy↑↑↑")
        
        # Clamp emotions to [0, 1]
        for emotion in self.emotions:
            self.emotions[emotion] = max(0.0, min(1.0, self.emotions[emotion]))
        
        # Apply decay over time (emotions fade)
        self._apply_decay()
        
        # Record state
        self.history.append(EmotionalState(
            timestamp=time.time(),
            curiosity=self.emotions['curiosity'],
            fear=self.emotions['fear'],
            joy=self.emotions['joy'],
            frustration=self.emotions['frustration'],
            confidence=self.emotions['confidence'],
            surprise=self.emotions['surprise']
        ))
    
    def _apply_decay(self, decay_rate: float = 0.02):
        """Emotions decay toward neutral over time"""
        for emotion, value in self.emotions.items():
            # Decay toward 0.5 (neutral)
            if value > 0.5:
                self.emotions[emotion] = max(0.5, value - decay_rate)
            elif value < 0.5:
                self.emotions[emotion] = min(0.5, value + decay_rate)
    
    def get_exploration_bias(self) -> float:
        """
        How much to explore vs exploit
        
        Returns:
            Exploration bias (0.0-1.0, higher = more exploration)
        """
        # High curiosity + low fear + high frustration = explore more
        exploration = (
            0.5 * self.emotions['curiosity'] +
            0.3 * (1.0 - self.emotions['fear']) +
            0.2 * self.emotions['frustration']
        )
        
        return max(0.0, min(1.0, exploration))
    
    def get_learning_rate_bias(self) -> float:
        """
        How to adjust learning rate based on emotions
        
        Returns:
            LR multiplier (0.5-2.0)
        """
        # High frustration = increase LR (try something radically different)
        # High confidence = decrease LR (fine-tune)
        # High fear = decrease LR (be cautious)
        
        if self.emotions['frustration'] > 0.7:
            return 1.8  # Increase LR significantly
        elif self.emotions['confidence'] > 0.8:
            return 0.7  # Decrease LR (refine)
        elif self.emotions['fear'] > 0.7:
            return 0.6  # Decrease LR (be safe)
        else:
            return 1.0  # Keep LR
    
    def get_risk_tolerance(self) -> float:
        """
        How much risk to take
        
        Returns:
            Risk tolerance (0.0-1.0)
        """
        # High confidence + low fear + high curiosity = take risks
        # High fear + low confidence = be conservative
        
        risk_tolerance = (
            0.4 * self.emotions['confidence'] +
            0.4 * (1.0 - self.emotions['fear']) +
            0.2 * self.emotions['curiosity']
        )
        
        return max(0.0, min(1.0, risk_tolerance))
    
    def should_try_new_strategy(self) -> bool:
        """Decide if should try completely new strategy"""
        # High frustration + low joy = time to try something new
        threshold = 0.6
        
        frustration_high = self.emotions['frustration'] > threshold
        joy_low = self.emotions['joy'] < 0.4
        
        return frustration_high and joy_low
    
    def get_motivation_level(self) -> float:
        """
        Overall motivation to continue
        
        Returns:
            Motivation (0.0-1.0)
        """
        # High when: high confidence, high curiosity, low frustration
        motivation = (
            0.3 * self.emotions['confidence'] +
            0.3 * self.emotions['curiosity'] +
            0.2 * self.emotions['joy'] +
            0.2 * (1.0 - self.emotions['frustration'])
        )
        
        return max(0.0, min(1.0, motivation))
    
    def get_emotional_summary(self) -> str:
        """Get human-readable emotional summary"""
        # Determine dominant emotion
        dominant = max(self.emotions.items(), key=lambda x: x[1])
        
        # Interpret state
        if self.emotions['joy'] > 0.7:
            return "😊 Happy and satisfied"
        elif self.emotions['curiosity'] > 0.7:
            return "🔍 Highly curious and exploring"
        elif self.emotions['frustration'] > 0.7:
            return "😤 Frustrated (needs change)"
        elif self.emotions['fear'] > 0.7:
            return "😰 Fearful and cautious"
        elif self.emotions['confidence'] > 0.8:
            return "💪 Highly confident"
        else:
            return f"😐 Neutral (dominant: {dominant[0]})"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get emotion system statistics"""
        return {
            'current_emotions': dict(self.emotions),
            'emotional_state': self.get_emotional_summary(),
            'exploration_bias': self.get_exploration_bias(),
            'learning_rate_bias': self.get_learning_rate_bias(),
            'risk_tolerance': self.get_risk_tolerance(),
            'motivation': self.get_motivation_level(),
            'event_counts': dict(self.event_counts),
            'history_length': len(self.history)
        }


if __name__ == "__main__":
    # Test artificial emotions
    print("❤️ Testing Artificial Emotions...")
    
    emotions = ArtificialEmotions()
    
    print(f"\n😐 Initial state: {emotions.get_emotional_summary()}")
    print(f"   Exploration bias: {emotions.get_exploration_bias():.2f}")
    print(f"   LR bias: {emotions.get_learning_rate_bias():.2f}")
    
    # Simulate events
    print("\n📋 Simulating events...")
    
    # Successes
    for i in range(3):
        emotions.update_emotions([{'type': 'unexpected_success'}])
        print(f"   After success {i+1}: {emotions.get_emotional_summary()}")
    
    # Failures
    for i in range(5):
        emotions.update_emotions([{'type': 'repeated_failure'}])
        if i % 2 == 0:
            print(f"   After failure {i+1}: {emotions.get_emotional_summary()}")
    
    print(f"\n   Should try new strategy? {emotions.should_try_new_strategy()}")
    
    # Breakthrough!
    emotions.update_emotions([{'type': 'breakthrough'}])
    print(f"\n   After breakthrough: {emotions.get_emotional_summary()}")
    
    # Final stats
    print("\n📊 Final statistics:")
    stats = emotions.get_statistics()
    for key, value in stats.items():
        if key not in ['current_emotions', 'event_counts']:
            print(f"   {key}: {value}")
    
    print("\n✅ Artificial emotions test complete")