#!/usr/bin/env python3
"""
🚀 ADVANCED INTEGRATION - P2 + P3
Integração completa de System Bridge, Collective Consciousness e Auto-Evolution
"""

import os
import logging
import torch

logger = logging.getLogger('AdvancedIntegration')

# Try imports
try:
    from system_bridge import create_brain_bridge
    _BRIDGE_AVAILABLE = True
except:
    _BRIDGE_AVAILABLE = False

try:
    from safe_collective_consciousness import get_collective
    _COLLECTIVE_AVAILABLE = True
except:
    _COLLECTIVE_AVAILABLE = False

try:
    from code_evolution_engine import CodeEvolutionEngine
    from true_godelian_incompleteness import TrueGodelianIncompleteness
    from meta_curiosity_module import MetaCuriosityModule
    _EVOLUTION_AVAILABLE = True
except:
    _EVOLUTION_AVAILABLE = False


class AdvancedIntegration:
    """Gerenciador de integrações avançadas P2+P3"""
    
    def __init__(self, brain_instance):
        self.brain = brain_instance
        
        # P2: System Integration
        self.bridge = None
        self.collective = None
        
        # P3: Auto-Evolution
        self.code_evolver = None
        self.true_godel = None
        self.meta_curiosity = None
        
        self.enabled = False
    
    def initialize(self):
        """Inicializa todas integrações"""
        # P2: System Bridge
        if _BRIDGE_AVAILABLE and os.getenv("ENABLE_BRIDGE", "1") == "1":
            try:
                self.bridge = create_brain_bridge()
                self.bridge.start()
                logger.info("✅ System Bridge initialized")
            except Exception as e:
                logger.warning(f"Bridge init failed: {e}")
        
        # P2: Collective Consciousness
        if _COLLECTIVE_AVAILABLE and os.getenv("ENABLE_COLLECTIVE", "1") == "1":
            try:
                self.collective = get_collective()
                self.collective.register_process("unified_brain", os.getpid())
                logger.info("✅ Collective Consciousness initialized")
            except Exception as e:
                logger.warning(f"Collective init failed: {e}")
        
        # P3: Auto-Evolution
        if _EVOLUTION_AVAILABLE and os.getenv("ENABLE_AUTO_EVOLUTION", "0") == "1":
            try:
                self.code_evolver = CodeEvolutionEngine()
                self.true_godel = TrueGodelianIncompleteness()
                self.meta_curiosity = MetaCuriosityModule(H=1024)
                logger.info("✅ Auto-Evolution initialized")
            except Exception as e:
                logger.warning(f"Auto-evolution init failed: {e}")
        
        self.enabled = True
    
    def on_episode_step(self, obs, action, next_obs, reward):
        """Hook chamado a cada step do episódio"""
        meta_reward = 0.0
        
        # P3: Meta-Curiosity
        if self.meta_curiosity:
            try:
                result = self.meta_curiosity(obs, action, next_obs)
                meta_reward = result['total_intrinsic'].item() * 0.01
                
                # Treinar predictor
                self.meta_curiosity.train_predictor(obs, action, next_obs)
            except Exception:
                pass
        
        return meta_reward
    
    def on_episode_end(self, episode, reward, loss, stats):
        """Hook chamado ao fim do episódio"""
        # P2: Publish metrics
        if self.bridge:
            try:
                self.bridge.publish_metrics({
                    'episode': episode,
                    'reward': reward,
                    'loss': loss,
                    'best_reward': self.brain.best_reward,
                    'avg_reward': stats.get('avg_reward_last_100', 0.0),
                    'step_time': stats.get('avg_time_per_step', 0.0)
                })
            except Exception:
                pass
        
        # P2: Update collective
        if self.collective:
            try:
                fitness = max(0.0, min(1.0, reward / 200.0))
                self.collective.update_fitness("unified_brain", fitness)
                
                # Broadcast milestone
                if reward > self.brain.best_reward * 0.9:
                    self.collective.broadcast_insight("unified_brain", {
                        'type': 'milestone',
                        'reward': reward,
                        'episode': episode
                    })
            except Exception:
                pass
        
        # P3: Meta-learning
        if self.meta_curiosity:
            try:
                self.meta_curiosity.learn_curiosity(reward)
            except Exception:
                pass
        
        # P3: Check fundamental limits (a cada 10 episódios)
        if self.true_godel and episode % 10 == 0:
            try:
                limit_result = self.true_godel.detect_fundamental_limit(
                    self.brain.hybrid.core,
                    "reinforcement learning cartpole"
                )
                
                if limit_result['is_limited']:
                    logger.warning(f"⚠️  Fundamental limit: {limit_result['limit_type']}")
            except Exception:
                pass
    
    def shutdown(self):
        """Desliga integrações"""
        if self.bridge:
            try:
                self.bridge.stop()
            except:
                pass
        
        logger.info("🛑 Advanced integrations shutdown")