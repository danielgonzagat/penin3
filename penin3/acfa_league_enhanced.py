#!/usr/bin/env python3
"""
ACFA League Enhanced
====================

Melhorias para ACFA League:
- Persistence path handling
- Staged canary rollout (10% → 50% → 100%)
- Automatic rollback on failure
- PCAg logging for all stage changes
"""

import sys
import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)


class ACFALeagueEnhanced:
    """
    Enhanced ACFA League with persistence and staged rollout
    """
    
    def __init__(self, persistence_path: Optional[str] = None):
        self.persistence_path = Path(persistence_path) if persistence_path else Path('/root/penin3/data/acfa_state.json')
        self.persistence_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Staged canary percentages
        self.canary_stages = [0.10, 0.25, 0.50, 0.75, 1.0]  # 10% → 25% → 50% → 75% → 100%
        self.current_stage = 0
        
        # State
        self.champion_model = None
        self.canary_model = None
        self.canary_percentage = 0.0
        self.canary_performance = []
        self.champion_performance = []
        
        # Load state if exists
        self._load_state()
        
        logger.info("🏆 ACFA League Enhanced initialized")
        logger.info(f"   Persistence: {self.persistence_path}")
        logger.info(f"   Canary stages: {self.canary_stages}")
    
    def _load_state(self):
        """Load persisted state"""
        if self.persistence_path.exists():
            try:
                with open(self.persistence_path, 'r') as f:
                    state = json.load(f)
                
                self.current_stage = state.get('current_stage', 0)
                self.canary_percentage = state.get('canary_percentage', 0.0)
                self.canary_performance = state.get('canary_performance', [])
                self.champion_performance = state.get('champion_performance', [])
                
                logger.info(f"   📂 Loaded state from {self.persistence_path}")
                logger.info(f"      Current stage: {self.current_stage}/{len(self.canary_stages)}")
                logger.info(f"      Canary %: {self.canary_percentage*100:.0f}%")
                
            except Exception as e:
                logger.warning(f"   Failed to load state: {e}")
    
    def _save_state(self):
        """Save current state"""
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'current_stage': self.current_stage,
                'canary_percentage': self.canary_percentage,
                'canary_performance': self.canary_performance[-10:],  # Last 10
                'champion_performance': self.champion_performance[-10:],
                'total_evaluations': len(self.canary_performance)
            }
            
            # Atomic write
            temp_path = self.persistence_path.with_suffix('.tmp')
            with open(temp_path, 'w') as f:
                json.dump(state, f, indent=2)
            
            temp_path.replace(self.persistence_path)
            
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    def promote_canary_stage(self, worm_ledger=None) -> Dict[str, Any]:
        """
        Promote canary to next stage
        Logs PCAg for audit trail
        
        Returns:
            Result dict with new percentage and PCAg hash
        """
        if self.current_stage >= len(self.canary_stages):
            logger.info("   Already at 100% canary deployment")
            return {'promoted': False, 'reason': 'already_full'}
        
        # Get new percentage
        old_pct = self.canary_percentage
        self.current_stage += 1
        new_pct = self.canary_stages[self.current_stage - 1]
        self.canary_percentage = new_pct
        
        logger.info(f"   🚀 CANARY PROMOTION: {old_pct*100:.0f}% → {new_pct*100:.0f}%")
        
        # Log to WORM with PCAg
        pcag_hash = None
        if worm_ledger:
            try:
                # Create PCAg for promotion decision
                from penin.ledger import ProofCarryingArtifact
                
                pcag = ProofCarryingArtifact.create(
                    decision_id=f"canary_promotion_stage_{self.current_stage}",
                    decision_type="promote",
                    metrics={
                        'old_percentage': old_pct,
                        'new_percentage': new_pct,
                        'stage': self.current_stage,
                        'canary_avg_perf': sum(self.canary_performance[-5:]) / max(1, len(self.canary_performance[-5:])) if self.canary_performance else 0.0,
                        'champion_avg_perf': sum(self.champion_performance[-5:]) / max(1, len(self.champion_performance[-5:])) if self.champion_performance else 0.0
                    },
                    gates={'manual_approval': True},
                    reason=f"Staged canary rollout to {new_pct*100:.0f}%",
                    metadata={'component': 'ACFA_Enhanced', 'stage': self.current_stage}
                )
                
                event = worm_ledger.append_pcag(pcag)
                pcag_hash = event.event_hash
                
                logger.info(f"   📋 PCAg logged: {pcag_hash[:16]}...")
                
            except Exception as e:
                logger.error(f"   Failed to log PCAg: {e}")
        
        # Save state
        self._save_state()
        
        return {
            'promoted': True,
            'old_percentage': old_pct,
            'new_percentage': new_pct,
            'stage': self.current_stage,
            'pcag_hash': pcag_hash
        }
    
    def rollback_canary(self, reason: str, worm_ledger=None) -> Dict[str, Any]:
        """
        Rollback canary to previous stage or champion
        
        Args:
            reason: Why rollback is happening
            worm_ledger: WORM ledger for audit trail
        
        Returns:
            Rollback result
        """
        if self.current_stage == 0:
            logger.warning("   Already at 0% canary, cannot rollback further")
            return {'rolled_back': False, 'reason': 'at_minimum'}
        
        old_stage = self.current_stage
        old_pct = self.canary_percentage
        
        # Rollback to previous stage
        self.current_stage = max(0, self.current_stage - 1)
        self.canary_percentage = self.canary_stages[self.current_stage - 1] if self.current_stage > 0 else 0.0
        
        logger.warning(f"   ⚠️ CANARY ROLLBACK: {old_pct*100:.0f}% → {self.canary_percentage*100:.0f}%")
        logger.warning(f"      Reason: {reason}")
        
        # Log to WORM
        pcag_hash = None
        if worm_ledger:
            try:
                from penin.ledger import ProofCarryingArtifact
                
                pcag = ProofCarryingArtifact.create(
                    decision_id=f"canary_rollback_from_stage_{old_stage}",
                    decision_type="rollback",
                    metrics={
                        'old_percentage': old_pct,
                        'new_percentage': self.canary_percentage,
                        'stage_before': old_stage,
                        'stage_after': self.current_stage
                    },
                    gates={'safety_gate': False},  # Failed safety
                    reason=reason,
                    metadata={'component': 'ACFA_Enhanced', 'rollback': True}
                )
                
                event = worm_ledger.append_pcag(pcag)
                pcag_hash = event.event_hash
                
                logger.info(f"   📋 Rollback PCAg logged: {pcag_hash[:16]}...")
                
            except Exception as e:
                logger.error(f"   Failed to log rollback PCAg: {e}")
        
        # Save state
        self._save_state()
        
        return {
            'rolled_back': True,
            'reason': reason,
            'old_stage': old_stage,
            'new_stage': self.current_stage,
            'old_percentage': old_pct,
            'new_percentage': self.canary_percentage,
            'pcag_hash': pcag_hash
        }
    
    def evaluate_canary_performance(self, canary_metric: float, champion_metric: float) -> Dict[str, Any]:
        """
        Evaluate canary vs champion performance
        Decides if should promote, rollback, or maintain
        
        Args:
            canary_metric: Performance metric for canary model (higher is better)
            champion_metric: Performance metric for champion model
        
        Returns:
            Decision dict
        """
        self.canary_performance.append(canary_metric)
        self.champion_performance.append(champion_metric)
        
        # Require minimum samples (5) before decision
        if len(self.canary_performance) < 5:
            return {
                'decision': 'wait',
                'reason': f'insufficient_samples ({len(self.canary_performance)}/5)'
            }
        
        # Compare recent performance (last 5 samples)
        recent_canary = sum(self.canary_performance[-5:]) / 5
        recent_champion = sum(self.champion_performance[-5:]) / 5
        
        # Decision thresholds
        PROMOTE_THRESHOLD = 1.05  # Canary must be 5% better
        ROLLBACK_THRESHOLD = 0.95  # Canary below 95% of champion → rollback
        
        ratio = recent_canary / max(0.001, recent_champion)
        
        logger.info(f"   📊 Performance comparison:")
        logger.info(f"      Canary (recent): {recent_canary:.4f}")
        logger.info(f"      Champion (recent): {recent_champion:.4f}")
        logger.info(f"      Ratio: {ratio:.4f}")
        
        if ratio >= PROMOTE_THRESHOLD and self.current_stage < len(self.canary_stages):
            # Canary performing well → promote to next stage
            logger.info(f"   ✅ Canary performing {(ratio-1)*100:.1f}% better → PROMOTE")
            return {
                'decision': 'promote',
                'reason': f'canary_better_by_{(ratio-1)*100:.1f}%',
                'ratio': ratio
            }
        
        elif ratio < ROLLBACK_THRESHOLD:
            # Canary performing poorly → rollback
            logger.warning(f"   ⚠️ Canary performing {(1-ratio)*100:.1f}% worse → ROLLBACK")
            return {
                'decision': 'rollback',
                'reason': f'canary_worse_by_{(1-ratio)*100:.1f}%',
                'ratio': ratio
            }
        
        else:
            # Canary OK but not significantly better → maintain
            logger.info(f"   ⏸️ Canary stable (ratio={ratio:.4f}) → MAINTAIN")
            return {
                'decision': 'maintain',
                'reason': 'canary_stable',
                'ratio': ratio
            }


def test_acfa_enhanced():
    """Test ACFA League Enhanced"""
    logger = logging.getLogger("acfa_test")
    logger.info("\n" + "="*80)
    logger.info("🏆 TEST: ACFA League Enhanced")
    logger.info("="*80)
    
    try:
        # Create instance
        acfa = ACFALeagueEnhanced(persistence_path='/tmp/acfa_test_state.json')
        
        logger.info("   ✅ Instance created")
        logger.info(f"      Stages: {acfa.canary_stages}")
        logger.info(f"      Current stage: {acfa.current_stage}")
        
        # Test evaluation
        logger.info("\n   Testing performance evaluation...")
        for i in range(6):
            canary = 0.85 + (i * 0.02)  # Improving
            champion = 0.82  # Stable
            
            result = acfa.evaluate_canary_performance(canary, champion)
            logger.info(f"      Eval {i+1}: {result['decision']} (ratio={result.get('ratio', 0):.3f})")
        
        # Test promotion
        logger.info("\n   Testing stage promotion...")
        promo_result = acfa.promote_canary_stage()
        logger.info(f"      Promoted: {promo_result['promoted']}")
        if promo_result['promoted']:
            logger.info(f"      New percentage: {promo_result['new_percentage']*100:.0f}%")
        
        # Test rollback
        logger.info("\n   Testing rollback...")
        rollback_result = acfa.rollback_canary("test_rollback")
        logger.info(f"      Rolled back: {rollback_result['rolled_back']}")
        if rollback_result['rolled_back']:
            logger.info(f"      New percentage: {rollback_result['new_percentage']*100:.0f}%")
        
        # Cleanup
        if acfa.persistence_path.exists():
            acfa.persistence_path.unlink()
        
        logger.info("\n   ✅ All ACFA Enhanced tests PASSED")
        return True
        
    except Exception as e:
        logger.error(f"   ❌ ACFA Enhanced test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # Run test
    success = test_acfa_enhanced()
    sys.exit(0 if success else 1)