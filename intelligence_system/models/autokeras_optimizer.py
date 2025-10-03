"""
AutoKeras Integration - AutoML and Neural Architecture Search
Merge do autokeras para otimização automática de arquiteturas
"""
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

try:
    import autokeras as ak
    import tensorflow as tf
    AUTOKERAS_AVAILABLE = True
except ImportError:
    logger.warning("AutoKeras not available - using manual architecture")
    AUTOKERAS_AVAILABLE = False

class AutoKerasOptimizer:
    """
    AutoKeras-based architecture search
    Automatically discovers optimal neural architectures
    """
    
    def __init__(self, task_type: str = 'classification',
                 max_trials: int = 10,
                 checkpoint_dir: Path = None):
        self.autokeras_available = AUTOKERAS_AVAILABLE
        self.task_type = task_type
        self.max_trials = max_trials
        self.checkpoint_dir = checkpoint_dir or Path('data/autokeras')
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.searcher = None
        self.best_architecture = None
        
        if AUTOKERAS_AVAILABLE:
            self._init_autokeras()
        else:
            logger.info("⚠️  Using manual architecture (no AutoKeras)")
        
        logger.info(f"🔄 AutoKeras Optimizer initialized (AutoKeras: {AUTOKERAS_AVAILABLE})")
    
    def _init_autokeras(self):
        """Initialize AutoKeras searcher"""
        try:
            if self.task_type == 'classification':
                # MNIST AutoKeras
                self.searcher = ak.ImageClassifier(
                    max_trials=self.max_trials,
                    overwrite=False,
                    directory=str(self.checkpoint_dir),
                    project_name='mnist_nas'
                )
            elif self.task_type == 'rl':
                # Future: Custom AutoKeras for RL architectures
                logger.info("RL AutoKeras not yet implemented")
            
            logger.info(f"✅ AutoKeras searcher initialized (max_trials={self.max_trials})")
        except Exception as e:
            logger.warning(f"AutoKeras initialization failed: {e}")
            self.autokeras_available = False
    
    def search_architecture(self, X_train: np.ndarray, y_train: np.ndarray,
                          X_val: np.ndarray, y_val: np.ndarray,
                          epochs: int = 10) -> Optional[Dict[str, Any]]:
        """
        Search for optimal architecture
        
        Args:
            X_train, y_train: Training data
            X_val, y_val: Validation data
            epochs: Max epochs per trial
        
        Returns:
            Best architecture info
        """
        if not self.autokeras_available or self.searcher is None:
            logger.info("⚠️  AutoKeras search not available")
            return None
        
        try:
            logger.info(f"🔍 Starting AutoKeras search (max_trials={self.max_trials})...")
            
            # Search
            self.searcher.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                verbose=1
            )
            
            # Get best model
            best_model = self.searcher.export_model()
            
            # Evaluate
            val_loss, val_acc = best_model.evaluate(X_val, y_val, verbose=0)
            
            self.best_architecture = {
                'model': best_model,
                'val_loss': float(val_loss),
                'val_acc': float(val_acc),
                'summary': self._get_model_summary(best_model)
            }
            
            logger.info(f"✅ AutoKeras search complete: val_acc={val_acc:.4f}")
            return self.best_architecture
        
        except Exception as e:
            logger.error(f"AutoKeras search failed: {e}")
            return None
    
    def _get_model_summary(self, model) -> Dict[str, Any]:
        """Get model architecture summary"""
        try:
            # Count parameters
            total_params = sum([np.prod(v.shape) for v in model.trainable_weights])
            
            return {
                'total_params': int(total_params),
                'num_layers': len(model.layers),
                'layer_types': [layer.__class__.__name__ for layer in model.layers]
            }
        except:
            return {'error': 'Could not extract summary'}
    
    def convert_to_pytorch(self, architecture: Dict[str, Any]) -> Optional[object]:
        """
        Convert discovered architecture to PyTorch
        (Future feature - for now returns None)
        """
        logger.info("🔄 Architecture conversion to PyTorch (future feature)")
        return None
    
    def get_search_stats(self) -> Dict[str, Any]:
        """Get architecture search statistics"""
        stats = {
            'autokeras_available': self.autokeras_available,
            'task_type': self.task_type,
            'max_trials': self.max_trials,
            'search_completed': self.best_architecture is not None
        }
        
        if self.best_architecture:
            stats['best_val_acc'] = self.best_architecture['val_acc']
            stats['best_architecture'] = self.best_architecture['summary']
        
        return stats
    
    def should_trigger_search(self, current_performance: float,
                             cycles_stagnant: int,
                             threshold: int = 20) -> bool:
        """
        Decide if we should trigger AutoKeras search
        
        Args:
            current_performance: Current model performance
            cycles_stagnant: Number of cycles without improvement
            threshold: Stagnation threshold to trigger search
        
        Returns:
            True if search should be triggered
        """
        # Trigger search if:
        # 1. Performance < 90% AND stagnant for 20+ cycles
        # 2. Never searched before
        
        if not self.autokeras_available:
            return False
        
        never_searched = self.best_architecture is None
        poor_performance = current_performance < 90.0
        highly_stagnant = cycles_stagnant >= threshold
        
        should_search = never_searched or (poor_performance and highly_stagnant)
        
        if should_search:
            logger.info(f"🎯 AutoKeras search triggered: perf={current_performance:.1f}%, stagnant={cycles_stagnant}")
        
        return should_search

