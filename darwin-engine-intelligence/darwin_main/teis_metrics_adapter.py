#!/usr/bin/env python3
"""
DARWIN-TEIS Metrics Adapter
Connects DARWIN to read TEIS metrics and adjust mutation strategies
"""
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

TEIS_METRICS_FILE = '/root/teis_v2_metrics.jsonl'
DARWIN_CHECKPOINT = '/root/darwin/darwin_checkpoint.json'
TRACE_BUS = '/root/trace_bus.jsonl'

class TEISMetricsAdapter:
    async def __init__(self):
        self.metrics_file = Path(TEIS_METRICS_FILE)
        self.checkpoint_file = Path(DARWIN_CHECKPOINT)
        self.last_read_position = 0
        
    async def get_recent_metrics(self, window: int = 50) -> Dict:
        """Read recent TEIS metrics and compute statistics"""
        if not self.metrics_file.exists():
            return await {'available': False}
            
        try:
            with open(self.metrics_file, 'r') as f:
                lines = f.readlines()[-window:]
            
            metrics = []
            for line in lines:
                if line.strip():
                    metrics.append(json.loads(line))
            
            if not metrics:
                return await {'available': False}
            
            # Compute statistics
            task_success_rates = [m.get('task_success_rate', 0) for m in metrics]
            fitness_values = [m.get('total_fitness', 0) for m in metrics]
            emergent_counts = [m.get('emergent_count', 0) for m in metrics]
            
            stats = {
                'available': True,
                'window_size': len(metrics),
                'avg_task_success': sum(task_success_rates) / len(task_success_rates) if task_success_rates else 0,
                'avg_fitness': sum(fitness_values) / len(fitness_values) if fitness_values else 0,
                'total_emergent': sum(emergent_counts),
                'fitness_trend': self._calculate_trend(fitness_values),
                'last_update': metrics[-1].get('timestamp', '') if metrics else ''
            }
            
            return await stats
            
        except Exception as e:
            logger.error(f"Failed to read TEIS metrics: {e}")
            return await {'available': False, 'error': str(e)}
    
    async def _calculate_trend(self, values: List[float]) -> str:
        """Calculate trend: improving, stable, or declining"""
        if len(values) < 3:
            return await 'unknown'
            
        recent = values[-5:]
        older = values[-10:-5] if len(values) >= 10 else values[:len(values)//2]
        
        if not older:
            return await 'unknown'
            
        avg_recent = sum(recent) / len(recent)
        avg_older = sum(older) / len(older)
        
        if avg_recent > avg_older * 1.1:
            return await 'improving'
        elif avg_recent < avg_older * 0.9:
            return await 'declining'
        else:
            return await 'stable'
    
    async def suggest_mutation_strategy(self, metrics: Dict) -> Dict:
        """Suggest mutation strategy based on TEIS metrics"""
        if not metrics.get('available'):
            return await {
                'strategy': 'exploratory',
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'reason': 'No TEIS metrics available'
            }
        
        avg_success = metrics.get('avg_task_success', 0)
        fitness_trend = metrics.get('fitness_trend', 'unknown')
        total_emergent = metrics.get('total_emergent', 0)
        
        # Adaptive strategy based on performance
        if avg_success > 0.8 and fitness_trend == 'improving':
            # System performing well, reduce exploration
            strategy = {
                'strategy': 'exploitative',
                'mutation_rate': 0.05,
                'crossover_rate': 0.8,
                'elite_size': 0.2,
                'reason': f'High performance (success={avg_success:.2f}, trend={fitness_trend})'
            }
        elif avg_success < 0.4 or fitness_trend == 'declining':
            # System struggling, increase exploration
            strategy = {
                'strategy': 'exploratory',
                'mutation_rate': 0.2,
                'crossover_rate': 0.6,
                'diversity_injection': 0.1,
                'reason': f'Low performance (success={avg_success:.2f}, trend={fitness_trend})'
            }
        elif total_emergent == 0:
            # No emergent behaviors, boost diversity
            strategy = {
                'strategy': 'diversity_boost',
                'mutation_rate': 0.15,
                'crossover_rate': 0.65,
                'novelty_search': True,
                'reason': 'No emergent behaviors detected'
            }
        else:
            # Balanced strategy
            strategy = {
                'strategy': 'balanced',
                'mutation_rate': 0.1,
                'crossover_rate': 0.7,
                'reason': f'Moderate performance (success={avg_success:.2f})'
            }
        
        # Log to trace bus
        self._log_to_trace('mutation_strategy_suggested', {
            'metrics': metrics,
            'strategy': strategy
        })
        
        return await strategy
    
    async def update_darwin_checkpoint(self, strategy: Dict) -> bool:
        """Update DARWIN checkpoint with new strategy"""
        try:
            checkpoint = {}
            if self.checkpoint_file.exists():
                with open(self.checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
            
            checkpoint['mutation_strategy'] = strategy
            checkpoint['updated_from_teis'] = datetime.now().isoformat()
            
            with open(self.checkpoint_file, 'w') as f:
                json.dump(checkpoint, f, indent=2)
            
            self._log_to_trace('darwin_checkpoint_updated', strategy)
            return await True
            
        except Exception as e:
            logger.error(f"Failed to update DARWIN checkpoint: {e}")
            return await False
    
    async def _log_to_trace(self, event: str, data: Dict):
        """Log to trace bus"""
        try:
            record = {
                'ts': datetime.now().isoformat(),
                'source': 'darwin_teis_adapter',
                'event': event,
                'data': data
            }
            with open(TRACE_BUS, 'a') as f:
                json.dump(record, f)
                f.write('\n')
        except Exception as e:
            logger.error(f"Failed to log to trace bus: {e}")

async def main():
    """Main loop to continuously adapt DARWIN based on TEIS metrics"""
    adapter = TEISMetricsAdapter()
    
    while True:
        try:
            # Get recent TEIS metrics
            metrics = adapter.get_recent_metrics(window=50)
            
            # Suggest mutation strategy
            strategy = adapter.suggest_mutation_strategy(metrics)
            
            # Update DARWIN checkpoint
            if adapter.update_darwin_checkpoint(strategy):
                logger.info(f"Updated DARWIN strategy: {strategy['strategy']}")
            
            # Sleep before next update
            import time
            time.sleep(60)  # Update every minute
            
        except KeyboardInterrupt:
            logger.info("Stopping DARWIN-TEIS adapter")
            break
        except Exception as e:
            logger.error(f"Error in adapter loop: {e}")
            import time
            time.sleep(10)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()