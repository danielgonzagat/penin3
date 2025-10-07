#!/usr/bin/env python3
"""
PENIN Trace Bus Consumer
Allows PENIN to consume trace bus events and adapt policies
"""
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
import sqlite3

logger = logging.getLogger(__name__)

TRACE_BUS = Path('/root/trace_bus.jsonl')
PENIN_DB = Path('/root/penin_ultra_evolution.db')
POLICY_CACHE = Path('/root/.penin_omega/adaptive_policies.json')

class PeninTraceBusConsumer:
    async def __init__(self):
        self.last_position = 0
        self.policies = self._load_policies()
        self.event_buffer = []
        self.adaptation_threshold = 10  # Events before adaptation
        
    async def _load_policies(self) -> Dict:
        """Load current PENIN policies"""
        if POLICY_CACHE.exists():
            with open(POLICY_CACHE, 'r') as f:
                return await json.load(f)
        return await {
            'exploration_rate': 0.1,
            'exploitation_rate': 0.9,
            'learning_rate': 0.001,
            'batch_size': 32,
            'update_frequency': 100
        }
    
    async def _save_policies(self):
        """Save updated policies"""
        with open(POLICY_CACHE, 'w') as f:
            json.dump(self.policies, f, indent=2)
    
    async def read_new_events(self) -> List[Dict]:
        """Read new events from trace bus"""
        if not TRACE_BUS.exists():
            return await []
        
        new_events = []
        with open(TRACE_BUS, 'r') as f:
            # Seek to last position
            f.seek(self.last_position)
            
            for line in f:
                if line.strip():
                    try:
                        event = json.loads(line)
                        new_events.append(event)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse trace bus line: {line}")
            
            # Update position
            self.last_position = f.tell()
        
        return await new_events
    
    async def analyze_events(self, events: List[Dict]) -> Dict[str, Any]:
        """Analyze events for patterns and metrics"""
        analysis = {
            'total_events': len(events),
            'sources': {},
            'event_types': {},
            'performance_indicators': {}
        }
        
        for event in events:
            # Count by source
            source = event.get('source', 'unknown')
            analysis['sources'][source] = analysis['sources'].get(source, 0) + 1
            
            # Count by event type
            event_type = event.get('event', 'unknown')
            analysis['event_types'][event_type] = analysis['event_types'].get(event_type, 0) + 1
            
            # Extract performance indicators
            data = event.get('data', {})
            
            # From TEIS
            if source == 'teis_v2' and 'task_success_rate' in data:
                if 'task_success_rates' not in analysis['performance_indicators']:
                    analysis['performance_indicators']['task_success_rates'] = []
                analysis['performance_indicators']['task_success_rates'].append(data['task_success_rate'])
            
            # From Darwin
            if source == 'darwin_teis_adapter' and event_type == 'mutation_strategy_suggested':
                analysis['performance_indicators']['darwin_strategy'] = data.get('strategy', {}).get('strategy', 'unknown')
            
            # From IA3
            if source == 'ia3_fusion_validator' and event_type == 'variant_validated':
                if 'ia3_validations' not in analysis['performance_indicators']:
                    analysis['performance_indicators']['ia3_validations'] = {'approved': 0, 'rejected': 0}
                if data.get('approved'):
                    analysis['performance_indicators']['ia3_validations']['approved'] += 1
                else:
                    analysis['performance_indicators']['ia3_validations']['rejected'] += 1
        
        # Compute averages
        if 'task_success_rates' in analysis['performance_indicators']:
            rates = analysis['performance_indicators']['task_success_rates']
            analysis['performance_indicators']['avg_task_success'] = sum(rates) / len(rates) if rates else 0
        
        return await analysis
    
    async def adapt_policies(self, analysis: Dict[str, Any]):
        """Adapt PENIN policies based on event analysis"""
        indicators = analysis.get('performance_indicators', {})
        
        # Adapt based on task success rate
        avg_success = indicators.get('avg_task_success', 0.5)
        if avg_success < 0.4:
            # Poor performance - increase exploration
            self.policies['exploration_rate'] = min(0.3, self.policies['exploration_rate'] + 0.05)
            self.policies['exploitation_rate'] = max(0.7, self.policies['exploitation_rate'] - 0.05)
            self.policies['learning_rate'] = min(0.01, self.policies['learning_rate'] * 1.2)
            logger.info(f"Increased exploration due to low success rate: {avg_success:.2f}")
        elif avg_success > 0.8:
            # Good performance - increase exploitation
            self.policies['exploration_rate'] = max(0.05, self.policies['exploration_rate'] - 0.02)
            self.policies['exploitation_rate'] = min(0.95, self.policies['exploitation_rate'] + 0.02)
            logger.info(f"Increased exploitation due to high success rate: {avg_success:.2f}")
        
        # Adapt based on Darwin strategy
        darwin_strategy = indicators.get('darwin_strategy', '')
        if darwin_strategy == 'exploratory':
            self.policies['batch_size'] = min(64, self.policies['batch_size'] + 8)
            self.policies['update_frequency'] = max(50, self.policies['update_frequency'] - 10)
        elif darwin_strategy == 'exploitative':
            self.policies['batch_size'] = max(16, self.policies['batch_size'] - 8)
            self.policies['update_frequency'] = min(200, self.policies['update_frequency'] + 10)
        
        # Adapt based on IA3 validations
        ia3_validations = indicators.get('ia3_validations', {})
        if ia3_validations:
            approval_rate = ia3_validations['approved'] / (ia3_validations['approved'] + ia3_validations['rejected'] + 1)
            if approval_rate < 0.3:
                # Many rejections - be more conservative
                self.policies['learning_rate'] = max(0.0001, self.policies['learning_rate'] * 0.8)
                logger.info(f"Reduced learning rate due to low IA3 approval: {approval_rate:.2f}")
        
        # Save updated policies
        self._save_policies()
        
        # Log adaptation
        self._log_to_trace('policies_adapted', {
            'analysis': analysis,
            'new_policies': self.policies
        })
    
    async def _log_to_trace(self, event: str, data: Dict):
        """Log back to trace bus"""
        record = {
            'ts': datetime.now().isoformat(),
            'source': 'penin_trace_consumer',
            'event': event,
            'data': data
        }
        with open(TRACE_BUS, 'a') as f:
            json.dump(record, f)
            f.write('\n')
    
    async def update_penin_db(self):
        """Update PENIN database with adapted policies"""
        if not PENIN_DB.exists():
            logger.warning("PENIN database not found")
            return
        
        try:
            conn = sqlite3.connect(PENIN_DB)
            cursor = conn.cursor()
            
            # Check if policies table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS adaptive_policies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT,
                    policies TEXT,
                    source TEXT
                )
            """)
            
            # Insert new policy record
            cursor.execute("""
                INSERT INTO adaptive_policies (timestamp, policies, source)
                VALUES (?, ?, ?)
            """, (datetime.now().isoformat(), json.dumps(self.policies), 'trace_bus_consumer'))
            
            conn.commit()
            conn.close()
            
            logger.info("Updated PENIN database with new policies")
            
        except Exception as e:
            logger.error(f"Failed to update PENIN database: {e}")
    
    async def run_cycle(self):
        """Run one consumption and adaptation cycle"""
        # Read new events
        new_events = self.read_new_events()
        
        if new_events:
            self.event_buffer.extend(new_events)
            logger.info(f"Consumed {len(new_events)} new events from trace bus")
            
            # Adapt policies if buffer threshold reached
            if len(self.event_buffer) >= self.adaptation_threshold:
                analysis = self.analyze_events(self.event_buffer)
                self.adapt_policies(analysis)
                self.update_penin_db()
                
                # Clear buffer
                self.event_buffer = []
    
    async def run(self, interval: int = 10):
        """Main consumer loop"""
        logger.info("Starting PENIN Trace Bus Consumer")
        
        try:
            while True:
                self.run_cycle()
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        except Exception as e:
            logger.error(f"Consumer error: {e}")
            time.sleep(30)  # Wait before retry

async def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    consumer = PeninTraceBusConsumer()
    consumer.run(interval=15)  # Check every 15 seconds

if __name__ == "__main__":
    main()