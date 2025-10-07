"""
MULTI-SYSTEM COORDINATOR - Extraído de real_intelligence_system
Coordenação de múltiplos sistemas de inteligência em paralelo
"""
import logging
import threading
import queue
import time
from typing import Dict, List, Any, Callable, Optional
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

class SystemModule:
    """
    Base class for system modules
    """
    
    def __init__(self, name: str):
        self.name = name
        self.active = False
        self.metrics = {
            'cycles': 0,
            'performance': 0.0,
            'last_update': None
        }
    
    def process(self, input_data: Any) -> Any:
        """Process input data"""
        raise NotImplementedError
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get module metrics"""
        return self.metrics


class MultiSystemCoordinator:
    """
    Coordinator for multiple intelligent systems running in parallel
    Extracted from: real_intelligence_system/unified_real_intelligence.py
    
    Features:
    - Parallel execution of multiple systems
    - Inter-system communication via queues
    - Metric aggregation
    - Thread-safe coordination
    """
    
    def __init__(self, max_systems: int = 5):
        self.max_systems = max_systems
        self.systems: Dict[str, SystemModule] = {}
        self.communication_queue = queue.Queue()
        self.running = False
        self.threads: List[threading.Thread] = []
        
        # Metrics aggregation
        self.global_metrics = {
            'total_cycles': 0,
            'systems_active': 0,
            'avg_performance': 0.0,
            'coordination_events': 0,
            'emergence_detected': 0
        }
        
        # Communication buffer
        self.communication_buffer = deque(maxlen=1000)
        
        logger.info("🎯 Multi-System Coordinator initialized")
        logger.info(f"   Max systems: {max_systems}")
    
    def register_system(self, name: str, system: SystemModule):
        """
        Register a new system module
        
        Args:
            name: System name
            system: System module instance
        """
        if len(self.systems) >= self.max_systems:
            logger.warning(f"⚠️  Max systems reached ({self.max_systems})")
            return False
        
        self.systems[name] = system
        logger.info(f"✅ Registered system: {name}")
        return True
    
    def start_system(self, name: str):
        """Start a registered system in its own thread"""
        if name not in self.systems:
            logger.error(f"❌ System not found: {name}")
            return
        
        system = self.systems[name]
        
        def system_loop():
            """System execution loop"""
            system.active = True
            logger.info(f"🚀 Started system: {name}")
            
            while self.running:
                try:
                    # Get input from queue (with timeout)
                    try:
                        input_data = self.communication_queue.get(timeout=1.0)
                        
                        # Process
                        output = system.process(input_data)
                        
                        # Send output back to queue for other systems
                        if output:
                            self.communication_buffer.append({
                                'source': name,
                                'data': output,
                                'timestamp': time.time()
                            })
                        
                        # Update metrics
                        system.metrics['cycles'] += 1
                        system.metrics['last_update'] = time.time()
                        
                    except queue.Empty:
                        # No input available, just sleep
                        time.sleep(0.1)
                
                except Exception as e:
                    logger.error(f"❌ Error in system {name}: {e}")
                    time.sleep(1.0)
            
            system.active = False
            logger.info(f"⏹️  Stopped system: {name}")
        
        # Create and start thread
        thread = threading.Thread(target=system_loop, daemon=True)
        thread.start()
        self.threads.append(thread)
    
    def start_all(self):
        """Start all registered systems"""
        self.running = True
        
        for name in self.systems.keys():
            self.start_system(name)
        
        self.global_metrics['systems_active'] = len(self.systems)
        
        logger.info(f"🚀 All {len(self.systems)} systems started")
    
    def stop_all(self):
        """Stop all systems"""
        self.running = False
        
        # Wait for threads to finish
        for thread in self.threads:
            thread.join(timeout=5.0)
        
        logger.info("⏹️  All systems stopped")
    
    def send_to_all(self, data: Any):
        """Send data to all systems"""
        for _ in range(len(self.systems)):
            self.communication_queue.put(data)
    
    def aggregate_metrics(self) -> Dict[str, Any]:
        """Aggregate metrics from all systems"""
        total_cycles = 0
        total_performance = 0.0
        active_count = 0
        
        for system in self.systems.values():
            metrics = system.get_metrics()
            total_cycles += metrics['cycles']
            total_performance += metrics['performance']
            if system.active:
                active_count += 1
        
        self.global_metrics['total_cycles'] = total_cycles
        self.global_metrics['systems_active'] = active_count
        self.global_metrics['avg_performance'] = total_performance / len(self.systems) if self.systems else 0.0
        
        return self.global_metrics
    
    def detect_emergence(self) -> bool:
        """
        Detect emergent behavior from system interactions
        
        Returns:
            True if emergence detected
        """
        if len(self.communication_buffer) < 100:
            return False
        
        # Simple emergence detection: high communication rate
        recent = list(self.communication_buffer)[-100:]
        
        # Check communication diversity
        sources = set(msg['source'] for msg in recent)
        
        # Emergence = many systems communicating rapidly
        if len(sources) >= 3 and len(recent) >= 100:
            self.global_metrics['emergence_detected'] += 1
            logger.info("✨ EMERGENCE DETECTED!")
            return True
        
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get coordinator statistics"""
        return {
            'systems_registered': len(self.systems),
            'systems_active': sum(1 for s in self.systems.values() if s.active),
            'threads_alive': sum(1 for t in self.threads if t.is_alive()),
            'queue_size': self.communication_queue.qsize(),
            'buffer_size': len(self.communication_buffer),
            'global_metrics': self.global_metrics
        }
