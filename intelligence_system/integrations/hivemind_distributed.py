"""
Hivemind Integration - COMPLETO - Distributed Deep Learning
Merge REAL do /root/hivemind com distributed training completo
"""
import logging
import sys
from typing import Dict, Any, Optional, List
from pathlib import Path
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Try to import from installed Hivemind
try:
    sys.path.insert(0, '/root/hivemind')
    import hivemind
    from hivemind import DHT, get_dht_time
    from hivemind.optim import DecentralizedOptimizer
    HIVEMIND_AVAILABLE = True
    logger.info("✅ Hivemind imported successfully from /root/hivemind")
except ImportError as e:
    logger.warning(f"Hivemind not available: {e}")
    HIVEMIND_AVAILABLE = False

class HivemindDistributed:
    """
    Hivemind-based distributed training - PRODUCTION READY
    Peer-to-peer distributed deep learning across multiple machines
    """
    
    def __init__(self, peer_mode: bool = False,
                 initial_peers: Optional[List[str]] = None,
                 host_maddrs: Optional[List[str]] = None):
        self.hivemind_available = HIVEMIND_AVAILABLE
        self.peer_mode = peer_mode
        self.dht = None
        self.initial_peers = initial_peers or []
        self.host_maddrs = host_maddrs or ["/ip4/0.0.0.0/tcp/0"]
        self.peer_count_history = []
        
        if HIVEMIND_AVAILABLE and peer_mode:
            self._init_hivemind()
        
        logger.info(f"🐝 Hivemind Distributed initialized (available: {HIVEMIND_AVAILABLE})")
    
    def _init_hivemind(self):
        """Initialize Hivemind DHT for peer discovery"""
        try:
            # Create DHT with configuration
            self.dht = DHT(
                initial_peers=self.initial_peers,
                host_maddrs=self.host_maddrs,
                start=True
            )
            logger.info("✅ Hivemind DHT started")
            logger.info(f"   Host: {self.host_maddrs}")
            logger.info(f"   Initial peers: {len(self.initial_peers)}")
            
            # Wait for DHT to be ready
            self.dht.wait_until_ready()
            logger.info("✅ DHT ready for peer discovery")
            
        except Exception as e:
            logger.error(f"Hivemind DHT initialization failed: {e}")
            self.hivemind_available = False
    
    def create_distributed_optimizer(self, 
                                     optimizer: torch.optim.Optimizer,
                                     scheduler: Optional[object] = None,
                                     matchmaking_time: float = 5.0,
                                     averaging_timeout: float = 30.0,
                                     **kwargs):
        """
        Create Hivemind decentralized optimizer
        
        Args:
            optimizer: Base PyTorch optimizer
            scheduler: Optional LR scheduler
            matchmaking_time: Time to wait for peers
            averaging_timeout: Timeout for gradient averaging
        
        Returns:
            Decentralized optimizer wrapper
        """
        if not self.hivemind_available or self.dht is None:
            logger.warning("Hivemind not available, returning base optimizer")
            return optimizer
        
        try:
            distributed_opt = DecentralizedOptimizer(
                optimizer=optimizer,
                dht=self.dht,
                scheduler=scheduler,
                matchmaking_time=matchmaking_time,
                averaging_timeout=averaging_timeout,
                **kwargs
            )
            
            logger.info("✅ Decentralized optimizer created")
            logger.info(f"   Matchmaking time: {matchmaking_time}s")
            logger.info(f"   Averaging timeout: {averaging_timeout}s")
            
            return distributed_opt
            
        except Exception as e:
            logger.error(f"Distributed optimizer creation failed: {e}")
            return optimizer
    
    def get_peer_count(self) -> int:
        """Get number of currently visible peers"""
        if not self.hivemind_available or self.dht is None:
            return 0
        
        try:
            peers = self.dht.get_visible_peers()
            count = len(peers)
            self.peer_count_history.append(count)
            if len(self.peer_count_history) > 100:
                self.peer_count_history = self.peer_count_history[-100:]
            return count
        except Exception as e:
            logger.error(f"Failed to get peer count: {e}")
            return 0
    
    def get_peer_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about peers"""
        if not self.hivemind_available or self.dht is None:
            return []
        
        try:
            peers = self.dht.get_visible_peers()
            return [
                {
                    'peer_id': str(peer),
                    'timestamp': get_dht_time()
                }
                for peer in peers
            ]
        except Exception as e:
            logger.error(f"Failed to get peer info: {e}")
            return []
    
    def announce(self, key: str, value: Any, expiration_time: Optional[float] = None):
        """
        Announce information to the DHT
        
        Args:
            key: Key to announce
            value: Value to store
            expiration_time: Optional expiration time
        """
        if not self.hivemind_available or self.dht is None:
            logger.warning("Cannot announce: DHT not available")
            return False
        
        try:
            self.dht.store(key, value, expiration_time=expiration_time)
            logger.info(f"✅ Announced to DHT: {key}")
            return True
        except Exception as e:
            logger.error(f"DHT announce failed: {e}")
            return False
    
    def get_value(self, key: str) -> Optional[Any]:
        """Get value from DHT"""
        if not self.hivemind_available or self.dht is None:
            return None
        
        try:
            value = self.dht.get(key)
            return value
        except Exception as e:
            logger.error(f"DHT get failed: {e}")
            return None
    
    def shutdown(self):
        """Gracefully shutdown DHT"""
        if self.dht is not None:
            try:
                self.dht.shutdown()
                logger.info("✅ Hivemind DHT shut down")
            except Exception as e:
                logger.error(f"DHT shutdown failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive Hivemind statistics"""
        peer_count = self.get_peer_count()
        
        stats = {
            'hivemind_available': self.hivemind_available,
            'peer_mode': self.peer_mode,
            'dht_initialized': self.dht is not None,
            'peers_connected': peer_count,
            'initial_peers': len(self.initial_peers),
            'host_addresses': self.host_maddrs
        }
        
        if len(self.peer_count_history) > 0:
            import statistics
            stats['peer_count_avg'] = statistics.mean(self.peer_count_history)
            stats['peer_count_max'] = max(self.peer_count_history)
        
        return stats
    
    def is_ready(self) -> bool:
        """Check if Hivemind is ready for distributed training"""
        return self.hivemind_available and self.dht is not None

