#!/usr/bin/env python3
"""
🧠 BRAIN DAEMON - Sistema 24/7
Cérebro ativo continuamente, conectado a todos os sistemas
"""

import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')
sys.path.insert(0, '/root')

import torch
import time
from pathlib import Path
import signal
import json
from datetime import datetime

from unified_brain_core import CoreSoupHybrid
from brain_system_integration import UnifiedSystemController
from brain_logger import brain_logger

class BrainDaemon:
    """Daemon que roda o cérebro 24/7"""
    
    def __init__(self):
        self.running = True
        self.hybrid = None
        self.controller = None
        self.stats = {
            'start_time': datetime.now().isoformat(),
            'total_steps': 0,
            'total_time': 0.0,
            'errors': 0
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.shutdown)
        signal.signal(signal.SIGTERM, self.shutdown)
        
        brain_logger.info("Brain Daemon initializing...")
    
    def shutdown(self, signum, frame):
        """Graceful shutdown"""
        brain_logger.info(f"Received signal {signum}, shutting down gracefully...")
        self.running = False
    
    def initialize(self):
        """Inicializa o cérebro e conexões"""
        brain_logger.info("Loading brain...")
        
        # Carrega cérebro
        self.hybrid = CoreSoupHybrid(H=1024)
        
        snapshot_path = Path("/root/UNIFIED_BRAIN/snapshots/initial_state_registry.json")
        if snapshot_path.exists():
            self.hybrid.core.registry.load_with_adapters(str(snapshot_path))
            self.hybrid.core.initialize_router()
            brain_logger.info(f"Brain loaded: {self.hybrid.core.registry.count()['total']} neurons")
        else:
            brain_logger.warning("No snapshot found, starting fresh")
        
        # Setup controller
        self.controller = UnifiedSystemController(self.hybrid.core)
        self.controller.connect_v7(obs_dim=4, act_dim=2)
        
        brain_logger.info("Brain daemon ready!")
    
    def run_step(self):
        """Um ciclo de processamento"""
        try:
            # Gera observação (simulated environment)
            obs = torch.randn(1, 4)
            
            # Processa
            start = time.time()
            result = self.controller.step(
                obs=obs,
                penin_metrics={
                    'L_infinity': torch.rand(1).item(),
                    'CAOS_plus': torch.rand(1).item(),
                    'SR_Omega_infinity': torch.rand(1).item()
                },
                reward=torch.rand(1).item()
            )
            elapsed = time.time() - start
            
            self.stats['total_steps'] += 1
            self.stats['total_time'] += elapsed
            
            # Log periódico
            if self.stats['total_steps'] % 100 == 0:
                avg_time = self.stats['total_time'] / self.stats['total_steps']
                brain_logger.info(
                    f"Step {self.stats['total_steps']}: "
                    f"IA³={result['ia3_signal']:.3f}, "
                    f"avg_latency={avg_time*1000:.1f}ms"
                )
                
                # Salva checkpoint a cada 1000 steps
                if self.stats['total_steps'] % 1000 == 0:
                    self.save_checkpoint()
            
        except Exception as e:
            self.stats['errors'] += 1
            brain_logger.error(f"Error in step: {e}")
            
            if self.stats['errors'] > 100:
                brain_logger.critical("Too many errors, shutting down")
                self.running = False
    
    def save_checkpoint(self):
        """Salva checkpoint do daemon"""
        checkpoint = {
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        checkpoint_path = Path("/root/UNIFIED_BRAIN/daemon_checkpoint.json")
        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        brain_logger.info(f"Checkpoint saved: {self.stats['total_steps']} steps")
    
    def run(self):
        """Loop principal 24/7"""
        self.initialize()
        
        brain_logger.info("Brain daemon running 24/7...")
        brain_logger.info("Press Ctrl+C to stop")
        
        while self.running:
            self.run_step()
            try:
                # Manutenção periódica do core/soup
                if self.hybrid:
                    self.hybrid.tick_maintenance()
            except Exception:
                pass
            
            # Small sleep para não saturar CPU
            time.sleep(0.01)
        
        # Cleanup
        self.save_checkpoint()
        brain_logger.info(f"Brain daemon stopped. Total steps: {self.stats['total_steps']}")


if __name__ == "__main__":
    daemon = BrainDaemon()
    daemon.run()
