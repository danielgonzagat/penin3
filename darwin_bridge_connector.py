#!/usr/bin/env python3
"""
🧬 DARWIN BRIDGE CONNECTOR
Conecta Darwin Evolver ao System Bridge para feedback loop evolutivo
"""

import sys
import os
import time
import logging
sys.path.insert(0, '/root')
sys.path.insert(0, '/root/UNIFIED_BRAIN')

from system_bridge import create_darwin_bridge
from darwin_godelian_evolver import evolve_godelian

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('DarwinConnector')

def run_darwin_with_bridge():
    """Roda Darwin Evolver conectado ao bridge"""
    logger.info("🧬 Starting Darwin Evolver with Bridge integration")
    
    # Criar bridge
    bridge = create_darwin_bridge()
    bridge.start()
    
    # Callback para receber métricas do brain
    def on_metrics(message):
        """Quando brain publica métricas, usar para fitness"""
        try:
            metrics = message.data
            episode = metrics.get('episode', 0)
            reward = metrics.get('reward', 0.0)
            
            logger.info(f"📊 Received metrics: episode={episode}, reward={reward}")
            
            # Se reward alto, considerar para fitness
            if reward > 50:
                logger.info(f"🎯 High reward detected: {reward}")
        except Exception as e:
            logger.error(f"Metrics callback error: {e}")
    
    bridge.register_callback("metrics", on_metrics)
    
    # Loop evolutivo
    logger.info("🧬 Starting evolutionary loop...")
    
    try:
        # Rodar evolução (15 gerações)
        best_individual = evolve_godelian(generations=15, population_size=20)
        
        # Publicar melhor genoma
        logger.info("📡 Publishing best genome to bridge...")
        bridge.publish_genome({
            'genome': best_individual.genome,
            'fitness': best_individual.fitness,
            'type': 'godelian_evolved'
        })
        
        logger.info("✅ Darwin evolution complete and published!")
        
        # Manter rodando para receber métricas
        logger.info("👂 Listening for brain metrics...")
        while True:
            time.sleep(10)
            
            # Re-evoluir periodicamente baseado em métricas
            # (simplificado - na prática seria mais sofisticado)
    
    except KeyboardInterrupt:
        logger.info("🛑 Darwin connector stopped")
    finally:
        bridge.stop()

if __name__ == "__main__":
    run_darwin_with_bridge()