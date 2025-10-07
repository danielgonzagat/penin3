#!/usr/bin/env python3
"""
✅ RECURSIVE IMPROVEMENT DAEMON
Loop 24/7 de auto-melhoria contínua
"""

import sys
sys.path.insert(0, '/root/UNIFIED_BRAIN')
sys.path.insert(0, '/root')

import time
import sqlite3
from datetime import datetime
from pathlib import Path
from brain_logger import brain_logger

try:
    from recursive_improvement import RecursiveImprovementEngine, SelfImprovementLoop
    RECURSIVE_AVAILABLE = True
except Exception as e:
    brain_logger.error(f"Recursive engine import failed: {e}")
    RECURSIVE_AVAILABLE = False

class RecursiveDaemon:
    """Daemon de melhoria recursiva 24/7"""
    
    def __init__(self):
        if not RECURSIVE_AVAILABLE:
            self.enabled = False
            return
        
        self.engine = RecursiveImprovementEngine()
        self.loop = SelfImprovementLoop(self.engine)
        self.db_path = '/root/intelligence_system/data/intelligence.db'
        self.enabled = True
        
        brain_logger.info("♾️ Recursive Daemon initialized")
    
    def get_recent_performance(self) -> dict:
        """Busca performance recente do brain"""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.cursor()
            
            cur.execute("""
                SELECT episode, coherence, novelty, ia3_signal, timestamp
                FROM brain_metrics
                ORDER BY timestamp DESC
                LIMIT 100
            """)
            
            rows = cur.fetchall()
            conn.close()
            
            if not rows:
                return {}
            
            return {
                'episodes': [r[0] for r in rows],
                'coherence': [r[1] for r in rows],
                'novelty': [r[2] for r in rows],
                'ia3': [r[3] or 0 for r in rows],
                'latest_episode': rows[0][0],
                'latest_timestamp': rows[0][4]
            }
        except Exception as e:
            brain_logger.error(f"Performance fetch failed: {e}")
            return {}
    
    def run_forever(self):
        """Loop infinito de auto-melhoria"""
        if not self.enabled:
            brain_logger.error("❌ Recursive daemon disabled")
            return
        
        cycle = 0
        brain_logger.info("♾️ Starting recursive improvement loop...")
        
        while True:
            try:
                cycle += 1
                brain_logger.info(f"♾️ Recursive cycle {cycle}")
                
                # 1. Observar performance
                perf = self.get_recent_performance()
                
                if not perf or len(perf.get('episodes', [])) == 0:
                    brain_logger.warning("No performance data, waiting...")
                    time.sleep(60)
                    continue
                
                # 2. Criar dados do ciclo
                cycle_data = {
                    'cycle': cycle,
                    'timestamp': datetime.now().isoformat(),
                    'episodes': len(perf['episodes']),
                    'avg_coherence': sum(perf['coherence']) / len(perf['coherence']),
                    'avg_novelty': sum(perf['novelty']) / len(perf['novelty']),
                    'avg_ia3': sum(perf['ia3']) / len(perf['ia3']) if perf['ia3'] else 0,
                    'strategy': 'exploration'  # Default
                }
                
                # 3. Observar
                self.engine.observe_cycle(cycle_data)
                
                # 4. Meta-learn a cada 10 ciclos
                if cycle % 10 == 0:
                    insights = self.engine.meta_learn()
                    
                    brain_logger.warning(
                        f"♾️ Meta-insights: {len(insights.get('patterns', []))} patterns, "
                        f"{len(insights.get('recommendations', []))} recommendations"
                    )
                    
                    # Log recomendações HIGH priority
                    for rec in insights.get('recommendations', []):
                        if rec.get('priority') == 'HIGH':
                            brain_logger.warning(
                                f"♾️ RECOMMENDATION: {rec['action']} - {rec['reason']}"
                            )
                
                # 5. Self-improvement a cada 5 ciclos
                if cycle % 5 == 0:
                    improvements = self.loop.iterate({
                        'performance_data': perf,
                        'cycle': cycle
                    })
                    
                    if improvements.get('actions_generated', 0) > 0:
                        brain_logger.warning(
                            f"♾️ SELF-IMPROVEMENT: "
                            f"{improvements['actions_generated']} actions generated"
                        )
                
                # 6. Checkpoint a cada 20 ciclos
                if cycle % 20 == 0:
                    self._save_checkpoint(cycle)
                
                # Aguardar próximo ciclo (2 minutos)
                time.sleep(120)
                
            except KeyboardInterrupt:
                brain_logger.info("♾️ Recursive daemon stopping (KeyboardInterrupt)...")
                break
            except Exception as e:
                brain_logger.error(f"♾️ Recursive cycle error: {e}")
                time.sleep(60)  # Esperar antes de retry
    
    def _save_checkpoint(self, cycle):
        """Salva checkpoint do estado recursivo"""
        try:
            checkpoint = {
                'cycle': cycle,
                'timestamp': datetime.now().isoformat(),
                'patterns': self.engine.meta_history.meta_patterns,
                'improvement_cycles': len(self.engine.improvement_cycles),
                'meta_updates': self.engine.meta_history.history
            }
            
            import json
            checkpoint_path = Path('/root/UNIFIED_BRAIN/recursive_checkpoint.json')
            
            with open(checkpoint_path, 'w') as f:
                # Serializar apenas o essencial (evitar OOM)
                essential = {
                    'cycle': checkpoint['cycle'],
                    'timestamp': checkpoint['timestamp'],
                    'patterns_count': len(checkpoint['patterns']),
                    'improvement_cycles': checkpoint['improvement_cycles']
                }
                json.dump(essential, f, indent=2)
            
            brain_logger.info(f"♾️ Recursive checkpoint saved: cycle {cycle}")
            
        except Exception as e:
            brain_logger.error(f"Checkpoint save failed: {e}")

if __name__ == "__main__":
    daemon = RecursiveDaemon()
    
    if daemon.enabled:
        brain_logger.info("♾️ Starting recursive improvement daemon...")
        print("✅ Recursive daemon starting...")
        print("📊 Will analyze and improve system every 2 minutes")
        print("⚠️  Press Ctrl+C to stop")
        daemon.run_forever()
    else:
        print("❌ Recursive daemon disabled (dependencies missing)")