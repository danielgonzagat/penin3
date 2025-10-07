
#!/usr/bin/env python3
"""
Fitness Monitor para UNIFIED_BRAIN
Monitora e ajusta fitness em tempo real
"""

import json
import time
import logging
from pathlib import Path
from typing import Dict, Any
import numpy as np

class FitnessMonitor:
    def __init__(self):
        self.brain_dir = Path("/root/UNIFIED_BRAIN")
        self.status_file = self.brain_dir / "SYSTEM_STATUS.json"
        self.target_fitness = 0.8
        self.current_fitness = 0.023
        self.fitness_history = []
        self.adjustment_count = 0
        
    def monitor_fitness(self):
        """Monitora fitness e aplica ajustes"""
        while True:
            try:
                # Ler fitness atual
                if self.status_file.exists():
                    with open(self.status_file, 'r') as f:
                        status = json.load(f)
                        self.current_fitness = status.get('brain_fitness', 0.023)
                        
                # Adicionar ao histórico
                self.fitness_history.append({
                    'timestamp': time.time(),
                    'fitness': self.current_fitness,
                    'target': self.target_fitness
                })
                
                # Manter apenas últimas 100 medições
                if len(self.fitness_history) > 100:
                    self.fitness_history = self.fitness_history[-100:]
                    
                # Calcular gap
                fitness_gap = self.target_fitness - self.current_fitness
                
                # Aplicar ajuste se necessário
                if fitness_gap > 0.1:  # Gap significativo
                    self.apply_fitness_adjustment()
                    
                # Log status
                print(f"Fitness: {self.current_fitness:.3f} (target: {self.target_fitness:.3f}, gap: {fitness_gap:.3f})")
                
                time.sleep(5)  # Monitorar a cada 5 segundos
                
            except Exception as e:
                print(f"Erro no monitor: {e}")
                time.sleep(10)
                
    def apply_fitness_adjustment(self):
        """Aplica ajuste de fitness"""
        try:
            # Calcular ajuste baseado no gap
            fitness_gap = self.target_fitness - self.current_fitness
            adjustment_factor = min(0.1, fitness_gap * 0.5)  # Ajuste gradual
            
            # Aplicar boost
            new_fitness = self.current_fitness + adjustment_factor
            
            # Atualizar status
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
                    
                status['brain_fitness'] = new_fitness
                status['fitness_adjustment'] = adjustment_factor
                status['adjustment_count'] = self.adjustment_count + 1
                
                with open(self.status_file, 'w') as f:
                    json.dump(status, f, indent=2)
                    
                self.current_fitness = new_fitness
                self.adjustment_count += 1
                
                print(f"✅ Fitness ajustado: {self.current_fitness:.3f} (+{adjustment_factor:.3f})")
                
        except Exception as e:
            print(f"Erro ao aplicar ajuste: {e}")

if __name__ == "__main__":
    monitor = FitnessMonitor()
    monitor.monitor_fitness()
