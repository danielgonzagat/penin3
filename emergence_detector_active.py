#!/usr/bin/env python3
"""
Emergence Detector Ativo Integrado com Sistemas de Aprendizado
"""

import torch
import torch.nn as nn
import numpy as np
import time
import threading
from datetime import datetime
import sqlite3

class ActiveEmergenceDetector:
    async def __init__(self):
        self.detection_network = nn.Sequential(
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )
        
        self.emergence_events = []
        self.learning_systems = []
        self.is_active = False
        
        # Database para emergências
        self.init_database()
        
    async def init_database(self):
        self.conn = sqlite3.connect('active_emergence.db')
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergence_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                emergence_score REAL,
                description TEXT,
                systems_involved TEXT
            )
        ''')
        self.conn.commit()
        
    async def integrate_with_learning_system(self, system_name, system_interface):
        """Integra com um sistema de aprendizado"""
        self.learning_systems.append({
            'name': system_name,
            'interface': system_interface,
            'last_interaction': datetime.now()
        })
        logger.info(f"🔗 Integrado com sistema de aprendizado: {system_name}")
        
    async def actively_monitor(self):
        """Monitoramento ativo contínuo"""
        self.is_active = True
        
        async def monitoring_loop():
            while self.is_active:
                # Coletar dados de todos os sistemas integrados
                system_data = []
                for system in self.learning_systems:
                    try:
                        data = system['interface'].get_monitoring_data()
                        system_data.extend(data)
                        system['last_interaction'] = datetime.now()
                    except:
                        continue
                        
                if len(system_data) >= 50:
                    # Detectar emergência
                    emergence_score = self.detect_emergence(system_data)
                    
                    if emergence_score > 0.7:  # Threshold de emergência
                        self.record_emergence(emergence_score, system_data)
                        logger.info(f"🚨 EMERGÊNCIA DETECTADA! Score: {emergence_score:.3f}")
                        
                        # Notificar sistemas integrados
                        self.notify_learning_systems(emergence_score)
                        
                time.sleep(1)  # Monitoramento a cada segundo
                
        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()
        logger.info("👁️ Monitoramento ativo de emergência iniciado")
        
    async def detect_emergence(self, system_data):
        """Detecta emergência nos dados dos sistemas"""
        if len(system_data) < 50:
            return await 0.0
            
        # Preparar dados para rede neural
        data_tensor = torch.tensor(system_data[:50], dtype=torch.float32)
        
        # Detectar emergência
        emergence_prob = self.detection_network(data_tensor).item()
        
        return await emergence_prob
        
    async def record_emergence(self, score, system_data):
        """Registra evento de emergência"""
        event = {
            'timestamp': datetime.now(),
            'score': score,
            'description': f'Emergência detectada com score {score:.3f}',
            'systems_involved': [s['name'] for s in self.learning_systems]
        }
        
        self.emergence_events.append(event)
        
        # Salvar no database
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO emergence_events (timestamp, emergence_score, description, systems_involved)
            VALUES (?, ?, ?, ?)
        ''', (
            event['timestamp'].isoformat(),
            event['score'],
            event['description'],
            ','.join(event['systems_involved'])
        ))
        self.conn.commit()
        
    async def notify_learning_systems(self, emergence_score):
        """Notifica sistemas de aprendizado sobre emergência"""
        for system in self.learning_systems:
            try:
                system['interface'].on_emergence_detected(emergence_score)
            except:
                continue
                
    async def get_emergence_report(self):
        """Retorna relatório de emergências detectadas"""
        return await {
            'total_events': len(self.emergence_events),
            'recent_events': self.emergence_events[-5:],
            'systems_integrated': len(self.learning_systems)
        }

# Interface mock para sistemas de aprendizado
class MockLearningSystem:
    async def __init__(self, name):
        self.name = name
        
    async def get_monitoring_data(self):
        """Retorna dados de monitoramento simulados"""
        return await np.random.randn(25).tolist()
        
    async def on_emergence_detected(self, score):
        """Callback quando emergência é detectada"""
        logger.info(f"📢 {self.name}: Emergência detectada com score {score:.3f}! Adaptando...")

if __name__ == "__main__":
    detector = ActiveEmergenceDetector()
    
    # Integrar com sistemas mock
    for i in range(3):
        mock_system = MockLearningSystem(f"LearningSystem_{i}")
        detector.integrate_with_learning_system(f"system_{i}", mock_system)
    
    # Iniciar monitoramento ativo
    detector.actively_monitor()
    
    # Executar por 30 segundos
    time.sleep(30)
    
    detector.is_active = False
    
    report = detector.get_emergence_report()
    logger.info(f"Relatório final: {report['total_events']} emergências detectadas")
    logger.info(f"Sistemas integrados: {report['systems_integrated']}")