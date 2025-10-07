#!/usr/bin/env python3
"""
SISTEMA DE MONITORAMENTO DE EMERGÊNCIA 24H+
==========================================
Monitora continuamente por sinais de emergência de inteligência real
"""

import json
import time
import psutil
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("EmergenceMonitor")

class EmergenceMonitor:
    def __init__(self):
        self.monitor_dir = Path("/root/emergence_monitoring")
        self.monitor_dir.mkdir(exist_ok=True)
        self.running = False
        self.start_time = datetime.now()
        self.emergence_events = []
        
    def detect_emergence_patterns(self):
        """Detecta padrões de emergência em tempo real"""
        patterns = {
            "sudden_performance_jump": False,
            "novel_behavior": False,
            "self_modification": False,
            "unexpected_optimization": False,
            "cross_system_communication": False
        }
        
        try:
            # Check Neural Farm for sudden fitness jumps
            db_path = "/root/neural_farm_prod/neural_farm.db"
            if Path(db_path).exists():
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT max_fit FROM evolution ORDER BY generation DESC LIMIT 5")
                recent_fitness = [row[0] for row in cursor.fetchall()]
                
                if len(recent_fitness) >= 2:
                    improvement = recent_fitness[0] - recent_fitness[-1]
                    if improvement > 10:  # Sudden jump threshold
                        patterns["sudden_performance_jump"] = True
                        self.emergence_events.append({
                            "timestamp": datetime.now().isoformat(),
                            "type": "sudden_performance_jump",
                            "system": "neural_farm",
                            "improvement": improvement
                        })
                
                conn.close()
            
            # Check for novel file creation (self-modification)
            recent_files = []
            for file_path in Path("/root").glob("*"):
                if file_path.is_file():
                    mod_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                    if datetime.now() - mod_time < timedelta(minutes=10):
                        recent_files.append(file_path)
            
            if len(recent_files) > 5:  # Many new files = potential self-modification
                patterns["self_modification"] = True
                self.emergence_events.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "self_modification",
                    "new_files": len(recent_files)
                })
            
            # Check for unexpected CPU optimization
            high_cpu_procs = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent']):
                if proc.info['cpu_percent'] > 200:  # High CPU usage
                    high_cpu_procs.append(proc.info)
            
            if len(high_cpu_procs) > 3:
                patterns["unexpected_optimization"] = True
                self.emergence_events.append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "unexpected_optimization",
                    "high_cpu_processes": len(high_cpu_procs)
                })
            
        except Exception as e:
            logger.error(f"Error detecting emergence: {e}")
        
        return patterns
    
    def log_system_state(self):
        """Registra estado completo do sistema"""
        state = {
            "timestamp": datetime.now().isoformat(),
            "uptime_hours": (datetime.now() - self.start_time).total_seconds() / 3600,
            "cpu_usage": psutil.cpu_percent(interval=1),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "process_count": len(psutil.pids()),
            "emergence_events_count": len(self.emergence_events)
        }
        
        # Log to file
        log_file = self.monitor_dir / "system_state.jsonl"
        with open(log_file, 'a') as f:
            f.write(json.dumps(state) + '\n')
        
        return state
    
    def monitor_loop(self):
        """Loop principal de monitoramento"""
        logger.info("🔍 Iniciando monitoramento de emergência 24h+")
        
        while self.running:
            try:
                # Detect emergence patterns
                patterns = self.detect_emergence_patterns()
                
                # Log system state
                state = self.log_system_state()
                
                # Check for emergence
                emergence_detected = any(patterns.values())
                
                if emergence_detected:
                    logger.warning("🚨 EMERGÊNCIA DETECTADA!")
                    
                    emergence_report = {
                        "timestamp": datetime.now().isoformat(),
                        "patterns": patterns,
                        "system_state": state,
                        "events": self.emergence_events[-10:]  # Last 10 events
                    }
                    
                    # Save emergence report
                    report_file = self.monitor_dir / f"emergence_report_{int(time.time())}.json"
                    with open(report_file, 'w') as f:
                        json.dump(emergence_report, f, indent=2)
                
                # Sleep for monitoring interval
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(60)
    
    def start_monitoring(self, duration_hours=24):
        """Inicia monitoramento por período especificado"""
        self.running = True
        
        # Create monitoring thread
        monitor_thread = threading.Thread(target=self.monitor_loop)
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Create stop timer
        def stop_monitoring():
            time.sleep(duration_hours * 3600)
            self.running = False
            logger.info(f"🛑 Monitoramento finalizado após {duration_hours}h")
        
        stop_thread = threading.Thread(target=stop_monitoring)
        stop_thread.daemon = True
        stop_thread.start()
        
        return {
            "status": "started",
            "duration_hours": duration_hours,
            "start_time": self.start_time.isoformat(),
            "monitor_dir": str(self.monitor_dir)
        }
    
    def get_status(self):
        """Retorna status atual do monitoramento"""
        uptime = (datetime.now() - self.start_time).total_seconds() / 3600
        
        return {
            "running": self.running,
            "uptime_hours": uptime,
            "emergence_events": len(self.emergence_events),
            "last_events": self.emergence_events[-5:] if self.emergence_events else []
        }

def create_monitoring_service():
    """Cria serviço de monitoramento contínuo"""
    service_script = '''#!/usr/bin/env python3
import sys
sys.path.append('/root')
from emergence_monitor import EmergenceMonitor

if __name__ == "__main__":
    monitor = EmergenceMonitor()
    result = monitor.start_monitoring(duration_hours=24)
    
    print("🔍 MONITORAMENTO DE EMERGÊNCIA INICIADO")
    print("=" * 40)
    print(f"Duração: {result['duration_hours']}h")
    print(f"Início: {result['start_time']}")
    print(f"Logs: {result['monitor_dir']}")
    
    # Keep script running
    try:
        while monitor.running:
            status = monitor.get_status()
            print(f"\\rUptime: {status['uptime_hours']:.1f}h | Eventos: {status['emergence_events']}", end="")
            import time
            time.sleep(10)
    except KeyboardInterrupt:
        monitor.running = False
        print("\\n🛑 Monitoramento interrompido")
'''
    
    service_file = Path("/root/emergence_monitoring_service.py")
    with open(service_file, 'w') as f:
        f.write(service_script)
    
    service_file.chmod(0o755)
    return str(service_file)

if __name__ == "__main__":
    # Create and start monitoring
    monitor = EmergenceMonitor()
    service_file = create_monitoring_service()
    
    # Start short test monitoring (5 minutes for demo)
    result = monitor.start_monitoring(duration_hours=0.083)  # 5 minutes
    
    print("🔍 MONITORAMENTO DE EMERGÊNCIA - TESTE")
    print("=" * 40)
    print(f"✅ Serviço criado: {service_file}")
    print(f"✅ Monitoramento iniciado: {result['duration_hours']*60:.0f} minutos")
    print(f"📁 Logs em: {result['monitor_dir']}")
    
    # Monitor for test period
    import time
    test_duration = 30  # 30 seconds for demo
    for i in range(test_duration):
        status = monitor.get_status()
        print(f"\rTempo: {i+1}s | Eventos: {status['emergence_events']}", end="")
        time.sleep(1)
    
    monitor.running = False
    print(f"\n✅ Teste concluído. Para monitoramento 24h, execute: python3 {service_file}")
