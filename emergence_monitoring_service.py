#!/usr/bin/env python3
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
            print(f"\rUptime: {status['uptime_hours']:.1f}h | Eventos: {status['emergence_events']}", end="")
            import time
            time.sleep(10)
    except KeyboardInterrupt:
        monitor.running = False
        print("\n🛑 Monitoramento interrompido")
