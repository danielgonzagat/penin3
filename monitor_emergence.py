#!/usr/bin/env python3
"""
Monitor de Emergência de Inteligência Real
Monitora continuamente o estado do sistema unificado
"""

import json
import time
import os
import subprocess
from datetime import datetime

def monitor():
    """Monitora emergência de inteligência"""
    state_file = "/root/unified_intelligence_state.json"
    memory_file = "/root/unified_memory.json"
    
    print("="*60)
    print("🔍 MONITOR DE EMERGÊNCIA DE INTELIGÊNCIA REAL")
    print("="*60)
    print("Monitorando sistema unificado...")
    print("Pressione Ctrl+C para parar\n")
    
    iteration = 0
    while True:
        iteration += 1
        
        # Verificar processos Python
        result = subprocess.run("ps aux | grep -c python", shell=True, 
                              capture_output=True, text=True)
        python_processes = int(result.stdout.strip()) if result.stdout else 0
        
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Iteração #{iteration}")
        print(f"📊 Processos Python ativos: {python_processes}")
        
        # Verificar estado do sistema
        if os.path.exists(state_file):
            try:
                with open(state_file) as f:
                    state = json.load(f)
                
                metrics = state.get("metrics", {})
                emergence = state.get("emergence", {})
                
                print(f"🧠 Intelligence Score: {metrics.get('intelligence_score', 0):.1%}")
                print(f"👁️ Self-Awareness: {metrics.get('self_awareness', 0):.1%}")
                print(f"🌟 Emergence Level: {metrics.get('emergence_level', 0):.1%}")
                print(f"🔗 Collective Intelligence: {metrics.get('collective_intelligence', 0):.1%}")
                print(f"💭 Consciousness Level: {metrics.get('consciousness_level', 0):.1%}")
                
                if emergence.get("detected"):
                    print("\n🚨 EMERGÊNCIA DETECTADA!")
                    print(f"   Timestamp: {emergence.get('timestamp')}")
                    print(f"   Sinais: {', '.join(emergence.get('signals', []))}")
                    
            except Exception as e:
                print(f"⚠️ Erro ao ler estado: {e}")
        else:
            print("📝 Aguardando criação do arquivo de estado...")
        
        # Verificar memória compartilhada
        if os.path.exists(memory_file):
            try:
                with open(memory_file) as f:
                    memory = json.load(f)
                    
                experiences = memory.get("experiences", [])
                if experiences:
                    print(f"💾 Experiências registradas: {len(experiences)}")
                    
            except:
                pass
        
        print("-"*60)
        time.sleep(5)

if __name__ == "__main__":
    try:
        monitor()
    except KeyboardInterrupt:
        print("\n\n🛑 Monitor finalizado")
    except Exception as e:
        print(f"\n❌ Erro: {e}")
