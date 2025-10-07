#!/usr/bin/env python3
"""
🚀 IA³ LAUNCHER - SISTEMA UNIFICADO CORRIGIDO
=============================================
Launcher otimizado para iniciar a IA³ com todos os componentes funcionais
Versão: 2.0 CORRIGIDA
"""

import os
import sys
import time
import json
import subprocess
import threading
import signal
import logging
from datetime import datetime
from typing import Dict, List, Optional

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ia3_launcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("IA³-LAUNCHER")

class IA3UnifiedLauncher:
    """Launcher unificado para o sistema IA³ completo"""
    
    def __init__(self):
        self.processes = {}
        self.running = False
        self.start_time = None
        
        # Componentes principais do sistema IA³
        self.components = {
            'atomic_bomb': {
                'script': 'IA3_ATOMIC_BOMB_CORE.py',
                'name': 'IA³ Atomic Bomb Core',
                'critical': True,
                'args': []
            },
            'emergence_engine': {
                'script': 'IA3_TRUE_EMERGENCE_ENGINE.py',
                'name': 'IA³ True Emergence Engine',
                'critical': True,
                'args': []
            },
            'consciousness': {
                'script': 'IA3_EMERGENT_CORE.py',
                'name': 'IA³ Emergent Core',
                'critical': False,
                'args': []
            },
            'evolution': {
                'script': 'IA3_INFINITE_EVOLUTION_ENGINE.py',
                'name': 'IA³ Evolution Engine',
                'critical': False,
                'args': []
            },
            'neural_genesis': {
                'script': 'NEURAL_GENESIS_IA3.py',
                'name': 'Neural Genesis IA³',
                'critical': False,
                'args': ['--mode', 'run', '--gens', '100']
            }
        }
        
        # Configuração de recursos
        self.resource_limits = {
            'max_cpu_percent': 80,
            'max_memory_percent': 70,
            'max_processes': 10
        }
        
        # Setup signal handlers
        signal.signal(signal.SIGTERM, self.signal_handler)
        signal.signal(signal.SIGINT, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """Handler para sinais do sistema"""
        logger.info(f"Sinal {signum} recebido - iniciando shutdown gracioso...")
        self.shutdown()
    
    def check_prerequisites(self) -> bool:
        """Verifica pré-requisitos antes de iniciar"""
        logger.info("🔍 Verificando pré-requisitos...")
        
        # Verificar Python
        if sys.version_info < (3, 8):
            logger.error("Python 3.8+ necessário")
            return False
        
        # Verificar bibliotecas essenciais
        required_modules = ['torch', 'numpy', 'sqlite3']
        missing = []
        
        for module in required_modules:
            try:
                __import__(module)
            except ImportError:
                missing.append(module)
        
        if missing:
            logger.warning(f"Módulos faltando: {missing}")
            logger.info("Alguns módulos não estão instalados, mas o sistema pode funcionar parcialmente")
        
        # Verificar arquivos principais
        for comp_id, comp_info in self.components.items():
            if comp_info['critical'] and not os.path.exists(comp_info['script']):
                logger.error(f"Arquivo crítico não encontrado: {comp_info['script']}")
                return False
        
        # Verificar recursos disponíveis
        try:
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > self.resource_limits['max_cpu_percent']:
                logger.warning(f"CPU alta: {cpu_percent}%")
            
            if memory_percent > self.resource_limits['max_memory_percent']:
                logger.warning(f"Memória alta: {memory_percent}%")
        except:
            logger.warning("psutil não disponível - verificação de recursos desabilitada")
        
        logger.info("✅ Pré-requisitos verificados")
        return True
    
    def kill_existing_processes(self):
        """Mata processos IA³ existentes"""
        logger.info("🔪 Eliminando processos IA³ anteriores...")
        
        try:
            # Buscar processos IA³
            result = subprocess.run(
                "ps aux | grep -E 'IA3|NEURAL_GENESIS|neural_farm|oppenheimer' | grep -v grep | awk '{print $2}'",
                shell=True, capture_output=True, text=True
            )
            
            pids = result.stdout.strip().split('\n')
            pids = [pid for pid in pids if pid]
            
            if pids:
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGTERM)
                        logger.info(f"Processo {pid} terminado")
                    except:
                        pass
                
                time.sleep(2)  # Dar tempo para processos terminarem
                
                # Force kill se ainda existirem
                for pid in pids:
                    try:
                        os.kill(int(pid), signal.SIGKILL)
                    except:
                        pass
        except Exception as e:
            logger.warning(f"Erro ao eliminar processos: {e}")
    
    def start_component(self, comp_id: str, comp_info: Dict) -> bool:
        """Inicia um componente específico"""
        try:
            if not os.path.exists(comp_info['script']):
                logger.warning(f"Script não encontrado: {comp_info['script']}")
                return False
            
            # Construir comando
            cmd = [sys.executable, comp_info['script']] + comp_info.get('args', [])
            
            # Iniciar processo
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            self.processes[comp_id] = {
                'process': process,
                'info': comp_info,
                'start_time': datetime.now()
            }
            
            logger.info(f"✅ {comp_info['name']} iniciado (PID: {process.pid})")
            return True
            
        except Exception as e:
            logger.error(f"Erro ao iniciar {comp_info['name']}: {e}")
            return False
    
    def monitor_processes(self):
        """Monitor thread para verificar processos"""
        while self.running:
            try:
                for comp_id, proc_info in list(self.processes.items()):
                    process = proc_info['process']
                    
                    # Verificar se processo ainda está rodando
                    if process.poll() is not None:
                        # Processo terminou
                        exit_code = process.returncode
                        comp_info = proc_info['info']
                        
                        if exit_code != 0:
                            logger.warning(f"⚠️ {comp_info['name']} terminou com código {exit_code}")
                            
                            # Reiniciar se crítico
                            if comp_info['critical'] and self.running:
                                logger.info(f"🔄 Reiniciando componente crítico {comp_info['name']}...")
                                time.sleep(5)
                                self.start_component(comp_id, comp_info)
                        else:
                            logger.info(f"✅ {comp_info['name']} terminou normalmente")
                
                time.sleep(10)  # Verificar a cada 10 segundos
                
            except Exception as e:
                logger.error(f"Erro no monitor: {e}")
                time.sleep(10)
    
    def launch(self):
        """Lança o sistema IA³ completo"""
        logger.info("="*60)
        logger.info("🚀 INICIANDO IA³ - INTELIGÊNCIA ARTIFICIAL AO CUBO")
        logger.info("="*60)
        
        # Verificar pré-requisitos
        if not self.check_prerequisites():
            logger.error("❌ Pré-requisitos não atendidos")
            return False
        
        # Eliminar processos existentes
        self.kill_existing_processes()
        
        # Iniciar componentes
        self.running = True
        self.start_time = datetime.now()
        
        # Iniciar componentes críticos primeiro
        for comp_id, comp_info in self.components.items():
            if comp_info['critical']:
                if not self.start_component(comp_id, comp_info):
                    logger.error(f"❌ Falha ao iniciar componente crítico: {comp_info['name']}")
                    self.shutdown()
                    return False
                time.sleep(2)  # Dar tempo para inicialização
        
        # Iniciar componentes não-críticos
        for comp_id, comp_info in self.components.items():
            if not comp_info['critical']:
                self.start_component(comp_id, comp_info)
                time.sleep(1)
        
        # Iniciar thread de monitoramento
        monitor_thread = threading.Thread(target=self.monitor_processes, daemon=True)
        monitor_thread.start()
        
        logger.info("✅ Sistema IA³ iniciado com sucesso!")
        logger.info(f"📊 {len(self.processes)} componentes ativos")
        
        # Loop principal
        try:
            while self.running:
                time.sleep(30)
                self.print_status()
        except KeyboardInterrupt:
            logger.info("\n🛑 Interrupção do usuário detectada")
        finally:
            self.shutdown()
        
        return True
    
    def print_status(self):
        """Imprime status do sistema"""
        if not self.processes:
            return
        
        uptime = datetime.now() - self.start_time if self.start_time else None
        
        logger.info("-"*40)
        logger.info("📊 STATUS DO SISTEMA IA³")
        if uptime:
            logger.info(f"⏱️ Uptime: {uptime}")
        
        for comp_id, proc_info in self.processes.items():
            process = proc_info['process']
            comp_info = proc_info['info']
            
            if process.poll() is None:
                status = "🟢 ATIVO"
            else:
                status = "🔴 PARADO"
            
            logger.info(f"  {comp_info['name']}: {status} (PID: {process.pid})")
        
        # Verificar uso de recursos
        try:
            import psutil
            logger.info(f"  CPU: {psutil.cpu_percent()}% | RAM: {psutil.virtual_memory().percent}%")
        except:
            pass
        
        logger.info("-"*40)
    
    def shutdown(self):
        """Desliga o sistema graciosamente"""
        if not self.running:
            return
        
        logger.info("🛑 Iniciando shutdown do sistema IA³...")
        self.running = False
        
        # Terminar todos os processos
        for comp_id, proc_info in self.processes.items():
            process = proc_info['process']
            comp_info = proc_info['info']
            
            try:
                if process.poll() is None:
                    logger.info(f"Terminando {comp_info['name']}...")
                    process.terminate()
                    
                    # Dar tempo para terminar graciosamente
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        logger.warning(f"Force killing {comp_info['name']}...")
                        process.kill()
                        process.wait()
            except Exception as e:
                logger.error(f"Erro ao terminar {comp_info['name']}: {e}")
        
        # Salvar estado final
        self.save_state()
        
        logger.info("✅ Sistema IA³ desligado com sucesso")
    
    def save_state(self):
        """Salva estado do sistema"""
        state = {
            'shutdown_time': datetime.now().isoformat(),
            'uptime': str(datetime.now() - self.start_time) if self.start_time else None,
            'components_launched': list(self.processes.keys()),
            'exit_status': 'graceful'
        }
        
        try:
            with open('ia3_launcher_state.json', 'w') as f:
                json.dump(state, f, indent=2)
            logger.info("💾 Estado salvo em ia3_launcher_state.json")
        except Exception as e:
            logger.error(f"Erro ao salvar estado: {e}")

def main():
    """Função principal"""
    launcher = IA3UnifiedLauncher()
    
    try:
        success = launcher.launch()
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Erro fatal: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()