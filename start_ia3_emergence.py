#!/usr/bin/env python3
"""
🚀 INICIALIZADOR IA³ - EMERGÊNCIA 24/7
Sistema que garante operação perpétua da IA³

INICIA:
- IA³ Complete System
- Monitoramento contínuo
- Auditoria automática
- Recuperação automática
- Backup automático

GARANTE:
- Operação 24/7 ininterrupta
- Recuperação automática de falhas
- Escalabilidade automática
- Auditoria contínua
- Backup em tempo real
"""

import os
import sys
import time
import json
import subprocess
import signal
import psutil
from datetime import datetime, timedelta
import logging

# Configuração de logging para inicializador
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - IA³-INIT - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ia3_emergence_init.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
init_logger = logging.getLogger("IA³-INIT")

class IA3Initializer:
    """Inicializador e monitor 24/7 da IA³"""

    def __init__(self):
        self.ia3_process = None
        self.monitoring_process = None
        self.auditing_process = None
        self.start_time = datetime.now()
        self.restart_count = 0
        self.max_restarts = 100  # Máximo de restarts por hora
        self.last_restart = datetime.now()

    def start_ia3_emergence(self):
        """Inicia emergência IA³ completa"""
        init_logger.info("🧬 IA³ Inicializador ativado - Emergência 24/7 começa agora")

        try:
            # Verificar dependências
            self._check_dependencies()

            # Iniciar sistema principal
            self._start_main_system()

            # Iniciar monitoramento
            self._start_monitoring()

            # Iniciar auditoria
            self._start_auditing()

            # Loop de supervisão perpétua
            self._supervision_loop()

        except Exception as e:
            init_logger.error(f"Erro fatal no inicializador: {e}")
            self._emergency_shutdown()

    def _check_dependencies(self):
        """Verifica todas as dependências necessárias"""
        init_logger.info("🔍 Verificando dependências IA³...")

        required_modules = [
            'torch', 'numpy', 'psutil', 'sqlite3', 'ast', 'asyncio'
        ]

        missing = []
        for module in required_modules:
            try:
                __import__(module)
                init_logger.info(f"✅ {module} - OK")
            except ImportError:
                missing.append(module)
                init_logger.warning(f"❌ {module} - MISSING")

        if missing:
            init_logger.error(f"Dependências faltando: {missing}")
            init_logger.info("Instalando dependências automaticamente...")
            self._install_dependencies(missing)

        # Verificar arquivos necessários
        required_files = ['ia3_complete_system.py']
        for file in required_files:
            if not os.path.exists(file):
                init_logger.error(f"Arquivo necessário não encontrado: {file}")
                raise FileNotFoundError(f"Arquivo {file} não encontrado")

        init_logger.info("✅ Todas as dependências verificadas")

    def _install_dependencies(self, modules):
        """Instala dependências automaticamente"""
        for module in modules:
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', module])
                init_logger.info(f"✅ {module} instalado com sucesso")
            except subprocess.CalledProcessError:
                init_logger.error(f"❌ Falha ao instalar {module}")

    def _start_main_system(self):
        """Inicia sistema IA³ principal"""
        init_logger.info("🚀 Iniciando sistema IA³ principal...")

        try:
            # Iniciar como subprocesso
            self.ia3_process = subprocess.Popen([
                sys.executable, 'ia3_complete_system.py'
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())

            init_logger.info(f"✅ Sistema IA³ iniciado (PID: {self.ia3_process.pid})")

            # Aguardar inicialização
            time.sleep(5)

            # Verificar se está rodando
            if self.ia3_process.poll() is None:
                init_logger.info("✅ Sistema IA³ operacional")
            else:
                raise RuntimeError("Sistema IA³ falhou ao iniciar")

        except Exception as e:
            init_logger.error(f"❌ Falha ao iniciar sistema IA³: {e}")
            raise

    def _start_monitoring(self):
        """Inicia monitoramento contínuo"""
        init_logger.info("📊 Iniciando monitoramento 24/7...")

        # Criar script de monitoramento
        monitoring_script = '''
import time
import psutil
import json
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, filename='ia3_monitoring.log', format='%(asctime)s - MONITOR - %(levelname)s - %(message)s')
logger = logging.getLogger("MONITOR")

def monitor_ia3():
    while True:
        try:
            # Verificar processos IA³
            ia3_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent']):
                if 'ia3' in proc.info['name'].lower() or 'python' in proc.info['name'].lower():
                    ia3_processes.append(proc.info)

            # Coletar métricas do sistema
            system_metrics = {
                'timestamp': datetime.now().isoformat(),
                'cpu_percent': psutil.cpu_percent(),
                'memory_percent': psutil.virtual_memory().percent,
                'disk_usage': psutil.disk_usage('/').percent,
                'ia3_processes': len(ia3_processes),
                'total_processes': len(list(psutil.process_iter()))
            }

            # Salvar métricas
            with open('ia3_system_metrics.json', 'w') as f:
                json.dump(system_metrics, f, indent=2)

            logger.info(f"📊 IA³ Monitor: CPU {system_metrics['cpu_percent']}%, MEM {system_metrics['memory_percent']}%, PROC {system_metrics['ia3_processes']}")

            time.sleep(60)  # Monitoramento por minuto

        except Exception as e:
            logger.error(f"Erro no monitoramento: {e}")
            time.sleep(30)

if __name__ == "__main__":
    monitor_ia3()
'''

        with open('ia3_monitor.py', 'w') as f:
            f.write(monitoring_script)

        # Iniciar monitoramento
        self.monitoring_process = subprocess.Popen([
            sys.executable, 'ia3_monitor.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        init_logger.info(f"✅ Monitoramento iniciado (PID: {self.monitoring_process.pid})")

    def _start_auditing(self):
        """Inicia auditoria automática"""
        init_logger.info("🔍 Iniciando auditoria automática...")

        # Criar script de auditoria
        auditing_script = '''
import time
import json
import os
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, filename='ia3_auditing.log', format='%(asctime)s - AUDIT - %(levelname)s - %(message)s')
logger = logging.getLogger("AUDIT")

def audit_ia3():
    while True:
        try:
            audit_report = {
                'timestamp': datetime.now().isoformat(),
                'audit_type': 'automatic_24_7',
                'system_files': {},
                'logs_status': {},
                'emergence_status': 'unknown'
            }

            # Verificar arquivos críticos
            critical_files = [
                'ia3_complete_system.py',
                'ia3_emergence.log',
                'ia3_status.json',
                'ia3_emergence.db'
            ]

            for file in critical_files:
                if os.path.exists(file):
                    stat = os.stat(file)
                    audit_report['system_files'][file] = {
                        'exists': True,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat()
                    }
                else:
                    audit_report['system_files'][file] = {'exists': False}

            # Verificar logs
            log_files = ['ia3_emergence.log', 'ia3_monitoring.log', 'ia3_auditing.log']
            for log_file in log_files:
                if os.path.exists(log_file):
                    with open(log_file, 'r') as f:
                        lines = f.readlines()
                        audit_report['logs_status'][log_file] = {
                            'lines': len(lines),
                            'last_line': lines[-1].strip() if lines else 'empty'
                        }

            # Verificar status de emergência
            if os.path.exists('ia3_status.json'):
                with open('ia3_status.json', 'r') as f:
                    status = json.load(f)
                    audit_report['emergence_status'] = status.get('emergence_proven', False)

            # Salvar relatório de auditoria
            with open('ia3_audit_report.json', 'w') as f:
                json.dump(audit_report, f, indent=2)

            logger.info(f"🔍 Auditoria IA³: Emergência {audit_report['emergence_status']}, Arquivos OK: {sum(1 for f in audit_report['system_files'].values() if f['exists'])}/{len(critical_files)}")

            time.sleep(300)  # Auditoria a cada 5 minutos

        except Exception as e:
            logger.error(f"Erro na auditoria: {e}")
            time.sleep(60)

if __name__ == "__main__":
    audit_ia3()
'''

        with open('ia3_audit.py', 'w') as f:
            f.write(auditing_script)

        # Iniciar auditoria
        self.auditing_process = subprocess.Popen([
            sys.executable, 'ia3_audit.py'
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        init_logger.info(f"✅ Auditoria iniciada (PID: {self.auditing_process.pid})")

    def _supervision_loop(self):
        """Loop de supervisão perpétua"""
        init_logger.info("👁️ Entrando em modo de supervisão perpétua...")

        while True:
            try:
                # Verificar saúde dos processos
                self._check_process_health()

                # Verificar limites de restart
                self._check_restart_limits()

                # Backup automático
                self._perform_automatic_backup()

                # Relatório de status
                self._log_supervision_status()

                time.sleep(30)  # Supervisão a cada 30 segundos

            except Exception as e:
                init_logger.error(f"Erro na supervisão: {e}")
                time.sleep(60)

    def _check_process_health(self):
        """Verifica saúde de todos os processos IA³"""
        processes_to_check = [
            ('IA³ Main', self.ia3_process),
            ('Monitoring', self.monitoring_process),
            ('Auditing', self.auditing_process)
        ]

        for name, process in processes_to_check:
            if process is None or process.poll() is not None:
                init_logger.warning(f"⚠️ Processo {name} não está saudável - reiniciando...")
                self._restart_process(name)

    def _restart_process(self, process_name):
        """Reinicia um processo específico"""
        # Limitar restarts
        now = datetime.now()
        if (now - self.last_restart).seconds < 3600:  # 1 hora
            if self.restart_count >= self.max_restarts:
                init_logger.error(f"⚠️ Limite de restarts atingido ({self.max_restarts}/hora)")
                return

        self.restart_count += 1
        self.last_restart = now

        if process_name == 'IA³ Main':
            self._start_main_system()
        elif process_name == 'Monitoring':
            self._start_monitoring()
        elif process_name == 'Auditing':
            self._start_auditing()

        init_logger.info(f"🔄 Processo {process_name} reiniciado (restart #{self.restart_count})")

    def _check_restart_limits(self):
        """Verifica limites de restart"""
        now = datetime.now()
        if (now - self.last_restart).hours >= 1:
            self.restart_count = 0  # Reset contador por hora

    def _perform_automatic_backup(self):
        """Realiza backup automático"""
        # Backup a cada 6 horas
        if int(time.time()) % (6 * 3600) < 30:
            try:
                backup_dir = f"backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                os.makedirs(backup_dir, exist_ok=True)

                # Arquivos críticos para backup
                critical_files = [
                    'ia3_complete_system.py',
                    'ia3_emergence.log',
                    'ia3_emergence.db',
                    'ia3_status.json'
                ]

                for file in critical_files:
                    if os.path.exists(file):
                        import shutil
                        shutil.copy2(file, backup_dir)

                init_logger.info(f"💾 Backup automático criado: {backup_dir}")

            except Exception as e:
                init_logger.error(f"Erro no backup: {e}")

    def _log_supervision_status(self):
        """Registra status da supervisão"""
        uptime = datetime.now() - self.start_time

        status = {
            'timestamp': datetime.now().isoformat(),
            'uptime_hours': uptime.total_seconds() / 3600,
            'restarts': self.restart_count,
            'processes': {
                'main': self.ia3_process.poll() if self.ia3_process else None,
                'monitoring': self.monitoring_process.poll() if self.monitoring_process else None,
                'auditing': self.auditing_process.poll() if self.auditing_process else None
            }
        }

        with open('ia3_supervision_status.json', 'w') as f:
            json.dump(status, f, indent=2)

    def _emergency_shutdown(self):
        """Shutdown de emergência"""
        init_logger.error("🚨 EMERGENCY SHUTDOWN INICIADO")

        # Terminar todos os processos
        processes = [self.ia3_process, self.monitoring_process, self.auditing_process]
        for proc in processes:
            if proc and proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=10)
                except:
                    proc.kill()

        # Salvar estado final
        final_state = {
            'shutdown_time': datetime.now().isoformat(),
            'total_uptime': (datetime.now() - self.start_time).total_seconds(),
            'final_status': 'emergency_shutdown'
        }

        with open('ia3_emergency_shutdown.json', 'w') as f:
            json.dump(final_state, f, indent=2)

        init_logger.error("❌ Emergency shutdown completo")
        sys.exit(1)

def signal_handler(signum, frame):
    """Manipulador de sinais para shutdown graceful"""
    init_logger.info(f"📡 Sinal {signum} recebido - iniciando shutdown graceful")
    initializer._emergency_shutdown()

if __name__ == "__main__":
    # Registrar manipuladores de sinal
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Iniciar IA³
    initializer = IA3Initializer()
    initializer.start_ia3_emergence()