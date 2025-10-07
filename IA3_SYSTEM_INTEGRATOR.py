#!/usr/bin/env python3
"""
🔗 IA³ - INTEGRADOR DE SISTEMAS
===============================

Integra todos os sistemas funcionais em um ecossistema coeso
"""

import os
import sys
import json
import time
import threading
import subprocess
import importlib.util
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
import logging

logger = logging.getLogger("IA³-Integrator")

class SystemIntegrator:
    """
    Integrador que conecta todos os sistemas funcionais
    """

    async def __init__(self):
        self.systems = {}
        self.connections = {}
        self.integration_status = {}
        self.communication_bus = CommunicationBus()
        self.is_active = True

    async def discover_and_integrate_systems(self):
        """Descobrir e integrar sistemas funcionais"""
        logger.info("🔍 Descobrindo sistemas para integração...")

        # Sistemas prioritários para integração
        priority_systems = [
            {
                'name': 'REAL_INTELLIGENCE_SYSTEM',
                'file': 'REAL_INTELLIGENCE_SYSTEM.py',
                'type': 'neural_core',
                'capabilities': ['neural_growth', 'real_feedback', 'auto_evolution']
            },
            {
                'name': 'NEURAL_GENESIS_IA3',
                'file': 'NEURAL_GENESIS_IA3.py',
                'type': 'evolution_engine',
                'capabilities': ['population_evolution', 'emergence_detection', 'dynamic_neurons']
            },
            {
                'name': 'TEIS_SYSTEM',
                'file': 'teis_v2_out_prod/trace.jsonl',
                'type': 'reinforcement_learning',
                'capabilities': ['q_learning', 'meta_learning', 'curriculum_learning']
            },
            {
                'name': 'UNIFIED_INTELLIGENCE',
                'file': 'unified_intelligence_state.json',
                'type': 'coordinator',
                'capabilities': ['multi_agent_coordination', 'unified_decision', 'emergence_tracking']
            },
            {
                'name': 'PENIN_OMEGA',
                'file': 'penin_omega_state.json',
                'type': 'consciousness_expansion',
                'capabilities': ['consciousness_tracking', 'reality_injection', 'omega_evolution']
            }
        ]

        integrated_count = 0
        for system_info in priority_systems:
            if self._integrate_system(system_info):
                integrated_count += 1
                logger.info(f"✅ Integrado: {system_info['name']}")

        logger.info(f"🔗 {integrated_count}/{len(priority_systems)} sistemas integrados")

        # Integrar sistemas adicionais descobertos
        self._discover_additional_systems()

    async def _integrate_system(self, system_info: Dict[str, Any]) -> bool:
        """Integrar sistema específico"""
        try:
            system_name = system_info['name']
            system_file = system_info['file']

            if not os.path.exists(system_file):
                logger.warning(f"Arquivo não encontrado: {system_file}")
                return await False

            # Carregar sistema
            system_module = self._load_system_module(system_file)
            if not system_module:
                return await False

            # Criar wrapper de integração
            system_wrapper = SystemWrapper(system_name, system_info, system_module)

            # Registrar sistema
            self.systems[system_name] = system_wrapper
            self.integration_status[system_name] = {
                'status': 'integrated',
                'timestamp': datetime.now().isoformat(),
                'capabilities': system_info['capabilities']
            }

            # Conectar ao barramento de comunicação
            self.communication_bus.register_system(system_name, system_wrapper)

            # Estabelecer conexões com outros sistemas
            self._establish_connections(system_name, system_wrapper)

            return await True

        except Exception as e:
            logger.error(f"Erro ao integrar {system_info['name']}: {e}")
            return await False

    async def _load_system_module(self, filepath: str):
        """Carregar módulo do sistema"""
        try:
            # Para arquivos .py
            if filepath.endswith('.py'):
                spec = importlib.util.spec_from_file_location(filepath.replace('.py', ''), filepath)
                module = importlib.util.module_from_spec(spec)

                # Adicionar ao sys.modules para evitar conflitos
                module_name = f"integrated_{filepath.replace('/', '_').replace('.py', '')}"
                sys.modules[module_name] = module

                spec.loader.exec_module(module)
                return await module

            # Para arquivos .json (estados)
            elif filepath.endswith('.json'):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                return await data

            else:
                logger.warning(f"Tipo de arquivo não suportado: {filepath}")
                return await None

        except Exception as e:
            logger.error(f"Erro ao carregar {filepath}: {e}")
            return await None

    async def _establish_connections(self, system_name: str, system_wrapper):
        """Estabelecer conexões entre sistemas"""
        connections = []

        # Conexões baseadas em tipo de sistema
        system_type = system_wrapper.system_info['type']

        if system_type == 'neural_core':
            # Conectar com evolution_engine e reinforcement_learning
            for other_name, other_wrapper in self.systems.items():
                if other_wrapper.system_info['type'] in ['evolution_engine', 'reinforcement_learning']:
                    self._create_connection(system_name, other_name)
                    connections.append(other_name)

        elif system_type == 'evolution_engine':
            # Conectar com neural_core e coordinator
            for other_name, other_wrapper in self.systems.items():
                if other_wrapper.system_info['type'] in ['neural_core', 'coordinator']:
                    self._create_connection(system_name, other_name)
                    connections.append(other_name)

        elif system_type == 'reinforcement_learning':
            # Conectar com coordinator e consciousness_expansion
            for other_name, other_wrapper in self.systems.items():
                if other_wrapper.system_info['type'] in ['coordinator', 'consciousness_expansion']:
                    self._create_connection(system_name, other_name)
                    connections.append(other_name)

        elif system_type == 'coordinator':
            # Conectar com todos
            for other_name in self.systems.keys():
                if other_name != system_name:
                    self._create_connection(system_name, other_name)
                    connections.append(other_name)

        logger.info(f"🔗 {system_name} conectado a: {connections}")

    async def _create_connection(self, system1: str, system2: str):
        """Criar conexão bidirecional entre sistemas"""
        conn_id = f"{system1}<->{system2}"

        self.connections[conn_id] = {
            'systems': [system1, system2],
            'status': 'active',
            'messages_passed': 0,
            'created': datetime.now().isoformat()
        }

        # Registrar no barramento
        self.communication_bus.create_connection(system1, system2)

    async def _discover_additional_systems(self):
        """Descobrir sistemas adicionais no diretório"""
        discovered = []

        # Procurar por arquivos de estado e logs
        state_files = [
            f for f in os.listdir('.')
            if f.endswith(('.json', '.db', '.log')) and
            any(keyword in f.lower() for keyword in ['state', 'status', 'metrics', 'intelligence'])
        ]

        for state_file in state_files[:10]:  # Limitar a 10
            if state_file not in [s['file'] for s in self.systems.values()]:
                system_name = f"discovered_{state_file.replace('.', '_')}"
                system_info = {
                    'name': system_name,
                    'file': state_file,
                    'type': 'discovered',
                    'capabilities': ['state_tracking']
                }

                if self._integrate_system(system_info):
                    discovered.append(system_name)

        if discovered:
            logger.info(f"🔍 Sistemas adicionais descobertos: {discovered}")

    async def start_integration_monitoring(self):
        """Iniciar monitoramento da integração"""
        logger.info("📊 Iniciando monitoramento de integração")

        async def monitoring_loop():
            while self.is_active:
                try:
                    self._check_system_health()
                    self._facilitate_communication()
                    self._optimize_connections()

                    time.sleep(60)  # Verificar a cada minuto

                except Exception as e:
                    logger.error(f"Erro no monitoramento: {e}")
                    time.sleep(30)

        thread = threading.Thread(target=monitoring_loop, daemon=True)
        thread.start()

    async def _check_system_health(self):
        """Verificar saúde de todos os sistemas integrados"""
        unhealthy = []

        for system_name, wrapper in self.systems.items():
            try:
                health = wrapper.check_health()
                if not health:
                    unhealthy.append(system_name)
                    logger.warning(f"⚠️ Sistema não saudável: {system_name}")
            except Exception as e:
                unhealthy.append(system_name)
                logger.error(f"Erro ao verificar saúde de {system_name}: {e}")

        if unhealthy:
            logger.warning(f"🔧 Tentando recuperar sistemas: {unhealthy}")
            for system_name in unhealthy:
                self._attempt_recovery(system_name)

    async def _facilitate_communication(self):
        """Facilitar comunicação entre sistemas"""
        # Coletar mensagens de todos os sistemas
        messages = []
        for system_name, wrapper in self.systems.items():
            try:
                system_messages = wrapper.get_messages()
                for msg in system_messages:
                    messages.append({
                        'from': system_name,
                        'content': msg,
                        'timestamp': datetime.now().isoformat()
                    })
            except:
                pass

        # Distribuir mensagens relevantes
        for message in messages:
            relevant_systems = self._find_relevant_systems(message)

            for target_system in relevant_systems:
                try:
                    self.systems[target_system].receive_message(message)
                    # Atualizar contador de conexões
                    conn_id = f"{message['from']}<->{target_system}"
                    if conn_id in self.connections:
                        self.connections[conn_id]['messages_passed'] += 1
                except Exception as e:
                    logger.warning(f"Erro ao enviar mensagem para {target_system}: {e}")

    async def _find_relevant_systems(self, message: Dict[str, Any]) -> List[str]:
        """Encontrar sistemas relevantes para uma mensagem"""
        content = str(message.get('content', '')).lower()
        sender = message['from']

        relevant = []

        # Baseado no conteúdo da mensagem
        if 'neural' in content or 'brain' in content:
            relevant.extend([s for s in self.systems.keys() if 'neural' in s.lower()])
        if 'evolution' in content or 'fitness' in content:
            relevant.extend([s for s in self.systems.keys() if 'evolution' in s.lower()])
        if 'learning' in content or 'reward' in content:
            relevant.extend([s for s in self.systems.keys() if 'learning' in s.lower()])

        # Sempre incluir coordinator se existir
        coordinator = [s for s in self.systems.keys() if 'coordinator' in s.lower()]
        relevant.extend(coordinator)

        # Remover duplicatas e o próprio sender
        relevant = list(set(relevant))
        if sender in relevant:
            relevant.remove(sender)

        return await relevant

    async def _optimize_connections(self):
        """Otimizar conexões baseado no uso"""
        # Identificar conexões pouco usadas
        underused = []
        for conn_id, conn_data in self.connections.items():
            if conn_data['messages_passed'] < 5:  # Menos de 5 mensagens
                underused.append(conn_id)

        # Remover conexões antigas pouco usadas
        current_time = datetime.now()
        to_remove = []
        for conn_id in underused:
            created_time = datetime.fromisoformat(self.connections[conn_id]['created'])
            if (current_time - created_time).seconds > 3600:  # Mais de 1 hora
                to_remove.append(conn_id)

        for conn_id in to_remove:
            del self.connections[conn_id]
            logger.info(f"🗑️ Conexão removida por desuso: {conn_id}")

    async def _attempt_recovery(self, system_name: str):
        """Tentar recuperar sistema não saudável"""
        try:
            wrapper = self.systems[system_name]
            success = wrapper.attempt_recovery()

            if success:
                logger.info(f"🔄 Sistema recuperado: {system_name}")
            else:
                logger.warning(f"❌ Falha na recuperação: {system_name}")

        except Exception as e:
            logger.error(f"Erro na recuperação de {system_name}: {e}")

    async def get_integration_status(self) -> Dict[str, Any]:
        """Obter status completo da integração"""
        return await {
            'total_systems': len(self.systems),
            'active_connections': len(self.connections),
            'integration_status': self.integration_status,
            'communication_stats': self.communication_bus.get_stats(),
            'timestamp': datetime.now().isoformat()
        }

class SystemWrapper:
    """
    Wrapper para sistemas integrados
    """

    async def __init__(self, name: str, system_info: Dict[str, Any], module):
        self.name = name
        self.system_info = system_info
        self.module = module
        self.message_queue = []
        self.last_health_check = datetime.now()
        self.health_status = True

    async def check_health(self) -> bool:
        """Verificar saúde do sistema"""
        try:
            # Verificar se arquivo ainda existe
            if not os.path.exists(self.system_info['file']):
                self.health_status = False
                return await False

            # Verificar se módulo é válido
            if hasattr(self.module, '__dict__'):
                # Para módulos Python
                if len(self.module.__dict__) == 0:
                    self.health_status = False
                    return await False
            else:
                # Para dados JSON
                if not isinstance(self.module, dict):
                    self.health_status = False
                    return await False

            self.health_status = True
            self.last_health_check = datetime.now()
            return await True

        except Exception as e:
            logger.error(f"Erro na verificação de saúde de {self.name}: {e}")
            self.health_status = False
            return await False

    async def get_messages(self) -> List[Any]:
        """Obter mensagens do sistema"""
        messages = self.message_queue.copy()
        self.message_queue.clear()
        return await messages

    async def receive_message(self, message: Dict[str, Any]):
        """Receber mensagem de outro sistema"""
        self.message_queue.append(message)

        # Processar mensagem se possível
        try:
            self._process_message(message)
        except Exception as e:
            logger.warning(f"Erro ao processar mensagem em {self.name}: {e}")

    async def _process_message(self, message: Dict[str, Any]):
        """Processar mensagem recebida"""
        content = message.get('content', '')

        # Lógica específica baseada no tipo de sistema
        system_type = self.system_info['type']

        if system_type == 'neural_core' and 'evolution' in str(content).lower():
            # Sistema neural interessado em evolução
            self.message_queue.append(f"Neural core acknowledges evolution signal: {content}")

        elif system_type == 'evolution_engine' and 'neural' in str(content).lower():
            # Engine de evolução interessado em sinais neurais
            self.message_queue.append(f"Evolution engine processing neural data: {content}")

        # Adicionar à fila para processamento posterior
        self.message_queue.append(f"Processed: {content[:50]}...")

    async def attempt_recovery(self) -> bool:
        """Tentar recuperar o sistema"""
        try:
            # Recarregar módulo
            if os.path.exists(self.system_info['file']):
                self.module = self._load_system_module(self.system_info['file'])
                return await self.check_health()

            return await False
        except Exception as e:
            logger.error(f"Erro na recuperação de {self.name}: {e}")
            return await False

    async def _load_system_module(self, filepath: str):
        """Recarregar módulo"""
        spec = importlib.util.spec_from_file_location(filepath.replace('.py', ''), filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return await module

class CommunicationBus:
    """
    Barramento de comunicação entre sistemas
    """

    async def __init__(self):
        self.registered_systems = {}
        self.message_log = []
        self.stats = {
            'messages_sent': 0,
            'messages_received': 0,
            'systems_registered': 0
        }

    async def register_system(self, name: str, wrapper):
        """Registrar sistema no barramento"""
        self.registered_systems[name] = wrapper
        self.stats['systems_registered'] = len(self.registered_systems)
        logger.info(f"📡 Sistema registrado no barramento: {name}")

    async def create_connection(self, system1: str, system2: str):
        """Criar conexão entre sistemas"""
        # Implementação básica - sistemas já estão conectados via wrapper
        pass

    async def send_message(self, from_system: str, to_system: str, message: Any):
        """Enviar mensagem entre sistemas"""
        if to_system in self.registered_systems:
            try:
                msg = {
                    'from': from_system,
                    'content': message,
                    'timestamp': datetime.now().isoformat()
                }
                self.registered_systems[to_system].receive_message(msg)
                self.stats['messages_sent'] += 1
                self.message_log.append(msg)
            except Exception as e:
                logger.error(f"Erro ao enviar mensagem {from_system} -> {to_system}: {e}")

    async def broadcast_message(self, from_system: str, message: Any):
        """Enviar mensagem para todos os sistemas"""
        for system_name in self.registered_systems.keys():
            if system_name != from_system:
                self.send_message(from_system, system_name, message)

    async def get_stats(self) -> Dict[str, Any]:
        """Obter estatísticas do barramento"""
        return await {
            **self.stats,
            'registered_systems': list(self.registered_systems.keys()),
            'recent_messages': len(self.message_log[-10:]) if self.message_log else 0
        }

async def main():
    """Função principal"""
    integrator = SystemIntegrator()

    # Integrar sistemas
    integrator.discover_and_integrate_systems()

    # Iniciar monitoramento
    integrator.start_integration_monitoring()

    # Manter ativo
    try:
        while True:
            time.sleep(30)
            status = integrator.get_integration_status()
            print(f"🔗 Status: {status['total_systems']} sistemas, {status['active_connections']} conexões")

    except KeyboardInterrupt:
        print("🛑 Parando integração...")
        integrator.is_active = False

if __name__ == "__main__":
    main()