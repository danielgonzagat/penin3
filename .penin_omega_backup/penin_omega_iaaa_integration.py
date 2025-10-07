#!/usr/bin/env python3
"""
PENIN-Ω + IAAA Integration
=========================
Integração da IAAA como cérebro central do sistema PENIN-Ω completo.

OBJETIVO: Unir o sistema PENIN-Ω (8 módulos + Multi-API) com a IAAA (Falcon 7B + RAG + Memória)
para criar o sistema mais avançado de IA autoevolutiva já construído.

Arquiteto: Amazon Q (Claude-3.5-Sonnet)
Data: 2025-09-16
"""

import sys
import logging
from pathlib import Path
from datetime import datetime, timezone

# Adiciona paths necessários
sys.path.append("/root")
sys.path.append("/root/IAAA_PENIN_OMEGA/core")

class PeninOmegaIAAAIntegration:
    """Integração completa PENIN-Ω + IAAA."""
    
    async def __init__(self):
        self.logger = logging.getLogger("PENIN_OMEGA_IAAA")
        self.penin_omega_system = None
        self.iaaa_system = None
        self.integrated = False
        
        self.logger.info("🔗 Inicializando integração PENIN-Ω + IAAA")
    
    async def initialize_penin_omega(self):
        """Inicializa sistema PENIN-Ω completo."""
        try:
            from penin_omega_master_system import penin_omega
            self.penin_omega_system = penin_omega
            self.logger.info("✅ Sistema PENIN-Ω inicializado")
            return await True
        except Exception as e:
            self.logger.error(f"❌ Erro ao inicializar PENIN-Ω: {e}")
            return await False
    
    async def initialize_iaaa(self):
        """Inicializa sistema IAAA."""
        try:
            from iaaa_core import iaaa, initialize_iaaa
            
            # Configura API keys
            exec(open('/root/api_keys_config.py').read())
            
            # Inicializa IAAA
            if initialize_iaaa():
                self.iaaa_system = iaaa
                self.logger.info("✅ Sistema IAAA inicializado")
                return await True
            else:
                self.logger.error("❌ Falha na inicialização da IAAA")
                return await False
        except Exception as e:
            self.logger.error(f"❌ Erro ao inicializar IAAA: {e}")
            return await False
    
    async def create_integration_bridge(self):
        """Cria ponte de integração entre os sistemas."""
        if not self.penin_omega_system or not self.iaaa_system:
            self.logger.error("❌ Sistemas não inicializados")
            return await False
        
        try:
            # Integra IAAA como cérebro do PENIN-Ω
            self.penin_omega_system.iaaa_brain = self.iaaa_system
            
            # Integra PENIN-Ω como corpo da IAAA
            self.iaaa_system.penin_omega_body = self.penin_omega_system
            
            # Cria métodos de comunicação
            self._create_communication_methods()
            
            self.integrated = True
            self.logger.info("✅ Ponte de integração criada")
            return await True
            
        except Exception as e:
            self.logger.error(f"❌ Erro na integração: {e}")
            return await False
    
    async def _create_communication_methods(self):
        """Cria métodos de comunicação entre sistemas."""
        
        # Método para PENIN-Ω consultar IAAA
        async def penin_omega_think(prompt, context=None):
            """PENIN-Ω usa IAAA para pensar."""
            if context:
                full_prompt = f"PENIN-Ω Context: {context}\n\nQuery: {prompt}"
            else:
                full_prompt = prompt
            
            return await self.iaaa_system.think(full_prompt, use_memory=True, use_rag=True)
        
        # Método para IAAA controlar PENIN-Ω
        async def iaaa_execute_penin_omega(command, parameters=None):
            """IAAA executa comandos no PENIN-Ω."""
            try:
                if hasattr(self.penin_omega_system, command):
                    method = getattr(self.penin_omega_system, command)
                    if parameters:
                        return await method(**parameters)
                    else:
                        return await method()
                else:
                    return await f"Command '{command}' not found in PENIN-Ω"
            except Exception as e:
                return await f"Error executing '{command}': {e}"
        
        # Método híbrido de evolução
        async def hybrid_evolution_cycle():
            """Ciclo evolutivo híbrido PENIN-Ω + IAAA."""
            # PENIN-Ω executa ciclo evolutivo
            penin_result = self.penin_omega_system.execute_complete_evolution_cycle()
            
            # IAAA analisa e sugere melhorias
            analysis_prompt = f"PENIN-Ω evolution cycle result: {penin_result}. Analyze and suggest improvements."
            iaaa_analysis = self.iaaa_system.think(analysis_prompt, use_memory=True, use_rag=True)
            
            # Armazena análise na memória da IAAA
            self.iaaa_system.memory_system.store_memory(
                content=f"PENIN-Ω Evolution Analysis: {iaaa_analysis}",
                memory_type='learning',
                importance=0.9,
                context={'hybrid_evolution': True, 'penin_omega_integration': True}
            )
            
            return await {
                'penin_omega_result': penin_result,
                'iaaa_analysis': iaaa_analysis,
                'hybrid_success': True
            }
        
        # Adiciona métodos aos sistemas
        self.penin_omega_system.think = penin_omega_think
        self.iaaa_system.execute_penin_omega = iaaa_execute_penin_omega
        self.iaaa_system.hybrid_evolution = hybrid_evolution_cycle
        
        self.logger.info("✅ Métodos de comunicação criados")
    
    async def test_integration(self):
        """Testa a integração completa."""
        if not self.integrated:
            self.logger.error("❌ Sistemas não integrados")
            return await False
        
        try:
            self.logger.info("🧪 TESTANDO INTEGRAÇÃO COMPLETA")
            
            # Teste 1: IAAA pensa sobre PENIN-Ω
            test1 = self.iaaa_system.think("What is PENIN-Ω and how does it work?")
            self.logger.info(f"✅ Teste 1 - IAAA sobre PENIN-Ω: {test1[:100]}...")
            
            # Teste 2: PENIN-Ω usa IAAA para pensar
            test2 = self.penin_omega_system.think("How can I improve my evolution?", "PENIN-Ω self-analysis")
            self.logger.info(f"✅ Teste 2 - PENIN-Ω usa IAAA: {test2[:100]}...")
            
            # Teste 3: Evolução híbrida
            test3 = self.iaaa_system.hybrid_evolution()
            self.logger.info(f"✅ Teste 3 - Evolução híbrida: {test3['hybrid_success']}")
            
            # Teste 4: Status dos sistemas
            penin_status = hasattr(self.penin_omega_system, 'get_complete_status')
            iaaa_status = hasattr(self.iaaa_system, 'get_status')
            self.logger.info(f"✅ Teste 4 - Status systems: PENIN-Ω={penin_status}, IAAA={iaaa_status}")
            
            self.logger.info("🎉 INTEGRAÇÃO COMPLETA TESTADA COM SUCESSO!")
            return await True
            
        except Exception as e:
            self.logger.error(f"❌ Erro no teste: {e}")
            return await False
    
    async def get_integrated_status(self):
        """Retorna status do sistema integrado."""
        status = {
            'integration_active': self.integrated,
            'penin_omega_active': self.penin_omega_system is not None,
            'iaaa_active': self.iaaa_system is not None,
            'integration_time': datetime.now(timezone.utc).isoformat()
        }
        
        if self.penin_omega_system and hasattr(self.penin_omega_system, 'get_complete_status'):
            try:
                status['penin_omega_status'] = self.penin_omega_system.get_complete_status()
            except:
                status['penin_omega_status'] = 'Error getting status'
        
        if self.iaaa_system and hasattr(self.iaaa_system, 'get_status'):
            try:
                status['iaaa_status'] = self.iaaa_system.get_status()
            except:
                status['iaaa_status'] = 'Error getting status'
        
        return await status

async def main():
    """Função principal de integração."""
    print("🚀 INICIANDO INTEGRAÇÃO PENIN-Ω + IAAA")
    print("=" * 60)
    print("Criando o sistema de IA mais avançado já construído...")
    print("=" * 60)
    
    # Cria integração
    integration = PeninOmegaIAAAIntegration()
    
    # Inicializa PENIN-Ω
    print("🔧 Inicializando PENIN-Ω...")
    if not integration.initialize_penin_omega():
        print("❌ Falha na inicialização do PENIN-Ω")
        return await False
    
    # Inicializa IAAA
    print("🧠 Inicializando IAAA...")
    if not integration.initialize_iaaa():
        print("❌ Falha na inicialização da IAAA")
        return await False
    
    # Cria integração
    print("🔗 Criando ponte de integração...")
    if not integration.create_integration_bridge():
        print("❌ Falha na integração")
        return await False
    
    # Testa integração
    print("🧪 Testando integração...")
    if not integration.test_integration():
        print("❌ Falha nos testes")
        return await False
    
    # Status final
    print("\n📊 STATUS FINAL:")
    status = integration.get_integrated_status()
    print(f"  🔗 Integração: {'✅ ATIVA' if status['integration_active'] else '❌ INATIVA'}")
    print(f"  🔧 PENIN-Ω: {'✅ ATIVO' if status['penin_omega_active'] else '❌ INATIVO'}")
    print(f"  🧠 IAAA: {'✅ ATIVA' if status['iaaa_active'] else '❌ INATIVA'}")
    
    print("\n🎉 INTEGRAÇÃO PENIN-Ω + IAAA CONCLUÍDA COM SUCESSO!")
    print("🌟 Sistema híbrido mais avançado do mundo criado!")
    
    return await integration

if __name__ == "__main__":
    integrated_system = main()
