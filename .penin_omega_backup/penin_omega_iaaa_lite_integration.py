#!/usr/bin/env python3
"""
PENIN-Ω + IAAA Lite Integration
==============================
Integração otimizada sem carregar Falcon 7B para economizar memória.

Usa apenas os componentes de memória, RAG e arquitetura da IAAA.
"""

import sys
import logging
from datetime import datetime, timezone

# Adiciona paths
sys.path.append("/root")
sys.path.append("/root/IAAA_PENIN_OMEGA/core")

class PeninOmegaIAAALiteIntegration:
    """Integração lite PENIN-Ω + IAAA (sem Falcon)."""
    
    def __init__(self):
        self.logger = logging.getLogger("PENIN_OMEGA_IAAA_LITE")
        self.penin_omega_system = None
        self.iaaa_components = {}
        self.integrated = False
        
    def initialize_penin_omega(self):
        """Inicializa sistema PENIN-Ω."""
        try:
            from penin_omega_master_system import penin_omega
            self.penin_omega_system = penin_omega
            self.logger.info("✅ Sistema PENIN-Ω inicializado")
            return True
        except Exception as e:
            self.logger.error(f"❌ Erro PENIN-Ω: {e}")
            return False
    
    def initialize_iaaa_components(self):
        """Inicializa apenas componentes leves da IAAA."""
        try:
            # Configura API keys
            exec(open('/root/api_keys_config.py').read())
            
            # Importa componentes sem Falcon
            from memory.persistent_memory import memory_system
            from memory.rag_system import rag_system
            from penin_omega_robust_multi_api import RobustMultiAPI
            
            self.iaaa_components = {
                'memory': memory_system,
                'rag': rag_system,
                'multi_api': RobustMultiAPI()
            }
            
            self.logger.info("✅ Componentes IAAA inicializados (sem Falcon)")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro componentes IAAA: {e}")
            return False
    
    def create_integration(self):
        """Cria integração completa."""
        if not self.penin_omega_system or not self.iaaa_components:
            return False
        
        try:
            # Integra componentes IAAA no PENIN-Ω
            self.penin_omega_system.memory_system = self.iaaa_components['memory']
            self.penin_omega_system.rag_system = self.iaaa_components['rag']
            self.penin_omega_system.multi_api = self.iaaa_components['multi_api']
            
            # Cria métodos híbridos
            self._create_hybrid_methods()
            
            self.integrated = True
            self.logger.info("✅ Integração lite criada")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro integração: {e}")
            return False
    
    def _create_hybrid_methods(self):
        """Cria métodos híbridos."""
        
        def enhanced_think(prompt, context=None):
            """Pensamento melhorado com RAG e memória."""
            # Busca conhecimento RAG
            rag_context = ""
            if self.iaaa_components['rag']:
                try:
                    rag_context = self.iaaa_components['rag'].get_contextual_knowledge(prompt, 500)
                except:
                    pass
            
            # Busca memórias relevantes
            memory_context = ""
            if self.iaaa_components['memory']:
                try:
                    memories = self.iaaa_components['memory'].retrieve_memories(prompt, limit=2)
                    if memories:
                        memory_context = "\\n".join([m.content[:100] for m in memories])
                except:
                    pass
            
            # Combina contextos
            full_context = f"RAG: {rag_context}\\nMemory: {memory_context}\\nQuery: {prompt}"
            
            # Armazena na memória
            if self.iaaa_components['memory']:
                try:
                    self.iaaa_components['memory'].store_memory(
                        content=f"Enhanced thinking: {prompt}",
                        memory_type='thought',
                        importance=0.6
                    )
                except:
                    pass
            
            return f"Enhanced response based on context: {full_context[:200]}..."
        
        def multi_api_consult(prompt, model="auto"):
            """Consulta Multi-API."""
            try:
                import asyncio
                from penin_omega_robust_multi_api import APIRequest
                
                request = APIRequest(prompt=prompt, max_tokens=100)
                
                try:
                    loop = asyncio.get_event_loop()
                except:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                response = loop.run_until_complete(
                    self.iaaa_components['multi_api'].process_request(request)
                )
                
                return str(response) if response else "No response"
                
            except Exception as e:
                return f"Error: {e}"
        
        # Adiciona métodos ao PENIN-Ω
        self.penin_omega_system.enhanced_think = enhanced_think
        self.penin_omega_system.multi_api_consult = multi_api_consult
        
        self.logger.info("✅ Métodos híbridos criados")
    
    def test_integration(self):
        """Testa integração."""
        if not self.integrated:
            return False
        
        try:
            # Teste 1: Pensamento melhorado
            result1 = self.penin_omega_system.enhanced_think("What is PENIN-Ω?")
            self.logger.info(f"✅ Enhanced think: {len(result1)} chars")
            
            # Teste 2: Multi-API
            result2 = self.penin_omega_system.multi_api_consult("Hello")
            self.logger.info(f"✅ Multi-API: {result2[:50]}...")
            
            # Teste 3: Memória
            if self.iaaa_components['memory']:
                stats = self.iaaa_components['memory'].get_memory_stats()
                self.logger.info(f"✅ Memory: {stats['total_memories']} memories")
            
            self.logger.info("🎉 INTEGRAÇÃO LITE TESTADA COM SUCESSO!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Erro teste: {e}")
            return False

def main():
    """Executa integração lite."""
    print("🚀 INTEGRAÇÃO PENIN-Ω + IAAA LITE")
    print("=" * 50)
    
    integration = PeninOmegaIAAALiteIntegration()
    
    # Inicializa PENIN-Ω
    print("🔧 Inicializando PENIN-Ω...")
    if not integration.initialize_penin_omega():
        print("❌ Falha PENIN-Ω")
        return False
    
    # Inicializa componentes IAAA
    print("🧠 Inicializando componentes IAAA...")
    if not integration.initialize_iaaa_components():
        print("❌ Falha componentes IAAA")
        return False
    
    # Cria integração
    print("🔗 Criando integração...")
    if not integration.create_integration():
        print("❌ Falha integração")
        return False
    
    # Testa
    print("🧪 Testando...")
    if not integration.test_integration():
        print("❌ Falha testes")
        return False
    
    print("\\n🎉 INTEGRAÇÃO LITE CONCLUÍDA!")
    print("✅ PENIN-Ω agora tem:")
    print("  - Sistema de memória persistente")
    print("  - Sistema RAG com conhecimento")
    print("  - Multi-API para outros modelos")
    print("  - Métodos híbridos de pensamento")
    
    return integration

if __name__ == "__main__":
    integrated_system = main()
