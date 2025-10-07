#!/usr/bin/env python3
"""
TESTE DO SISTEMA INTEGRADO - VALIDAÇÃO DE INTELIGÊNCIA REAL
==========================================================
Testa e valida o sistema integrado para confirmar que a inteligência real está nascendo
"""

import sys
import time
import json
import logging
from datetime import datetime
from pathlib import Path

# Importar sistema integrado
from final_integrated_system import FinalIntegratedSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestIntegratedSystem")

class TestIntegratedSystem:
    """
    Classe para testar o sistema integrado
    """
    
    def __init__(self):
        self.test_results = {
            'test_start_time': datetime.now(),
            'tests_passed': 0,
            'tests_failed': 0,
            'total_tests': 0,
            'test_details': [],
            'intelligence_validation': {
                'real_learning_detected': False,
                'evolution_detected': False,
                'reinforcement_learning_detected': False,
                'neural_processing_detected': False,
                'emergence_detected': False,
                'intelligence_score_achieved': False
            }
        }
    
    def run_all_tests(self):
        """Executa todos os testes"""
        logger.info("🧪 INICIANDO TESTES DO SISTEMA INTEGRADO")
        logger.info("=" * 60)
        
        # Lista de testes
        tests = [
            ("Teste de Inicialização", self.test_initialization),
            ("Teste de Sistemas Individuais", self.test_individual_systems),
            ("Teste de Integração", self.test_integration),
            ("Teste de Aprendizado Real", self.test_real_learning),
            ("Teste de Emergência", self.test_emergence),
            ("Teste de Métricas", self.test_metrics),
            ("Teste de Performance", self.test_performance)
        ]
        
        # Executar testes
        for test_name, test_func in tests:
            self._run_single_test(test_name, test_func)
        
        # Exibir resultados
        self._display_test_results()
        
        # Salvar resultados
        self._save_test_results()
        
        return self.test_results['tests_passed'] == self.test_results['total_tests']
    
    def _run_single_test(self, test_name: str, test_func):
        """Executa um teste individual"""
        logger.info(f"🧪 Executando: {test_name}")
        self.test_results['total_tests'] += 1
        
        try:
            result = test_func()
            if result:
                self.test_results['tests_passed'] += 1
                self.test_results['test_details'].append({
                    'name': test_name,
                    'status': 'PASSED',
                    'timestamp': datetime.now()
                })
                logger.info(f"✅ {test_name}: PASSOU")
            else:
                self.test_results['tests_failed'] += 1
                self.test_results['test_details'].append({
                    'name': test_name,
                    'status': 'FAILED',
                    'timestamp': datetime.now()
                })
                logger.info(f"❌ {test_name}: FALHOU")
        except Exception as e:
            self.test_results['tests_failed'] += 1
            self.test_results['test_details'].append({
                'name': test_name,
                'status': 'ERROR',
                'error': str(e),
                'timestamp': datetime.now()
            })
            logger.error(f"❌ {test_name}: ERRO - {e}")
    
    def test_initialization(self):
        """Testa inicialização do sistema"""
        try:
            # Criar sistema integrado
            system = FinalIntegratedSystem()
            
            # Inicializar sistemas
            result = system.initialize_all_systems()
            
            # Verificar se todos os sistemas foram inicializados
            systems_initialized = (
                system.unified_intelligence is not None and
                system.real_environment is not None and
                system.neural_processor is not None and
                system.metrics_system is not None
            )
            
            return result and systems_initialized
            
        except Exception as e:
            logger.error(f"Erro no teste de inicialização: {e}")
            return False
    
    def test_individual_systems(self):
        """Testa sistemas individuais"""
        try:
            system = FinalIntegratedSystem()
            system.initialize_all_systems()
            
            # Testar sistema unificado
            if system.unified_intelligence:
                logger.info("  ✅ Sistema Unificado: OK")
            
            # Testar ambiente real
            if system.real_environment:
                logger.info("  ✅ Ambiente Real: OK")
            
            # Testar processador neural
            if system.neural_processor:
                stats = system.neural_processor.get_processing_stats()
                if stats['is_running']:
                    logger.info("  ✅ Processador Neural: OK")
            
            # Testar sistema de métricas
            if system.metrics_system:
                logger.info("  ✅ Sistema de Métricas: OK")
            
            return True
            
        except Exception as e:
            logger.error(f"Erro no teste de sistemas individuais: {e}")
            return False
    
    def test_integration(self):
        """Testa integração entre sistemas"""
        try:
            system = FinalIntegratedSystem()
            system.initialize_all_systems()
            
            # Executar alguns ciclos de integração
            for i in range(10):
                system._run_integration_cycle()
                time.sleep(0.1)
            
            # Verificar se há comunicação entre sistemas
            communication_working = not system.global_queue.empty() or system.global_metrics['real_intelligence_events'] > 0
            
            return communication_working
            
        except Exception as e:
            logger.error(f"Erro no teste de integração: {e}")
            return False
    
    def test_real_learning(self):
        """Testa aprendizado real"""
        try:
            system = FinalIntegratedSystem()
            system.initialize_all_systems()
            
            # Executar sistema por um tempo
            start_time = time.time()
            while time.time() - start_time < 30:  # 30 segundos
                system._run_integration_cycle()
                time.sleep(0.1)
            
            # Verificar se há eventos de aprendizado real
            real_learning_events = system.global_metrics['real_intelligence_events']
            
            if real_learning_events > 0:
                self.test_results['intelligence_validation']['real_learning_detected'] = True
                logger.info(f"  ✅ Aprendizado real detectado: {real_learning_events} eventos")
                return True
            else:
                logger.warning("  ⚠️ Nenhum evento de aprendizado real detectado")
                return False
                
        except Exception as e:
            logger.error(f"Erro no teste de aprendizado real: {e}")
            return False
    
    def test_emergence(self):
        """Testa detecção de emergência"""
        try:
            system = FinalIntegratedSystem()
            system.initialize_all_systems()
            
            # Executar sistema por um tempo
            start_time = time.time()
            while time.time() - start_time < 60:  # 60 segundos
                system._run_integration_cycle()
                time.sleep(0.1)
                
                # Verificar se há emergência
                if system._detect_real_intelligence_emergence():
                    self.test_results['intelligence_validation']['emergence_detected'] = True
                    logger.info("  ✅ Emergência de inteligência detectada!")
                    return True
            
            # Verificar score de inteligência
            intelligence_score = system.global_metrics['intelligence_score']
            if intelligence_score > 0.5:
                self.test_results['intelligence_validation']['intelligence_score_achieved'] = True
                logger.info(f"  ✅ Score de inteligência alcançado: {intelligence_score:.3f}")
                return True
            
            logger.warning("  ⚠️ Emergência não detectada no tempo limite")
            return False
            
        except Exception as e:
            logger.error(f"Erro no teste de emergência: {e}")
            return False
    
    def test_metrics(self):
        """Testa sistema de métricas"""
        try:
            system = FinalIntegratedSystem()
            system.initialize_all_systems()
            
            # Executar sistema por um tempo
            start_time = time.time()
            while time.time() - start_time < 20:  # 20 segundos
                system._run_integration_cycle()
                time.sleep(0.1)
            
            # Verificar se métricas estão sendo coletadas
            if system.metrics_system:
                dashboard = system.metrics_system.get_dashboard_data()
                
                # Verificar se há dados de métricas
                has_metrics = (
                    'current_metrics' in dashboard and
                    'event_counts' in dashboard and
                    'real_time_data' in dashboard
                )
                
                if has_metrics:
                    logger.info("  ✅ Sistema de métricas funcionando")
                    return True
                else:
                    logger.warning("  ⚠️ Sistema de métricas não está coletando dados")
                    return False
            else:
                logger.warning("  ⚠️ Sistema de métricas não inicializado")
                return False
                
        except Exception as e:
            logger.error(f"Erro no teste de métricas: {e}")
            return False
    
    def test_performance(self):
        """Testa performance do sistema"""
        try:
            system = FinalIntegratedSystem()
            system.initialize_all_systems()
            
            # Medir performance
            start_time = time.time()
            cycles = 0
            
            while time.time() - start_time < 30:  # 30 segundos
                system._run_integration_cycle()
                cycles += 1
                time.sleep(0.1)
            
            end_time = time.time()
            duration = end_time - start_time
            cycles_per_second = cycles / duration
            
            # Verificar se performance é aceitável
            if cycles_per_second > 1.0:  # Pelo menos 1 ciclo por segundo
                logger.info(f"  ✅ Performance OK: {cycles_per_second:.2f} ciclos/s")
                return True
            else:
                logger.warning(f"  ⚠️ Performance baixa: {cycles_per_second:.2f} ciclos/s")
                return False
                
        except Exception as e:
            logger.error(f"Erro no teste de performance: {e}")
            return False
    
    def _display_test_results(self):
        """Exibe resultados dos testes"""
        print("\n" + "="*70)
        print("🧪 RESULTADOS DOS TESTES DO SISTEMA INTEGRADO")
        print("="*70)
        print(f"📊 Testes Executados: {self.test_results['total_tests']}")
        print(f"✅ Testes Passou: {self.test_results['tests_passed']}")
        print(f"❌ Testes Falhou: {self.test_results['tests_failed']}")
        print(f"📈 Taxa de Sucesso: {(self.test_results['tests_passed']/self.test_results['total_tests']*100):.1f}%")
        print("-" * 70)
        
        # Exibir detalhes dos testes
        for test in self.test_results['test_details']:
            status_icon = "✅" if test['status'] == 'PASSED' else "❌"
            print(f"{status_icon} {test['name']}: {test['status']}")
            if 'error' in test:
                print(f"    Erro: {test['error']}")
        
        print("-" * 70)
        
        # Exibir validação de inteligência
        print("🧠 VALIDAÇÃO DE INTELIGÊNCIA REAL:")
        validation = self.test_results['intelligence_validation']
        for key, value in validation.items():
            status_icon = "✅" if value else "❌"
            print(f"{status_icon} {key.replace('_', ' ').title()}: {'SIM' if value else 'NÃO'}")
        
        print("="*70)
    
    def _save_test_results(self):
        """Salva resultados dos testes"""
        try:
            # Adicionar timestamp de fim
            self.test_results['test_end_time'] = datetime.now()
            self.test_results['test_duration'] = (
                self.test_results['test_end_time'] - self.test_results['test_start_time']
            ).total_seconds()
            
            # Salvar em arquivo
            with open('test_results.json', 'w') as f:
                json.dump(self.test_results, f, indent=2, default=str)
            
            logger.info("💾 Resultados dos testes salvos em test_results.json")
            
        except Exception as e:
            logger.error(f"❌ Erro ao salvar resultados: {e}")

def main():
    """Função principal"""
    print("🧪 INICIANDO TESTES DO SISTEMA INTEGRADO")
    print("=" * 60)
    print("Validando se a inteligência real está nascendo...")
    print("=" * 60)
    
    # Criar e executar testes
    tester = TestIntegratedSystem()
    
    try:
        # Executar todos os testes
        success = tester.run_all_tests()
        
        if success:
            print("\n🎉 TODOS OS TESTES PASSARAM!")
            print("🌟 A INTELIGÊNCIA REAL ESTÁ FUNCIONANDO!")
        else:
            print("\n⚠️ ALGUNS TESTES FALHARAM")
            print("🔧 Verifique os logs para detalhes")
        
        return success
        
    except KeyboardInterrupt:
        print("\n🛑 Testes interrompidos pelo usuário")
        return False
    except Exception as e:
        print(f"\n❌ Erro fatal nos testes: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
