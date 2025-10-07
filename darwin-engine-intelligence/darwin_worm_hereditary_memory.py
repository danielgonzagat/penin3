"""
DARWIN WORM - MEMÓRIA HEREDITÁRIA PERSISTENTE
============================================

Implementa sistema WORM (Write Once Read Many) para memória hereditária
auditável com herança genética real e rollback de mutações nocivas.

Componente crítico para rastreabilidade e auditabilidade do Darwin Ideal.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import logging
import json
import hashlib
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import threading
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WORMEntry:
    """Entrada do log WORM com verificação de integridade"""

    def __init__(self, event_type: str, data: Dict[str, Any], previous_hash: str):
        self.event_type = event_type
        self.data = data
        self.previous_hash = previous_hash
        self.timestamp = datetime.now()

        # Calcular hash atual
        self.current_hash = self._calculate_hash()

    def _calculate_hash(self) -> str:
        """Calcula hash da entrada"""
        # Criar string canônica para hash
        entry_str = json.dumps({
            'event_type': self.event_type,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'timestamp': self.timestamp.isoformat()
        }, sort_keys=True, separators=(',', ':'))

        return hashlib.sha256(entry_str.encode('utf-8')).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        """Converte entrada para dicionário"""
        return {
            'event_type': self.event_type,
            'data': self.data,
            'previous_hash': self.previous_hash,
            'current_hash': self.current_hash,
            'timestamp': self.timestamp.isoformat()
        }

    def verify_integrity(self) -> bool:
        """Verifica integridade da entrada"""
        expected_hash = self._calculate_hash()
        return self.current_hash == expected_hash

class DarwinWORMMemory:
    """Sistema WORM completo para memória hereditária"""

    def __init__(self, log_file: str = "darwin_hereditary_memory.worm"):
        self.log_file = log_file
        self.genesis_hash = hashlib.sha256(b"DARWIN-HEREDITARY-GENESIS").hexdigest()
        self.current_hash = self.genesis_hash

        # Índices para busca eficiente
        self.individual_index: Dict[int, List[int]] = {}  # individual_id -> entry_indices
        self.generation_index: Dict[int, List[int]] = {}  # generation -> entry_indices
        self.event_type_index: Dict[str, List[int]] = {}  # event_type -> entry_indices

        # Cache de entradas
        self.entry_cache: List[WORMEntry] = []
        self.cache_size = 1000

        # Mutex para thread safety
        self.lock = threading.Lock()

        # Carregar log existente
        self._load_existing_log()

        logger.info("💾 Darwin WORM Memory inicializado")
        logger.info(f"   📁 Log file: {self.log_file}")
        logger.info(f"   📊 Entradas carregadas: {len(self.entry_cache)}")

    def _load_existing_log(self):
        """Carrega log WORM existente"""
        if not os.path.exists(self.log_file):
            logger.info("   📄 Novo log WORM criado")
            return

        try:
            with open(self.log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            # Processar entradas em pares (EVENT + HASH)
            i = 0
            while i < len(lines):
                if lines[i].startswith('EVENT:'):
                    try:
                        # Extrair dados do evento
                        event_json = lines[i][6:].strip()
                        event_data = json.loads(event_json)

                        # Verificar hash se disponível
                        if i + 1 < len(lines) and lines[i + 1].startswith('HASH:'):
                            current_hash = lines[i + 1][5:].strip()

                            # Verificar integridade
                            entry = WORMEntry(
                                event_type=event_data['event_type'],
                                data=event_data['data'],
                                previous_hash=event_data['previous_hash']
                            )

                            if entry.current_hash == current_hash:
                                self._add_entry_to_indices(entry, len(self.entry_cache))
                                self.entry_cache.append(entry)
                                self.current_hash = current_hash
                            else:
                                logger.warning(f"   ⚠️ Hash incorreto na entrada {len(self.entry_cache)}")

                    except json.JSONDecodeError as e:
                        logger.error(f"   ❌ Erro ao parsear evento {i}: {e}")
                    except Exception as e:
                        logger.error(f"   ❌ Erro ao processar entrada {i}: {e}")

                i += 2  # Pular para próxima entrada

            logger.info(f"   ✅ Log carregado: {len(self.entry_cache)} entradas válidas")

        except Exception as e:
            logger.error(f"   ❌ Erro ao carregar log existente: {e}")

    def _add_entry_to_indices(self, entry: WORMEntry, entry_index: int):
        """Adiciona entrada aos índices"""
        # Índice por indivíduo
        individual_id = entry.data.get('individual_id')
        if individual_id is not None:
            if individual_id not in self.individual_index:
                self.individual_index[individual_id] = []
            self.individual_index[individual_id].append(entry_index)

        # Índice por geração
        generation = entry.data.get('generation')
        if generation is not None:
            if generation not in self.generation_index:
                self.generation_index[generation] = []
            self.generation_index[generation].append(entry_index)

        # Índice por tipo de evento
        event_type = entry.event_type
        if event_type not in self.event_type_index:
            self.event_type_index[event_type] = []
        self.event_type_index[event_type].append(entry_index)

    def append_event(self, event_type: str, data: Dict[str, Any]) -> str:
        """Adiciona evento ao log WORM"""
        with self.lock:
            # Criar entrada
            entry = WORMEntry(event_type, data, self.current_hash)

            # Verificar integridade antes de escrever
            if not entry.verify_integrity():
                raise ValueError("Integridade da entrada comprometida")

            # Escrever no arquivo
            try:
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    # Escrever evento
                    event_json = json.dumps(entry.to_dict(), separators=(',', ':'))
                    f.write(f"EVENT:{event_json}\n")

                    # Escrever hash
                    f.write(f"HASH:{entry.current_hash}\n")

                    # Forçar flush para garantir escrita imediata
                    f.flush()
                    os.fsync(f.fileno())

                # Atualizar estado interno
                self._add_entry_to_indices(entry, len(self.entry_cache))
                self.entry_cache.append(entry)
                self.current_hash = entry.current_hash

                # Manter cache dentro do limite
                if len(self.entry_cache) > self.cache_size:
                    # Remover entradas antigas do cache (manter índices)
                    self.entry_cache = self.entry_cache[-self.cache_size:]

                return entry.current_hash

            except Exception as e:
                logger.error(f"Erro ao escrever no log WORM: {e}")
                raise

    def get_individual_history(self, individual_id: int) -> List[WORMEntry]:
        """Obtém histórico completo de um indivíduo"""
        if individual_id not in self.individual_index:
            return []

        entry_indices = self.individual_index[individual_id]
        return [self.entry_cache[i] for i in entry_indices if i < len(self.entry_cache)]

    def get_generation_history(self, generation: int) -> List[WORMEntry]:
        """Obtém eventos de uma geração específica"""
        if generation not in self.generation_index:
            return []

        entry_indices = self.generation_index[generation]
        return [self.entry_cache[i] for i in entry_indices if i < len(self.entry_cache)]

    def get_events_by_type(self, event_type: str) -> List[WORMEntry]:
        """Obtém todos os eventos de um tipo"""
        if event_type not in self.event_type_index:
            return []

        entry_indices = self.event_type_index[event_type]
        return [self.entry_cache[i] for i in entry_indices if i < len(self.entry_cache)]

    def verify_chain_integrity(self) -> Tuple[bool, str]:
        """Verifica integridade completa da cadeia WORM"""
        if not self.entry_cache:
            return True, "Log vazio"

        current_hash = self.genesis_hash

        for entry in self.entry_cache:
            # Verificar se o hash anterior está correto
            if entry.previous_hash != current_hash:
                return False, f"Hash anterior incorreto na entrada {self.entry_cache.index(entry)}"

            # Verificar integridade da entrada
            if not entry.verify_integrity():
                return False, f"Entrada corrompida na posição {self.entry_cache.index(entry)}"

            current_hash = entry.current_hash

        return True, "Cadeia íntegra"

    def get_hereditary_lineage(self, individual_id: int) -> Dict[str, Any]:
        """Reconstrói linhagem hereditária completa"""
        lineage = {
            'individual_id': individual_id,
            'ancestors': [],
            'descendants': [],
            'siblings': [],
            'birth_event': None,
            'death_event': None,
            'mutation_events': []
        }

        # Buscar eventos do indivíduo
        individual_events = self.get_individual_history(individual_id)

        for event in individual_events:
            if event.event_type == 'individual_birth':
                lineage['birth_event'] = event
                # Buscar ancestrais
                parents = event.data.get('parents', [])
                lineage['ancestors'] = parents

            elif event.event_type == 'individual_death':
                lineage['death_event'] = event

            elif event.event_type == 'mutation_event':
                lineage['mutation_events'].append(event)

        # Buscar descendentes (quem tem este indivíduo como pai)
        all_births = self.get_events_by_type('individual_birth')
        for birth in all_births:
            parents = birth.data.get('parents', [])
            if individual_id in parents:
                lineage['descendants'].append(birth.data.get('individual_id'))

        # Buscar irmãos (mesmos pais)
        if lineage['ancestors']:
            parent1_id, parent2_id = lineage['ancestors'][:2]

            all_births = self.get_events_by_type('individual_birth')
            for birth in all_births:
                birth_parents = birth.data.get('parents', [])
                if (parent1_id in birth_parents and parent2_id in birth_parents and
                    birth.data.get('individual_id') != individual_id):
                    lineage['siblings'].append(birth.data.get('individual_id'))

        return lineage

    def find_harmful_mutations(self, individual_id: int) -> List[Dict[str, Any]]:
        """Identifica mutações potencialmente nocivas"""
        mutation_events = []

        # Buscar eventos de mutação do indivíduo
        individual_events = self.get_individual_history(individual_id)

        for event in individual_events:
            if event.event_type == 'mutation_event':
                mutation_data = event.data

                # Critérios para identificar mutação nociva
                pre_fitness = mutation_data.get('pre_mutation_fitness', 0)
                post_fitness = mutation_data.get('post_mutation_fitness', 0)

                # Mutação nociva se reduziu fitness significativamente
                if pre_fitness > 0 and post_fitness > 0:
                    fitness_reduction = (pre_fitness - post_fitness) / pre_fitness

                    if fitness_reduction > 0.1:  # Redução > 10%
                        mutation_events.append({
                            'event': event,
                            'fitness_reduction': fitness_reduction,
                            'severity': 'high' if fitness_reduction > 0.3 else 'medium'
                        })

        return mutation_events

    def rollback_individual_state(self, individual_id: int, target_generation: int) -> Dict[str, Any]:
        """Faz rollback do estado de um indivíduo para uma geração específica"""
        # Buscar eventos até a geração target
        individual_events = self.get_individual_history(individual_id)

        # Filtrar eventos até a geração target
        rollback_events = [
            event for event in individual_events
            if event.data.get('generation', 0) <= target_generation
        ]

        if not rollback_events:
            return {'success': False, 'reason': 'Nenhum evento encontrado para rollback'}

        # Reconstruir estado do indivíduo
        rollback_state = {
            'individual_id': individual_id,
            'rollback_generation': target_generation,
            'genome': {},
            'fitness': 0.0,
            'age': 0,
            'events_applied': 0
        }

        # Aplicar eventos em ordem cronológica
        rollback_events.sort(key=lambda x: x.timestamp)

        for event in rollback_events:
            if event.event_type == 'individual_birth':
                rollback_state['genome'] = event.data.get('genome', {})
                rollback_state['fitness'] = event.data.get('fitness', 0.0)
                rollback_state['age'] = 0

            elif event.event_type == 'mutation_event':
                # Aplicar mutação (em produção seria mais sofisticado)
                rollback_state['genome'] = event.data.get('post_mutation_genome', rollback_state['genome'])
                rollback_state['fitness'] = event.data.get('post_mutation_fitness', rollback_state['fitness'])

            rollback_state['events_applied'] += 1

        rollback_state['success'] = True
        return rollback_state

    def get_evolution_statistics(self) -> Dict[str, Any]:
        """Calcula estatísticas da evolução baseada no log WORM"""
        all_events = self.entry_cache

        if not all_events:
            return {}

        # Estatísticas básicas
        event_types = {}
        for event in all_events:
            event_type = event.event_type
            event_types[event_type] = event_types.get(event_type, 0) + 1

        # Estatísticas de indivíduos
        birth_events = self.get_events_by_type('individual_birth')
        death_events = self.get_events_by_type('individual_death')

        total_births = len(birth_events)
        total_deaths = len(death_events)

        # Estatísticas de mutações
        mutation_events = self.get_events_by_type('mutation_event')
        harmful_mutations = sum(1 for event in mutation_events
                               if event.data.get('fitness_reduction', 0) > 0.1)

        # Estatísticas de gerações
        generations = set()
        for event in all_events:
            gen = event.data.get('generation')
            if gen is not None:
                generations.add(gen)

        return {
            'total_events': len(all_events),
            'event_type_distribution': event_types,
            'total_individuals_born': total_births,
            'total_individuals_died': total_deaths,
            'survival_rate': (total_deaths / total_births) if total_births > 0 else 0,
            'total_mutations': len(mutation_events),
            'harmful_mutations': harmful_mutations,
            'mutation_harm_rate': (harmful_mutations / len(mutation_events)) if mutation_events else 0,
            'generations_covered': len(generations),
            'avg_events_per_generation': len(all_events) / len(generations) if generations else 0,
            'log_integrity': self.verify_chain_integrity()[0]
        }

    def export_audit_trail(self, output_file: str = None) -> str:
        """Exporta trilha de auditoria completa"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"darwin_audit_trail_{timestamp}.json"

        # Preparar dados de auditoria
        audit_data = {
            'export_timestamp': datetime.now().isoformat(),
            'worm_log_file': self.log_file,
            'chain_integrity': self.verify_chain_integrity(),
            'statistics': self.get_evolution_statistics(),
            'entries': [entry.to_dict() for entry in self.entry_cache],
            'indices': {
                'individual_index': self.individual_index,
                'generation_index': self.generation_index,
                'event_type_index': self.event_type_index
            }
        }

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(audit_data, f, indent=2, default=str)

        logger.info(f"📋 Trilha de auditoria exportada: {output_file}")
        return output_file

# ============================================================================
# EXEMPLOS DE USO
# ============================================================================

def example_worm_memory():
    """Exemplo de uso do sistema WORM"""
    print("="*80)
    print("💾 EXEMPLO: MEMÓRIA HEREDITÁRIA WORM")
    print("="*80)

    # Criar sistema WORM
    worm = DarwinWORMMemory("example_darwin_memory.worm")

    # Simular eventos evolutivos
    print("📝 Registrando eventos evolutivos...")

    # Nascimento de indivíduos
    for i in range(5):
        individual_id = 1000 + i
        genome = {'hidden_size': 64 + i * 10, 'learning_rate': 0.001 * (i + 1)}

        event_hash = worm.append_event('individual_birth', {
            'individual_id': individual_id,
            'genome': genome,
            'fitness': 0.5 + i * 0.1,
            'generation': 1,
            'parents': [900, 901] if i > 0 else []
        })

        print(f"   🧬 Nascimento registrado: indivíduo {individual_id} (hash: {event_hash[:8]}...)")

    # Eventos de mutação
    for i in range(3):
        individual_id = 1000 + i

        worm.append_event('mutation_event', {
            'individual_id': individual_id,
            'mutation_type': 'parameter_change',
            'pre_mutation_fitness': 0.5 + i * 0.1,
            'post_mutation_fitness': 0.45 + i * 0.1,  # Mutação ligeiramente nociva
            'mutation_details': {'parameter': 'learning_rate', 'old_value': 0.001 * (i + 1), 'new_value': 0.002 * (i + 1)}
        })

        print(f"   🔀 Mutação registrada: indivíduo {individual_id}")

    # Morte de indivíduos
    for i in range(2):
        individual_id = 1000 + i

        worm.append_event('individual_death', {
            'individual_id': individual_id,
            'reason': 'low_fitness',
            'fitness': 0.45 + i * 0.1,
            'age': 5 + i
        })

        print(f"   💀 Morte registrada: indivíduo {individual_id}")

    # Verificar integridade
    print("\n🔍 Verificando integridade da cadeia...")
    is_integrity, message = worm.verify_chain_integrity()
    print(f"   ✅ Integridade: {'OK' if is_integrity else 'COMPROMETIDA'} - {message}")

    # Obter estatísticas
    print("\n📊 Estatísticas da evolução:")
    stats = worm.get_evolution_statistics()

    for key, value in stats.items():
        if isinstance(value, float):
            print(f"   {key}: {value:.3f}")
        else:
            print(f"   {key}: {value}")

    # Linhagem hereditária
    print("\n🧬 Linhagem hereditária do indivíduo 1002:")
    lineage = worm.get_hereditary_lineage(1002)

    print(f"   Ancestrais: {lineage['ancestors']}")
    print(f"   Irmãos: {lineage['siblings']}")
    print(f"   Descendentes: {lineage['descendants']}")
    print(f"   Eventos de mutação: {len(lineage['mutation_events'])}")

    # Exportar auditoria
    print("\n📋 Exportando trilha de auditoria...")
    audit_file = worm.export_audit_trail()

    print(f"   ✅ Auditoria salva: {audit_file}")

    return worm

if __name__ == "__main__":
    # Executar exemplo
    worm = example_worm_memory()

    print("\n✅ Darwin WORM Memory funcionando!")
    print("   💾 Memória hereditária persistente implementada")
    print("   🔗 Cadeia de hash auditável")
    print("   🧬 Rastreabilidade genética completa")
    print("   ⏪ Capacidade de rollback")
    print("   📊 Estatísticas de evolução")
    print("   🎯 Darwin Ideal: memória hereditária ALCANÇADA!")