"""
DARWINACCI UNIVERSAL CONNECTOR
===============================
Conecta Darwinacci-Ω como núcleo sináptico universal de TODOS os sistemas

Como sinapses conectam neurônios em um cérebro, este módulo conecta:
- V7 Intelligence System
- Brain Daemon (UNIFIED_BRAIN)
- Darwin Evolution (original)
- Meta-Learner
- Novelty System
- TEIS Agents
- Database/Telemetry
- LLM Systems

Todos os sistemas trocam informação via Darwinacci como hub central.
"""

import sys
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

sys.path.insert(0, '/root')

logger = logging.getLogger(__name__)


class UniversalConnector:
    """
    Darwinacci como núcleo sináptico universal
    
    Conecta TODOS sistemas via protocolo comum:
    - Extração de genomes de sistemas fontes
    - Injeção de genomes evoluídos em sistemas alvos
    - Sincronização bidirecional contínua
    """
    
    def __init__(self, darwinacci_engine=None):
        self.darwinacci = darwinacci_engine
        self.connections = {}  # {system_name: connection_info}
        self.synapses = {}     # {source → target: synapse_strength}
        self.active = False
        
        # Telemetry
        self.sync_count = 0
        self.extraction_count = 0
        self.injection_count = 0
        
        logger.info("🧠 Universal Connector initialized")
    
    def connect_v7(self, v7_system):
        """Conecta V7 Intelligence System ao Darwinacci"""
        try:
            logger.info("🔌 Connecting V7 → Darwinacci...")
            
            # Check if V7 has darwin_real
            if hasattr(v7_system, 'darwin_real'):
                # Inject Darwinacci engine into V7
                if self.darwinacci:
                    v7_system.darwin_darwinacci = self.darwinacci
                    v7_system.use_darwinacci = True
                    logger.info("   ✅ Injected Darwinacci engine into V7")
                
                # Create synapse
                self.synapses['darwinacci→v7'] = 1.0
                self.synapses['v7→darwinacci'] = 0.8
                
                self.connections['v7'] = {
                    'system': v7_system,
                    'type': 'bidirectional',
                    'extract_fn': self._extract_from_v7,
                    'inject_fn': self._inject_to_v7,
                    'active': True
                }
                
                logger.info("   ✅ V7 connected (bidirectional synapse)")
                return True
        except Exception as e:
            logger.error(f"❌ V7 connection failed: {e}")
            return False
    
    def connect_brain_daemon(self, brain_daemon=None):
        """Conecta Brain Daemon ao Darwinacci"""
        try:
            logger.info("🔌 Connecting Brain Daemon → Darwinacci...")
            
            # Brain daemon evolui hiperparâmetros via Darwinacci
            self.synapses['darwinacci→brain'] = 0.9
            self.synapses['brain→darwinacci'] = 0.7
            
            self.connections['brain'] = {
                'system': brain_daemon,
                'type': 'bidirectional',
                'extract_fn': self._extract_from_brain,
                'inject_fn': self._inject_to_brain,
                'active': True
            }
            
            logger.info("   ✅ Brain Daemon connected")
            return True
        except Exception as e:
            logger.error(f"❌ Brain connection failed: {e}")
            return False
    
    def connect_darwin_runner(self):
        """Conecta Darwin Evolution runner"""
        try:
            logger.info("🔌 Connecting Darwin Runner → Darwinacci...")
            
            self.synapses['darwinacci→darwin_runner'] = 1.0
            
            self.connections['darwin_runner'] = {
                'type': 'replacement',  # Darwinacci substitui Darwin
                'active': True
            }
            
            logger.info("   ✅ Darwin Runner connected (replacement mode)")
            return True
        except Exception as e:
            logger.error(f"❌ Darwin connection failed: {e}")
            return False
    
    def connect_database(self, db_path=None):
        """Conecta sistema de telemetria/database"""
        try:
            logger.info("🔌 Connecting Database → Darwinacci...")
            
            if db_path is None:
                db_path = '/root/intelligence_system/data/intelligence.db'
            
            self.synapses['darwinacci→db'] = 0.5
            self.synapses['db→darwinacci'] = 0.3
            
            self.connections['database'] = {
                'db_path': db_path,
                'type': 'telemetry',
                'extract_fn': self._extract_from_db,
                'inject_fn': self._inject_to_db,
                'active': True
            }
            
            logger.info("   ✅ Database connected")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection failed: {e}")
            return False
    
    def connect_novelty_system(self, novelty_system):
        """Conecta Novelty Search ao Darwinacci"""
        try:
            logger.info("🔌 Connecting Novelty System ↔ Darwinacci...")
            
            self.synapses['novelty→darwinacci'] = 0.9
            self.synapses['darwinacci→novelty'] = 0.6
            
            self.connections['novelty'] = {
                'system': novelty_system,
                'type': 'bidirectional',
                'extract_fn': self._extract_from_novelty,
                'inject_fn': self._inject_to_novelty,
                'active': True
            }
            
            logger.info("   ✅ Novelty System connected")
            return True
        except Exception as e:
            logger.error(f"❌ Novelty connection failed: {e}")
            return False
    
    def activate(self):
        """Ativa todas conexões"""
        self.active = True
        logger.info(f"🧠 Universal Connector ACTIVATED")
        logger.info(f"   Connections: {len(self.connections)}")
        logger.info(f"   Synapses: {len(self.synapses)}")
        
        # Log synapse map
        logger.info("   Synapse map:")
        for synapse, strength in self.synapses.items():
            logger.info(f"      {synapse}: {strength:.2f}")
    
    def sync_all(self):
        """
        Sincronização completa de todos sistemas conectados
        Como neurotransmissores fluindo pelas sinapses
        """
        if not self.active:
            return {}
        
        self.sync_count += 1
        results = {'synced': [], 'failed': []}
        
        # EXTRAÇÃO: Coletar genomes de todos sistemas
        extracted_genomes = {}
        
        for sys_name, conn in self.connections.items():
            if not conn.get('active'):
                continue
            
            try:
                if 'extract_fn' in conn and conn['extract_fn']:
                    genomes = conn['extract_fn'](conn.get('system'))
                    extracted_genomes[sys_name] = genomes
                    self.extraction_count += len(genomes) if genomes else 0
                    results['synced'].append(f'extract:{sys_name}')
            except Exception as e:
                logger.debug(f"Extract from {sys_name} failed: {e}")
                results['failed'].append(f'extract:{sys_name}')
        
        # INJEÇÃO: Distribuir genomes evoluídos para todos sistemas
        if self.darwinacci and hasattr(self.darwinacci, 'archive'):
            # Pegar melhores genomes do Darwinacci
            try:
                best_cells = self.darwinacci.archive.bests()[:10]
                
                for sys_name, conn in self.connections.items():
                    if not conn.get('active'):
                        continue
                    
                    try:
                        if 'inject_fn' in conn and conn['inject_fn']:
                            success = conn['inject_fn'](conn.get('system'), best_cells)
                            if success:
                                self.injection_count += 1
                                results['synced'].append(f'inject:{sys_name}')
                    except Exception as e:
                        logger.debug(f"Inject to {sys_name} failed: {e}")
                        results['failed'].append(f'inject:{sys_name}')
            except Exception as e:
                logger.debug(f"Archive access failed: {e}")
        
        if results['synced']:
            logger.debug(f"🔄 Sync #{self.sync_count}: {len(results['synced'])} operations")
        
        return results
    
    # === EXTRACTION FUNCTIONS ===
    
    def _extract_from_v7(self, v7_system) -> List[Dict]:
        """Extrai genomes do V7"""
        genomes = []
        
        try:
            # Extrair best MNIST hyperparameters
            if hasattr(v7_system, 'best') and v7_system.best:
                genome = {
                    'source': 'v7_mnist',
                    'mnist_accuracy': v7_system.best.get('mnist', 0.0),
                    'v7_cycle': v7_system.cycle,
                }
                genomes.append(genome)
            
            # Extrair Darwin population se existir
            if hasattr(v7_system, 'darwin_real') and v7_system.darwin_real:
                darwin = v7_system.darwin_real
                if hasattr(darwin, 'population') and darwin.population:
                    for ind in darwin.population[:5]:  # Top 5
                        if hasattr(ind, 'genome'):
                            genome = dict(ind.genome)
                            genome['source'] = 'v7_darwin'
                            genome['fitness'] = getattr(ind, 'fitness', 0.0)
                            genomes.append(genome)
        except Exception as e:
            logger.debug(f"V7 extraction error: {e}")
        
        return genomes
    
    def _extract_from_brain(self, brain_daemon) -> List[Dict]:
        """Extrai hyperparameters do Brain Daemon"""
        genomes = []
        
        try:
            # Extrair hiperparâmetros atuais
            genome = {
                'source': 'brain_daemon',
                'avg_reward': 0.0,  # Placeholder
                'curiosity_weight': 0.1,  # Placeholder
                'top_k': 8,
            }
            genomes.append(genome)
        except Exception as e:
            logger.debug(f"Brain extraction error: {e}")
        
        return genomes
    
    def _extract_from_db(self, system) -> List[Dict]:
        """Extrai métricas históricas do database"""
        genomes = []
        
        try:
            import sqlite3
            db_path = self.connections['database']['db_path']
            
            with sqlite3.connect(db_path, timeout=2.0) as conn:
                # Pegar top episodes por reward
                results = conn.execute("""
                    SELECT episode, energy, avg_competence, top_k, temperature
                    FROM brain_metrics
                    ORDER BY energy DESC
                    LIMIT 5
                """).fetchall()
                
                for row in results:
                    genome = {
                        'source': 'database_history',
                        'episode': row[0],
                        'energy': row[1],
                        'avg_competence': row[2],
                        'top_k': row[3] if row[3] else 8,
                        'temperature': row[4] if row[4] else 1.0,
                    }
                    genomes.append(genome)
        except Exception as e:
            logger.debug(f"Database extraction error: {e}")
        
        return genomes
    
    def _extract_from_novelty(self, novelty_system) -> List[Dict]:
        """Extrai behaviors do Novelty System"""
        genomes = []
        
        try:
            if hasattr(novelty_system, 'behavior_archive'):
                for i, behavior in enumerate(novelty_system.behavior_archive[-10:]):
                    genome = {
                        'source': 'novelty_archive',
                        'behavior_dim_0': float(behavior[0]) if len(behavior) > 0 else 0.0,
                        'behavior_dim_1': float(behavior[1]) if len(behavior) > 1 else 0.0,
                        'novelty_index': i,
                    }
                    genomes.append(genome)
        except Exception as e:
            logger.debug(f"Novelty extraction error: {e}")
        
        return genomes
    
    # === INJECTION FUNCTIONS ===
    
    def _inject_to_v7(self, v7_system, best_cells) -> bool:
        """Injeta genomes evoluídos no V7"""
        try:
            if not best_cells:
                return False
            
            # Pegar melhor champion
            best_cell = best_cells[0]
            if len(best_cell) < 2:
                return False
            
            champion_data = best_cell[1]
            
            # Aplicar evolved hyperparameters ao V7
            if hasattr(champion_data, 'genome'):
                genome = champion_data.genome
                
                # Ajustar learning rate se presente
                if 'lr' in genome and hasattr(v7_system, 'rl_agent'):
                    evolved_lr = float(genome['lr'])
                    old_lr = getattr(v7_system.rl_agent, 'lr', 0.0003)
                    # Mix suave: 70% old + 30% evolved
                    new_lr = 0.7 * old_lr + 0.3 * evolved_lr
                    v7_system.rl_agent.lr = new_lr
                    logger.debug(f"   V7 lr: {old_lr:.6f} → {new_lr:.6f}")
            
            return True
        except Exception as e:
            logger.debug(f"V7 injection error: {e}")
            return False
    
    def _inject_to_brain(self, brain_daemon, best_cells) -> bool:
        """Injeta evolved hyperparameters no Brain Daemon"""
        try:
            # Placeholder - implementar quando Brain Daemon estiver integrado
            return True
        except Exception as e:
            logger.debug(f"Brain injection error: {e}")
            return False
    
    def _inject_to_db(self, system, best_cells) -> bool:
        """Salva evolved genomes no database"""
        try:
            import sqlite3
            db_path = self.connections['database']['db_path']
            
            # Criar tabela para Darwinacci genomes
            with sqlite3.connect(db_path, timeout=2.0) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS darwinacci_genomes (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cycle INTEGER,
                        score REAL,
                        coverage REAL,
                        genome_json TEXT,
                        timestamp INTEGER
                    )
                """)
                
                # Salvar best genomes
                for cell in best_cells[:3]:
                    if len(cell) >= 2:
                        data = cell[1]
                        if hasattr(data, 'genome') and hasattr(data, 'best_score'):
                            conn.execute("""
                                INSERT INTO darwinacci_genomes 
                                (cycle, score, coverage, genome_json, timestamp)
                                VALUES (?, ?, ?, ?, ?)
                            """, (
                                self.sync_count,
                                data.best_score,
                                getattr(data, 'coverage', 0.0),
                                json.dumps(data.genome),
                                int(datetime.now().timestamp())
                            ))
                
                conn.commit()
            
            return True
        except Exception as e:
            logger.debug(f"Database injection error: {e}")
            return False
    
    def _inject_to_novelty(self, novelty_system, best_cells) -> bool:
        """Injeta behaviors evoluídos no Novelty System"""
        try:
            if not best_cells:
                return False
            
            # Extrair behaviors dos best genomes
            for cell in best_cells[:5]:
                if len(cell) >= 2:
                    data = cell[1]
                    if hasattr(data, 'behavior') and hasattr(novelty_system, 'add_behavior'):
                        novelty_system.add_behavior(data.behavior)
            
            return True
        except Exception as e:
            logger.debug(f"Novelty injection error: {e}")
            return False
    
    def get_synapse_map(self) -> Dict[str, float]:
        """Retorna mapa de sinapses (conexões) ativas"""
        return dict(self.synapses)
    
    def get_status(self) -> Dict[str, Any]:
        """Status completo do connector"""
        return {
            'active': self.active,
            'connections': list(self.connections.keys()),
            'synapses': len(self.synapses),
            'sync_count': self.sync_count,
            'extraction_count': self.extraction_count,
            'injection_count': self.injection_count,
            'connection_details': {
                name: {
                    'type': conn.get('type'),
                    'active': conn.get('active'),
                }
                for name, conn in self.connections.items()
            }
        }
    
    def visualize_network(self) -> str:
        """Retorna visualização ASCII do network sináptico"""
        output = []
        output.append("╔═══════════════════════════════════════════════════════════╗")
        output.append("║         DARWINACCI UNIVERSAL SYNAPSE NETWORK              ║")
        output.append("╚═══════════════════════════════════════════════════════════╝")
        output.append("")
        output.append("                    ┌──────────────────┐")
        output.append("                    │   DARWINACCI-Ω   │")
        output.append("                    │  (Núcleo Central) │")
        output.append("                    └────────┬─────────┘")
        output.append("                             │")
        output.append("        ┌────────────────────┼────────────────────┐")
        output.append("        │                    │                    │")
        
        connections = list(self.connections.keys())
        for i, conn_name in enumerate(connections):
            strength_in = self.synapses.get(f'{conn_name}→darwinacci', 0.0)
            strength_out = self.synapses.get(f'darwinacci→{conn_name}', 0.0)
            
            status = "✅" if self.connections[conn_name].get('active') else "❌"
            output.append(f"  ┌─────▼──────┐")
            output.append(f"  │ {conn_name:^10} │ {status} IN:{strength_in:.1f} OUT:{strength_out:.1f}")
            output.append(f"  └────────────┘")
        
        output.append("")
        output.append(f"Syncs: {self.sync_count} | Extractions: {self.extraction_count} | Injections: {self.injection_count}")
        
        return "\n".join(output)


# Global instance (singleton)
_UNIVERSAL_CONNECTOR = None

def get_universal_connector(darwinacci_engine=None):
    """Get or create universal connector instance"""
    global _UNIVERSAL_CONNECTOR
    if _UNIVERSAL_CONNECTOR is None:
        _UNIVERSAL_CONNECTOR = UniversalConnector(darwinacci_engine)
    return _UNIVERSAL_CONNECTOR