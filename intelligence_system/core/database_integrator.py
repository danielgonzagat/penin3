"""
DATABASE INTEGRATOR - Integração rigorosa de databases externas
Merge científico de models, trajectories e knowledge de databases antigas
"""
import logging
import sqlite3
import pickle
import json
from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class DatabaseIntegrator:
    """
    Integra dados de databases externas no sistema atual
    - Models/neurons treinados
    - Learning trajectories
    - Knowledge/patterns
    """
    
    def __init__(self, target_db_path: Path):
        self.target_db = target_db_path
        self.integrated_count = 0
        self.errors = []
        
        logger.info("🔗 Database Integrator initialized")
        logger.info(f"   Target: {target_db_path}")
    
    def integrate_database(self, source_db_path: str, db_type: str) -> Dict[str, Any]:
        """
        Integra uma database externa
        
        Args:
            source_db_path: Path para database fonte
            db_type: Tipo de dados ('models', 'trajectories', 'knowledge')
        
        Returns:
            Estatísticas de integração
        """
        if not Path(source_db_path).exists():
            return {'error': 'Database not found'}
        
        db_name = Path(source_db_path).name
        logger.info(f"📂 Integrating {db_name} ({db_type})...")
        
        stats = {
            'source': db_name,
            'type': db_type,
            'rows_integrated': 0,
            'tables_processed': 0,
            'errors': []
        }
        
        try:
            source_conn = sqlite3.connect(source_db_path)
            source_cursor = source_conn.cursor()
            
            # Get tables
            source_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in source_cursor.fetchall()]
            
            for table in tables:
                if table == 'sqlite_sequence':
                    continue
                
                try:
                    result = self._integrate_table(
                        source_cursor,
                        table,
                        db_type,
                        db_name
                    )
                    
                    stats['rows_integrated'] += result['rows']
                    stats['tables_processed'] += 1
                    
                except Exception as e:
                    error_msg = f"{table}: {str(e)[:50]}"
                    stats['errors'].append(error_msg)
                    logger.warning(f"   ⚠️  {error_msg}")
            
            source_conn.close()
            
            logger.info(f"   ✅ Integrated {stats['rows_integrated']} rows from {stats['tables_processed']} tables")
            
        except Exception as e:
            stats['errors'].append(str(e)[:100])
            logger.error(f"   ❌ Failed: {str(e)[:100]}")
        
        return stats
    
    def _integrate_table(self, cursor: sqlite3.Cursor, table: str, db_type: str, source_name: str) -> Dict[str, int]:
        """Integra uma table específica"""
        
        # Get all data
        cursor.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        
        if not rows:
            return {'rows': 0}
        
        # Get column names
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [col[1] for col in cursor.fetchall()]
        
        # Save to target database with metadata
        target_conn = sqlite3.connect(str(self.target_db))
        target_cursor = target_conn.cursor()
        
        # Create integrated_data table if not exists
        target_cursor.execute('''
            CREATE TABLE IF NOT EXISTS integrated_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_db TEXT,
                source_table TEXT,
                data_type TEXT,
                data_json TEXT,
                integrated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Insert each row as JSON
        for row in rows:
            row_dict = dict(zip(columns, row))
            row_json = json.dumps(row_dict, default=str)
            
            target_cursor.execute('''
                INSERT INTO integrated_data (source_db, source_table, data_type, data_json)
                VALUES (?, ?, ?, ?)
            ''', (source_name, table, db_type, row_json))
        
        target_conn.commit()
        target_conn.close()
        
        logger.info(f"      ✓ {table}: {len(rows)} rows")
        
        return {'rows': len(rows)}
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas de integração"""
        
        try:
            conn = sqlite3.connect(str(self.target_db))
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*), COUNT(DISTINCT source_db), COUNT(DISTINCT data_type) FROM integrated_data")
            total_rows, unique_dbs, unique_types = cursor.fetchone()
            
            conn.close()
            
            return {
                'total_rows_integrated': total_rows,
                'unique_databases': unique_dbs,
                'unique_data_types': unique_types
            }
        except:
            return {'total_rows_integrated': 0}


if __name__ == "__main__":
    # Test integration
    import sys
    from pathlib import Path as PathLib
    sys.path.insert(0, str(PathLib(__file__).parent.parent))
    
    from config.settings import DATABASE_PATH
    
    integrator = DatabaseIntegrator(DATABASE_PATH)
    
    # Top 5 databases
    top_dbs = [
        ('/root/sistema_real_24_7.db', 'models'),
        ('/root/test_memory.db', 'models'),
        ('/root/fazenda_memory.db', 'models'),
        ('/root/inteligencia_suprema_24_7.db', 'trajectories'),
        ('/root/true_emergent_real_intelligence.db', 'models')
    ]
    
    print("="*80)
    print("🔬 INTEGRATING TOP 5 DATABASES")
    print("="*80)
    print()
    
    for db_path, db_type in top_dbs:
        stats = integrator.integrate_database(db_path, db_type)
        print(f"✅ {Path(db_path).name}: {stats['rows_integrated']} rows integrated")
    
    print()
    final_stats = integrator.get_integration_stats()
    print(f"📊 TOTAL INTEGRATION:")
    print(f"   Rows: {final_stats['total_rows_integrated']}")
    print(f"   Databases: {final_stats['unique_databases']}")
    print(f"   Types: {final_stats['unique_data_types']}")
    print()
    print("="*80)
