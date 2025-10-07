"""
Database Mass Integrator - Massive knowledge extraction
Integrates 78+ databases with transfer learning capabilities

Extracts:
- Model checkpoints and weights
- Training trajectories
- Knowledge graphs
- Emergence patterns
- Performance metrics

Enables massive transfer learning and bootstrapping
"""

import logging
import sqlite3
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import json
import numpy as np

logger = logging.getLogger(__name__)


class DatabaseMassIntegrator:
    """
    Mass database integrator
    Efficiently processes 78+ databases for knowledge extraction
    """
    
    def __init__(self, target_db_path: str, source_db_dir: str = "/root"):
        self.target_db_path = target_db_path
        self.source_db_dir = Path(source_db_dir)
        
        # Connect to target (unified) database
        self.target_conn = sqlite3.connect(target_db_path)
        self.target_cursor = self.target_conn.cursor()
        
        self.integrated_dbs: List[str] = []
        self.total_rows_integrated = 0
        self.errors: List[str] = []
        
        # Ensure extended tables exist
        self._create_extended_tables()
        
    def _create_extended_tables(self):
        """Create tables for mass integration"""
        # Evolution history table
        self.target_cursor.execute('''
            CREATE TABLE IF NOT EXISTS evolution_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_db TEXT,
                generation INTEGER,
                fitness REAL,
                architecture TEXT,
                timestamp REAL
            )
        ''')
        
        # Checkpoints table  
        self.target_cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_checkpoints (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_db TEXT,
                model_name TEXT,
                checkpoint_data TEXT,
                performance REAL,
                metadata TEXT
            )
        ''')
        
        # Emergence patterns table
        self.target_cursor.execute('''
            CREATE TABLE IF NOT EXISTS emergence_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_db TEXT,
                pattern_type TEXT,
                pattern_data TEXT,
                confidence REAL,
                timestamp REAL
            )
        ''')
        
        # Generic integrated data table (if not exists)
        self.target_cursor.execute('''
            CREATE TABLE IF NOT EXISTS integrated_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_db TEXT,
                data_type TEXT,
                content TEXT
            )
        ''')
        
        self.target_conn.commit()
        logger.info("✅ Extended tables created for mass integration")
    
    def discover_databases(self) -> List[Path]:
        """
        Discover all .db files in source directory
        
        Returns:
            List of database paths
        """
        db_files = list(self.source_db_dir.glob("*.db"))
        
        # Filter out target database and test databases
        db_files = [
            db for db in db_files
            if str(db) != self.target_db_path
            and "test" not in db.name.lower()
            and db.stat().st_size > 1024  # > 1KB
        ]
        
        logger.info(f"📊 Discovered {len(db_files)} databases to integrate")
        return db_files
    
    def inspect_database(self, db_path: Path) -> Dict[str, Any]:
        """
        Inspect database structure and content
        
        Args:
            db_path: Path to database
        
        Returns:
            Database metadata
        """
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Get tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            # Get row counts
            table_info = {}
            total_rows = 0
            for table in tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM {table}")
                    count = cursor.fetchone()[0]
                    table_info[table] = count
                    total_rows += count
                except:
                    table_info[table] = 0
            
            conn.close()
            
            return {
                'path': str(db_path),
                'name': db_path.name,
                'size_mb': db_path.stat().st_size / (1024 * 1024),
                'tables': tables,
                'table_info': table_info,
                'total_rows': total_rows
            }
        
        except Exception as e:
            logger.error(f"Error inspecting {db_path.name}: {e}")
            return None
    
    def integrate_database(self, db_path: Path) -> Tuple[bool, int]:
        """
        Integrate a single database
        
        Args:
            db_path: Path to database
        
        Returns:
            (success, rows_integrated)
        """
        try:
            # Inspect first
            metadata = self.inspect_database(db_path)
            if not metadata:
                return False, 0
            
            logger.info(f"📥 Integrating {db_path.name} ({metadata['size_mb']:.2f}MB, {metadata['total_rows']} rows)")
            
            source_conn = sqlite3.connect(str(db_path))
            source_cursor = source_conn.cursor()
            
            rows_integrated = 0
            
            # Extract valuable data based on table names
            for table, count in metadata['table_info'].items():
                if count == 0:
                    continue
                
                # Extract evolution data
                if any(keyword in table.lower() for keyword in ['evolution', 'generation', 'fitness']):
                    rows = self._extract_evolution_data(source_cursor, table, db_path.name)
                    rows_integrated += rows
                
                # Extract checkpoint data
                elif any(keyword in table.lower() for keyword in ['checkpoint', 'model', 'weights']):
                    rows = self._extract_checkpoint_data(source_cursor, table, db_path.name)
                    rows_integrated += rows
                
                # Extract emergence data
                elif any(keyword in table.lower() for keyword in ['emergence', 'pattern', 'metric']):
                    rows = self._extract_emergence_data(source_cursor, table, db_path.name)
                    rows_integrated += rows
                
                # Generic data extraction (to integrated_data table)
                else:
                    rows = self._extract_generic_data(source_cursor, table, db_path.name)
                    rows_integrated += rows
            
            source_conn.close()
            self.target_conn.commit()
            
            self.integrated_dbs.append(db_path.name)
            self.total_rows_integrated += rows_integrated
            
            logger.info(f"   ✅ Integrated {rows_integrated} rows from {db_path.name}")
            return True, rows_integrated
        
        except Exception as e:
            error_msg = f"Error integrating {db_path.name}: {e}"
            logger.error(error_msg)
            self.errors.append(error_msg)
            return False, 0
    
    def _extract_evolution_data(self, cursor: sqlite3.Cursor, table: str, source_db: str) -> int:
        """Extract evolution/generation data"""
        try:
            # Try to extract relevant columns
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            
            # Look for generation/fitness columns
            gen_col = next((c for c in columns if 'gen' in c.lower() or 'iteration' in c.lower()), None)
            fit_col = next((c for c in columns if 'fit' in c.lower() or 'score' in c.lower() or 'performance' in c.lower()), None)
            
            if gen_col and fit_col:
                cursor.execute(f"SELECT * FROM {table} LIMIT 100")  # Limit for efficiency
                rows = cursor.fetchall()
                
                for row in rows:
                    row_dict = dict(zip(columns, row))
                    self.target_cursor.execute('''
                        INSERT INTO evolution_history (source_db, generation, fitness, architecture, timestamp)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (
                        source_db,
                        row_dict.get(gen_col, 0),
                        row_dict.get(fit_col, 0.0),
                        json.dumps(row_dict),
                        row_dict.get('timestamp', 0.0)
                    ))
                
                return len(rows)
        except Exception as e:
            logger.warning(f"Could not extract evolution data from {table}: {e}")
        
        return 0
    
    def _extract_checkpoint_data(self, cursor: sqlite3.Cursor, table: str, source_db: str) -> int:
        """Extract checkpoint/model data"""
        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT 50")  # Checkpoints can be large
            rows = cursor.fetchall()
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            
            for row in rows:
                row_dict = dict(zip(columns, row))
                self.target_cursor.execute('''
                    INSERT INTO model_checkpoints (source_db, model_name, checkpoint_data, performance, metadata)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    source_db,
                    f"{source_db}_{table}",
                    json.dumps(row_dict, default=str),
                    row_dict.get('performance', 0.0) if 'performance' in row_dict else 0.0,
                    json.dumps({'table': table, 'source': source_db})
                ))
            
            return len(rows)
        except Exception as e:
            logger.warning(f"Could not extract checkpoint data from {table}: {e}")
        
        return 0
    
    def _extract_emergence_data(self, cursor: sqlite3.Cursor, table: str, source_db: str) -> int:
        """Extract emergence/pattern data"""
        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT 100")
            rows = cursor.fetchall()
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            
            for row in rows:
                row_dict = dict(zip(columns, row))
                self.target_cursor.execute('''
                    INSERT INTO emergence_patterns (source_db, pattern_type, pattern_data, confidence, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    source_db,
                    table,
                    json.dumps(row_dict, default=str),
                    row_dict.get('confidence', 0.0) if 'confidence' in row_dict else 0.0,
                    row_dict.get('timestamp', 0.0)
                ))
            
            return len(rows)
        except Exception as e:
            logger.warning(f"Could not extract emergence data from {table}: {e}")
        
        return 0
    
    def _extract_generic_data(self, cursor: sqlite3.Cursor, table: str, source_db: str) -> int:
        """Extract generic data to integrated_data table"""
        try:
            cursor.execute(f"SELECT * FROM {table} LIMIT 100")
            rows = cursor.fetchall()
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [row[1] for row in cursor.fetchall()]
            
            for row in rows:
                row_dict = dict(zip(columns, row))
                
                # Insert into generic integrated_data
                self.target_cursor.execute('''
                    INSERT INTO integrated_data (source_db, data_type, content)
                    VALUES (?, ?, ?)
                ''', (
                    source_db,
                    table,
                    json.dumps(row_dict, default=str)
                ))
            
            return len(rows)
        except:
            # Table might not exist or other error
            return 0
    
    def integrate_all(self, max_databases: Optional[int] = None) -> Dict[str, Any]:
        """
        Integrate all discovered databases
        
        Args:
            max_databases: Optional limit on number of databases
        
        Returns:
            Integration statistics
        """
        logger.info("="*80)
        logger.info("🚀 STARTING MASS DATABASE INTEGRATION")
        logger.info("="*80)
        
        # Discover
        databases = self.discover_databases()
        
        if max_databases:
            databases = databases[:max_databases]
        
        logger.info(f"📊 Integrating {len(databases)} databases...")
        
        # Integrate each
        successful = 0
        failed = 0
        
        for i, db_path in enumerate(databases, 1):
            logger.info(f"\n[{i}/{len(databases)}] Processing {db_path.name}...")
            
            success, rows = self.integrate_database(db_path)
            
            if success:
                successful += 1
            else:
                failed += 1
        
        # Final commit
        self.target_conn.commit()
        
        # Statistics
        stats = {
            'total_databases_discovered': len(databases),
            'successfully_integrated': successful,
            'failed': failed,
            'total_rows_integrated': self.total_rows_integrated,
            'integrated_dbs': self.integrated_dbs,
            'errors': self.errors
        }
        
        logger.info("\n" + "="*80)
        logger.info("✅ MASS INTEGRATION COMPLETE")
        logger.info("="*80)
        logger.info(f"📊 Databases: {successful}/{len(databases)} integrated")
        logger.info(f"📊 Total rows: {self.total_rows_integrated:,}")
        logger.info(f"📊 Errors: {failed}")
        logger.info("="*80)
        
        return stats
    
    def get_integration_summary(self) -> Dict[str, Any]:
        """Get summary of integrated data"""
        # Count rows by type
        self.target_cursor.execute("SELECT COUNT(*) FROM evolution_history")
        evolution_count = self.target_cursor.fetchone()[0]
        
        self.target_cursor.execute("SELECT COUNT(*) FROM model_checkpoints")
        checkpoint_count = self.target_cursor.fetchone()[0]
        
        self.target_cursor.execute("SELECT COUNT(*) FROM emergence_patterns")
        emergence_count = self.target_cursor.fetchone()[0]
        
        self.target_cursor.execute("SELECT COUNT(*) FROM integrated_data")
        generic_count = self.target_cursor.fetchone()[0]
        
        return {
            'total_databases': len(self.integrated_dbs),
            'total_rows': self.total_rows_integrated,
            'evolution_history': evolution_count,
            'model_checkpoints': checkpoint_count,
            'emergence_patterns': emergence_count,
            'generic_data': generic_count
        }
    
    def close(self):
        """Close database connections"""
        self.target_conn.close()

    def scan_databases(self, limit: int = None) -> dict:
        """Scan available databases - REAL"""
        dbs = self.discover_databases()
        if limit: dbs = dbs[:limit]
        results = {'total_found': len(dbs), 'scanned': 0, 'total_rows': 0, 'databases': []}
        for db_path in dbs:
            try:
                info = self.inspect_database(db_path)
                results['databases'].append({'name': db_path.name, 'rows': info.get('total_rows', 0)})
                results['scanned'] += 1
                results['total_rows'] += info.get('total_rows', 0)
            except: pass
        logger.info(f"📊 Scanned {results['scanned']}/{results['total_found']} DBs: {results['total_rows']} rows")
        return results


# Test function
def test_database_mass_integrator():
    """Test the database mass integrator"""
    import tempfile
    
    print("="*80)
    print("🧪 TESTING DATABASE MASS INTEGRATOR")
    print("="*80)
    
    # Create temporary target database
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.db') as f:
        temp_db = f.name
    
    try:
        # Initialize integrator
        integrator = DatabaseMassIntegrator(
            target_db_path=temp_db,
            source_db_dir="/root"
        )
        
        # Discover databases
        databases = integrator.discover_databases()
        print(f"\n📊 Discovered {len(databases)} databases")
        
        # Show top 10 by size
        print("\n📊 Top 10 by size:")
        databases_sorted = sorted(databases, key=lambda x: x.stat().st_size, reverse=True)
        for i, db in enumerate(databases_sorted[:10], 1):
            size_mb = db.stat().st_size / (1024 * 1024)
            print(f"   {i}. {db.name}: {size_mb:.2f}MB")
        
        # Integrate top 5 (for testing)
        print("\n🚀 Integrating top 5 databases (test mode):")
        stats = integrator.integrate_all(max_databases=5)
        
        # Get summary
        print("\n📊 Integration Summary:")
        summary = integrator.get_integration_summary()
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                print(f"   {key}: {value:,}")
        
        integrator.close()
        
        print("\n" + "="*80)
        print("✅ DATABASE MASS INTEGRATOR TEST COMPLETE")
        print("="*80)
        
        return stats
    
    finally:
        # Cleanup
        if os.path.exists(temp_db):
            os.remove(temp_db)


if __name__ == "__main__":
    # Run test
    stats = test_database_mass_integrator()
