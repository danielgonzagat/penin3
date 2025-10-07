"""
Database Schema Migrations
Manages schema versions and upgrades
"""
import sqlite3
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class DatabaseMigrator:
    """Handles database schema migrations"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        self._ensure_version_table()
    
    def _ensure_version_table(self):
        """Create schema_version table if missing"""
        # Check if table exists
        cursor = self.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        exists = cursor.fetchone() is not None
        
        if not exists:
            self.conn.execute("""
                CREATE TABLE schema_version (
                    version INTEGER PRIMARY KEY,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    description TEXT
                )
            """)
            self.conn.commit()
        else:
            # Verify description column exists, add if missing
            cursor = self.conn.execute("PRAGMA table_info(schema_version)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'description' not in columns:
                self.conn.execute("ALTER TABLE schema_version ADD COLUMN description TEXT")
                self.conn.commit()
    
    def get_current_version(self) -> int:
        """Get current schema version"""
        try:
            cursor = self.conn.execute("SELECT version FROM schema_version WHERE id = 1")
            row = cursor.fetchone()
            return row['version'] if row and row['version'] is not None else 0
        except Exception:
            return 0
    
    def migrate_to_v2(self):
        """Migrate from v1 to v2: Add indices for errors, APIs, events"""
        current = self.get_current_version()
        if current >= 2:
            logger.info(f"✅ Database already at v{current} (>= v2)")
            return True
        
        logger.info("🔧 Migrating database to v2...")
        
        try:
            # Add events table if missing
            self.conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle INTEGER,
                    event_type TEXT,
                    payload TEXT,
                    timestamp INTEGER
                )
            """)
            
            # Add indices for fast queries (safe: skip if table/column doesn't exist)
            indices = [
                ("CREATE INDEX IF NOT EXISTS idx_errors_cycle ON errors(cycle)", "errors", "cycle"),
                ("CREATE INDEX IF NOT EXISTS idx_errors_component ON errors(component)", "errors", "component"),
                ("CREATE INDEX IF NOT EXISTS idx_events_cycle ON events(cycle)", "events", "cycle"),
                ("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)", "events", "event_type"),
                ("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)", "events", "timestamp"),
            ]
            
            for idx_sql, table, column in indices:
                try:
                    # Verify table and column exist
                    cursor = self.conn.execute(f"PRAGMA table_info({table})")
                    columns = [row[1] for row in cursor.fetchall()]
                    if column not in columns:
                        logger.warning(f"   ⚠️  Skipping index on {table}.{column} (column doesn't exist)")
                        continue
                    self.conn.execute(idx_sql)
                    logger.info(f"   ✅ {idx_sql.split('idx_')[1].split(' ')[0]}")
                except Exception as e:
                    logger.warning(f"   ⚠️  Index creation skipped: {e}")
            
            # Record migration (UPDATE existing row with id=1)
            self.conn.execute(
                "UPDATE schema_version SET version = ?, description = ? WHERE id = 1",
                (2, "Added indices for errors, API responses, and events table")
            )
            
            self.conn.commit()
            logger.info("✅ Migration to v2 complete")
            return True
            
        except Exception as e:
            logger.error(f"❌ Migration failed: {e}")
            self.conn.rollback()
            return False
    
    def migrate_to_latest(self):
        """Migrate to latest schema version"""
        current = self.get_current_version()
        logger.info(f"📊 Current schema version: v{current}")
        
        if current < 2:
            if not self.migrate_to_v2():
                return False
        
        final = self.get_current_version()
        logger.info(f"✅ Database at v{final}")
        return True
    
    def close(self):
        """Close database connection"""
        self.conn.close()


def migrate_database(db_path: Path) -> bool:
    """Helper function to migrate database to latest schema"""
    migrator = DatabaseMigrator(db_path)
    try:
        return migrator.migrate_to_latest()
    finally:
        migrator.close()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config.settings import DATABASE_PATH
    
    logging.basicConfig(level=logging.INFO)
    
    print("🔧 Starting database migration...")
    success = migrate_database(DATABASE_PATH)
    
    if success:
        print("✅ Migration complete!")
        sys.exit(0)
    else:
        print("❌ Migration failed!")
        sys.exit(1)