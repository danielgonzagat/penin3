"""
Emergence Tracker SIMPLIFIED - No numpy, minimal dependencies
"""
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class EmergenceTracker:
    """Tracks statistical surprises and emergent patterns"""
    
    def __init__(self, surprises_db: Path, connections_db: Path):
        self.surprises_db = str(surprises_db)
        self.connections_db = str(connections_db)
        
        # Rolling statistics for surprise detection
        self.metric_history = defaultdict(list)
        self.max_history = 100
        
        # Initialize DBs immediately
        self._init_dbs()
    
    def _init_dbs(self):
        """Initialize databases"""
        try:
            # Surprises DB
            conn1 = sqlite3.connect(self.surprises_db, timeout=5.0)
            conn1.execute("""
                CREATE TABLE IF NOT EXISTS surprises (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_type TEXT NOT NULL,
                    metric_name TEXT NOT NULL,
                    expected_value REAL,
                    actual_value REAL,
                    sigma REAL NOT NULL,
                    episode INTEGER,
                    timestamp INTEGER NOT NULL
                )
            """)
            conn1.execute("CREATE INDEX IF NOT EXISTS idx_surprises_sigma ON surprises(sigma DESC)")
            conn1.execute("CREATE INDEX IF NOT EXISTS idx_surprises_timestamp ON surprises(timestamp DESC)")
            conn1.commit()
            conn1.close()
            
            # Connections DB
            conn2 = sqlite3.connect(self.connections_db, timeout=5.0)
            conn2.execute("""
                CREATE TABLE IF NOT EXISTS connections (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_system TEXT NOT NULL,
                    target_system TEXT NOT NULL,
                    connection_type TEXT,
                    strength REAL,
                    data_flow TEXT,
                    timestamp INTEGER NOT NULL
                )
            """)
            conn2.execute("CREATE INDEX IF NOT EXISTS idx_connections_timestamp ON connections(timestamp DESC)")
            conn2.commit()
            conn2.close()
            
            logger.info("✅ Emergence tracking DBs initialized (SIMPLE)")
            
        except Exception as e:
            logger.error(f"Failed to init emergence DBs: {e}")
            # Don't raise - fail gracefully
    
    def track_metric(self, metric_name: str, value: float, episode: Optional[int] = None):
        """Track a metric and detect statistical surprises"""
        try:
            # Add to history
            self.metric_history[metric_name].append(value)
            if len(self.metric_history[metric_name]) > self.max_history:
                self.metric_history[metric_name].pop(0)
            
            # Need at least 10 samples for statistics
            history = self.metric_history[metric_name]
            if len(history) < 10:
                return
            
            # Compute statistics (pure Python, no numpy)
            history_prev = history[:-1]
            mean = sum(history_prev) / len(history_prev)
            
            # Compute std dev
            variance = sum((x - mean) ** 2 for x in history_prev) / len(history_prev)
            std = variance ** 0.5
            
            if std < 1e-6:  # Avoid division by zero
                return
            
            # Compute z-score (sigma)
            z_score = abs((value - mean) / std)
            
            # If >3σ, it's a surprise!
            if z_score > 3.0:
                self.record_surprise(
                    event_type="statistical_anomaly",
                    metric_name=metric_name,
                    expected_value=float(mean),
                    actual_value=float(value),
                    sigma=float(z_score),
                    episode=episode
                )
                logger.warning(
                    f"🌟 SURPRISE: {metric_name} = {value:.3f} "
                    f"(expected {mean:.3f}, {z_score:.1f}σ deviation!)"
                )
        
        except Exception as e:
            logger.error(f"Failed to track metric {metric_name}: {e}")
    
    def record_surprise(self, event_type: str, metric_name: str,
                       expected_value: float, actual_value: float,
                       sigma: float, episode: Optional[int] = None):
        """Record a surprise event"""
        try:
            conn = sqlite3.connect(self.surprises_db, timeout=5.0)
            conn.execute("""
                INSERT INTO surprises (
                    event_type, metric_name, expected_value, actual_value,
                    sigma, episode, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (event_type, metric_name, expected_value, actual_value,
                 sigma, episode, int(datetime.now().timestamp())))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to record surprise: {e}")
    
    def record_connection(self, source_system: str, target_system: str,
                         connection_type: str = "data_flow",
                         strength: float = 1.0, data_flow: Optional[str] = None):
        """Record a system connection"""
        try:
            conn = sqlite3.connect(self.connections_db, timeout=5.0)
            conn.execute("""
                INSERT INTO connections (
                    source_system, target_system, connection_type,
                    strength, data_flow, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (source_system, target_system, connection_type,
                 strength, data_flow, int(datetime.now().timestamp())))
            conn.commit()
            conn.close()
        except Exception as e:
            logger.error(f"Failed to record connection: {e}")
    
    def get_high_sigma_surprises(self, min_sigma: float = 5.0, limit: int = 20) -> List[Dict]:
        """Get high-sigma surprises"""
        try:
            conn = sqlite3.connect(self.surprises_db, timeout=5.0)
            conn.row_factory = sqlite3.Row
            results = conn.execute("""
                SELECT * FROM surprises 
                WHERE sigma >= ?
                ORDER BY sigma DESC 
                LIMIT ?
            """, (min_sigma, limit)).fetchall()
            conn.close()
            
            return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get surprises: {e}")
            return []
    
    def get_connection_graph(self) -> Dict[str, List[str]]:
        """Get system connection graph"""
        try:
            graph = defaultdict(list)
            conn = sqlite3.connect(self.connections_db, timeout=5.0)
            results = conn.execute("""
                SELECT source_system, target_system 
                FROM connections
                ORDER BY timestamp DESC
                LIMIT 1000
            """).fetchall()
            conn.close()
            
            for source, target in results:
                if target not in graph[source]:
                    graph[source].append(target)
            
            return dict(graph)
        except Exception as e:
            logger.error(f"Failed to get connection graph: {e}")
            return {}
