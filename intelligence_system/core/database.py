"""
Professional Database Manager
Clean abstraction, proper error handling
"""
import sqlite3
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class Database:
    """Clean database interface with proper error handling"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._conn = None  # FIX: For cleanup compatibility
        self._init_db()
    
    @property
    def conn(self):
        """FIX: Property for legacy cleanup code"""
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn
    
    def _init_db(self):
        """Initialize database schema"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS cycles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cycle INTEGER NOT NULL,
                        mnist_accuracy REAL,
                        cartpole_reward REAL,
                        cartpole_avg_reward REAL,
                        timestamp INTEGER NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS api_responses (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cycle INTEGER NOT NULL,
                        api_name TEXT NOT NULL,
                        prompt TEXT,
                        response TEXT,
                        used_for TEXT,
                        timestamp INTEGER NOT NULL
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS errors (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        cycle INTEGER,
                        component TEXT NOT NULL,
                        error_type TEXT NOT NULL,
                        error_message TEXT,
                        traceback TEXT,
                        timestamp INTEGER NOT NULL
                    )
                """)
                
                conn.commit()
                logger.info(f"✅ Database initialized: {self.db_path}")
                
        except Exception as e:
            logger.error(f"❌ Database init failed: {e}", exc_info=True)
            raise
    
    def save_cycle(self, cycle: int, mnist: Optional[float] = None, 
                   cartpole: Optional[float] = None, 
                   cartpole_avg: Optional[float] = None) -> bool:
        """Save cycle metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO cycles (cycle, mnist_accuracy, cartpole_reward, 
                                      cartpole_avg_reward, timestamp)
                    VALUES (?, ?, ?, ?, ?)
                """, (cycle, mnist, cartpole, cartpole_avg, int(datetime.now().timestamp())))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save cycle: {e}")
            return False
    
    def save_api_response(self, cycle: int, api_name: str, prompt: str, 
                         response: str, used_for: str) -> bool:
        """Save API interaction"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO api_responses (cycle, api_name, prompt, response, 
                                             used_for, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (cycle, api_name, prompt, response, used_for, 
                     int(datetime.now().timestamp())))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save API response: {e}")
            return False
    
    def save_error(self, component: str, error_type: str, 
                   error_message: str, traceback: str = "", 
                   cycle: Optional[int] = None) -> bool:
        """Save error for debugging"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO errors (cycle, component, error_type, 
                                      error_message, traceback, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (cycle, component, error_type, error_message, traceback,
                     int(datetime.now().timestamp())))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save error: {e}")
            return False
    
    def get_last_cycle(self) -> int:
        """Get last cycle number"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute(
                    "SELECT MAX(cycle) FROM cycles"
                ).fetchone()
                return result[0] or 0
        except Exception as e:
            logger.error(f"Failed to get last cycle: {e}")
            return 0
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                result = conn.execute("""
                    SELECT 
                        MAX(mnist_accuracy) as best_mnist,
                        MAX(cartpole_avg_reward) as best_cartpole
                    FROM cycles
                """).fetchone()
                
                return {
                    "mnist": result[0] or 0.0,
                    "cartpole": result[1] or 0.0
                }
        except Exception as e:
            logger.error(f"Failed to get best metrics: {e}")
            return {"mnist": 0.0, "cartpole": 0.0}
    
    def get_recent_cycles(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent cycle data"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                results = conn.execute("""
                    SELECT * FROM cycles 
                    ORDER BY cycle DESC 
                    LIMIT ?
                """, (limit,)).fetchall()
                
                return [dict(row) for row in results]
        except Exception as e:
            logger.error(f"Failed to get recent cycles: {e}")
            return []
    
    def get_stagnation_score(self, window: int = 10) -> float:
        """Check if system is stagnating (0=improving, 1=stagnant)"""
        try:
            recent = self.get_recent_cycles(window)
            if len(recent) < window:
                return 0.0
            
            # Check variance in recent performance
            mnist_vals = [r["mnist_accuracy"] for r in recent if r["mnist_accuracy"]]
            if len(mnist_vals) < 2:
                return 0.0
            
            variance = sum((x - sum(mnist_vals)/len(mnist_vals))**2 for x in mnist_vals) / len(mnist_vals)
            
            # Low variance = stagnation
            return 1.0 if variance < 0.01 else 0.0
            
        except Exception as e:
            logger.error(f"Failed to compute stagnation: {e}")
            return 0.0
