#!/usr/bin/env python3
"""
🔍 MEMORY LEAK DETECTOR
BLOCO 2 - TAREFA 26

Tracks memory usage over time and detects leaks.
"""

__version__ = "1.0.0"

import psutil
import time
import sqlite3
import sys
import os
import signal
from pathlib import Path
from datetime import datetime

DB_PATH = Path("/root/memory_profile.db")
LOG_PATH = Path("/root/memory_profile.log")

def setup_db():
    """Initialize memory profiling database"""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp INTEGER NOT NULL,
            rss_mb REAL NOT NULL,
            vms_mb REAL NOT NULL,
            percent REAL NOT NULL,
            num_fds INTEGER,
            num_threads INTEGER
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_memory_timestamp 
        ON memory_samples(timestamp)
    """)
    conn.commit()
    conn.close()

def log(msg):
    """Simple logging"""
    line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')

def monitor_memory(target_pid=None, interval=60):
    """
    Monitor memory usage and detect leaks.
    
    Args:
        target_pid: PID to monitor (default: self)
        interval: Sample interval in seconds
    """
    pid_file = Path("/root/memory_profiler.pid")
    pid_file.write_text(str(os.getpid()))
    
    # Signal handler
    def cleanup(signum, frame):
        log("🛑 Stopping memory profiler...")
        pid_file.unlink(missing_ok=True)
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    
    # Setup
    setup_db()
    
    if target_pid:
        process = psutil.Process(target_pid)
        log(f"📊 Monitoring process: {target_pid} ({process.name()})")
    else:
        process = psutil.Process()
        log(f"📊 Monitoring self: {os.getpid()}")
    
    log(f"Sampling interval: {interval}s")
    log("")
    
    try:
        while True:
            # Collect metrics
            try:
                mem = process.memory_info()
                percent = process.memory_percent()
                num_fds = process.num_fds() if hasattr(process, 'num_fds') else 0
                num_threads = process.num_threads()
                
                # Save to DB
                conn = sqlite3.connect(DB_PATH)
                conn.execute("""
                    INSERT INTO memory_samples 
                    (timestamp, rss_mb, vms_mb, percent, num_fds, num_threads)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    int(time.time()),
                    mem.rss / 1024**2,  # MB
                    mem.vms / 1024**2,  # MB
                    percent,
                    num_fds,
                    num_threads
                ))
                conn.commit()
                
                # Check for leak (growth > 10MB/hour)
                cursor = conn.execute("""
                    SELECT rss_mb FROM memory_samples 
                    WHERE timestamp > ? 
                    ORDER BY timestamp ASC 
                    LIMIT 1
                """, (int(time.time()) - 3600,))
                
                row = cursor.fetchone()
                if row:
                    growth_mb = (mem.rss / 1024**2) - row[0]
                    if growth_mb > 10:
                        log(f"⚠️ Memory leak detected: +{growth_mb:.1f}MB in last hour")
                        log(f"   Current: {mem.rss / 1024**2:.1f}MB ({percent:.1f}%)")
                
                conn.close()
                
                # Log sample
                if time.time() % 600 < interval:  # Every 10min
                    log(f"📊 RSS: {mem.rss / 1024**2:.1f}MB | {percent:.1f}% | FDs: {num_fds} | Threads: {num_threads}")
                
            except psutil.NoSuchProcess:
                log(f"❌ Process {process.pid} died")
                break
            except Exception as e:
                log(f"⚠️ Sample error: {e}")
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        pass
    finally:
        pid_file.unlink(missing_ok=True)
        log("✅ Memory profiler stopped")

if __name__ == "__main__":
    target = int(sys.argv[1]) if len(sys.argv) > 1 else None
    interval = int(sys.argv[2]) if len(sys.argv) > 2 else 60
    
    monitor_memory(target_pid=target, interval=interval)
