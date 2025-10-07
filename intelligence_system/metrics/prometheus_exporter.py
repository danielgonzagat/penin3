#!/usr/bin/env python3
"""
📊 PROMETHEUS METRICS EXPORTER
BLOCO 2 - TAREFA 23

Exposes Intelligence System metrics in Prometheus format.
Access at: http://127.0.0.1:9090/metrics
"""

__version__ = "1.0.0"

from http.server import HTTPServer, BaseHTTPRequestHandler
import sqlite3
import json
import sys
import os
import signal
from pathlib import Path

class MetricsHandler(BaseHTTPRequestHandler):
    """Serve Prometheus metrics"""
    
    def do_GET(self):
        if self.path == '/metrics':
            metrics = self.collect_metrics()
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; version=0.0.4')
            self.end_headers()
            self.wfile.write(metrics.encode('utf-8'))
        elif self.path == '/health':
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        else:
            self.send_response(404)
            self.end_headers()
    
    def collect_metrics(self):
        """Collect all metrics in Prometheus format"""
        lines = []
        
        # Brain checkpoint metrics
        try:
            ckpt_path = Path('/root/UNIFIED_BRAIN/real_env_checkpoint_v3.json')
            if ckpt_path.exists():
                ckpt = json.load(open(ckpt_path))
                
                lines.append('# HELP brain_episode Current episode number')
                lines.append('# TYPE brain_episode counter')
                lines.append(f"brain_episode {ckpt.get('episode', 0)}")
                
                lines.append('# HELP brain_best_reward Best reward achieved')
                lines.append('# TYPE brain_best_reward gauge')
                lines.append(f"brain_best_reward {ckpt.get('best_reward', 0)}")
                
                lines.append('# HELP brain_avg_reward_100 Average reward last 100 episodes')
                lines.append('# TYPE brain_avg_reward_100 gauge')
                lines.append(f"brain_avg_reward_100 {ckpt['stats'].get('avg_reward_last_100', 0)}")
                
                lines.append('# HELP brain_learning_progress Learning progress (0-1)')
                lines.append('# TYPE brain_learning_progress gauge')
                lines.append(f"brain_learning_progress {ckpt['stats'].get('learning_progress', 0)}")
        except Exception:
            pass
        
        # Brain metrics from DB
        try:
            conn = sqlite3.connect('/root/intelligence_system/data/intelligence.db')
            cursor = conn.execute("""
                SELECT coherence, novelty, ia3_signal, num_active_neurons
                FROM brain_metrics
                ORDER BY episode DESC LIMIT 1
            """)
            row = cursor.fetchone()
            if row:
                lines.append('# HELP brain_coherence Current coherence')
                lines.append('# TYPE brain_coherence gauge')
                lines.append(f"brain_coherence {row[0] or 0}")
                
                lines.append('# HELP brain_novelty Current novelty')
                lines.append('# TYPE brain_novelty gauge')
                lines.append(f"brain_novelty {row[1] or 0}")
                
                lines.append('# HELP brain_ia3_signal IA³ signal strength')
                lines.append('# TYPE brain_ia3_signal gauge')
                lines.append(f"brain_ia3_signal {row[2] or 0}")
                
                lines.append('# HELP brain_active_neurons Number of active neurons')
                lines.append('# TYPE brain_active_neurons gauge')
                lines.append(f"brain_active_neurons {row[3] or 0}")
            conn.close()
        except Exception:
            pass
        
        # Surprises
        try:
            conn = sqlite3.connect('/root/emergence_surprises.db')
            cursor = conn.execute('SELECT COUNT(*), MAX(surprise_score) FROM surprises')
            count, max_score = cursor.fetchone()
            conn.close()
            
            lines.append('# HELP surprises_total Total surprises detected')
            lines.append('# TYPE surprises_total counter')
            lines.append(f"surprises_total {count or 0}")
            
            lines.append('# HELP surprises_max_score Maximum surprise score (sigma)')
            lines.append('# TYPE surprises_max_score gauge')
            lines.append(f"surprises_max_score {max_score or 0}")
        except Exception:
            pass
        
        # System connections
        try:
            conn = sqlite3.connect('/root/system_connections.db')
            cursor = conn.execute('SELECT COUNT(*) FROM connections')
            count = cursor.fetchone()[0]
            conn.close()
            
            lines.append('# HELP system_connections_total Total system connections')
            lines.append('# TYPE system_connections_total counter')
            lines.append(f"system_connections_total {count or 0}")
        except Exception:
            pass
        
        # V7 core cycle and bests
        try:
            conn = sqlite3.connect('/root/intelligence_system/data/intelligence.db')
            cur = conn.cursor()
            cur.execute('SELECT MAX(cycle) FROM cycles')
            last_cycle = cur.fetchone()[0] or 0
            lines.append('# HELP v7_cycle Last recorded V7 cycle')
            lines.append('# TYPE v7_cycle gauge')
            lines.append(f"v7_cycle {last_cycle}")

            cur.execute('SELECT MAX(mnist_accuracy), MAX(cartpole_reward) FROM cycles')
            best_mnist, best_cart = cur.fetchone()
            lines.append('# HELP v7_mnist_best Best MNIST accuracy observed')
            lines.append('# TYPE v7_mnist_best gauge')
            lines.append(f"v7_mnist_best {best_mnist or 0}")
            lines.append('# HELP v7_cartpole_best Best CartPole reward observed')
            lines.append('# TYPE v7_cartpole_best gauge')
            lines.append(f"v7_cartpole_best {best_cart or 0}")

            # Darwinacci transfer count from events table
            try:
                cur.execute("SELECT COUNT(*) FROM events WHERE event_type = 'darwinacci_transfer_ingested'")
                transfers = cur.fetchone()[0] or 0
                lines.append('# HELP v7_darwinacci_transfers_applied Total daemon transfers ingested')
                lines.append('# TYPE v7_darwinacci_transfers_applied counter')
                lines.append(f"v7_darwinacci_transfers_applied {transfers}")
            except Exception:
                pass
            conn.close()
        except Exception:
            pass

        # V7 RL knobs
        try:
            # Read a small state file if present (optional), else skip
            import json as _json
            # Expose only safe static approximations: skip if not accessible
            # Instead, rely on last applied transfer as proxy for lr/entropy
            tf = Path('/root/intelligence_system/data/darwin_transfer_latest.json')
            if tf.exists():
                d = _json.loads(tf.read_text())
                g = d.get('genome') or {}
                lr = float(g.get('lr', g.get('learning_rate', 0.0)) or 0.0)
                ent = float((g.get('entropy') or g.get('entropy_coef') or 0.0) or 0.0)
                lines.append('# HELP v7_lr Proxy: last applied lr from transfer')
                lines.append('# TYPE v7_lr gauge')
                lines.append(f"v7_lr {lr}")
                lines.append('# HELP v7_entropy_coef Proxy: last applied entropy from transfer')
                lines.append('# TYPE v7_entropy_coef gauge')
                lines.append(f"v7_entropy_coef {ent}")
        except Exception:
            pass

        # Darwinacci coverage (proxy from transfer file)
        try:
            import json as _json
            tf = Path('/root/intelligence_system/data/darwin_transfer_latest.json')
            if tf.exists():
                d = _json.loads(tf.read_text())
                st = d.get('stats') or {}
                cov = float(st.get('coverage', 0.0))
                lines.append('# HELP darwinacci_coverage Population coverage from latest transfer')
                lines.append('# TYPE darwinacci_coverage gauge')
                lines.append(f"darwinacci_coverage {cov}")
        except Exception:
            pass

        # V7 runtime (aux tasks + knn delta)
        try:
            rt = Path('/root/intelligence_system/data/v7_runtime.json')
            if rt.exists():
                d = json.load(open(rt))
                if 'knn_delta' in d:
                    lines.append('# HELP v7_knn_delta kNN retrieval estimated uplift')
                    lines.append('# TYPE v7_knn_delta gauge')
                    lines.append(f"v7_knn_delta {float(d['knn_delta'])}")
                if 'lander_avg' in d and d['lander_avg'] is not None:
                    lines.append('# HELP v7_lander_avg LunarLander average reward (aux)')
                    lines.append('# TYPE v7_lander_avg gauge')
                    lines.append(f"v7_lander_avg {float(d['lander_avg'])}")
                if 'mcar_avg' in d and d['mcar_avg'] is not None:
                    lines.append('# HELP v7_mcar_avg MountainCar average reward (aux)')
                    lines.append('# TYPE v7_mcar_avg gauge')
                    lines.append(f"v7_mcar_avg {float(d['mcar_avg'])}")
        except Exception:
            pass

        return '\n'.join(lines) + '\n'
    
    def log_message(self, format, *args):
        pass  # Suppress HTTP request logs

def run_server(port=9090):
    """Run Prometheus exporter HTTP server"""
    pid_file = Path("/root/prometheus_exporter.pid")
    pid_file.write_text(str(os.getpid()))
    
    def cleanup(signum, frame):
        print(f"\n🛑 Stopping Prometheus exporter...")
        pid_file.unlink(missing_ok=True)
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, cleanup)
    signal.signal(signal.SIGINT, cleanup)
    
    server = HTTPServer(('127.0.0.1', port), MetricsHandler)
    print(f"📊 Prometheus exporter v{__version__}")
    print(f"   Metrics: http://localhost:{port}/metrics")
    print(f"   Health:  http://localhost:{port}/health")
    print(f"   PID: {os.getpid()}")
    print("")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        pid_file.unlink(missing_ok=True)
        print("✅ Exporter stopped")

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9090
    run_server(port)
