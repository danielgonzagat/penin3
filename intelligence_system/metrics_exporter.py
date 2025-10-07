import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Dict, Any
from intelligence_system.core.database import Database
from intelligence_system.config.settings import DATABASE_PATH

LATEST_METRICS: Dict[str, Any] = {}

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != '/metrics':
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header('Content-Type', 'text/plain; version=0.0.4')
        self.end_headers()
        lines = []
        m = LATEST_METRICS.copy()
        # Basic gauges
        def g(name, value, help_text=""):
            if help_text:
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} gauge")
            lines.append(f"{name} {value}")
        g("v7_best_mnist", m.get('best_mnist', 0.0))
        g("v7_best_cartpole", m.get('best_cartpole', 0.0))
        g("v7_ia3_score", m.get('ia3_score', 0.0))
        g("penin_caos", m.get('caos', 0.0))
        g("penin_linf", m.get('linf', 0.0))
        g("penin_consciousness", m.get('consciousness', 0.0))
        self.wfile.write("\n".join(lines).encode('utf-8'))

class MetricsServer:
    def __init__(self, port: int = 9108):
        self.port = port
        self.thread: threading.Thread | None = None
        self.httpd: HTTPServer | None = None

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.httpd = HTTPServer(('0.0.0.0', self.port), MetricsHandler)
        self.thread = threading.Thread(target=self.httpd.serve_forever, name='MetricsServer')
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        try:
            if self.httpd:
                self.httpd.shutdown()
        except Exception:
            pass

METRICS_SERVER = MetricsServer()


def export_metrics_to_db(metrics: Dict[str, Any]) -> bool:
    """Persist a minimal snapshot of metrics into the DB (best-effort)."""
    try:
        db = Database(DATABASE_PATH)
        cycle = db.get_last_cycle() + 1
        mnist = float(metrics.get('best_mnist', 0.0) or 0.0)
        cart_last = float(metrics.get('cartpole_last', 0.0) or 0.0)
        cart_avg = float(metrics.get('best_cartpole', 0.0) or 0.0)
        return db.save_cycle(cycle, mnist, cart_last, cart_avg)
    except Exception:
        return False
