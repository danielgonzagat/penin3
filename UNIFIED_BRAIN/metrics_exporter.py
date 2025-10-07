import json, time, pathlib
from typing import Any, Dict

# Minimal no-op metrics server placeholders to satisfy imports in core
class _NoopMetricsServer:
    def __init__(self, port: int = 0) -> None:
        self.port = int(port)
        self.running = False
    def start(self) -> None:
        self.running = True
    def stop(self) -> None:
        self.running = False

class _NoopMetrics:
    def observe(self, name: str, value: float, labels: Dict[str, Any] | None = None) -> None:
        pass

METRICS_SERVER = _NoopMetricsServer
Metrics = _NoopMetrics

def write_brain_status(fitness: float, path: str = '/root/UNIFIED_BRAIN/SYSTEM_STATUS.json') -> None:
    """Write unified brain status JSON for external consumers."""
    payload = {
        'ts': int(time.time()),
        'brain_fitness': float(fitness),
    }
    p = pathlib.Path(path)
    p.write_text(json.dumps(payload))
