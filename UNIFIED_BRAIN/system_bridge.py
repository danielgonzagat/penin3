#!/usr/bin/env python3
"""
🌉 SYSTEM BRIDGE - Comunicação ZMQ entre todos sistemas
Conecta: UNIFIED_BRAIN ↔ THE_NEEDLE ↔ Darwin Evolver ↔ AGI Systems
"""

import zmq
import json
import threading
import time
import logging
from typing import Dict, Any, Callable, Optional
from collections import deque
from dataclasses import dataclass, asdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SystemBridge')

@dataclass
class Message:
    source: str
    timestamp: float
    type: str
    data: Dict[str, Any]

class UnifiedSystemBridge:
    """Ponte de comunicação ZMQ entre todos sistemas inteligentes"""
    
    def __init__(self, component_name: str = "unnamed"):
        self.component_name = component_name
        self.context = zmq.Context()
        
        # Sockets
        self.metrics_pub = None
        self.metrics_sub = None
        self.genome_pub = None
        self.genome_sub = None
        self.insights_pub = None
        self.insights_sub = None
        
        # Callbacks
        self.callbacks = {}
        
        # Message history
        self.message_history = deque(maxlen=10000)
        
        # Running flag
        self.running = False
        self.threads = []
        
        logger.info(f"🌉 SystemBridge initialized for: {component_name}")
    
    def initialize_publisher(self, port: int, topic: str):
        """Inicializa socket de publicação"""
        socket = self.context.socket(zmq.PUB)
        socket.bind(f"tcp://127.0.0.1:{port}")
        logger.info(f"📤 Publisher bound: {topic} @ port {port}")
        
        if topic == "metrics":
            self.metrics_pub = socket
        elif topic == "genome":
            self.genome_pub = socket
        elif topic == "insights":
            self.insights_pub = socket
        
        return socket
    
    def initialize_subscriber(self, port: int, topic: str):
        """Inicializa socket de subscrição"""
        socket = self.context.socket(zmq.SUB)
        socket.connect(f"tcp://127.0.0.1:{port}")
        socket.setsockopt_string(zmq.SUBSCRIBE, topic)
        logger.info(f"📥 Subscriber connected: {topic} @ port {port}")
        
        if topic == "metrics":
            self.metrics_sub = socket
        elif topic == "genome":
            self.genome_sub = socket
        elif topic == "insights":
            self.insights_sub = socket
        
        return socket
    
    def publish_metrics(self, metrics: Dict[str, Any]):
        """Publica métricas do sistema"""
        if self.metrics_pub is None:
            return
        
        msg = Message(
            source=self.component_name,
            timestamp=time.time(),
            type="metrics",
            data=metrics
        )
        
        msg_json = json.dumps(asdict(msg))
        self.metrics_pub.send_string(f"metrics {msg_json}")
        self.message_history.append(msg)
    
    def publish_genome(self, genome: Dict[str, Any]):
        """Publica genoma evoluído"""
        if self.genome_pub is None:
            return
        
        msg = Message(
            source=self.component_name,
            timestamp=time.time(),
            type="genome",
            data=genome
        )
        
        msg_json = json.dumps(asdict(msg))
        self.genome_pub.send_string(f"genome {msg_json}")
        self.message_history.append(msg)
    
    def publish_insight(self, insight: Dict[str, Any]):
        """Publica insight/descoberta"""
        if self.insights_pub is None:
            return
        
        msg = Message(
            source=self.component_name,
            timestamp=time.time(),
            type="insight",
            data=insight
        )
        
        msg_json = json.dumps(asdict(msg))
        self.insights_pub.send_string(f"insights {msg_json}")
        self.message_history.append(msg)
    
    def register_callback(self, topic: str, callback: Callable[[Message], None]):
        """Registra callback para mensagens de um tópico"""
        self.callbacks[topic] = callback
        logger.info(f"📌 Callback registered for topic: {topic}")
    
    def _subscriber_loop(self, socket: zmq.Socket, topic: str):
        """Loop de recebimento de mensagens"""
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        
        while self.running:
            try:
                socks = dict(poller.poll(timeout=1000))
                if socket in socks and socks[socket] == zmq.POLLIN:
                    msg_str = socket.recv_string(flags=zmq.NOBLOCK)
                    
                    # Parse mensagem
                    parts = msg_str.split(" ", 1)
                    if len(parts) == 2:
                        topic_received, data_json = parts
                        data = json.loads(data_json)
                        
                        msg = Message(**data)
                        self.message_history.append(msg)
                        
                        # Chamar callback se registrado
                        if topic in self.callbacks:
                            try:
                                self.callbacks[topic](msg)
                            except Exception as e:
                                logger.error(f"Callback error: {e}")
            
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(f"Subscriber loop error: {e}")
                time.sleep(0.1)
    
    def start(self):
        """Inicia threads de subscrição"""
        self.running = True
        
        if self.metrics_sub:
            t = threading.Thread(target=self._subscriber_loop, args=(self.metrics_sub, "metrics"), daemon=True)
            t.start()
            self.threads.append(t)
        
        if self.genome_sub:
            t = threading.Thread(target=self._subscriber_loop, args=(self.genome_sub, "genome"), daemon=True)
            t.start()
            self.threads.append(t)
        
        if self.insights_sub:
            t = threading.Thread(target=self._subscriber_loop, args=(self.insights_sub, "insights"), daemon=True)
            t.start()
            self.threads.append(t)
        
        logger.info(f"✅ SystemBridge started: {len(self.threads)} subscriber threads")
    
    def stop(self):
        """Para threads e fecha sockets"""
        self.running = False
        time.sleep(1)
        
        for socket in [self.metrics_pub, self.metrics_sub, self.genome_pub, self.genome_sub, self.insights_pub, self.insights_sub]:
            if socket:
                socket.close()
        
        self.context.term()
        logger.info("🛑 SystemBridge stopped")
    
    def get_recent_messages(self, count: int = 100, source: Optional[str] = None) -> list:
        """Retorna mensagens recentes"""
        messages = list(self.message_history)[-count:]
        
        if source:
            messages = [m for m in messages if m.source == source]
        
        return messages


# ============================================================================
# CONFIGURAÇÃO PADRÃO DOS COMPONENTES
# ============================================================================

def create_brain_bridge() -> UnifiedSystemBridge:
    """Cria bridge para UNIFIED_BRAIN"""
    bridge = UnifiedSystemBridge("unified_brain")
    
    # Publisher de métricas
    bridge.initialize_publisher(5555, "metrics")
    
    # Subscriber de genomas
    bridge.initialize_subscriber(5556, "genome")
    
    # Subscriber de insights
    bridge.initialize_subscriber(5557, "insights")
    
    return bridge

def create_darwin_bridge() -> UnifiedSystemBridge:
    """Cria bridge para Darwin Evolver"""
    bridge = UnifiedSystemBridge("darwin_evolver")
    
    # Subscriber de métricas
    bridge.initialize_subscriber(5555, "metrics")
    
    # Publisher de genomas
    bridge.initialize_publisher(5556, "genome")
    
    return bridge

def create_needle_bridge() -> UnifiedSystemBridge:
    """Cria bridge para THE_NEEDLE"""
    bridge = UnifiedSystemBridge("the_needle")
    
    # Subscriber de métricas
    bridge.initialize_subscriber(5555, "metrics")
    
    # Publisher de insights
    bridge.initialize_publisher(5557, "insights")
    
    return bridge


if __name__ == "__main__":
    # Teste
    bridge = create_brain_bridge()
    bridge.start()
    
    # Simular publicação
    for i in range(5):
        bridge.publish_metrics({
            'episode': i,
            'reward': 100 + i * 10,
            'loss': 50 - i * 5
        })
        time.sleep(1)
    
    bridge.stop()