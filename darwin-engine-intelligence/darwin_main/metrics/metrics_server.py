# /root/darwin/metrics/metrics_server.py
# Servidor de métricas Prometheus para Darwin v4 - Neurogênese Viva

import time
import argparse
import json
from pathlib import Path
from typing import Dict, Any

try:
    from prometheus_client import start_http_server, Gauge, Counter, Histogram
    PROMETHEUS_AVAILABLE = True
except ImportError:
    logger.info("⚠️ prometheus_client não disponível")
    PROMETHEUS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS DARWIN V4
# ═══════════════════════════════════════════════════════════════════════════════

if PROMETHEUS_AVAILABLE:
    # Counters de eventos
    g_survivals = Counter("darwin_survivals_total", "Total de sobrevivências (E=1)")
    g_deaths = Counter("darwin_deaths_total", "Total de mortes (E=0)")
    g_births = Counter("darwin_births_total", "Total de nascimentos (spawn)")
    g_rounds = Counter("darwin_rounds_total", "Total de rounds executados")
    g_mutations = Counter("darwin_mutations_total", "Total de auto-mutações")
    
    # Gauges de estado atual
    g_death_counter = Gauge("darwin_death_counter", "Contador de mortes desde último nascimento")
    g_val_loss = Gauge("darwin_neuron_val_loss", "Val loss do neurônio corrente")
    g_novelty = Gauge("darwin_neuron_novelty", "Novidade medida no passo")
    g_consciousness = Gauge("darwin_consciousness_level", "Nível de consciência neural")
    g_ia3_score = Gauge("darwin_ia3_score", "Score IA³ composto")
    g_neurons_alive = Gauge("darwin_neurons_alive", "Neurônios vivos atualmente")
    
    # Histogramas de distribuições
    h_val_loss = Histogram(
        "darwin_val_loss_distribution",
        "Distribuição de validation losses",
        buckets=(0.001, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0)
    )
    
    h_novelty = Histogram(
        "darwin_novelty_distribution", 
        "Distribuição de scores de novidade",
        buckets=(0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5)
    )
    
    h_consciousness = Histogram(
        "darwin_consciousness_distribution",
        "Distribuição de níveis de consciência",
        buckets=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    )

else:
    # Dummies se Prometheus não disponível
    class DummyMetric:
        async def inc(self, value=1): pass
        async def set(self, value): pass
        async def observe(self, value): pass
        async def dec(self, value=1): pass
    
    g_survivals = g_deaths = g_births = g_rounds = g_mutations = DummyMetric()
    g_death_counter = g_val_loss = g_novelty = g_consciousness = DummyMetric()
    g_ia3_score = g_neurons_alive = DummyMetric()
    h_val_loss = h_novelty = h_consciousness = DummyMetric()

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE PÚBLICA PARA ATUALIZAÇÕES
# ═══════════════════════════════════════════════════════════════════════════════

async def update_survival(neuron_id: str, val_loss: float, novelty: float, consciousness: float, ia3_score: float):
    """Atualiza métricas para sobrevivência"""
    g_survivals.inc()
    g_val_loss.set(val_loss)
    g_novelty.set(novelty)
    g_consciousness.set(consciousness)
    g_ia3_score.set(ia3_score)
    
    # Histogramas
    h_val_loss.observe(val_loss)
    h_novelty.observe(novelty)
    h_consciousness.observe(consciousness)

async def update_death(neuron_id: str, reason: str):
    """Atualiza métricas para morte"""
    g_deaths.inc()
    g_death_counter.inc()

async def update_birth(neuron_id: str, reason: str):
    """Atualiza métricas para nascimento"""
    g_births.inc()
    if reason == "death_counter_reset":
        g_death_counter.set(0)  # Reset contador

async def update_round_start(neurons_alive: int):
    """Atualiza métricas no início de round"""
    g_rounds.inc()
    g_neurons_alive.set(neurons_alive)

async def update_mutations(count: int):
    """Atualiza contador de mutações"""
    g_mutations.inc(count)

async def get_current_metrics() -> Dict[str, float]:
    """Retorna snapshot das métricas atuais"""
    if not PROMETHEUS_AVAILABLE:
        return await {"error": "Prometheus não disponível"}
    
    # Simular valores (em produção, usar _value.get())
    return await {
        "survivals_total": 0,
        "deaths_total": 0,
        "births_total": 0,
        "rounds_total": 0,
        "death_counter": 0,
        "current_val_loss": 0.0,
        "current_novelty": 0.0,
        "current_consciousness": 0.0,
        "current_ia3_score": 0.0,
        "neurons_alive": 0
    }

# ═══════════════════════════════════════════════════════════════════════════════
# SERVIDOR PRINCIPAL
# ═══════════════════════════════════════════════════════════════════════════════

async def serve_forever(port=9092, state_file="/root/darwin/logs/state.json"):
    """
    Inicia servidor de métricas e monitora estado Darwin
    """
    if not PROMETHEUS_AVAILABLE:
        logger.info(f"⚠️ Prometheus não disponível, servidor mock na porta {port}")
        while True:
            time.sleep(60)
    
    logger.info(f"✅ Iniciando servidor de métricas Darwin v4 na porta {port}")
    logger.info(f"   URL: http://localhost:{port}/metrics")
    logger.info(f"   Monitorando: {state_file}")
    
    # Iniciar servidor HTTP
    start_http_server(port)
    
    # Loop de monitoramento
    last_state = {}
    
    try:
        while True:
            # Ler estado Darwin se disponível
            if Path(state_file).exists():
                try:
                    with open(state_file, 'r') as f:
                        current_state = json.load(f)
                    
                    # Atualizar métricas baseadas no estado
                    if current_state != last_state:
                        deaths = current_state.get("deaths", 0)
                        births = current_state.get("births", 0)
                        survivals = current_state.get("survivals", 0)
                        
                        # Atualizar counters (simplificado)
                        g_death_counter.set(deaths % 10)  # Contador cíclico
                        
                        last_state = current_state
                        logger.info(f"📊 Estado atualizado: {survivals} vivos, {deaths} mortes, {births} nascimentos")
                
                except Exception as e:
                    logger.info(f"⚠️ Erro ao ler estado: {e}")
            
            time.sleep(5)  # Verificar estado a cada 5 segundos
    
    except KeyboardInterrupt:
        logger.info(f"\n🛑 Servidor de métricas interrompido")

async def test_metrics(port=9092):
    """Testa o servidor de métricas com dados simulados"""
    logger.info(f"🧪 Testando métricas Darwin v4 na porta {port}...")
    
    if not PROMETHEUS_AVAILABLE:
        logger.info("⚠️ Prometheus não disponível para teste")
        return
    
    start_http_server(port)
    
    # Simular eventos
    for i in range(10):
        if i % 3 == 0:
            update_survival(
                neuron_id=f"test_{i}",
                val_loss=1.0 - i*0.1,
                novelty=0.05 + i*0.01,
                consciousness=0.3 + i*0.05,
                ia3_score=0.6 + i*0.03
            )
        else:
            update_death(f"test_{i}", "failed_ia3")
        
        if i % 5 == 0:
            update_birth(f"newborn_{i}", "death_counter_reset")
        
        update_mutations(random.randint(1, 5))
        time.sleep(0.1)
    
    logger.info(f"✅ Teste concluído. Métricas em http://localhost:{port}/metrics")
    
    # Aguardar
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info(f"\n🛑 Teste interrompido")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="🧬 Servidor de Métricas Darwin v4")
    parser.add_argument("--port", type=int, default=9092, help="Porta do servidor")
    parser.add_argument("--state-file", default="/root/darwin/logs/state.json", 
                       help="Arquivo de estado Darwin")
    parser.add_argument("--test", action="store_true", help="Executar teste")
    
    args = parser.parse_args()
    
    if args.test:
        import random
        test_metrics(args.port)
    else:
        serve_forever(args.port, args.state_file)