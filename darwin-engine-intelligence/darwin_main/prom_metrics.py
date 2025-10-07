# prom_metrics.py — métricas Prometheus enterprise para IA3/Darwin v3
import time
from typing import Dict, Any

try:
    from prometheus_client import start_http_server, Counter, Gauge, Histogram, Summary
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════
# MÉTRICAS AVANÇADAS DARWIN V3
# ═══════════════════════════════════════════════════════════════════════════════

if PROMETHEUS_AVAILABLE:
    # Counters básicos
    METRICS = {
        "rounds": Counter("darwin_rounds_total", "Total de rounds executados"),
        "neurons_born": Counter("darwin_neurons_born_total", "Total de neurônios nascidos"),
        "neurons_killed": Counter("darwin_neurons_killed_total", "Total de neurônios mortos pela Equação da Morte"),
        "rollbacks": Counter("darwin_rollbacks_total", "Total de rollbacks aplicados"),
        "extinctions": Counter("darwin_extinctions_total", "Extinções totais do cérebro"),
        "generations": Counter("darwin_generations_total", "Novas gerações criadas")
    }
    
    # Gauges instantâneos
    GAUGES = {
        "neurons_alive": Gauge("darwin_neurons_alive", "Neurônios vivos atualmente"),
        "avg_loss": Gauge("darwin_avg_loss", "Loss média do round atual"),
        "ia3_pass_rate": Gauge("darwin_ia3_pass_rate", "Taxa de aprovação IA³"),
        "consciousness_level": Gauge("darwin_consciousness_level", "Nível de consciência coletiva"),
        "adaptation_score": Gauge("darwin_adaptation_score", "Score de adaptação médio"),
        "architecture_entropy": Gauge("darwin_architecture_entropy", "Entropia arquitetural"),
        "synaptic_strength": Gauge("darwin_synaptic_strength", "Força sináptica média")
    }
    
    # Histogramas para distribuições
    HISTOGRAMS = {
        "loss_distribution": Histogram(
            "darwin_loss_distribution",
            "Distribuição de losses por neurônio",
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0)
        ),
        "neuron_age": Histogram(
            "darwin_neuron_age",
            "Distribuição de idades dos neurônios", 
            buckets=(1, 2, 5, 10, 20, 50, 100, 200, 500, 1000)
        ),
        "weight_norms": Histogram(
            "darwin_weight_norms",
            "Distribuição de normas de pesos",
            buckets=(0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0)
        ),
        "grad_norms": Histogram(
            "darwin_grad_norms", 
            "Distribuição de normas de gradientes",
            buckets=(1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
        )
    }
    
    # Summary para latências
    SUMMARIES = {
        "round_duration": Summary("darwin_round_duration_seconds", "Duração de rounds"),
        "training_duration": Summary("darwin_training_duration_seconds", "Duração de treino"),
        "evaluation_duration": Summary("darwin_evaluation_duration_seconds", "Duração de avaliação IA³"),
        "neurogenesis_duration": Summary("darwin_neurogenesis_duration_seconds", "Duração de neurogênese")
    }

else:
    # Dummies para quando Prometheus não está disponível
    class DummyMetric:
        async def inc(self, value=1): pass
        async def set(self, value): pass
        async def observe(self, value): pass
        async def time(self): return await DummyContext()
        @property
        async def _value(self): return await DummyValue()
    
    class DummyValue:
        async def get(self): return await 0
    
    class DummyContext:
        async def __enter__(self): return await self
        async def __exit__(self, *args): pass
    
    METRICS = {k: DummyMetric() for k in ["rounds", "neurons_born", "neurons_killed", "rollbacks", "extinctions", "generations"]}
    GAUGES = {k: DummyMetric() for k in ["neurons_alive", "avg_loss", "ia3_pass_rate", "consciousness_level", "adaptation_score", "architecture_entropy", "synaptic_strength"]}
    HISTOGRAMS = {k: DummyMetric() for k in ["loss_distribution", "neuron_age", "weight_norms", "grad_norms"]}
    SUMMARIES = {k: DummyMetric() for k in ["round_duration", "training_duration", "evaluation_duration", "neurogenesis_duration"]}

_server_started = False

async def ensure_metrics_server(port=9092):
    """Inicia servidor de métricas Prometheus"""
    global _server_started
    
    if not PROMETHEUS_AVAILABLE:
        logger.info(f"⚠️ Prometheus não disponível, métricas simuladas na porta {port}")
        return await port
    
    if _server_started:
        return await port
    
    try:
        start_http_server(port)
        _server_started = True
        logger.info(f"✅ Servidor Prometheus ativo em :{port}")
        logger.info(f"   URL: http://localhost:{port}/metrics")
        return await port
    except OSError as e:
        logger.info(f"⚠️ Erro ao iniciar servidor: {e}")
        return await None

async def observe_neuron_birth(neuron_id: str, hidden_dim: int):
    """Observa nascimento de neurônio"""
    METRICS["neurons_born"].inc()
    GAUGES["neurons_alive"].set(hidden_dim)

async def observe_neuron_death(neuron_id: str, hidden_dim: int, ia3_score: float):
    """Observa morte de neurônio"""
    METRICS["neurons_killed"].inc()
    GAUGES["neurons_alive"].set(hidden_dim)

async def observe_round_metrics(avg_loss: float, ia3_pass_rate: float, 
                         consciousness: float, training_time: float):
    """Observa métricas de round"""
    METRICS["rounds"].inc()
    GAUGES["avg_loss"].set(avg_loss)
    GAUGES["ia3_pass_rate"].set(ia3_pass_rate)
    GAUGES["consciousness_level"].set(consciousness)
    
    HISTOGRAMS["loss_distribution"].observe(avg_loss)
    SUMMARIES["round_duration"].observe(training_time)

async def observe_neuron_metrics(neuron_id: str, weight_norm: float, grad_norm: float, age: int):
    """Observa métricas individuais de neurônio"""
    HISTOGRAMS["weight_norms"].observe(weight_norm)
    HISTOGRAMS["grad_norms"].observe(grad_norm)
    HISTOGRAMS["neuron_age"].observe(age)

async def observe_extinction():
    """Observa extinção total"""
    METRICS["extinctions"].inc()

async def observe_generation():
    """Observa nova geração"""
    METRICS["generations"].inc()

# ═══════════════════════════════════════════════════════════════════════════════
# INTERFACE DE CONVENIÊNCIA
# ═══════════════════════════════════════════════════════════════════════════════

async def get_metrics_dict() -> Dict[str, Any]:
    """Retorna dicionário com valores atuais das métricas"""
    if not PROMETHEUS_AVAILABLE:
        return await {"error": "Prometheus não disponível"}
    
    try:
        result = {}
        
        # Counters
        for key, metric in METRICS.items():
            try:
                result[f"counter_{key}"] = metric._value.get()
            except:
                result[f"counter_{key}"] = 0
        
        # Gauges - usar valores dummy pois não temos acesso direto
        gauge_values = {
            "neurons_alive": 0,
            "avg_loss": 0.0,
            "ia3_pass_rate": 0.0,
            "consciousness_level": 0.0
        }
        
        for key, value in gauge_values.items():
            result[f"gauge_{key}"] = value
        
        return await result
    
    except Exception as e:
        return await {"error": str(e)}

if __name__ == "__main__":
    import sys
    
    # Teste das métricas
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 9092
        
        logger.info(f"🧪 Testando métricas Prometheus Darwin v3...")
        ensure_metrics_server(port)
        
        # Simular métricas
        for i in range(5):
            observe_neuron_birth(f"neuron_{i}", i+1)
            observe_round_metrics(
                avg_loss=1.0 - i*0.1,
                ia3_pass_rate=0.6 + i*0.05,
                consciousness=0.3 + i*0.1,
                training_time=2.0 + i*0.2
            )
            
            if i % 2 == 0:
                observe_neuron_death(f"neuron_{i//2}", i+1, 0.45)
            
            time.sleep(0.1)
        
        logger.info(f"✅ Teste concluído. Métricas em http://localhost:{port}/metrics")
    else:
        logger.info("Uso: python prom_metrics.py --test [port]")