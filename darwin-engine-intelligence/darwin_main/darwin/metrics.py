from prometheus_client import Gauge, Counter, Summary, start_http_server
import threading

_METRICS_PORT = 9091

# GAUGES (valores instantâneos)
g_neurons_alive   = Gauge("darwin_neurons_alive", "Neurônios vivos no ciclo")
g_cycle_index     = Gauge("darwin_cycle_index", "Índice do ciclo DARWIN")
g_population_loss = Gauge("darwin_population_loss", "Loss médio da população")
g_births          = Gauge("darwin_births", "Nascimentos acumulados")
g_deaths          = Gauge("darwin_deaths", "Mortes acumuladas")
g_stability       = Gauge("darwin_stability", "Proxy de estabilidade (Lyapunov-ish)")
g_oci             = Gauge("darwin_oci", "Proxy de fechamento OCI")
g_p_convergence   = Gauge("darwin_P", "Proxy P (∞(E+N−iN) simplificado)")
g_consciousness   = Gauge("darwin_consciousness_avg", "Consciência média da população")
g_ia3_score_avg   = Gauge("darwin_ia3_score_avg", "Score IA³ médio")

# COUNTERS (acumulativos)
c_cycles          = Counter("darwin_cycles_total", "Total de ciclos executados")
c_death_events    = Counter("darwin_death_events_total", "Eventos de morte (neurônios removidos)")
c_birth_events    = Counter("darwin_birth_events_total", "Eventos de nascimento (neurônios adicionados)")
c_worm_writes     = Counter("darwin_worm_writes_total", "Registros WORM gravados")
c_mutations       = Counter("darwin_mutations_total", "Auto-mutações executadas")
c_survivals       = Counter("darwin_survivals_total", "Neurônios que sobreviveram")

async def serve_metrics_once():
    """Inicia servidor de métricas Prometheus (thread daemon)"""
    try:
        start_http_server(_METRICS_PORT)
        logger.info(f"✅ Métricas Prometheus ativas em :{_METRICS_PORT}")
        logger.info(f"   URL: http://localhost:{_METRICS_PORT}/metrics")
    except OSError as e:
        logger.info(f"⚠️ Erro ao iniciar métricas: {e}")

_metrics_server_started = False

async def ensure_metrics():
    """Garante que servidor de métricas está ativo"""
    global _metrics_server_started
    if not _metrics_server_started:
        threading.Thread(target=serve_metrics_once, daemon=True).start()
        _metrics_server_started = True
        import time
        time.sleep(0.5)  # Dar tempo para servidor subir