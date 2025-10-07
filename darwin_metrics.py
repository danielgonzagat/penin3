#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Darwin Metrics Exporter - Versão fusionada
Expõe métricas do DARWIN para Prometheus com fallback gracioso
"""

from typing import Optional
import time
import json
import os
from datetime import datetime

class _NoopMetrics:
    """Fallback quando prometheus_client não está disponível"""
    async def start(self, port: int): 
        logger.info(f"📊 Métricas NO-OP (prometheus_client não disponível) - porta {port}")
    async def inc_death(self, reason: str = "unknown"): pass
    async def inc_birth(self): pass
    async def set_survivors_active(self, n: int): pass
    async def set_mortality_last_run(self, rate: float): pass
    async def inc_gate_denies(self, reason: str): pass
    async def set_consecutive_i_failures(self, n: int): pass
    async def set_canary_progress(self, progress: float): pass
    async def observe_delta_linf(self, value: float): pass
    async def observe_survival_time(self, seconds: float): pass

async def _build_prom_metrics():
    """Constrói métricas reais do Prometheus"""
    from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, start_http_server

    class DarwinMetrics:
        async def __init__(self):
            self.registry = CollectorRegistry()
            
            # Contadores
            self.deaths_total = Counter(
                "darwin_deaths_total", 
                "Total de mortes decididas pelo DARWIN",
                ["reason"], 
                registry=self.registry
            )
            self.births_total = Counter(
                "darwin_births_total", 
                "Total de nascimentos (a cada X mortes)",
                registry=self.registry
            )
            self.gate_denies_total = Counter(
                "darwin_gate_denies_total", 
                "Negativas de gate por motivo",
                ["reason"], 
                registry=self.registry
            )
            
            # Gauges
            self.mortality_last_run = Gauge(
                "darwin_mortality_last_run",
                "Taxa de mortalidade na última rodada",
                registry=self.registry
            )
            self.survivors_active = Gauge(
                "darwin_survivors_active", 
                "Agentes ativos após a rodada",
                registry=self.registry
            )
            self.consecutive_i_failures = Gauge(
                "darwin_consecutive_i_failures",
                "Falhas consecutivas de I < 0.60 (kill switch)",
                registry=self.registry
            )
            self.canary_progress = Gauge(
                "darwin_canary_progress",
                "Progresso do canário (0.0 a 1.0)",
                registry=self.registry
            )
            self.time_since_last_birth_seconds = Gauge(
                "darwin_time_since_last_birth_seconds",
                "Tempo desde último nascimento em segundos",
                registry=self.registry
            )
            
            # Histogramas
            self.delta_linf_distribution = Histogram(
                "darwin_delta_linf_distribution",
                "Distribuição de valores delta L-infinity",
                buckets=(-0.1, -0.05, -0.01, 0, 0.01, 0.05, 0.1, 0.2, 0.5),
                registry=self.registry
            )
            self.survival_time_seconds = Histogram(
                "darwin_survival_time_seconds",
                "Tempo que agentes sobrevivem antes da morte",
                buckets=(60, 300, 600, 1800, 3600, 7200, 14400),
                registry=self.registry
            )
            
            self._start_http_server = start_http_server
            self._last_birth_time = time.time()

        async def start(self, port: int):
            """Inicia servidor HTTP de métricas"""
            try:
                self._start_http_server(port, registry=self.registry)
                logger.info(f"✅ Darwin Metrics Server rodando em :{port}")
            except Exception as e:
                logger.info(f"⚠️ Não foi possível iniciar servidor de métricas: {e}")

        async def inc_death(self, reason: str = "unknown"):
            self.deaths_total.labels(reason=reason).inc()

        async def inc_birth(self):
            self.births_total.inc()
            self._last_birth_time = time.time()

        async def set_survivors_active(self, n: int):
            self.survivors_active.set(float(n))

        async def set_mortality_last_run(self, rate: float):
            self.mortality_last_run.set(float(rate))

        async def inc_gate_denies(self, reason: str):
            self.gate_denies_total.labels(reason=reason).inc()
            
        async def set_consecutive_i_failures(self, n: int):
            self.consecutive_i_failures.set(float(n))
            
        async def set_canary_progress(self, progress: float):
            self.canary_progress.set(progress)
            
        async def observe_delta_linf(self, value: float):
            if value is not None:
                self.delta_linf_distribution.observe(value)
                
        async def observe_survival_time(self, seconds: float):
            if seconds > 0:
                self.survival_time_seconds.observe(seconds)
                
        async def update_time_since_birth(self):
            """Atualiza tempo desde último nascimento"""
            elapsed = time.time() - self._last_birth_time
            self.time_since_last_birth_seconds.set(elapsed)

    return await DarwinMetrics()

async def build_metrics():
    """Factory que retorna métricas reais ou NO-OP"""
    try:
        return await _build_prom_metrics()
    except ImportError:
        logger.info("⚠️ prometheus_client não disponível - usando métricas NO-OP")
        return await _NoopMetrics()
    except Exception as e:
        logger.info(f"⚠️ Erro ao criar métricas: {e} - usando NO-OP")
        return await _NoopMetrics()

# Exportador standalone opcional
async def export_static_metrics(worm_path="/root/darwin_worm.log"):
    """Exporta métricas estáticas para arquivo (alternativa ao servidor HTTP)"""
    metrics_data = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "deaths": 0,
        "births": 0,
        "survivors": 0,
        "mortality_rate": 0.0,
        "death_reasons": {}
    }
    
    if os.path.exists(worm_path):
        with open(worm_path, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    event = entry.get("event")
                    if event == "death":
                        metrics_data["deaths"] += 1
                        reasons = entry.get("deny_reasons", ["unknown"])
                        for r in reasons:
                            metrics_data["death_reasons"][r] = metrics_data["death_reasons"].get(r, 0) + 1
                    elif event == "survive":
                        metrics_data["survivors"] += 1
                    elif event == "birth_from_deaths":
                        metrics_data["births"] += 1
                except:
                    continue
    
    total = metrics_data["deaths"] + metrics_data["survivors"]
    if total > 0:
        metrics_data["mortality_rate"] = metrics_data["deaths"] / total
    
    # Salvar em formato Prometheus text
    prom_lines = []
    prom_lines.append(f'# HELP darwin_deaths_total Total deaths')
    prom_lines.append(f'# TYPE darwin_deaths_total counter')
    prom_lines.append(f'darwin_deaths_total {metrics_data["deaths"]}')
    
    prom_lines.append(f'# HELP darwin_births_total Total births')
    prom_lines.append(f'# TYPE darwin_births_total counter')
    prom_lines.append(f'darwin_births_total {metrics_data["births"]}')
    
    prom_lines.append(f'# HELP darwin_survivors_active Active survivors')
    prom_lines.append(f'# TYPE darwin_survivors_active gauge')
    prom_lines.append(f'darwin_survivors_active {metrics_data["survivors"]}')
    
    prom_lines.append(f'# HELP darwin_mortality_last_run Mortality rate')
    prom_lines.append(f'# TYPE darwin_mortality_last_run gauge')
    prom_lines.append(f'darwin_mortality_last_run {metrics_data["mortality_rate"]}')
    
    with open("/root/darwin_metrics.prom", 'w') as f:
        f.write('\n'.join(prom_lines))
    
    return await metrics_data

if __name__ == "__main__":
    # Teste ou execução standalone
    import sys
    if "--static" in sys.argv:
        data = export_static_metrics()
        logger.info(json.dumps(data, indent=2))
        logger.info("\n📊 Métricas exportadas para /root/darwin_metrics.prom")
    else:
        logger.info("Use --static para exportar métricas estáticas")
        logger.info("Ou importe este módulo no darwin_runner.py")