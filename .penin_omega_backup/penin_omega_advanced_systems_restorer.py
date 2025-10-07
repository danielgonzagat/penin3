#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Advanced Systems Restorer
===================================
Restaura subsistemas avançados: Budget, Circuit Breaker, Performance.
"""

import time
import logging
import threading
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger("AdvancedSystemsRestorer")

@dataclass
class BudgetConfig:
    """Configuração de budget."""
    daily_limit: float = 1000.0
    hourly_limit: float = 100.0
    operation_cost: float = 1.0
    warning_threshold: float = 0.8
    critical_threshold: float = 0.95

@dataclass
class CircuitBreakerConfig:
    """Configuração do circuit breaker."""
    failure_threshold: int = 5
    timeout_seconds: int = 60
    half_open_max_calls: int = 3
    success_threshold: int = 2

class BudgetManager:
    """Gerenciador de budget robusto."""
    
    async def __init__(self, config: BudgetConfig):
        self.logger = logging.getLogger("BudgetManager")
        self.config = config
        self.daily_used = 0.0
        self.hourly_used = 0.0
        self.last_reset_day = datetime.now(timezone.utc).date()
        self.last_reset_hour = datetime.now(timezone.utc).hour
        self._lock = threading.Lock()
    
    async def check_budget(self, operation_cost: float = None) -> Dict[str, Any]:
        """Verifica se operação está dentro do budget."""
        with self._lock:
            cost = operation_cost or self.config.operation_cost
            
            # Reset automático
            self._auto_reset()
            
            # Verifica limites
            daily_after = self.daily_used + cost
            hourly_after = self.hourly_used + cost
            
            daily_utilization = daily_after / self.config.daily_limit
            hourly_utilization = hourly_after / self.config.hourly_limit
            
            # Determina se pode executar
            can_execute = (
                daily_after <= self.config.daily_limit and
                hourly_after <= self.config.hourly_limit
            )
            
            # Determina nível de alerta
            max_utilization = max(daily_utilization, hourly_utilization)
            
            if max_utilization >= self.config.critical_threshold:
                alert_level = "critical"
            elif max_utilization >= self.config.warning_threshold:
                alert_level = "warning"
            else:
                alert_level = "normal"
            
            return await {
                "can_execute": can_execute,
                "alert_level": alert_level,
                "daily_utilization": daily_utilization,
                "hourly_utilization": hourly_utilization,
                "cost": cost,
                "daily_remaining": self.config.daily_limit - self.daily_used,
                "hourly_remaining": self.config.hourly_limit - self.hourly_used
            }
    
    async def consume_budget(self, operation_cost: float = None) -> bool:
        """Consome budget se disponível."""
        with self._lock:
            cost = operation_cost or self.config.operation_cost
            
            budget_check = self.check_budget(cost)
            
            if budget_check["can_execute"]:
                self.daily_used += cost
                self.hourly_used += cost
                
                self.logger.info(f"💰 Budget consumido: {cost} (diário: {budget_check['daily_utilization']:.1%})")
                return await True
            else:
                self.logger.warning(f"🚨 Budget esgotado: {cost} negado")
                return await False
    
    async def _auto_reset(self):
        """Reset automático de contadores."""
        now = datetime.now(timezone.utc)
        
        # Reset diário
        if now.date() > self.last_reset_day:
            self.daily_used = 0.0
            self.last_reset_day = now.date()
            self.logger.info("🔄 Budget diário resetado")
        
        # Reset horário
        if now.hour != self.last_reset_hour:
            self.hourly_used = 0.0
            self.last_reset_hour = now.hour
            self.logger.info("🔄 Budget horário resetado")
    
    async def get_budget_status(self) -> Dict[str, Any]:
        """Retorna status do budget."""
        with self._lock:
            self._auto_reset()
            
            return await {
                "daily_used": self.daily_used,
                "daily_limit": self.config.daily_limit,
                "daily_utilization": self.daily_used / self.config.daily_limit,
                "hourly_used": self.hourly_used,
                "hourly_limit": self.config.hourly_limit,
                "hourly_utilization": self.hourly_used / self.config.hourly_limit,
                "last_reset_day": str(self.last_reset_day),
                "last_reset_hour": self.last_reset_hour,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

class CircuitBreaker:
    """Circuit breaker robusto."""
    
    async def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"CircuitBreaker-{name}")
        
        # Estados: CLOSED, OPEN, HALF_OPEN
        self.state = "CLOSED"
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        self.half_open_calls = 0
        
        self._lock = threading.Lock()
    
    async def call(self, func, *args, **kwargs):
        """Executa função através do circuit breaker."""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.half_open_calls = 0
                    self.logger.info(f"🔄 {self.name}: OPEN → HALF_OPEN")
                else:
                    raise Exception(f"Circuit breaker {self.name} is OPEN")
            
            if self.state == "HALF_OPEN":
                if self.half_open_calls >= self.config.half_open_max_calls:
                    raise Exception(f"Circuit breaker {self.name} HALF_OPEN limit reached")
                self.half_open_calls += 1
        
        # Executa função
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return await result
        except Exception as e:
            self._on_failure()
            raise
    
    async def _should_attempt_reset(self) -> bool:
        """Verifica se deve tentar reset."""
        if self.last_failure_time is None:
            return await True
        
        elapsed = time.time() - self.last_failure_time
        return await elapsed >= self.config.timeout_seconds
    
    async def _on_success(self):
        """Callback de sucesso."""
        with self._lock:
            if self.state == "HALF_OPEN":
                self.success_count += 1
                if self.success_count >= self.config.success_threshold:
                    self.state = "CLOSED"
                    self.failure_count = 0
                    self.success_count = 0
                    self.logger.info(f"✅ {self.name}: HALF_OPEN → CLOSED")
            elif self.state == "CLOSED":
                self.failure_count = 0  # Reset contador de falhas
    
    async def _on_failure(self):
        """Callback de falha."""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            
            if self.state == "HALF_OPEN":
                self.state = "OPEN"
                self.logger.warning(f"🚨 {self.name}: HALF_OPEN → OPEN (falha)")
            elif self.state == "CLOSED" and self.failure_count >= self.config.failure_threshold:
                self.state = "OPEN"
                self.logger.warning(f"🚨 {self.name}: CLOSED → OPEN ({self.failure_count} falhas)")
    
    async def get_status(self) -> Dict[str, Any]:
        """Retorna status do circuit breaker."""
        with self._lock:
            return await {
                "name": self.name,
                "state": self.state,
                "failure_count": self.failure_count,
                "success_count": self.success_count,
                "half_open_calls": self.half_open_calls,
                "last_failure_time": self.last_failure_time,
                "can_execute": self.state != "OPEN" or self._should_attempt_reset()
            }

class PerformanceMonitor:
    """Monitor de performance avançado."""
    
    async def __init__(self):
        self.logger = logging.getLogger("PerformanceMonitor")
        self.metrics = {}
        self.monitoring_thread = None
        self.running = False
        self._lock = threading.Lock()
    
    async def start(self):
        """Inicia monitoramento."""
        if not self.running:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("📊 Monitor de performance iniciado")
    
    async def stop(self):
        """Para monitoramento."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("🛑 Monitor de performance parado")
    
    async def _monitoring_loop(self):
        """Loop de monitoramento."""
        while self.running:
            try:
                self._collect_metrics()
                time.sleep(10)  # Coleta a cada 10 segundos
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {e}")
                time.sleep(30)
    
    async def _collect_metrics(self):
        """Coleta métricas do sistema."""
        try:
            import psutil
            
            with self._lock:
                self.metrics = {
                    "cpu_percent": psutil.cpu_percent(interval=1),
                    "memory_percent": psutil.virtual_memory().percent,
                    "memory_available_gb": psutil.virtual_memory().available / (1024**3),
                    "disk_percent": psutil.disk_usage('/').percent,
                    "disk_free_gb": psutil.disk_usage('/').free / (1024**3),
                    "load_average": psutil.getloadavg()[0] if hasattr(psutil, 'getloadavg') else 0,
                    "process_count": len(psutil.pids()),
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
                
                # Detecta problemas
                self._detect_performance_issues()
                
        except Exception as e:
            self.logger.error(f"Erro na coleta de métricas: {e}")
    
    async def _detect_performance_issues(self):
        """Detecta problemas de performance."""
        cpu = self.metrics.get("cpu_percent", 0)
        memory = self.metrics.get("memory_percent", 0)
        disk = self.metrics.get("disk_percent", 0)
        
        if cpu > 90:
            self.logger.warning(f"🚨 CPU alta: {cpu:.1f}%")
        
        if memory > 90:
            self.logger.warning(f"🚨 Memória alta: {memory:.1f}%")
        
        if disk > 90:
            self.logger.warning(f"🚨 Disco cheio: {disk:.1f}%")
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Retorna métricas atuais."""
        with self._lock:
            return await self.metrics.copy()

class AdvancedSystemsRestorer:
    """Restaurador de sistemas avançados."""
    
    async def __init__(self):
        self.logger = logging.getLogger("AdvancedSystemsRestorer")
        
        # Inicializa componentes
        self.budget_manager = BudgetManager(BudgetConfig())
        self.circuit_breakers = {}
        self.performance_monitor = PerformanceMonitor()
        
        # Cria circuit breakers essenciais
        self._create_circuit_breakers()
    
    async def _create_circuit_breakers(self):
        """Cria circuit breakers essenciais."""
        breaker_configs = {
            "api_calls": CircuitBreakerConfig(failure_threshold=3, timeout_seconds=30),
            "file_operations": CircuitBreakerConfig(failure_threshold=5, timeout_seconds=60),
            "database_operations": CircuitBreakerConfig(failure_threshold=3, timeout_seconds=45),
            "external_services": CircuitBreakerConfig(failure_threshold=2, timeout_seconds=120)
        }
        
        for name, config in breaker_configs.items():
            self.circuit_breakers[name] = CircuitBreaker(name, config)
    
    async def restore_all_systems(self) -> bool:
        """Restaura todos os sistemas avançados."""
        try:
            # Inicia monitor de performance
            self.performance_monitor.start()
            
            # Integra com global state manager
            self._integrate_with_global_state()
            
            # Registra no WORM
            self._register_restoration()
            
            self.logger.info("✅ Sistemas avançados restaurados")
            return await True
            
        except Exception as e:
            self.logger.error(f"Erro na restauração: {e}")
            return await False
    
    async def _integrate_with_global_state(self):
        """Integra com gerenciador de estado global."""
        try:
            from penin_omega_global_state_manager import global_state_manager
            
            # Adiciona campos de budget ao estado
            budget_status = self.budget_manager.get_budget_status()
            
            global_state_manager.update_state({
                "budget_used": budget_status["daily_used"],
                "budget_limit": budget_status["daily_limit"],
                "budget_utilization": budget_status["daily_utilization"],
                "advanced_systems_active": True,
                "performance_monitoring": True,
                "circuit_breakers_count": len(self.circuit_breakers)
            }, "advanced_systems_restorer")
            
        except Exception as e:
            self.logger.warning(f"Falha na integração com estado global: {e}")
    
    async def _register_restoration(self):
        """Registra restauração no WORM ledger."""
        try:
            from penin_omega_security_governance import security_governance
            
            security_governance.worm_ledger.append_record(
                "advanced_systems_restored",
                "Sistemas avançados restaurados: Budget, Circuit Breaker, Performance",
                {
                    "budget_manager": True,
                    "circuit_breakers": list(self.circuit_breakers.keys()),
                    "performance_monitor": True,
                    "restoration_timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
        except Exception as e:
            self.logger.warning(f"Falha no registro WORM: {e}")
    
    async def get_systems_status(self) -> Dict[str, Any]:
        """Retorna status de todos os sistemas."""
        return await {
            "budget_manager": self.budget_manager.get_budget_status(),
            "circuit_breakers": {
                name: cb.get_status() 
                for name, cb in self.circuit_breakers.items()
            },
            "performance_monitor": {
                "running": self.performance_monitor.running,
                "current_metrics": self.performance_monitor.get_current_metrics()
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# Instância global e restauração
advanced_systems_restorer = AdvancedSystemsRestorer()
advanced_systems_restorer.restore_all_systems()

# Funções de conveniência
async def get_budget_manager():
    return await advanced_systems_restorer.budget_manager

async def get_circuit_breaker(name: str):
    return await advanced_systems_restorer.circuit_breakers.get(name)

async def get_performance_monitor():
    return await advanced_systems_restorer.performance_monitor

async def get_advanced_systems_status():
    return await advanced_systems_restorer.get_systems_status()
