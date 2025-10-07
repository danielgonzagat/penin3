#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Health Monitor - Monitoramento de Saúde do Sistema
===========================================================
Sistema robusto de monitoramento de saúde com métricas reais.
"""

import psutil
import threading
import time
import json
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"

@dataclass
class HealthMetric:
    """Representa uma métrica de saúde."""
    name: str
    value: float
    unit: str
    status: HealthStatus
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime
    description: str = ""

@dataclass
class SystemHealth:
    """Representa saúde geral do sistema."""
    overall_status: HealthStatus
    metrics: Dict[str, HealthMetric]
    timestamp: datetime
    uptime_seconds: float
    issues: List[str]

class HealthMonitor:
    """Monitor de saúde do sistema."""
    
    async def __init__(self):
        self.logger = logging.getLogger("HealthMonitor")
        self.metrics_history: List[SystemHealth] = []
        self.max_history = 1000
        
        # Paths
        self.health_dir = Path("/root/.penin_omega/health")
        self.health_dir.mkdir(parents=True, exist_ok=True)
        self.health_file = self.health_dir / "current_health.json"
        
        # Monitoring control
        self._monitoring = False
        self._monitor_thread = None
        self._start_time = time.time()
        
        # Thresholds
        self.thresholds = {
            "cpu_percent": {"warning": 80.0, "critical": 95.0},
            "memory_percent": {"warning": 85.0, "critical": 95.0},
            "disk_percent": {"warning": 90.0, "critical": 98.0},
            "load_average": {"warning": 2.0, "critical": 4.0},
            "open_files": {"warning": 1000, "critical": 2000},
            "thread_count": {"warning": 100, "critical": 200},
            "response_time_ms": {"warning": 1000, "critical": 5000}
        }
    
    async def start_monitoring(self, interval_seconds: int = 30):
        """Inicia monitoramento contínuo."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval_seconds,),
            daemon=True,
            name="HealthMonitor"
        )
        self._monitor_thread.start()
        self.logger.info(f"Health monitoring started (interval: {interval_seconds}s)")
    
    async def stop_monitoring(self):
        """Para monitoramento."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        self.logger.info("Health monitoring stopped")
    
    async def _monitor_loop(self, interval_seconds: int):
        """Loop principal de monitoramento."""
        while self._monitoring:
            try:
                health = self.get_current_health()
                self._save_health_snapshot(health)
                
                # Mantém histórico limitado
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history = self.metrics_history[-self.max_history:]
                
                # Log problemas críticos
                if health.overall_status == HealthStatus.CRITICAL:
                    self.logger.critical(f"System health CRITICAL: {health.issues}")
                elif health.overall_status == HealthStatus.WARNING:
                    self.logger.warning(f"System health WARNING: {health.issues}")
                
                time.sleep(interval_seconds)
                
            except Exception as e:
                self.logger.error(f"Error in health monitoring: {e}")
                time.sleep(interval_seconds)
    
    async def get_current_health(self) -> SystemHealth:
        """Obtém saúde atual do sistema."""
        try:
            metrics = {}
            issues = []
            
            # CPU
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics["cpu_percent"] = self._create_metric(
                "cpu_percent", cpu_percent, "%", "CPU usage percentage"
            )
            
            # Memory
            memory = psutil.virtual_memory()
            metrics["memory_percent"] = self._create_metric(
                "memory_percent", memory.percent, "%", "Memory usage percentage"
            )
            metrics["memory_available_gb"] = HealthMetric(
                name="memory_available_gb",
                value=memory.available / (1024**3),
                unit="GB",
                status=HealthStatus.HEALTHY,
                threshold_warning=1.0,
                threshold_critical=0.5,
                timestamp=datetime.now(timezone.utc),
                description="Available memory in GB"
            )
            
            # Disk
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics["disk_percent"] = self._create_metric(
                "disk_percent", disk_percent, "%", "Disk usage percentage"
            )
            
            # Load average (Linux/Unix)
            try:
                load_avg = psutil.getloadavg()[0]  # 1-minute load average
                metrics["load_average"] = self._create_metric(
                    "load_average", load_avg, "", "System load average (1min)"
                )
            except AttributeError:
                # Windows doesn't have load average
                pass
            
            # Process info
            process = psutil.Process()
            
            # Open files
            try:
                open_files = len(process.open_files())
                metrics["open_files"] = self._create_metric(
                    "open_files", open_files, "files", "Number of open files"
                )
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
            
            # Thread count
            try:
                thread_count = process.num_threads()
                metrics["thread_count"] = self._create_metric(
                    "thread_count", thread_count, "threads", "Number of threads"
                )
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                pass
            
            # PENIN-Ω specific metrics
            penin_metrics = self._get_penin_omega_metrics()
            metrics.update(penin_metrics)
            
            # Determine overall status
            overall_status = self._calculate_overall_status(metrics, issues)
            
            # Uptime
            uptime = time.time() - self._start_time
            
            health = SystemHealth(
                overall_status=overall_status,
                metrics=metrics,
                timestamp=datetime.now(timezone.utc),
                uptime_seconds=uptime,
                issues=issues
            )
            
            self.metrics_history.append(health)
            return await health
            
        except Exception as e:
            self.logger.error(f"Error getting system health: {e}")
            return await SystemHealth(
                overall_status=HealthStatus.UNKNOWN,
                metrics={},
                timestamp=datetime.now(timezone.utc),
                uptime_seconds=0,
                issues=[f"Health check failed: {str(e)}"]
            )
    
    async def _create_metric(self, name: str, value: float, unit: str, description: str) -> HealthMetric:
        """Cria métrica com status baseado em thresholds."""
        thresholds = self.thresholds.get(name, {"warning": float('inf'), "critical": float('inf')})
        
        if value >= thresholds["critical"]:
            status = HealthStatus.CRITICAL
        elif value >= thresholds["warning"]:
            status = HealthStatus.WARNING
        else:
            status = HealthStatus.HEALTHY
        
        return await HealthMetric(
            name=name,
            value=value,
            unit=unit,
            status=status,
            threshold_warning=thresholds["warning"],
            threshold_critical=thresholds["critical"],
            timestamp=datetime.now(timezone.utc),
            description=description
        )
    
    async def _get_penin_omega_metrics(self) -> Dict[str, HealthMetric]:
        """Obtém métricas específicas do PENIN-Ω."""
        metrics = {}
        
        try:
            # Database sizes
            db_paths = [
                "/root/.penin_omega/worm/worm_ledger.db",
                "/root/.penin_omega/state/global_state.db",
                "/root/.penin_omega/security/dlp_violations.db"
            ]
            
            total_db_size = 0
            for db_path in db_paths:
                if Path(db_path).exists():
                    total_db_size += Path(db_path).stat().st_size
            
            metrics["database_size_mb"] = HealthMetric(
                name="database_size_mb",
                value=total_db_size / (1024**2),
                unit="MB",
                status=HealthStatus.HEALTHY,
                threshold_warning=100.0,
                threshold_critical=500.0,
                timestamp=datetime.now(timezone.utc),
                description="Total database size"
            )
            
            # Log file sizes
            log_dir = Path("/root/.penin_omega/logs")
            total_log_size = 0
            if log_dir.exists():
                for log_file in log_dir.glob("*.log"):
                    total_log_size += log_file.stat().st_size
            
            metrics["log_size_mb"] = HealthMetric(
                name="log_size_mb",
                value=total_log_size / (1024**2),
                unit="MB",
                status=HealthStatus.HEALTHY,
                threshold_warning=50.0,
                threshold_critical=200.0,
                timestamp=datetime.now(timezone.utc),
                description="Total log files size"
            )
            
            # Module health (simplified check)
            try:
                from penin_omega_master_system import penin_omega
                status = penin_omega.get_life_status()
                
                module_health = 1.0 if status.get("status") == "ALIVE_AND_EVOLVING" else 0.0
                metrics["module_health"] = HealthMetric(
                    name="module_health",
                    value=module_health,
                    unit="score",
                    status=HealthStatus.HEALTHY if module_health > 0.5 else HealthStatus.CRITICAL,
                    threshold_warning=0.7,
                    threshold_critical=0.3,
                    timestamp=datetime.now(timezone.utc),
                    description="PENIN-Ω module health score"
                )
            except Exception:
                pass
            
        except Exception as e:
            self.logger.error(f"Error getting PENIN-Ω metrics: {e}")
        
        return await metrics
    
    async def _calculate_overall_status(self, metrics: Dict[str, HealthMetric], issues: List[str]) -> HealthStatus:
        """Calcula status geral baseado nas métricas."""
        critical_count = sum(1 for m in metrics.values() if m.status == HealthStatus.CRITICAL)
        warning_count = sum(1 for m in metrics.values() if m.status == HealthStatus.WARNING)
        
        if critical_count > 0:
            issues.extend([f"{m.name}: {m.value}{m.unit}" for m in metrics.values() if m.status == HealthStatus.CRITICAL])
            return await HealthStatus.CRITICAL
        elif warning_count > 2:  # Multiple warnings = critical
            issues.extend([f"{m.name}: {m.value}{m.unit}" for m in metrics.values() if m.status == HealthStatus.WARNING])
            return await HealthStatus.CRITICAL
        elif warning_count > 0:
            issues.extend([f"{m.name}: {m.value}{m.unit}" for m in metrics.values() if m.status == HealthStatus.WARNING])
            return await HealthStatus.WARNING
        else:
            return await HealthStatus.HEALTHY
    
    async def _save_health_snapshot(self, health: SystemHealth):
        """Salva snapshot de saúde."""
        try:
            health_data = {
                "overall_status": health.overall_status.value,
                "timestamp": health.timestamp.isoformat(),
                "uptime_seconds": health.uptime_seconds,
                "issues": health.issues,
                "metrics": {
                    name: {
                        "value": metric.value,
                        "unit": metric.unit,
                        "status": metric.status.value,
                        "description": metric.description
                    }
                    for name, metric in health.metrics.items()
                }
            }
            
            with open(self.health_file, 'w') as f:
                json.dump(health_data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Error saving health snapshot: {e}")
    
    async def get_health_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Obtém resumo de saúde das últimas horas."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_health = [h for h in self.metrics_history if h.timestamp >= cutoff_time]
        
        if not recent_health:
            return await {"error": "No health data available"}
        
        # Estatísticas
        total_checks = len(recent_health)
        healthy_checks = sum(1 for h in recent_health if h.overall_status == HealthStatus.HEALTHY)
        warning_checks = sum(1 for h in recent_health if h.overall_status == HealthStatus.WARNING)
        critical_checks = sum(1 for h in recent_health if h.overall_status == HealthStatus.CRITICAL)
        
        # Métricas médias
        avg_metrics = {}
        if recent_health:
            for metric_name in recent_health[0].metrics.keys():
                values = [h.metrics[metric_name].value for h in recent_health if metric_name in h.metrics]
                if values:
                    avg_metrics[metric_name] = {
                        "avg": sum(values) / len(values),
                        "min": min(values),
                        "max": max(values),
                        "unit": recent_health[0].metrics[metric_name].unit
                    }
        
        return await {
            "period_hours": hours,
            "total_checks": total_checks,
            "health_percentage": (healthy_checks / total_checks * 100) if total_checks > 0 else 0,
            "status_distribution": {
                "healthy": healthy_checks,
                "warning": warning_checks,
                "critical": critical_checks
            },
            "average_metrics": avg_metrics,
            "current_status": recent_health[-1].overall_status.value if recent_health else "unknown",
            "uptime_hours": recent_health[-1].uptime_seconds / 3600 if recent_health else 0
        }
    
    async def get_alerts(self) -> List[Dict[str, Any]]:
        """Obtém alertas ativos."""
        current_health = self.get_current_health()
        alerts = []
        
        for name, metric in current_health.metrics.items():
            if metric.status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                alerts.append({
                    "metric": name,
                    "value": metric.value,
                    "unit": metric.unit,
                    "status": metric.status.value,
                    "threshold": metric.threshold_critical if metric.status == HealthStatus.CRITICAL else metric.threshold_warning,
                    "description": metric.description,
                    "timestamp": metric.timestamp.isoformat()
                })
        
        return await alerts

# Instância global
health_monitor = HealthMonitor()

# Funções de conveniência
async def get_system_health() -> SystemHealth:
    """Função de conveniência para obter saúde do sistema."""
    return await health_monitor.get_current_health()

async def get_health_summary(hours: int = 24) -> Dict[str, Any]:
    """Função de conveniência para obter resumo de saúde."""
    return await health_monitor.get_health_summary(hours)

async def get_active_alerts() -> List[Dict[str, Any]]:
    """Função de conveniência para obter alertas ativos."""
    return await health_monitor.get_alerts()

async def start_health_monitoring(interval_seconds: int = 30):
    """Função de conveniência para iniciar monitoramento."""
    health_monitor.start_monitoring(interval_seconds)

if __name__ == "__main__":
    # Teste do monitor de saúde
    print("Testando monitor de saúde...")
    
    # Obtém saúde atual
    health = get_system_health()
    print(f"Status geral: {health.overall_status.value}")
    print(f"Métricas: {len(health.metrics)}")
    
    # Mostra algumas métricas
    for name, metric in list(health.metrics.items())[:3]:
        print(f"  {name}: {metric.value}{metric.unit} ({metric.status.value})")
    
    # Alertas
    alerts = get_active_alerts()
    print(f"Alertas ativos: {len(alerts)}")
    
    print("Monitor de saúde funcionando!")
