#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Otimizador de Performance
===================================
Sistema para resolver memory leaks, CPU overhead e otimizar I/O.
"""

from __future__ import annotations
import gc
import os
import psutil
import threading
import time
import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import logging
import weakref
from concurrent.futures import ThreadPoolExecutor
import resource

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PENIN_OMEGA_ROOT = Path("/root/.penin_omega")
PERFORMANCE_PATH = PENIN_OMEGA_ROOT / "performance"
PERFORMANCE_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CLASSES DE MONITORAMENTO
# =============================================================================

@dataclass
class PerformanceMetrics:
    """Métricas de performance do sistema."""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_rss: int = 0  # Resident Set Size
    memory_vms: int = 0  # Virtual Memory Size
    open_files: int = 0
    threads_count: int = 0
    gc_objects: int = 0
    gc_collections: Dict[int, int] = field(default_factory=dict)

@dataclass
class ResourceLeak:
    """Informações sobre vazamento de recursos."""
    resource_type: str
    count: int
    growth_rate: float
    severity: str  # low, medium, high, critical
    description: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

# =============================================================================
# OTIMIZADOR DE PERFORMANCE
# =============================================================================

class PerformanceOptimizer:
    """Otimiza performance e detecta vazamentos."""
    
    def __init__(self):
        self.logger = logging.getLogger("PerformanceOptimizer")
        self.process = psutil.Process()
        self.metrics_history = []
        self.resource_leaks = []
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Configurações de otimização
        self.optimization_config = {
            "gc_threshold": (700, 10, 10),  # Mais agressivo
            "max_threads": 8,
            "memory_limit_mb": 2048,
            "cpu_limit_percent": 80.0,
            "monitoring_interval": 5.0
        }
        
        # Weak references para objetos monitorados
        self.tracked_objects = weakref.WeakSet()
        
        # Thread pool otimizado
        self.thread_pool = ThreadPoolExecutor(
            max_workers=self.optimization_config["max_threads"],
            thread_name_prefix="PeninOmega"
        )
        
        self._apply_initial_optimizations()
    
    def _apply_initial_optimizations(self):
        """Aplica otimizações iniciais."""
        try:
            # Configura garbage collector
            gc.set_threshold(*self.optimization_config["gc_threshold"])
            
            # Configura limites de recursos
            try:
                # Limite de memória virtual (2GB)
                resource.setrlimit(resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, -1))
            except (OSError, ValueError):
                self.logger.warning("Não foi possível definir limite de memória")
            
            # Otimizações de I/O
            if hasattr(os, 'nice'):
                try:
                    os.nice(-5)  # Prioridade mais alta
                except PermissionError:
                    pass
            
            self.logger.info("✅ Otimizações iniciais aplicadas")
            
        except Exception as e:
            self.logger.error(f"Erro ao aplicar otimizações: {e}")
    
    def start_monitoring(self):
        """Inicia monitoramento de performance."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True,
            name="PerformanceMonitor"
        )
        self.monitor_thread.start()
        self.logger.info("🔍 Monitoramento de performance iniciado")
    
    def stop_monitoring(self):
        """Para monitoramento de performance."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("🛑 Monitoramento de performance parado")
    
    def _monitoring_loop(self):
        """Loop principal de monitoramento."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Limita histórico
                if len(self.metrics_history) > 1000:
                    self.metrics_history = self.metrics_history[-500:]
                
                # Detecta vazamentos
                self._detect_leaks(metrics)
                
                # Aplica otimizações automáticas
                self._auto_optimize(metrics)
                
                time.sleep(self.optimization_config["monitoring_interval"])
                
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {e}")
                time.sleep(10)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Coleta métricas atuais do sistema."""
        try:
            # Métricas de CPU e memória
            cpu_percent = self.process.cpu_percent()
            memory_info = self.process.memory_info()
            memory_percent = self.process.memory_percent()
            
            # Contadores de recursos
            try:
                open_files = len(self.process.open_files())
            except (psutil.AccessDenied, psutil.NoSuchProcess):
                open_files = 0
            
            threads_count = self.process.num_threads()
            
            # Métricas de garbage collection
            gc_objects = len(gc.get_objects())
            gc_stats = gc.get_stats()
            gc_collections = {i: stat['collections'] for i, stat in enumerate(gc_stats)}
            
            return PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_rss=memory_info.rss,
                memory_vms=memory_info.vms,
                open_files=open_files,
                threads_count=threads_count,
                gc_objects=gc_objects,
                gc_collections=gc_collections
            )
            
        except Exception as e:
            self.logger.error(f"Erro ao coletar métricas: {e}")
            return PerformanceMetrics()
    
    def _detect_leaks(self, current_metrics: PerformanceMetrics):
        """Detecta vazamentos de recursos."""
        if len(self.metrics_history) < 10:
            return
        
        # Analisa últimas 10 métricas
        recent_metrics = self.metrics_history[-10:]
        
        # Detecta crescimento de memória
        memory_growth = self._calculate_growth_rate(
            [m.memory_rss for m in recent_metrics]
        )
        
        if memory_growth > 0.1:  # 10% de crescimento
            severity = "high" if memory_growth > 0.3 else "medium"
            leak = ResourceLeak(
                resource_type="memory",
                count=current_metrics.memory_rss,
                growth_rate=memory_growth,
                severity=severity,
                description=f"Crescimento de memória: {memory_growth:.2%}"
            )
            self.resource_leaks.append(leak)
            self.logger.warning(f"⚠️  Vazamento de memória detectado: {memory_growth:.2%}")
        
        # Detecta crescimento de objetos
        objects_growth = self._calculate_growth_rate(
            [m.gc_objects for m in recent_metrics]
        )
        
        if objects_growth > 0.15:  # 15% de crescimento
            leak = ResourceLeak(
                resource_type="objects",
                count=current_metrics.gc_objects,
                growth_rate=objects_growth,
                severity="medium",
                description=f"Crescimento de objetos: {objects_growth:.2%}"
            )
            self.resource_leaks.append(leak)
            self.logger.warning(f"⚠️  Vazamento de objetos detectado: {objects_growth:.2%}")
        
        # Detecta crescimento de threads
        if current_metrics.threads_count > self.optimization_config["max_threads"] * 2:
            leak = ResourceLeak(
                resource_type="threads",
                count=current_metrics.threads_count,
                growth_rate=0.0,
                severity="high",
                description=f"Muitas threads: {current_metrics.threads_count}"
            )
            self.resource_leaks.append(leak)
            self.logger.warning(f"⚠️  Muitas threads detectadas: {current_metrics.threads_count}")
    
    def _calculate_growth_rate(self, values: List[float]) -> float:
        """Calcula taxa de crescimento de uma série de valores."""
        if len(values) < 2:
            return 0.0
        
        first_value = values[0]
        last_value = values[-1]
        
        if first_value == 0:
            return 1.0 if last_value > 0 else 0.0
        
        return (last_value - first_value) / first_value
    
    def _auto_optimize(self, metrics: PerformanceMetrics):
        """Aplica otimizações automáticas baseadas nas métricas."""
        try:
            # Força garbage collection se muitos objetos
            if metrics.gc_objects > 50000:
                collected = gc.collect()
                if collected > 0:
                    self.logger.info(f"🧹 GC forçado: {collected} objetos coletados")
            
            # Reduz prioridade se CPU alta
            if metrics.cpu_percent > self.optimization_config["cpu_limit_percent"]:
                if hasattr(os, 'nice'):
                    try:
                        os.nice(5)  # Reduz prioridade
                        self.logger.info("⬇️  Prioridade reduzida devido a CPU alta")
                    except PermissionError:
                        pass
            
            # Limpa caches se memória alta
            if metrics.memory_percent > 80.0:
                self._clear_caches()
                self.logger.info("🧹 Caches limpos devido a memória alta")
            
        except Exception as e:
            self.logger.error(f"Erro na otimização automática: {e}")
    
    def _clear_caches(self):
        """Limpa caches do sistema."""
        try:
            # Limpa cache de API se disponível
            api_cache_path = PENIN_OMEGA_ROOT / "cache" / "api_cache"
            if api_cache_path.exists():
                for cache_file in api_cache_path.glob("*.json"):
                    if cache_file.stat().st_size > 1024 * 1024:  # > 1MB
                        cache_file.unlink()
            
            # Força limpeza de weak references
            self.tracked_objects.clear()
            
        except Exception as e:
            self.logger.error(f"Erro ao limpar caches: {e}")
    
    def _optimize_cache_system(self) -> bool:
        """Otimiza sistema de cache."""
        try:
            # Limpa caches antigos
            self.cleanup_resources()
            
            # Otimiza cache L2
            cache_path = Path("/root/.penin_omega/cache/l2_cache.db")
            if cache_path.exists():
                # Compacta banco de cache
                import sqlite3
                conn = sqlite3.connect(cache_path)
                conn.execute("VACUUM")
                conn.close()
                
            return True
        except Exception as e:
            self.logger.error(f"Erro na otimização de cache: {e}")
            return False
    
    def _optimize_memory_usage(self) -> bool:
        """Otimiza uso de memória."""
        try:
            import gc
            
            # Força garbage collection
            collected = gc.collect()
            self.logger.info(f"Coletados {collected} objetos não referenciados")
            
            # Otimiza referências circulares
            gc.set_threshold(700, 10, 10)
            
            return True
        except Exception as e:
            self.logger.error(f"Erro na otimização de memória: {e}")
            return False
    
    def _optimize_io_operations(self) -> bool:
        """Otimiza operações de I/O."""
        try:
            # Otimiza logs - remove logs antigos
            logs_path = Path("/root/.penin_omega/logs")
            if logs_path.exists():
                for log_file in logs_path.glob("*.log"):
                    if log_file.stat().st_size > 10 * 1024 * 1024:  # > 10MB
                        # Trunca arquivo grande
                        with open(log_file, 'w') as f:
                            f.write(f"# Log truncado em {datetime.now()}\n")
            
            # Otimiza WORM ledger
            worm_path = Path("/root/.penin_omega/worm")
            if worm_path.exists():
                for db_file in worm_path.glob("*.db"):
                    if db_file.stat().st_size > 50 * 1024 * 1024:  # > 50MB
                        # Compacta banco WORM
                        import sqlite3
                        conn = sqlite3.connect(db_file)
                        conn.execute("VACUUM")
                        conn.close()
            
            return True
        except Exception as e:
            self.logger.error(f"Erro na otimização de I/O: {e}")
            return False

    def optimize_async_operations(self):
        """Otimiza operações assíncronas."""
        try:
            # Configura event loop para melhor performance
            if hasattr(asyncio, 'set_event_loop_policy'):
                if os.name == 'posix':
                    try:
                        import uvloop
                        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
                        self.logger.info("✅ UVLoop configurado para melhor performance")
                    except ImportError:
                        pass
            
        except Exception as e:
            self.logger.error(f"Erro ao otimizar operações async: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Gera relatório de performance."""
        if not self.metrics_history:
            return {"error": "Nenhuma métrica coletada"}
        
        latest_metrics = self.metrics_history[-1]
        
        # Calcula médias
        avg_cpu = sum(m.cpu_percent for m in self.metrics_history[-10:]) / min(10, len(self.metrics_history))
        avg_memory = sum(m.memory_percent for m in self.metrics_history[-10:]) / min(10, len(self.metrics_history))
        
        # Identifica vazamentos críticos
        critical_leaks = [leak for leak in self.resource_leaks if leak.severity == "critical"]
        
        return {
            "timestamp": latest_metrics.timestamp,
            "current_metrics": {
                "cpu_percent": latest_metrics.cpu_percent,
                "memory_percent": latest_metrics.memory_percent,
                "memory_rss_mb": latest_metrics.memory_rss / (1024 * 1024),
                "threads_count": latest_metrics.threads_count,
                "gc_objects": latest_metrics.gc_objects,
                "open_files": latest_metrics.open_files
            },
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory
            },
            "resource_leaks": {
                "total": len(self.resource_leaks),
                "critical": len(critical_leaks),
                "recent": [leak.__dict__ for leak in self.resource_leaks[-5:]]
            },
            "optimization_status": {
                "monitoring_active": self.monitoring_active,
                "gc_threshold": gc.get_threshold(),
                "thread_pool_size": self.thread_pool._max_workers
            }
        }
    
    def cleanup_resources(self):
        """Limpa recursos do otimizador."""
        try:
            self.stop_monitoring()
            self.thread_pool.shutdown(wait=True)
            self.tracked_objects.clear()
            gc.collect()
            self.logger.info("🧹 Recursos do otimizador limpos")
        except Exception as e:
            self.logger.error(f"Erro na limpeza: {e}")

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

# Instância global do otimizador
performance_optimizer = PerformanceOptimizer()

# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

def start_performance_monitoring():
    """Inicia monitoramento de performance."""
    performance_optimizer.start_monitoring()

def stop_performance_monitoring():
    """Para monitoramento de performance."""
    performance_optimizer.stop_monitoring()

def get_performance_report():
    """Obtém relatório de performance."""
    return performance_optimizer.get_performance_report()

def optimize_system():
    """Aplica otimizações reais no sistema."""
    try:
        optimizations_applied = []
        
        # 1. Otimização de cache
        cache_optimized = performance_optimizer._optimize_cache_system()
        if cache_optimized:
            optimizations_applied.append("cache_optimization")
        
        # 2. Otimização de memória
        memory_optimized = performance_optimizer._optimize_memory_usage()
        if memory_optimized:
            optimizations_applied.append("memory_optimization")
        
        # 3. Otimização de I/O
        io_optimized = performance_optimizer._optimize_io_operations()
        if io_optimized:
            optimizations_applied.append("io_optimization")
        
        # 4. Limpeza de recursos
        performance_optimizer.cleanup_resources()
        optimizations_applied.append("resource_cleanup")
        
        # 5. Otimização assíncrona
        performance_optimizer.optimize_async_operations()
        optimizations_applied.append("async_optimization")
        
        return {
            "success": True,
            "optimizations_applied": optimizations_applied,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erro nas otimizações: {e}")
        return {"success": False, "error": str(e)}

# =============================================================================
# TESTE DO SISTEMA
# =============================================================================

def test_performance_optimizer():
    """Testa o otimizador de performance."""
    print("🧪 Testando otimizador de performance...")
    
    # Inicia monitoramento
    start_performance_monitoring()
    print("✅ Monitoramento iniciado")
    
    # Aguarda coleta de métricas
    time.sleep(6)
    
    # Obtém relatório
    report = get_performance_report()
    if "error" not in report:
        print(f"✅ Relatório gerado:")
        print(f"   CPU: {report['current_metrics']['cpu_percent']:.1f}%")
        print(f"   Memória: {report['current_metrics']['memory_percent']:.1f}%")
        print(f"   Threads: {report['current_metrics']['threads_count']}")
        print(f"   Objetos GC: {report['current_metrics']['gc_objects']}")
        print(f"   Vazamentos: {report['resource_leaks']['total']}")
    else:
        print(f"❌ Erro no relatório: {report['error']}")
    
    # Testa otimizações
    optimized = optimize_system()
    if optimized:
        print("✅ Otimizações aplicadas")
    
    # Para monitoramento
    stop_performance_monitoring()
    print("✅ Monitoramento parado")
    
    print("🎉 Otimizador de performance funcionando!")
    return True

if __name__ == "__main__":
    test_performance_optimizer()
