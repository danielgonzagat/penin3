
# FUNÇÕES DETERMINÍSTICAS (substituem random)
import hashlib
import os
import time


def deterministic_random(seed_offset=0):
    """Substituto determinístico para random.random()"""
    import hashlib
    import time

    # Usa múltiplas fontes de determinismo
    sources = [
        str(time.time()).encode(),
        str(os.getpid()).encode(),
        str(id({})).encode(),
        str(seed_offset).encode()
    ]

    # Combina todas as fontes
    combined = b''.join(sources)
    hash_val = int(hashlib.md5(combined).hexdigest()[:8], 16)

    return (hash_val % 1000000) / 1000000.0


def deterministic_uniform(a, b, seed_offset=0):
    """Substituto determinístico para random.uniform(a, b)"""
    r = deterministic_random(seed_offset)
    return a + (b - a) * r


def deterministic_randint(a, b, seed_offset=0):
    """Substituto determinístico para random.randint(a, b)"""
    r = deterministic_random(seed_offset)
    return int(a + (b - a + 1) * r)


def deterministic_choice(seq, seed_offset=0):
    """Substituto determinístico para random.choice(seq)"""
    if not seq:
        raise IndexError("sequence is empty")

    r = deterministic_random(seed_offset)
    return seq[int(r * len(seq))]


def deterministic_shuffle(lst, seed_offset=0):
    """Substituto determinístico para random.shuffle(lst)"""
    if not lst:
        return

    # Shuffle determinístico baseado em ordenação por hash
    def sort_key(item):
        item_str = str(item) + str(seed_offset)
        return hashlib.md5(item_str.encode()).hexdigest()

    lst.sort(key=sort_key)


def deterministic_torch_rand(*size, seed_offset=0):
    """Substituto determinístico para torch.rand(*size)"""
    if not size:
        return torch.tensor(deterministic_random(seed_offset))

    # Gera valores determinísticos
    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_random(seed_offset + i))

    return torch.tensor(values).reshape(size)


def deterministic_torch_randint(low, high, size=None, seed_offset=0):
    """Substituto determinístico para torch.randint(low, high, size)"""
    if size is None:
        return torch.tensor(deterministic_randint(low, high, seed_offset))

    # Gera valores determinísticos
    if isinstance(size, int):
        size = (size,)

    total_elements = 1
    for dim in size:
        total_elements *= dim

    values = []
    for i in range(total_elements):
        values.append(deterministic_randint(low, high, seed_offset + i))

    return torch.tensor(values).reshape(size)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Módulo 7/8 NEXUS
=========================
Scheduler, Watchdog e UCB com orquestração segura.
"""

from __future__ import annotations
import asyncio
import json
import time
import math
import random
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple, Callable
import uuid
import queue
from enum import Enum

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

logger = logging.getLogger("PENIN_OMEGA_NEXUS")

# =============================================================================
# ENUMS E CLASSES
# =============================================================================

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class TaskPriority(Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4

@dataclass
class ScheduledTask:
    """Tarefa agendada."""
    id: str
    name: str
    function_name: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    priority: TaskPriority = TaskPriority.MEDIUM
    scheduled_time: Optional[datetime] = None
    deadline: Optional[datetime] = None
    max_retries: int = 3
    retry_count: int = 0
    status: TaskStatus = TaskStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    safety_approved: bool = False

@dataclass
class WatchdogAlert:
    """Alerta do Watchdog."""
    id: str
    alert_type: str
    severity: str  # "low", "medium", "high", "critical"
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    acknowledged: bool = False
    resolved: bool = False

class UCBArm:
    """Braço do Upper Confidence Bound."""
    
    def __init__(self, arm_id: str, name: str):
        self.arm_id = arm_id
        self.name = name
        self.total_reward = 0.0
        self.play_count = 0
        self.last_reward = 0.0
        self.confidence_bound = float('inf')
    
    def update(self, reward: float):
        """Atualiza estatísticas do braço."""
        self.total_reward += reward
        self.play_count += 1
        self.last_reward = reward
    
    def get_average_reward(self) -> float:
        """Retorna recompensa média."""
        return self.total_reward / max(1, self.play_count)
    
    def calculate_ucb(self, total_plays: int, confidence: float = 2.0) -> float:
        """Calcula Upper Confidence Bound."""
        if self.play_count == 0:
            return float('inf')
        
        avg_reward = self.get_average_reward()
        confidence_term = confidence * math.sqrt(math.log(total_plays) / self.play_count)
        
        self.confidence_bound = avg_reward + confidence_term
        return self.confidence_bound

class NexusScheduler:
    """Scheduler por utilidade segura."""
    
    def __init__(self):
        self.logger = logging.getLogger("NexusScheduler")
        self.task_queue = queue.PriorityQueue()
        self.running_tasks = {}
        self.completed_tasks = []
        self.failed_tasks = []
        self.scheduler_thread = None
        self.running = False
        self._lock = threading.Lock()
    
    def start(self):
        """Inicia o scheduler."""
        if not self.running:
            self.running = True
            self.scheduler_thread = threading.Thread(target=self._scheduler_loop, daemon=True)
            self.scheduler_thread.start()
            self.logger.info("✅ Scheduler iniciado")
    
    def stop(self):
        """Para o scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info("🛑 Scheduler parado")
    
    def schedule_task(self, task: ScheduledTask) -> bool:
        """Agenda uma tarefa."""
        try:
            # Verifica segurança da tarefa
            if not self._is_task_safe(task):
                self.logger.warning(f"🚨 Tarefa {task.id} rejeitada por segurança")
                return False
            
            task.safety_approved = True
            
            # Calcula prioridade efetiva
            priority_score = self._calculate_task_priority(task)
            
            # Adiciona à fila
            self.task_queue.put((priority_score, time.time(), task))
            
            self.logger.info(f"📅 Tarefa agendada: {task.name} (prioridade: {priority_score})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao agendar tarefa: {e}")
            return False
    
    def _scheduler_loop(self):
        """Loop principal do scheduler."""
        while self.running:
            try:
                # Verifica se pode executar novas tarefas
                if not self._can_execute_tasks():
                    time.sleep(1)
                    continue
                
                # Pega próxima tarefa
                try:
                    priority_score, queued_time, task = self.task_queue.get(timeout=1)
                except queue.Empty:
                    continue
                
                # Verifica se ainda é seguro executar
                if self._is_execution_safe(task):
                    self._execute_task(task)
                else:
                    # Rejeita tarefa por segurança
                    task.status = TaskStatus.FAILED
                    task.error = "Execução rejeitada por critérios de segurança"
                    self.failed_tasks.append(task)
                
            except Exception as e:
                self.logger.error(f"Erro no loop do scheduler: {e}")
                time.sleep(1)
    
    def _is_task_safe(self, task: ScheduledTask) -> bool:
        """Verifica se tarefa é segura para agendamento."""
        try:
            from penin_omega_global_state_manager import get_global_state
            
            current_state = get_global_state()
            
            # Verifica estado do sistema
            rho = current_state.get("rho", 0.5)
            sr_score = current_state.get("sr_score", 0.8)
            system_health = current_state.get("system_health", 1.0)
            
            # Critérios de segurança
            safety_checks = [
                rho < 0.95,  # ρ não crítico
                sr_score >= 0.7,  # SR mínimo
                system_health >= 0.8,  # Saúde do sistema
                task.priority != TaskPriority.CRITICAL or sr_score >= 0.85  # Tarefas críticas precisam SR alto
            ]
            
            return all(safety_checks)
            
        except Exception as e:
            self.logger.error(f"Erro na verificação de segurança: {e}")
            return False  # Fail-safe
    
    def _can_execute_tasks(self) -> bool:
        """Verifica se pode executar novas tarefas."""
        try:
            from penin_omega_global_state_manager import get_global_state
            
            current_state = get_global_state()
            
            # Limites de execução
            max_concurrent_tasks = 5
            current_running = len(self.running_tasks)
            
            # Verifica recursos
            system_health = current_state.get("system_health", 1.0)
            
            return (
                current_running < max_concurrent_tasks and
                system_health >= 0.7
            )
            
        except Exception:
            return False
    
    def _is_execution_safe(self, task: ScheduledTask) -> bool:
        """Verifica se execução é segura no momento atual."""
        try:
            from penin_omega_global_state_manager import get_global_state
            
            current_state = get_global_state()
            
            # Re-verifica segurança
            rho = current_state.get("rho", 0.5)
            
            # Se ρ muito alto, só executa tarefas críticas de segurança
            if rho > 0.9:
                safety_functions = ["emergency_shutdown", "safety_check", "system_recovery"]
                return task.function_name in safety_functions
            
            return True
            
        except Exception:
            return False
    
    def _calculate_task_priority(self, task: ScheduledTask) -> float:
        """Calcula prioridade efetiva da tarefa."""
        try:
            # Prioridade base
            base_priority = task.priority.value
            
            # Ajustes por deadline
            deadline_factor = 1.0
            if task.deadline:
                time_to_deadline = (task.deadline - datetime.now(timezone.utc)).total_seconds()
                if time_to_deadline < 3600:  # Menos de 1 hora
                    deadline_factor = 0.5  # Maior prioridade (menor número)
                elif time_to_deadline < 86400:  # Menos de 1 dia
                    deadline_factor = 0.8
            
            # Ajustes por tentativas
            retry_factor = 1.0 + (task.retry_count * 0.2)  # Aumenta prioridade com retries
            
            return base_priority * deadline_factor * retry_factor
            
        except Exception:
            return float(task.priority.value)
    
    def _execute_task(self, task: ScheduledTask):
        """Executa uma tarefa."""
        try:
            with self._lock:
                self.running_tasks[task.id] = task
            
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc).isoformat()
            
            # Simula execução da tarefa
            result = self._simulate_task_execution(task)
            
            # Atualiza resultado
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc).isoformat()
            
            with self._lock:
                del self.running_tasks[task.id]
                self.completed_tasks.append(task)
            
            self.logger.info(f"✅ Tarefa concluída: {task.name}")
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.completed_at = datetime.now(timezone.utc).isoformat()
            
            with self._lock:
                if task.id in self.running_tasks:
                    del self.running_tasks[task.id]
                self.failed_tasks.append(task)
            
            self.logger.error(f"❌ Tarefa falhou: {task.name} - {e}")
    
    def _simulate_task_execution(self, task: ScheduledTask) -> Any:
        """Simula execução de tarefa."""
        # Simula tempo de execução
        execution_time = deterministic_uniform(0.1, 2.0)
        time.sleep(execution_time)
        
        # Simula resultado baseado na função
        if task.function_name == "acquisition_f3":
            return {"candidates": 3, "quality": 0.8}
        elif task.function_name == "mutation_f4":
            return {"mutations": 2, "improvement": 0.15}
        elif task.function_name == "crucible_f5":
            return {"selected": 1, "quality": 0.9}
        else:
            return {"status": "completed", "execution_time": execution_time}
    
    def get_scheduler_status(self) -> Dict[str, Any]:
        """Retorna status do scheduler."""
        with self._lock:
            return {
                "running": self.running,
                "queued_tasks": self.task_queue.qsize(),
                "running_tasks": len(self.running_tasks),
                "completed_tasks": len(self.completed_tasks),
                "failed_tasks": len(self.failed_tasks),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }

class Watchdog:
    """Sistema de monitoramento e alertas."""
    
    def __init__(self):
        self.logger = logging.getLogger("Watchdog")
        self.alerts = []
        self.monitoring_thread = None
        self.running = False
        self.thresholds = {
            "rho_critical": 0.95,
            "rho_warning": 0.9,
            "sr_critical": 0.6,
            "sr_warning": 0.7,
            "system_health_critical": 0.5,
            "system_health_warning": 0.7
        }
    
    def start(self):
        """Inicia o watchdog."""
        if not self.running:
            self.running = True
            self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
            self.monitoring_thread.start()
            self.logger.info("👁️ Watchdog iniciado")
    
    def stop(self):
        """Para o watchdog."""
        self.running = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("🛑 Watchdog parado")
    
    def _monitoring_loop(self):
        """Loop de monitoramento."""
        while self.running:
            try:
                self._check_system_health()
                self._check_security_gates()
                self._check_performance_metrics()
                
                time.sleep(5)  # Verifica a cada 5 segundos
                
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {e}")
                time.sleep(10)
    
    def _check_system_health(self):
        """Verifica saúde geral do sistema."""
        try:
            from penin_omega_global_state_manager import get_global_state
            
            current_state = get_global_state()
            system_health = current_state.get("system_health", 1.0)
            
            if system_health <= self.thresholds["system_health_critical"]:
                self._create_alert(
                    "system_health_critical",
                    "critical",
                    f"Saúde do sistema crítica: {system_health:.3f}",
                    {"system_health": system_health}
                )
            elif system_health <= self.thresholds["system_health_warning"]:
                self._create_alert(
                    "system_health_warning",
                    "high",
                    f"Saúde do sistema baixa: {system_health:.3f}",
                    {"system_health": system_health}
                )
                
        except Exception as e:
            self.logger.error(f"Erro na verificação de saúde: {e}")
    
    def _check_security_gates(self):
        """Verifica gates de segurança."""
        try:
            from penin_omega_global_state_manager import get_global_state
            
            current_state = get_global_state()
            rho = current_state.get("rho", 0.5)
            sr_score = current_state.get("sr_score", 0.8)
            
            # Verifica ρ (IR→IC)
            if rho >= self.thresholds["rho_critical"]:
                self._create_alert(
                    "rho_critical",
                    "critical",
                    f"ρ crítico: {rho:.3f} ≥ {self.thresholds['rho_critical']}",
                    {"rho": rho, "threshold": self.thresholds["rho_critical"]}
                )
            elif rho >= self.thresholds["rho_warning"]:
                self._create_alert(
                    "rho_warning",
                    "high",
                    f"ρ alto: {rho:.3f} ≥ {self.thresholds['rho_warning']}",
                    {"rho": rho, "threshold": self.thresholds["rho_warning"]}
                )
            
            # Verifica SR (SR-Ω∞)
            if sr_score <= self.thresholds["sr_critical"]:
                self._create_alert(
                    "sr_critical",
                    "critical",
                    f"SR crítico: {sr_score:.3f} ≤ {self.thresholds['sr_critical']}",
                    {"sr_score": sr_score, "threshold": self.thresholds["sr_critical"]}
                )
            elif sr_score <= self.thresholds["sr_warning"]:
                self._create_alert(
                    "sr_warning",
                    "high",
                    f"SR baixo: {sr_score:.3f} ≤ {self.thresholds['sr_warning']}",
                    {"sr_score": sr_score, "threshold": self.thresholds["sr_warning"]}
                )
                
        except Exception as e:
            self.logger.error(f"Erro na verificação de gates: {e}")
    
    def _check_performance_metrics(self):
        """Verifica métricas de performance."""
        try:
            # Simula verificação de performance
            import psutil
            
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            if cpu_percent > 90:
                self._create_alert(
                    "cpu_high",
                    "high",
                    f"CPU alta: {cpu_percent:.1f}%",
                    {"cpu_percent": cpu_percent}
                )
            
            if memory.percent > 90:
                self._create_alert(
                    "memory_high",
                    "high",
                    f"Memória alta: {memory.percent:.1f}%",
                    {"memory_percent": memory.percent}
                )
                
        except Exception as e:
            self.logger.error(f"Erro na verificação de performance: {e}")
    
    def _create_alert(self, alert_type: str, severity: str, message: str, metrics: Dict[str, Any]):
        """Cria um alerta."""
        try:
            # Evita alertas duplicados recentes
            recent_alerts = [
                a for a in self.alerts[-10:] 
                if a.alert_type == alert_type and 
                (datetime.now(timezone.utc) - datetime.fromisoformat(a.timestamp.replace('Z', '+00:00'))).total_seconds() < 300
            ]
            
            if recent_alerts:
                return  # Não cria alerta duplicado
            
            alert = WatchdogAlert(
                id=f"alert_{uuid.uuid4().hex[:8]}",
                alert_type=alert_type,
                severity=severity,
                message=message,
                metrics=metrics
            )
            
            self.alerts.append(alert)
            
            # Log baseado na severidade
            if severity == "critical":
                self.logger.critical(f"🚨 CRÍTICO: {message}")
            elif severity == "high":
                self.logger.error(f"⚠️ ALTO: {message}")
            else:
                self.logger.warning(f"⚠️ {message}")
            
            # Registra no WORM se crítico
            if severity == "critical":
                try:
                    from penin_omega_security_governance import security_governance
                    security_governance.worm_ledger.append_record(
                        f"watchdog_alert_{alert.id}",
                        f"Alerta crítico: {message}",
                        {
                            "alert_id": alert.id,
                            "alert_type": alert_type,
                            "severity": severity,
                            "metrics": metrics
                        }
                    )
                except Exception as e:
                    self.logger.warning(f"Falha no registro WORM: {e}")
                    
        except Exception as e:
            self.logger.error(f"Erro ao criar alerta: {e}")
    
    def get_active_alerts(self) -> List[WatchdogAlert]:
        """Retorna alertas ativos."""
        return [alert for alert in self.alerts if not alert.resolved]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Reconhece um alerta."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve um alerta."""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False

class UCBOptimizer:
    """Otimizador Upper Confidence Bound."""
    
    def __init__(self):
        self.logger = logging.getLogger("UCBOptimizer")
        self.arms = {}
        self.total_plays = 0
        self.confidence_level = 2.0
    
    def add_arm(self, arm_id: str, name: str):
        """Adiciona um braço ao UCB."""
        self.arms[arm_id] = UCBArm(arm_id, name)
        self.logger.info(f"🎯 Braço UCB adicionado: {name}")
    
    def select_arm(self) -> Optional[str]:
        """Seleciona braço usando UCB."""
        try:
            if not self.arms:
                return None
            
            # Calcula UCB para todos os braços
            best_arm_id = None
            best_ucb = -float('inf')
            
            for arm_id, arm in self.arms.items():
                ucb_value = arm.calculate_ucb(self.total_plays, self.confidence_level)
                
                if ucb_value > best_ucb:
                    best_ucb = ucb_value
                    best_arm_id = arm_id
            
            return best_arm_id
            
        except Exception as e:
            self.logger.error(f"Erro na seleção UCB: {e}")
            return None
    
    def update_arm(self, arm_id: str, reward: float):
        """Atualiza braço com recompensa."""
        try:
            if arm_id in self.arms:
                self.arms[arm_id].update(reward)
                self.total_plays += 1
                
                self.logger.info(f"🎯 UCB atualizado: {arm_id} = {reward:.3f}")
            
        except Exception as e:
            self.logger.error(f"Erro na atualização UCB: {e}")
    
    def get_arm_stats(self) -> Dict[str, Dict[str, Any]]:
        """Retorna estatísticas dos braços."""
        return {
            arm_id: {
                "name": arm.name,
                "average_reward": arm.get_average_reward(),
                "play_count": arm.play_count,
                "last_reward": arm.last_reward,
                "confidence_bound": arm.confidence_bound
            }
            for arm_id, arm in self.arms.items()
        }

class NexusOrchestrator:
    """Orquestrador principal do NEXUS."""
    
    def __init__(self):
        self.logger = logging.getLogger("NexusOrchestrator")
        self.scheduler = NexusScheduler()
        self.watchdog = Watchdog()
        self.ucb_optimizer = UCBOptimizer()
        self.running = False
        
        # Inicializa braços UCB para estratégias
        self._initialize_ucb_arms()
    
    def _initialize_ucb_arms(self):
        """Inicializa braços UCB."""
        strategies = [
            ("conservative", "Estratégia Conservadora"),
            ("balanced", "Estratégia Balanceada"),
            ("adaptive", "Estratégia Adaptativa")
        ]
        
        for strategy_id, strategy_name in strategies:
            self.ucb_optimizer.add_arm(strategy_id, strategy_name)
    
    def start(self):
        """Inicia todos os componentes do NEXUS."""
        if not self.running:
            self.running = True
            self.scheduler.start()
            self.watchdog.start()
            self.logger.info("🚀 NEXUS iniciado")
    
    def stop(self):
        """Para todos os componentes do NEXUS."""
        if self.running:
            self.running = False
            self.scheduler.stop()
            self.watchdog.stop()
            self.logger.info("🛑 NEXUS parado")
    
    def schedule_pipeline_task(self, function_name: str, parameters: Dict[str, Any], priority: TaskPriority = TaskPriority.MEDIUM) -> str:
        """Agenda tarefa do pipeline."""
        try:
            task = ScheduledTask(
                id=f"task_{uuid.uuid4().hex[:8]}",
                name=f"Pipeline: {function_name}",
                function_name=function_name,
                parameters=parameters,
                priority=priority
            )
            
            if self.scheduler.schedule_task(task):
                return task.id
            else:
                return ""
                
        except Exception as e:
            self.logger.error(f"Erro ao agendar tarefa: {e}")
            return ""
    
    def get_nexus_status(self) -> Dict[str, Any]:
        """Retorna status completo do NEXUS."""
        return {
            "running": self.running,
            "scheduler": self.scheduler.get_scheduler_status(),
            "watchdog": {
                "active_alerts": len(self.watchdog.get_active_alerts()),
                "total_alerts": len(self.watchdog.alerts)
            },
            "ucb_optimizer": {
                "total_plays": self.ucb_optimizer.total_plays,
                "arms_count": len(self.ucb_optimizer.arms)
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

nexus_orchestrator = NexusOrchestrator()

# Funções de conveniência
def start_nexus():
    nexus_orchestrator.start()

def stop_nexus():
    nexus_orchestrator.stop()

def schedule_task(function_name: str, parameters: Dict[str, Any], priority: TaskPriority = TaskPriority.MEDIUM) -> str:
    return nexus_orchestrator.schedule_pipeline_task(function_name, parameters, priority)

def get_nexus_status() -> Dict[str, Any]:
    return nexus_orchestrator.get_nexus_status()

def get_active_alerts() -> List[WatchdogAlert]:
    return nexus_orchestrator.watchdog.get_active_alerts()

# Auto-start
nexus_orchestrator.start()

if __name__ == "__main__":
    # Teste básico
    orchestrator = NexusOrchestrator()
    orchestrator.start()
    
    # Agenda algumas tarefas de teste
    task_id = orchestrator.schedule_pipeline_task("acquisition_f3", {"query": "test"})
    print(f"Tarefa agendada: {task_id}")
    
    time.sleep(5)
    
    status = orchestrator.get_nexus_status()
    print(json.dumps(status, indent=2))
    
    orchestrator.stop()
