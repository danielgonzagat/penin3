#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Thread Manager - Gerenciamento Robusto de Threading
============================================================
Sistema robusto de gerenciamento de threads e pools.
"""

import threading
import concurrent.futures
import queue
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class Task:
    """Representa uma tarefa para execução."""
    id: str
    function: Callable
    args: tuple = ()
    kwargs: dict = None
    priority: int = 0
    timeout: Optional[float] = None
    retry_count: int = 0
    max_retries: int = 3
    status: TaskStatus = TaskStatus.PENDING
    result: Any = None
    error: Optional[Exception] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    async def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)

class ThreadManager:
    """Gerenciador robusto de threads."""
    
    async def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.logger = logging.getLogger("ThreadManager")
        
        # Thread pools
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="PeninOmega"
        )
        
        # Task management
        self.tasks: Dict[str, Task] = {}
        self.task_queue = queue.PriorityQueue()
        self.active_tasks: Dict[str, concurrent.futures.Future] = {}
        
        # Monitoring
        self.stats = {
            "tasks_submitted": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "tasks_cancelled": 0
        }
        
        # Control
        self._shutdown = False
        self._monitor_thread = None
        self._start_monitoring()
    
    async def _start_monitoring(self):
        """Inicia thread de monitoramento."""
        if self._monitor_thread is None or not self._monitor_thread.is_alive():
            self._monitor_thread = threading.Thread(
                target=self._monitor_tasks,
                daemon=True,
                name="PeninOmega-Monitor"
            )
            self._monitor_thread.start()
    
    async def _monitor_tasks(self):
        """Monitora tarefas em execução."""
        while not self._shutdown:
            try:
                # Verifica tarefas ativas
                completed_tasks = []
                
                for task_id, future in self.active_tasks.items():
                    if future.done():
                        completed_tasks.append(task_id)
                        self._handle_completed_task(task_id, future)
                
                # Remove tarefas completadas
                for task_id in completed_tasks:
                    del self.active_tasks[task_id]
                
                # Processa fila de tarefas
                self._process_task_queue()
                
                time.sleep(0.1)  # Intervalo de monitoramento
                
            except Exception as e:
                self.logger.error(f"Erro no monitoramento: {e}")
                time.sleep(1)
    
    async def _process_task_queue(self):
        """Processa fila de tarefas pendentes."""
        while not self.task_queue.empty() and len(self.active_tasks) < self.max_workers:
            try:
                priority, task_id = self.task_queue.get_nowait()
                task = self.tasks.get(task_id)
                
                if task and task.status == TaskStatus.PENDING:
                    self._execute_task(task)
                    
            except queue.Empty:
                break
            except Exception as e:
                self.logger.error(f"Erro ao processar fila: {e}")
    
    async def _execute_task(self, task: Task):
        """Executa uma tarefa."""
        try:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now(timezone.utc)
            
            # Submete para executor
            future = self.executor.submit(
                self._safe_task_execution,
                task
            )
            
            self.active_tasks[task.id] = future
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = e
            task.completed_at = datetime.now(timezone.utc)
            self.stats["tasks_failed"] += 1
            self.logger.error(f"Erro ao executar tarefa {task.id}: {e}")
    
    async def _safe_task_execution(self, task: Task) -> Any:
        """Execução segura de tarefa com timeout."""
        try:
            # Aplica timeout se especificado
            if task.timeout:
                import signal
                
                async def timeout_handler(signum, frame):
                    raise TimeoutError(f"Task {task.id} timed out after {task.timeout}s")
                
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(task.timeout))
            
            # Executa função
            result = task.function(*task.args, **task.kwargs)
            
            if task.timeout:
                signal.alarm(0)  # Cancela timeout
            
            return await result
            
        except Exception as e:
            if task.timeout:
                signal.alarm(0)
            raise e
    
    async def _handle_completed_task(self, task_id: str, future: concurrent.futures.Future):
        """Manipula tarefa completada."""
        task = self.tasks.get(task_id)
        if not task:
            return
        
        task.completed_at = datetime.now(timezone.utc)
        
        try:
            if future.cancelled():
                task.status = TaskStatus.CANCELLED
                self.stats["tasks_cancelled"] += 1
            elif future.exception():
                exception = future.exception()
                task.status = TaskStatus.FAILED
                task.error = exception
                self.stats["tasks_failed"] += 1
                
                # Retry logic
                if task.retry_count < task.max_retries:
                    task.retry_count += 1
                    task.status = TaskStatus.PENDING
                    task.started_at = None
                    task.completed_at = None
                    
                    # Re-enfileira para retry
                    self.task_queue.put((task.priority, task.id))
                    self.logger.warning(f"Retry {task.retry_count}/{task.max_retries} for task {task.id}")
                else:
                    self.logger.error(f"Task {task.id} failed permanently: {exception}")
            else:
                task.result = future.result()
                task.status = TaskStatus.COMPLETED
                self.stats["tasks_completed"] += 1
                
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = e
            self.stats["tasks_failed"] += 1
            self.logger.error(f"Erro ao processar resultado da tarefa {task.id}: {e}")
    
    async def submit_task(self, function: Callable, *args, task_id: Optional[str] = None,
                   priority: int = 0, timeout: Optional[float] = None,
                   max_retries: int = 3, **kwargs) -> str:
        """Submete tarefa para execução."""
        if task_id is None:
            task_id = f"task_{int(time.time() * 1000000)}"
        
        task = Task(
            id=task_id,
            function=function,
            args=args,
            kwargs=kwargs,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries
        )
        
        self.tasks[task_id] = task
        self.task_queue.put((priority, task_id))
        self.stats["tasks_submitted"] += 1
        
        return await task_id
    
    async def get_task_status(self, task_id: str) -> Optional[TaskStatus]:
        """Obtém status de uma tarefa."""
        task = self.tasks.get(task_id)
        return await task.status if task else None
    
    async def get_task_result(self, task_id: str, timeout: Optional[float] = None) -> Any:
        """Obtém resultado de uma tarefa (bloqueia até completar)."""
        task = self.tasks.get(task_id)
        if not task:
            raise ValueError(f"Task {task_id} not found")
        
        start_time = time.time()
        while task.status in [TaskStatus.PENDING, TaskStatus.RUNNING]:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"Timeout waiting for task {task_id}")
            time.sleep(0.01)
        
        if task.status == TaskStatus.COMPLETED:
            return await task.result
        elif task.status == TaskStatus.FAILED:
            raise task.error
        elif task.status == TaskStatus.CANCELLED:
            raise RuntimeError(f"Task {task_id} was cancelled")
        else:
            raise RuntimeError(f"Task {task_id} in unexpected state: {task.status}")
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancela uma tarefa."""
        # Cancela se estiver ativa
        if task_id in self.active_tasks:
            future = self.active_tasks[task_id]
            return await future.cancel()
        
        # Remove da fila se estiver pendente
        task = self.tasks.get(task_id)
        if task and task.status == TaskStatus.PENDING:
            task.status = TaskStatus.CANCELLED
            task.completed_at = datetime.now(timezone.utc)
            self.stats["tasks_cancelled"] += 1
            return await True
        
        return await False
    
    async def wait_for_all(self, timeout: Optional[float] = None) -> bool:
        """Aguarda todas as tarefas completarem."""
        start_time = time.time()
        
        while self.active_tasks or not self.task_queue.empty():
            if timeout and (time.time() - start_time) > timeout:
                return await False
            time.sleep(0.01)
        
        return await True
    
    async def get_stats(self) -> Dict[str, Any]:
        """Obtém estatísticas do gerenciador."""
        return await {
            **self.stats,
            "active_tasks": len(self.active_tasks),
            "pending_tasks": self.task_queue.qsize(),
            "total_tasks": len(self.tasks),
            "max_workers": self.max_workers
        }
    
    async def cleanup_completed_tasks(self, max_age_hours: int = 24):
        """Remove tarefas antigas completadas."""
        cutoff_time = datetime.now(timezone.utc).timestamp() - (max_age_hours * 3600)
        
        tasks_to_remove = []
        for task_id, task in self.tasks.items():
            if (task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED] and
                task.completed_at and task.completed_at.timestamp() < cutoff_time):
                tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del self.tasks[task_id]
        
        return await len(tasks_to_remove)
    
    async def shutdown(self, wait: bool = True, timeout: Optional[float] = None):
        """Encerra o gerenciador de threads."""
        self._shutdown = True
        
        if wait:
            self.wait_for_all(timeout)
        
        self.executor.shutdown(wait=wait, timeout=timeout)

# Instância global
thread_manager = ThreadManager()

# Funções de conveniência
async def submit_task(function: Callable, *args, **kwargs) -> str:
    """Função de conveniência para submeter tarefa."""
    return await thread_manager.submit_task(function, *args, **kwargs)

async def get_task_result(task_id: str, timeout: Optional[float] = None) -> Any:
    """Função de conveniência para obter resultado."""
    return await thread_manager.get_task_result(task_id, timeout)

async def get_thread_stats() -> Dict[str, Any]:
    """Função de conveniência para obter estatísticas."""
    return await thread_manager.get_stats()

if __name__ == "__main__":
    # Teste do gerenciador de threads
    import random
    
    async def test_function(duration: float, should_fail: bool = False):
        """Função de teste."""
        time.sleep(duration)
        if should_fail:
            raise ValueError("Test error")
        return await f"Completed after {duration}s"
    
    print("Testando gerenciador de threads...")
    
    # Submete várias tarefas
    task_ids = []
    for i in range(5):
        duration = random.uniform(0.1, 1.0)
        task_id = submit_task(test_function, duration, should_fail=(i == 2))
        task_ids.append(task_id)
        print(f"Submitted task {task_id}")
    
    # Aguarda resultados
    for task_id in task_ids:
        try:
            result = get_task_result(task_id, timeout=5.0)
            print(f"Task {task_id}: {result}")
        except Exception as e:
            print(f"Task {task_id} failed: {e}")
    
    # Estatísticas
    stats = get_thread_stats()
    print(f"Stats: {stats}")
    
    print("Teste concluído!")
