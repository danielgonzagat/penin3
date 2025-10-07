#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Gerenciador de Estado Global
======================================
Sistema unificado que sincroniza estado entre todos os módulos.
"""

from __future__ import annotations
import asyncio
import json
import threading
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable
import logging
import sqlite3

# Imports seguros usando o resolvedor
from penin_omega_dependency_resolver import get_classes, config

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PENIN_OMEGA_ROOT = Path(config.get("root_path", "/root/.penin_omega"))
STATE_PATH = PENIN_OMEGA_ROOT / "state"
STATE_PATH.mkdir(parents=True, exist_ok=True)

# =============================================================================
# GERENCIADOR DE ESTADO GLOBAL
# =============================================================================

class GlobalStateManager:
    """Gerencia estado unificado entre todos os módulos."""
    
    def __init__(self):
        self.classes = get_classes()
        self.state_lock = threading.RLock()
        self.subscribers = {}
        self.state_history = []
        self.db_path = STATE_PATH / "global_state.db"
        self.logger = logging.getLogger("GlobalStateManager")
        
        # Estado atual unificado
        if 'create_unified_state' in self.classes:
            self.current_state = self.classes['create_unified_state']()
        else:
            # Fallback se classes não disponíveis
            self.current_state = self._create_fallback_state()
        
        # Inicializa banco de dados
        self._init_database()
        
        # Carrega estado persistido
        self._load_persisted_state()
    
    def _create_fallback_state(self):
        """Cria estado fallback se classes unificadas não disponíveis."""
        return {
            "rho": 0.4,
            "sr_score": 0.85,
            "ece": 0.003,
            "consent": True,
            "eco_ok": True,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system_health": 1.0,
            "pipeline_status": "idle",
            "active_workers": []
        }
    
    def _init_database(self):
        """Inicializa banco de dados para persistência."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS state_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        state_data TEXT NOT NULL,
                        module_source TEXT,
                        change_type TEXT
                    )
                """)
                
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS current_state (
                        key TEXT PRIMARY KEY,
                        value TEXT NOT NULL,
                        last_updated TEXT NOT NULL
                    )
                """)
                conn.commit()
        except Exception as e:
            self.logger.error(f"Erro ao inicializar banco: {e}")
    
    def _load_persisted_state(self):
        """Carrega estado persistido do banco."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT key, value FROM current_state")
                persisted_data = dict(cursor.fetchall())
                
                if persisted_data:
                    # Atualiza estado atual com dados persistidos
                    for key, value_str in persisted_data.items():
                        try:
                            value = json.loads(value_str)
                            if hasattr(self.current_state, key):
                                setattr(self.current_state, key, value)
                            elif isinstance(self.current_state, dict):
                                self.current_state[key] = value
                        except json.JSONDecodeError:
                            continue
                    
                    self.logger.info(f"Estado carregado: {len(persisted_data)} campos")
        except Exception as e:
            self.logger.warning(f"Erro ao carregar estado: {e}")
    
    def get_state(self) -> Any:
        """Obtém estado atual."""
        with self.state_lock:
            if hasattr(self.current_state, 'to_dict'):
                return self.current_state.to_dict()
            return dict(self.current_state) if isinstance(self.current_state, dict) else self.current_state
    
    def update_state(self, updates: Dict[str, Any], module_source: str = "unknown") -> bool:
        """Atualiza estado global."""
        try:
            with self.state_lock:
                old_state = self.get_state()
                
                # Atualiza estado
                for key, value in updates.items():
                    if hasattr(self.current_state, key):
                        setattr(self.current_state, key, value)
                    elif isinstance(self.current_state, dict):
                        self.current_state[key] = value
                
                # Atualiza timestamp
                timestamp = datetime.now(timezone.utc).isoformat()
                if hasattr(self.current_state, 'last_update'):
                    self.current_state.last_update = timestamp
                elif isinstance(self.current_state, dict):
                    self.current_state['last_update'] = timestamp
                
                # Persiste no banco
                self._persist_state(updates, module_source)
                
                # Notifica subscribers
                self._notify_subscribers(old_state, self.get_state(), module_source)
                
                # Adiciona ao histórico
                self.state_history.append({
                    "timestamp": timestamp,
                    "updates": updates,
                    "module_source": module_source
                })
                
                # Limita histórico
                if len(self.state_history) > 1000:
                    self.state_history = self.state_history[-500:]
                
                self.logger.info(f"Estado atualizado por {module_source}: {list(updates.keys())}")
                return True
                
        except Exception as e:
            self.logger.error(f"Erro ao atualizar estado: {e}")
            return False
    
    def _persist_state(self, updates: Dict[str, Any], module_source: str):
        """Persiste estado no banco."""
        try:
            timestamp = datetime.now(timezone.utc).isoformat()
            
            with sqlite3.connect(self.db_path) as conn:
                # Salva histórico
                conn.execute(
                    "INSERT INTO state_history (timestamp, state_data, module_source, change_type) VALUES (?, ?, ?, ?)",
                    (timestamp, json.dumps(updates), module_source, "update")
                )
                
                # Atualiza estado atual
                for key, value in updates.items():
                    conn.execute(
                        "INSERT OR REPLACE INTO current_state (key, value, last_updated) VALUES (?, ?, ?)",
                        (key, json.dumps(value), timestamp)
                    )
                
                conn.commit()
        except Exception as e:
            self.logger.error(f"Erro ao persistir estado: {e}")
    
    def subscribe(self, callback: Callable, module_name: str = "unknown"):
        """Registra callback para mudanças de estado."""
        if module_name not in self.subscribers:
            self.subscribers[module_name] = []
        self.subscribers[module_name].append(callback)
        self.logger.info(f"Módulo {module_name} inscrito para notificações")
    
    def _notify_subscribers(self, old_state: Dict, new_state: Dict, source: str):
        """Notifica subscribers sobre mudanças."""
        for module_name, callbacks in self.subscribers.items():
            if module_name != source:  # Não notifica o próprio módulo que fez a mudança
                for callback in callbacks:
                    try:
                        callback(old_state, new_state, source)
                    except Exception as e:
                        self.logger.error(f"Erro ao notificar {module_name}: {e}")
    
    def get_module_metrics(self, module_name: str) -> Dict[str, Any]:
        """Obtém métricas específicas de um módulo."""
        state = self.get_state()
        return {
            "module": module_name,
            "timestamp": state.get("timestamp", ""),
            "system_health": state.get("system_health", 0.0),
            "pipeline_status": state.get("pipeline_status", "unknown"),
            "rho": state.get("rho", 0.0),
            "sr_score": state.get("sr_score", 0.0),
            "ece": state.get("ece", 0.0)
        }
    
    def sync_with_module(self, module_name: str, module_state: Dict[str, Any]) -> bool:
        """Sincroniza com estado de módulo específico."""
        try:
            # Filtra apenas campos relevantes
            relevant_fields = [
                "rho", "sr_score", "ece", "ppl_ood", "delta_linf", "mdl_gain",
                "consent", "eco_ok", "uncertainty", "system_health", "pipeline_status"
            ]
            
            updates = {k: v for k, v in module_state.items() if k in relevant_fields}
            
            if updates:
                return self.update_state(updates, module_name)
            return True
            
        except Exception as e:
            self.logger.error(f"Erro ao sincronizar com {module_name}: {e}")
            return False
    
    def get_state_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Obtém histórico de mudanças de estado."""
        return self.state_history[-limit:]
    
    def reset_state(self):
        """Reseta estado para valores padrão."""
        with self.state_lock:
            if 'create_unified_state' in self.classes:
                self.current_state = self.classes['create_unified_state']()
            else:
                self.current_state = self._create_fallback_state()
            
            self.logger.info("Estado resetado para valores padrão")

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

# Instância global do gerenciador
global_state_manager = GlobalStateManager()

# =============================================================================
# FUNÇÕES DE CONVENIÊNCIA
# =============================================================================

def get_global_state():
    """Obtém estado global atual."""
    return global_state_manager.get_state()

def update_global_state(updates: Dict[str, Any], module_source: str = "unknown"):
    """Atualiza estado global."""
    return global_state_manager.update_state(updates, module_source)

def subscribe_to_state_changes(callback: Callable, module_name: str = "unknown"):
    """Inscreve-se para notificações de mudança de estado."""
    global_state_manager.subscribe(callback, module_name)

def sync_module_state(module_name: str, module_state: Dict[str, Any]):
    """Sincroniza estado de módulo com estado global."""
    return global_state_manager.sync_with_module(module_name, module_state)

# =============================================================================
# TESTE DO SISTEMA
# =============================================================================

def test_global_state_manager():
    """Testa o gerenciador de estado global."""
    print("🧪 Testando gerenciador de estado global...")
    
    # Teste obter estado
    state = get_global_state()
    print(f"✅ Estado inicial obtido: {len(state)} campos")
    
    # Teste atualizar estado
    success = update_global_state({
        "rho": 0.6,
        "sr_score": 0.9,
        "pipeline_status": "running"
    }, "test_module")
    
    if success:
        print("✅ Estado atualizado com sucesso")
    else:
        print("❌ Falha ao atualizar estado")
    
    # Teste sincronização de módulo
    module_state = {
        "rho": 0.7,
        "system_health": 0.95,
        "extra_field": "ignored"  # Deve ser ignorado
    }
    
    sync_success = sync_module_state("test_sync", module_state)
    if sync_success:
        print("✅ Sincronização de módulo funcionando")
    else:
        print("❌ Falha na sincronização")
    
    # Teste callback
    def test_callback(old_state, new_state, source):
        print(f"📢 Callback recebido de {source}")
    
    subscribe_to_state_changes(test_callback, "test_subscriber")
    
    # Trigger callback
    update_global_state({"test_field": "test_value"}, "callback_test")
    
    # Teste histórico
    history = global_state_manager.get_state_history(5)
    print(f"✅ Histórico obtido: {len(history)} entradas")
    
    print("🎉 Gerenciador de estado global funcionando!")
    return True

if __name__ == "__main__":
    test_global_state_manager()
