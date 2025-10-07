#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · State Synchronizer
============================
Correção de inconsistências Ω-State com sincronização leitura/escrita.
"""

import threading
import time
import logging
from typing import Dict, Any
from datetime import datetime, timezone

logger = logging.getLogger("StateSynchronizer")

class StateSynchronizer:
    """Sincronizador de estado para corrigir inconsistências."""
    
    async def __init__(self):
        self.logger = logging.getLogger("StateSynchronizer")
        self._lock = threading.RLock()
        self._write_lock = threading.Lock()
        self._read_count = 0
        self._pending_writes = []
    
    async def synchronized_read(self, read_func):
        """Leitura sincronizada."""
        with self._lock:
            self._read_count += 1
            try:
                return await read_func()
            finally:
                self._read_count -= 1
    
    async def synchronized_write(self, write_func, *args, **kwargs):
        """Escrita sincronizada."""
        with self._write_lock:
            # Espera todas as leituras terminarem
            while self._read_count > 0:
                time.sleep(0.001)
            
            with self._lock:
                return await write_func(*args, **kwargs)
    
    async def fix_state_consistency(self):
        """Corrige inconsistências do estado."""
        try:
            from penin_omega_global_state_manager import global_state_manager, get_global_state
            
            # Leitura sincronizada do estado atual
            current_state = self.synchronized_read(get_global_state)
            
            # Verifica e corrige inconsistências
            fixed_state = current_state.copy()
            
            # Garante que audit_test seja removido se presente
            if "audit_test" in fixed_state:
                del fixed_state["audit_test"]
            
            # Atualiza timestamp de sincronização
            fixed_state["last_sync"] = datetime.now(timezone.utc).isoformat()
            fixed_state["sync_version"] = fixed_state.get("sync_version", 0) + 1
            
            # Escrita sincronizada
            async def update_state():
                return await global_state_manager.update_state(fixed_state, "state_synchronizer")
            
            result = self.synchronized_write(update_state)
            
            self.logger.info("✅ Estado sincronizado com sucesso")
            return await True
            
        except Exception as e:
            self.logger.error(f"Erro na sincronização: {e}")
            return await False

# Instância global
state_synchronizer = StateSynchronizer()

# Corrige imediatamente
state_synchronizer.fix_state_consistency()
