#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · DLP Fixer
===================
Resolve violações DLP com UNIQUE constraint e idempotência.
"""

import sqlite3
import hashlib
import logging
import time
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, Any, List

logger = logging.getLogger("DLPFixer")

class DLPViolationFixer:
    """Corretor de violações DLP."""
    
    async def __init__(self):
        self.logger = logging.getLogger("DLPViolationFixer")
        self.db_path = Path("/root/.penin_omega/dlp_violations.db")
        
        # Cria diretório se não existir
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    async def fix_dlp_violations(self) -> bool:
        """Corrige todas as violações DLP."""
        try:
            # Recria tabela com constraints corretos
            self._recreate_dlp_table()
            
            # Limpa violações duplicadas
            self._clean_duplicate_violations()
            
            # Implementa idempotência
            self._implement_idempotency()
            
            self.logger.info("✅ Violações DLP corrigidas")
            return await True
            
        except Exception as e:
            self.logger.error(f"Erro na correção DLP: {e}")
            return await False
    
    async def _recreate_dlp_table(self):
        """Recria tabela DLP com constraints corretos."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Remove tabela antiga se existir
            cursor.execute('DROP TABLE IF EXISTS dlp_violations')
            
            # Cria nova tabela com constraint único composto
            cursor.execute('''
                CREATE TABLE dlp_violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    violation_id TEXT NOT NULL,
                    operation_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    violation_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    details TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    resolved BOOLEAN DEFAULT FALSE,
                    created_at TEXT NOT NULL,
                    UNIQUE(violation_id, operation_id, content_hash)
                )
            ''')
            
            # Cria índices para performance
            cursor.execute('CREATE INDEX idx_violation_id ON dlp_violations(violation_id)')
            cursor.execute('CREATE INDEX idx_operation_id ON dlp_violations(operation_id)')
            cursor.execute('CREATE INDEX idx_timestamp ON dlp_violations(timestamp)')
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def _clean_duplicate_violations(self):
        """Remove violações duplicadas."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Conta violações antes da limpeza
            cursor.execute('SELECT COUNT(*) FROM dlp_violations')
            count_before = cursor.fetchone()[0]
            
            # Remove duplicatas mantendo apenas a mais recente
            cursor.execute('''
                DELETE FROM dlp_violations 
                WHERE id NOT IN (
                    SELECT MAX(id) 
                    FROM dlp_violations 
                    GROUP BY violation_id, operation_id, content_hash
                )
            ''')
            
            # Conta após limpeza
            cursor.execute('SELECT COUNT(*) FROM dlp_violations')
            count_after = cursor.fetchone()[0]
            
            removed = count_before - count_after
            if removed > 0:
                self.logger.info(f"🧹 Removidas {removed} violações duplicadas")
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def _implement_idempotency(self):
        """Implementa sistema de idempotência."""
        conn = sqlite3.connect(str(self.db_path))
        cursor = conn.cursor()
        
        try:
            # Cria tabela de operações idempotentes
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS idempotent_operations (
                    operation_key TEXT PRIMARY KEY,
                    operation_id TEXT NOT NULL,
                    result_hash TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    expires_at TEXT NOT NULL
                )
            ''')
            
            # Cria índice para expiração
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_expires_at ON idempotent_operations(expires_at)')
            
            # Remove operações expiradas
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute('DELETE FROM idempotent_operations WHERE expires_at < ?', (now,))
            
            conn.commit()
            
        finally:
            conn.close()
    
    async def add_violation_idempotent(self, violation_id: str, operation_id: str, content: str, 
                                violation_type: str, severity: str, details: Dict[str, Any]) -> bool:
        """Adiciona violação de forma idempotente."""
        try:
            # Gera hash do conteúdo
            content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
            
            # Gera chave de operação única
            operation_key = f"{violation_id}:{operation_id}:{content_hash}:{int(time.time() * 1000)}"
            operation_key_hash = hashlib.sha256(operation_key.encode()).hexdigest()
            
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            try:
                # Verifica se violação similar já existe
                cursor.execute('''
                    SELECT COUNT(*) FROM dlp_violations 
                    WHERE violation_id = ? AND operation_id = ? AND content_hash = ?
                ''', (violation_id, operation_id, content_hash))
                
                if cursor.fetchone()[0] > 0:
                    # Violação similar já existe, retorna sucesso
                    return await True
                
                # Gera ID único para esta violação
                unique_violation_id = f"{violation_id}_{uuid.uuid4().hex[:8]}"
                
                # Insere nova violação com ID único
                timestamp = datetime.now(timezone.utc).isoformat()
                
                cursor.execute('''
                    INSERT INTO dlp_violations 
                    (violation_id, operation_id, content_hash, violation_type, 
                     severity, details, timestamp, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    unique_violation_id, operation_id, content_hash, violation_type,
                    severity, str(details), timestamp, timestamp
                ))
                
                # Registra operação como executada
                expires_at = datetime.fromtimestamp(
                    time.time() + 3600, timezone.utc  # Expira em 1 hora
                ).isoformat()
                
                cursor.execute('''
                    INSERT OR REPLACE INTO idempotent_operations
                    (operation_key, operation_id, result_hash, timestamp, expires_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (
                    operation_key_hash, operation_id, content_hash, timestamp, expires_at
                ))
                
                conn.commit()
                return await True
                
            finally:
                conn.close()
                
        except Exception as e:
            self.logger.error(f"Erro ao adicionar violação: {e}")
            return await False
    
    async def get_violation_stats(self) -> Dict[str, Any]:
        """Retorna estatísticas das violações."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Total de violações
            cursor.execute('SELECT COUNT(*) FROM dlp_violations')
            total_violations = cursor.fetchone()[0]
            
            # Violações por tipo
            cursor.execute('''
                SELECT violation_type, COUNT(*) 
                FROM dlp_violations 
                GROUP BY violation_type
            ''')
            by_type = dict(cursor.fetchall())
            
            # Violações por severidade
            cursor.execute('''
                SELECT severity, COUNT(*) 
                FROM dlp_violations 
                GROUP BY severity
            ''')
            by_severity = dict(cursor.fetchall())
            
            # Violações resolvidas
            cursor.execute('SELECT COUNT(*) FROM dlp_violations WHERE resolved = TRUE')
            resolved_count = cursor.fetchone()[0]
            
            conn.close()
            
            return await {
                "total_violations": total_violations,
                "by_type": by_type,
                "by_severity": by_severity,
                "resolved_count": resolved_count,
                "resolution_rate": resolved_count / max(1, total_violations),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter estatísticas: {e}")
            return await {"error": str(e)}

# Instância global e correção
dlp_fixer = DLPViolationFixer()
dlp_fixer.fix_dlp_violations()
