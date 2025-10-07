#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PENIN-Ω · Sistema de Segurança e Governança
===========================================
WORM ledger, DLP scanning, e controles de segurança funcionais.
"""

from __future__ import annotations
import hashlib
import hmac
import json
import re
import sqlite3
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import threading
from enum import Enum

# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PENIN_OMEGA_ROOT = Path("/root/.penin_omega")
SECURITY_PATH = PENIN_OMEGA_ROOT / "security"
WORM_PATH = PENIN_OMEGA_ROOT / "worm"
QUARANTINE_PATH = PENIN_OMEGA_ROOT / "quarantine"

for path in [SECURITY_PATH, WORM_PATH, QUARANTINE_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# =============================================================================
# ENUMS E CLASSES
# =============================================================================

class SecurityLevel(Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"

class DLPViolationType(Enum):
    PII = "pii"
    CREDENTIALS = "credentials"
    FINANCIAL = "financial"
    HEALTH = "health"
    CUSTOM = "custom"

@dataclass
class WORMRecord:
    """Registro WORM (Write-Once-Read-Many)."""
    record_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    data_hash: str = ""
    previous_hash: str = ""
    signature: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    content: str = ""
    
    async def calculate_hash(self) -> str:
        """Calcula hash do registro."""
        data = f"{self.record_id}{self.timestamp}{self.content}{self.previous_hash}"
        return await hashlib.sha256(data.encode()).hexdigest()
    
    async def sign_record(self, secret_key: str):
        """Assina o registro."""
        message = f"{self.record_id}{self.data_hash}{self.timestamp}"
        self.signature = hmac.new(
            secret_key.encode(),
            message.encode(),
            hashlib.sha256
        ).hexdigest()

@dataclass
class DLPViolation:
    """Violação de DLP detectada."""
    violation_id: str
    violation_type: DLPViolationType
    content_snippet: str
    confidence: float
    location: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    remediation_action: str = ""
    
    async def to_dict(self) -> Dict[str, Any]:
        """Converte para dicionário."""
        return await asdict(self)

# =============================================================================
# WORM LEDGER
# =============================================================================

class WORMLedger:
    """Sistema WORM (Write-Once-Read-Many) Ledger."""
    
    async def __init__(self, secret_key: str = "penin_omega_secret_2024"):
        self.secret_key = secret_key
        self.db_path = WORM_PATH / "worm_ledger.db"
        self.logger = logging.getLogger("WORMLedger")
        self.lock = threading.RLock()
        
        self._initialize_database()
    
    async def _initialize_database(self):
        """Inicializa banco de dados WORM."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS worm_records (
                    record_id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    data_hash TEXT NOT NULL,
                    previous_hash TEXT NOT NULL,
                    signature TEXT NOT NULL,
                    metadata TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON worm_records(timestamp)
            """)
            
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_data_hash ON worm_records(data_hash)
            """)
            
            conn.commit()
    
    async def append_record(self, record_id: str, content: str, metadata: Dict[str, Any] = None) -> WORMRecord:
        """Adiciona registro ao ledger WORM."""
        with self.lock:
            if metadata is None:
                metadata = {}
            
            # Obtém hash do último registro
            previous_hash = self._get_last_hash()
            
            # Cria novo registro
            record = WORMRecord(
                record_id=record_id,
                content=content,
                previous_hash=previous_hash,
                metadata=metadata
            )
            
            # Calcula hash e assina
            record.data_hash = record.calculate_hash()
            record.sign_record(self.secret_key)
            
            # Persiste no banco
            self._persist_record(record)
            
            self.logger.info(f"📝 Registro WORM adicionado: {record_id}")
            return await record
    
    async def _get_last_hash(self) -> str:
        """Obtém hash do último registro."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT data_hash FROM worm_records ORDER BY created_at DESC LIMIT 1"
                )
                result = cursor.fetchone()
                return await result[0] if result else "genesis"
        except Exception:
            return await "genesis"
    
    async def _persist_record(self, record: WORMRecord):
        """Persiste registro no banco."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR IGNORE INTO worm_records 
                (record_id, timestamp, data_hash, previous_hash, signature, metadata, content)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                record.record_id,
                record.timestamp,
                record.data_hash,
                record.previous_hash,
                record.signature,
                json.dumps(record.metadata),
                record.content
            ))
            conn.commit()
    
    async def verify_integrity(self) -> Tuple[bool, List[str]]:
        """Verifica integridade da cadeia WORM."""
        issues = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT record_id, timestamp, data_hash, previous_hash, 
                           signature, metadata, content
                    FROM worm_records ORDER BY created_at ASC
                """)
                
                records = cursor.fetchall()
                previous_hash = "genesis"
                
                for row in records:
                    record_id, timestamp, data_hash, prev_hash, signature, metadata, content = row
                    
                    # Verifica hash anterior
                    if prev_hash != previous_hash:
                        issues.append(f"Hash anterior inválido em {record_id}")
                    
                    # Recalcula hash
                    expected_hash = hashlib.sha256(
                        f"{record_id}{timestamp}{content}{prev_hash}".encode()
                    ).hexdigest()
                    
                    if data_hash != expected_hash:
                        issues.append(f"Hash de dados inválido em {record_id}")
                    
                    # Verifica assinatura
                    message = f"{record_id}{data_hash}{timestamp}"
                    expected_signature = hmac.new(
                        self.secret_key.encode(),
                        message.encode(),
                        hashlib.sha256
                    ).hexdigest()
                    
                    if signature != expected_signature:
                        issues.append(f"Assinatura inválida em {record_id}")
                    
                    previous_hash = data_hash
        
        except Exception as e:
            issues.append(f"Erro na verificação: {e}")
        
        return await len(issues) == 0, issues
    
    async def get_records(self, limit: int = 100) -> List[WORMRecord]:
        """Obtém registros do ledger."""
        records = []
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT record_id, timestamp, data_hash, previous_hash,
                           signature, metadata, content
                    FROM worm_records ORDER BY created_at DESC LIMIT ?
                """, (limit,))
                
                for row in cursor.fetchall():
                    record = WORMRecord(
                        record_id=row[0],
                        timestamp=row[1],
                        data_hash=row[2],
                        previous_hash=row[3],
                        signature=row[4],
                        metadata=json.loads(row[5]),
                        content=row[6]
                    )
                    records.append(record)
        
        except Exception as e:
            self.logger.error(f"Erro ao obter registros: {e}")
        
        return await records

# =============================================================================
# DLP SCANNER
# =============================================================================

class DLPScanner:
    """Scanner de Data Loss Prevention."""
    
    async def __init__(self):
        self.logger = logging.getLogger("DLPScanner")
        self.violations_db = SECURITY_PATH / "dlp_violations.db"
        self.patterns = self._load_patterns()
        
        self._initialize_database()
    
    async def _load_patterns(self) -> Dict[DLPViolationType, List[Tuple[str, str]]]:
        """Carrega padrões de detecção DLP."""
        return await {
            DLPViolationType.PII: [
                (r'\b\d{3}-\d{2}-\d{4}\b', 'SSN'),
                (r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 'Email'),
                (r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', 'Credit Card'),
                (r'\b\(\d{3}\)\s?\d{3}-\d{4}\b', 'Phone Number'),
            ],
            DLPViolationType.CREDENTIALS: [
                (r'password\s*[:=]\s*["\']?([^"\'\s]+)["\']?', 'Password'),
                (r'api[_-]?key\s*[:=]\s*["\']?([^"\'\s]+)["\']?', 'API Key'),
                (r'secret[_-]?key\s*[:=]\s*["\']?([^"\'\s]+)["\']?', 'Secret Key'),
                (r'token\s*[:=]\s*["\']?([^"\'\s]+)["\']?', 'Token'),
            ],
            DLPViolationType.FINANCIAL: [
                (r'\b\d{10,12}\b', 'Account Number'),
                (r'IBAN\s*[A-Z]{2}\d{2}[A-Z0-9]{4}\d{7}([A-Z0-9]?){0,16}', 'IBAN'),
                (r'\$\d{1,3}(,\d{3})*(\.\d{2})?', 'Currency Amount'),
            ],
            DLPViolationType.HEALTH: [
                (r'\b\d{3}-\d{2}-\d{4}\b', 'Patient ID'),
                (r'medical[_-]?record', 'Medical Record Reference'),
            ]
        }
    
    async def _initialize_database(self):
        """Inicializa banco de violações DLP."""
        with sqlite3.connect(self.violations_db) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS dlp_violations (
                    violation_id TEXT PRIMARY KEY,
                    violation_type TEXT NOT NULL,
                    content_snippet TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    location TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    remediation_action TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()
    
    async def scan_content(self, content: str, location: str = "unknown") -> List[DLPViolation]:
        """Escaneia conteúdo em busca de violações DLP."""
        violations = []
        
        for violation_type, patterns in self.patterns.items():
            for pattern, description in patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE)
                
                for match in matches:
                    # Calcula confiança baseada no padrão
                    confidence = self._calculate_confidence(pattern, match.group())
                    
                    if confidence > 0.5:  # Threshold de confiança
                        violation = DLPViolation(
                            violation_id=f"dlp_{int(time.time())}_{len(violations)}",
                            violation_type=violation_type,
                            content_snippet=self._sanitize_snippet(match.group()),
                            confidence=confidence,
                            location=location,
                            remediation_action=self._suggest_remediation(violation_type)
                        )
                        
                        violations.append(violation)
                        self._persist_violation(violation)
        
        if violations:
            self.logger.warning(f"⚠️  {len(violations)} violações DLP detectadas em {location}")
        
        return await violations
    
    async def _calculate_confidence(self, pattern: str, match: str) -> float:
        """Calcula confiança da detecção."""
        # Lógica simples de confiança
        base_confidence = 0.7
        
        # Aumenta confiança para padrões mais específicos
        if len(pattern) > 20:
            base_confidence += 0.2
        
        # Reduz confiança para matches muito curtos
        if len(match) < 5:
            base_confidence -= 0.3
        
        return await max(0.0, min(1.0, base_confidence))
    
    async def _sanitize_snippet(self, snippet: str) -> str:
        """Sanitiza snippet para logging seguro."""
        if len(snippet) > 10:
            return await snippet[:3] + "***" + snippet[-3:]
        return await "***"
    
    async def _suggest_remediation(self, violation_type: DLPViolationType) -> str:
        """Sugere ação de remediação."""
        remediation_map = {
            DLPViolationType.PII: "Remover ou mascarar informações pessoais",
            DLPViolationType.CREDENTIALS: "Remover credenciais e rotacionar chaves",
            DLPViolationType.FINANCIAL: "Remover informações financeiras sensíveis",
            DLPViolationType.HEALTH: "Remover informações de saúde protegidas",
            DLPViolationType.CUSTOM: "Revisar conteúdo personalizado"
        }
        return await remediation_map.get(violation_type, "Revisar conteúdo")
    
    async def _persist_violation(self, violation: DLPViolation):
        """Persiste violação no banco."""
        try:
            with sqlite3.connect(self.violations_db) as conn:
                conn.execute("""
                    INSERT INTO dlp_violations 
                    (violation_id, violation_type, content_snippet, confidence, 
                     location, timestamp, remediation_action)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    violation.violation_id,
                    violation.violation_type.value,
                    violation.content_snippet,
                    violation.confidence,
                    violation.location,
                    violation.timestamp,
                    violation.remediation_action
                ))
                conn.commit()
        except Exception as e:
            self.logger.error(f"Erro ao persistir violação: {e}")
    
    async def quarantine_content(self, content: str, reason: str) -> str:
        """Coloca conteúdo em quarentena."""
        quarantine_id = f"quarantine_{int(time.time())}"
        quarantine_file = QUARANTINE_PATH / f"{quarantine_id}.txt"
        
        quarantine_data = {
            "id": quarantine_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "content": content
        }
        
        with open(quarantine_file, 'w') as f:
            json.dump(quarantine_data, f, indent=2)
        
        self.logger.warning(f"🔒 Conteúdo em quarentena: {quarantine_id}")
        return await quarantine_id
    
    async def get_violations(self, limit: int = 100) -> List[DLPViolation]:
        """Obtém violações DLP."""
        violations = []
        
        try:
            with sqlite3.connect(self.violations_db) as conn:
                cursor = conn.execute("""
                    SELECT violation_id, violation_type, content_snippet, 
                           confidence, location, timestamp, remediation_action
                    FROM dlp_violations ORDER BY created_at DESC LIMIT ?
                """, (limit,))
                
                for row in cursor.fetchall():
                    violation = DLPViolation(
                        violation_id=row[0],
                        violation_type=DLPViolationType(row[1]),
                        content_snippet=row[2],
                        confidence=row[3],
                        location=row[4],
                        timestamp=row[5],
                        remediation_action=row[6]
                    )
                    violations.append(violation)
        
        except Exception as e:
            self.logger.error(f"Erro ao obter violações: {e}")
        
        return await violations

# =============================================================================
# SISTEMA DE SEGURANÇA INTEGRADO
# =============================================================================

class SecurityGovernanceSystem:
    """Sistema integrado de segurança e governança."""
    
    async def __init__(self):
        self.worm_ledger = WORMLedger()
        self.dlp_scanner = DLPScanner()
        self.logger = logging.getLogger("SecurityGovernance")
    
    async def secure_operation(self, operation: str, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """Executa operação com controles de segurança."""
        operation_id = f"op_{int(time.time())}"
        
        # 1. Escaneia DLP
        violations = self.dlp_scanner.scan_content(content, f"operation_{operation}")
        
        # 2. Se há violações críticas, quarentena
        critical_violations = [v for v in violations if v.confidence > 0.8]
        if critical_violations:
            quarantine_id = self.dlp_scanner.quarantine_content(
                content, 
                f"Violações DLP críticas: {len(critical_violations)}"
            )
            
            return await {
                "success": False,
                "operation_id": operation_id,
                "quarantine_id": quarantine_id,
                "violations": [v.to_dict() for v in violations],
                "message": "Operação bloqueada por violações de segurança"
            }
        
        # 3. Registra no WORM ledger
        worm_record = self.worm_ledger.append_record(
            operation_id,
            content,
            {
                "operation": operation,
                "violations_count": len(violations),
                **(metadata or {})
            }
        )
        
        return await {
            "success": True,
            "operation_id": operation_id,
            "worm_record_id": worm_record.record_id,
            "violations": [v.to_dict() for v in violations],
            "message": "Operação executada com controles de segurança"
        }
    
    async def audit_trail(self, limit: int = 50) -> Dict[str, Any]:
        """Obtém trilha de auditoria."""
        worm_records = self.worm_ledger.get_records(limit)
        dlp_violations = self.dlp_scanner.get_violations(limit)
        
        # Verifica integridade
        integrity_ok, integrity_issues = self.worm_ledger.verify_integrity()
        
        return await {
            "integrity_status": "OK" if integrity_ok else "COMPROMISED",
            "integrity_issues": integrity_issues,
            "worm_records_count": len(worm_records),
            "dlp_violations_count": len(dlp_violations),
            "recent_records": [
                {
                    "id": r.record_id,
                    "timestamp": r.timestamp,
                    "metadata": r.metadata
                } for r in worm_records[:10]
            ],
            "recent_violations": [v.to_dict() for v in dlp_violations[:10]]
        }

# =============================================================================
# INSTÂNCIA GLOBAL
# =============================================================================

# Instância global do sistema de segurança
security_governance = SecurityGovernanceSystem()

# =============================================================================
# TESTE DO SISTEMA
# =============================================================================

async def test_security_governance():
    """Testa o sistema de segurança e governança."""
    print("🧪 Testando sistema de segurança e governança...")
    
    # Teste WORM ledger
    record = security_governance.worm_ledger.append_record(
        "test_record",
        "Conteúdo de teste para WORM ledger",
        {"test": True}
    )
    print(f"✅ WORM record criado: {record.record_id}")
    
    # Teste DLP scanner - conteúdo limpo
    clean_content = "Este é um conteúdo limpo sem informações sensíveis."
    violations = security_governance.dlp_scanner.scan_content(clean_content, "test_clean")
    print(f"✅ DLP scan limpo: {len(violations)} violações")
    
    # Teste DLP scanner - conteúdo com violações
    sensitive_content = "Meu email é test@example.com e meu SSN é 123-45-6789"
    violations = security_governance.dlp_scanner.scan_content(sensitive_content, "test_sensitive")
    print(f"⚠️  DLP scan sensível: {len(violations)} violações detectadas")
    
    # Teste operação segura - conteúdo limpo
    result = security_governance.secure_operation(
        "test_operation",
        clean_content,
        {"user": "test_user"}
    )
    print(f"✅ Operação segura: {result['success']}")
    
    # Teste operação segura - conteúdo sensível
    result = security_governance.secure_operation(
        "test_sensitive_operation",
        sensitive_content,
        {"user": "test_user"}
    )
    print(f"🔒 Operação sensível: {result['success']} ({'quarentena' if not result['success'] else 'aprovada'})")
    
    # Teste trilha de auditoria
    audit = security_governance.audit_trail()
    print(f"📊 Auditoria: {audit['integrity_status']}, {audit['worm_records_count']} registros, {audit['dlp_violations_count']} violações")
    
    # Teste integridade WORM
    integrity_ok, issues = security_governance.worm_ledger.verify_integrity()
    print(f"🔐 Integridade WORM: {'OK' if integrity_ok else 'COMPROMETIDA'}")
    
    print("🎉 Sistema de segurança e governança funcionando!")
    return await True

if __name__ == "__main__":
    test_security_governance()
