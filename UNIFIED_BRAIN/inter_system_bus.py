#!/usr/bin/env python3
"""
🔗 INTER-SYSTEM MESSAGE BUS
Permite comunicação assíncrona entre sistemas via SQLite
Usado para coordenar: UnifiedBrain ↔ V7 ↔ Darwin ↔ TEIS
"""
import sqlite3
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class MessageBus:
    """
    Sistema de mensagens simples via SQLite para comunicação inter-sistemas
    Thread-safe, persist, auditável
    """
    
    def __init__(self, db_path: str = '/root/intelligence_system/data/message_bus.db'):
        self.db_path = db_path
        self._init_db()
        logger.info(f"✅ MessageBus initialized: {db_path}")
    
    def _init_db(self):
        """Cria tabela de mensagens se não existir"""
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sender TEXT NOT NULL,
                    receiver TEXT NOT NULL,
                    message_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    timestamp REAL NOT NULL,
                    processed INTEGER DEFAULT 0,
                    priority INTEGER DEFAULT 0
                )
            """)
            
            # Index para queries rápidas
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_receiver_processed 
                ON messages(receiver, processed, timestamp)
            """)
            
            conn.commit()
        finally:
            conn.close()
    
    def send(self, sender: str, receiver: str, message_type: str, 
             payload: Dict[str, Any], priority: int = 0):
        """
        Envia mensagem de um sistema para outro
        
        Args:
            sender: ID do sistema remetente (ex: 'unified_brain')
            receiver: ID do sistema destinatário (ex: 'darwinacci')
            message_type: Tipo da mensagem (ex: 'metrics', 'command', 'genome')
            payload: Dados da mensagem (dict)
            priority: 0=normal, 1=high, 2=critical
        """
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                """INSERT INTO messages 
                   (sender, receiver, message_type, payload, timestamp, priority) 
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (sender, receiver, message_type, json.dumps(payload), time.time(), priority)
            )
            conn.commit()
            logger.debug(f"📤 {sender} → {receiver}: {message_type}")
        finally:
            conn.close()
    
    def receive(self, receiver: str, message_type: Optional[str] = None, 
                limit: int = 100) -> List[Dict[str, Any]]:
        """
        Recebe mensagens não processadas para um sistema
        
        Args:
            receiver: ID do sistema
            message_type: Filtrar por tipo (None = todos)
            limit: Máximo de mensagens a retornar
            
        Returns:
            Lista de mensagens [{id, sender, type, payload, timestamp}, ...]
        """
        conn = sqlite3.connect(self.db_path)
        try:
            if message_type:
                query = """
                    SELECT id, sender, message_type, payload, timestamp, priority 
                    FROM messages 
                    WHERE receiver=? AND message_type=? AND processed=0 
                    ORDER BY priority DESC, timestamp ASC
                    LIMIT ?
                """
                rows = conn.execute(query, (receiver, message_type, limit)).fetchall()
            else:
                query = """
                    SELECT id, sender, message_type, payload, timestamp, priority 
                    FROM messages 
                    WHERE receiver=? AND processed=0 
                    ORDER BY priority DESC, timestamp ASC
                    LIMIT ?
                """
                rows = conn.execute(query, (receiver, limit)).fetchall()
            
            messages = []
            for row in rows:
                msg_id, sender, mtype, payload, ts, priority = row
                messages.append({
                    'id': msg_id,
                    'sender': sender,
                    'type': mtype,
                    'payload': json.loads(payload),
                    'timestamp': ts,
                    'priority': priority,
                })
                
                # Marcar como processado
                conn.execute("UPDATE messages SET processed=1 WHERE id=?", (msg_id,))
            
            conn.commit()
            
            if messages:
                logger.debug(f"📥 {receiver} recebeu {len(messages)} mensagens")
            
            return messages
        finally:
            conn.close()
    
    def peek(self, receiver: str, message_type: Optional[str] = None) -> int:
        """
        Conta mensagens pendentes sem marcá-las como processadas
        
        Returns:
            Número de mensagens não processadas
        """
        conn = sqlite3.connect(self.db_path)
        try:
            if message_type:
                query = "SELECT COUNT(*) FROM messages WHERE receiver=? AND message_type=? AND processed=0"
                count = conn.execute(query, (receiver, message_type)).fetchone()[0]
            else:
                query = "SELECT COUNT(*) FROM messages WHERE receiver=? AND processed=0"
                count = conn.execute(query, (receiver,)).fetchone()[0]
            return count
        finally:
            conn.close()
    
    def broadcast(self, sender: str, message_type: str, payload: Dict[str, Any], 
                  receivers: List[str], priority: int = 0):
        """
        Envia mesma mensagem para múltiplos destinatários
        """
        for receiver in receivers:
            self.send(sender, receiver, message_type, payload, priority)
        
        logger.info(f"📡 Broadcast: {sender} → {len(receivers)} receivers ({message_type})")
    
    def clear_old_messages(self, days: int = 7):
        """
        Remove mensagens processadas mais antigas que N dias
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cutoff = time.time() - (days * 86400)
            deleted = conn.execute(
                "DELETE FROM messages WHERE processed=1 AND timestamp < ?",
                (cutoff,)
            ).rowcount
            conn.commit()
            
            if deleted > 0:
                logger.info(f"🗑️  Removidas {deleted} mensagens antigas (>{days} dias)")
            
            return deleted
        finally:
            conn.close()
    
    def get_stats(self) -> Dict[str, int]:
        """Retorna estatísticas do bus"""
        conn = sqlite3.connect(self.db_path)
        try:
            total = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
            pending = conn.execute("SELECT COUNT(*) FROM messages WHERE processed=0").fetchone()[0]
            processed = conn.execute("SELECT COUNT(*) FROM messages WHERE processed=1").fetchone()[0]
            
            return {
                'total': total,
                'pending': pending,
                'processed': processed,
            }
        finally:
            conn.close()


# Singleton global
_BUS: Optional[MessageBus] = None

def get_message_bus() -> MessageBus:
    """Retorna instância singleton do MessageBus"""
    global _BUS
    if _BUS is None:
        _BUS = MessageBus()
    return _BUS


# ═══════════════════════════════════════════════════════════════════════════
# EXEMPLO DE USO
# ═══════════════════════════════════════════════════════════════════════════

def example_unified_brain_sender():
    """Exemplo: UnifiedBrain envia métricas para outros sistemas"""
    bus = get_message_bus()
    
    # Enviar métricas para Darwinacci
    bus.send(
        sender='unified_brain',
        receiver='darwinacci',
        message_type='metrics',
        payload={
            'router_grad_norm': 0.377,
            'avg_reward': 45.0,
            'coherence': 0.998,
            'novelty': 0.042,
        }
    )
    
    # Broadcast para todos
    bus.broadcast(
        sender='unified_brain',
        message_type='status',
        payload={'active': True, 'episode': 100},
        receivers=['darwinacci', 'v7', 'teis']
    )

def example_darwinacci_receiver():
    """Exemplo: Darwinacci recebe e processa mensagens"""
    bus = get_message_bus()
    
    # Verificar se há mensagens
    pending_count = bus.peek('darwinacci')
    if pending_count > 0:
        logger.info(f"📬 Darwinacci tem {pending_count} mensagens pendentes")
    
    # Receber mensagens de métricas
    messages = bus.receive('darwinacci', message_type='metrics')
    
    for msg in messages:
        sender = msg['sender']
        payload = msg['payload']
        
        # Usar dados para ajustar evolução
        if 'router_grad_norm' in payload:
            grad_norm = payload['router_grad_norm']
            # Usar como sinal de fitness
            logger.info(f"🧬 Darwinacci recebeu grad_norm={grad_norm:.4f} de {sender}")
        
        if 'avg_reward' in payload:
            reward = payload['avg_reward']
            # Incorporar na função de fitness
            logger.info(f"🧬 Darwinacci recebeu reward={reward:.2f} de {sender}")


if __name__ == '__main__':
    # Teste do MessageBus
    logging.basicConfig(level=logging.INFO)
    
    bus = get_message_bus()
    
    print("🔗 Testando MessageBus...")
    
    # Enviar mensagens de teste
    bus.send('unified_brain', 'darwinacci', 'metrics', {'test': 1.0})
    bus.send('darwinacci', 'unified_brain', 'genome', {'lr': 0.001})
    bus.send('v7', 'unified_brain', 'command', {'action': 'increase_exploration'})
    
    # Receber
    msgs_darwinacci = bus.receive('darwinacci')
    msgs_ubrain = bus.receive('unified_brain')
    
    print(f"✅ Darwinacci recebeu: {len(msgs_darwinacci)} mensagens")
    print(f"✅ UnifiedBrain recebeu: {len(msgs_ubrain)} mensagens")
    
    # Stats
    stats = bus.get_stats()
    print(f"📊 Total: {stats['total']}, Pending: {stats['pending']}, Processed: {stats['processed']}")
    
    print("✅ MessageBus funcionando corretamente!")