"""
WORM Ledger: Write-Once Read-Many Immutable Event Log

Implements a hash-chain based immutable ledger for auditability and
reproducibility. Each event is cryptographically linked to the previous one,
creating a tamper-evident history.
"""

import json
import hashlib
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class LedgerEntry:
    """
    A single entry in the WORM ledger.
    
    Attributes:
        index: Sequential index of this entry.
        timestamp: Unix timestamp when entry was created.
        event_type: Type of event (e.g., "generation", "elite_added", "rollback").
        data: Event-specific data.
        prev_hash: Hash of the previous entry.
        hash: Hash of this entry (computed from all above).
    """
    index: int
    timestamp: float
    event_type: str
    data: Dict[str, Any]
    prev_hash: str
    hash: str = ""
    
    def compute_hash(self) -> str:
        """
        Compute SHA-256 hash of this entry.
        
        The hash includes: index, timestamp, event_type, data, prev_hash.
        """
        content = {
            "index": self.index,
            "timestamp": self.timestamp,
            "event_type": self.event_type,
            "data": self.data,
            "prev_hash": self.prev_hash,
        }
        content_str = json.dumps(content, sort_keys=True)
        return hashlib.sha256(content_str.encode()).hexdigest()
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'LedgerEntry':
        """Create from dictionary."""
        return cls(**data)


class WormLedger:
    """
    Write-Once Read-Many immutable ledger using hash chains.
    
    Each event is appended to the ledger and cryptographically linked
    to the previous event. This creates an immutable, auditable history
    that cannot be modified without detection.
    
    Example:
        >>> ledger = WormLedger()
        >>> ledger.append("generation", {"gen": 1, "best_fitness": 0.5})
        >>> ledger.append("elite_added", {"fitness": 0.6, "niche": (5, 7)})
        >>> ledger.verify()
        True
    """
    
    def __init__(self):
        self.chain: List[LedgerEntry] = []
        self._create_genesis_block()
    
    def _create_genesis_block(self):
        """Create the first (genesis) block in the chain."""
        genesis = LedgerEntry(
            index=0,
            timestamp=time.time(),
            event_type="genesis",
            data={"message": "Fibonacci Engine WORM Ledger initialized"},
            prev_hash="0" * 64,  # No previous block
        )
        genesis.hash = genesis.compute_hash()
        self.chain.append(genesis)
    
    def append(self, event_type: str, data: Dict[str, Any]) -> LedgerEntry:
        """
        Append a new event to the ledger.
        
        Args:
            event_type: Type of event (e.g., "generation", "elite_added").
            data: Event-specific data (must be JSON-serializable).
            
        Returns:
            The created ledger entry.
            
        Raises:
            ValueError: If data is not JSON-serializable.
        """
        # Validate that data is JSON-serializable
        try:
            json.dumps(data)
        except (TypeError, ValueError) as e:
            raise ValueError(f"Data must be JSON-serializable: {e}")
        
        prev_entry = self.chain[-1]
        
        entry = LedgerEntry(
            index=len(self.chain),
            timestamp=time.time(),
            event_type=event_type,
            data=data,
            prev_hash=prev_entry.hash,
        )
        entry.hash = entry.compute_hash()
        
        self.chain.append(entry)
        return entry
    
    def verify(self) -> bool:
        """
        Verify the integrity of the entire chain.
        
        Returns:
            True if the chain is valid, False if tampering is detected.
        """
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            # Check that current entry's hash is correct
            if current.hash != current.compute_hash():
                return False
            
            # Check that current entry links to previous
            if current.prev_hash != previous.hash:
                return False
        
        return True
    
    def get_entries(
        self,
        event_type: Optional[str] = None,
        start_index: int = 0,
        end_index: Optional[int] = None
    ) -> List[LedgerEntry]:
        """
        Query ledger entries.
        
        Args:
            event_type: Filter by event type (optional).
            start_index: Start index (inclusive).
            end_index: End index (exclusive). None means all.
            
        Returns:
            List of matching ledger entries.
        """
        if end_index is None:
            end_index = len(self.chain)
        
        entries = self.chain[start_index:end_index]
        
        if event_type:
            entries = [e for e in entries if e.event_type == event_type]
        
        return entries
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get ledger statistics.
        
        Returns:
            Dictionary with statistics.
        """
        event_types = {}
        for entry in self.chain[1:]:  # Skip genesis
            event_types[entry.event_type] = event_types.get(entry.event_type, 0) + 1
        
        return {
            "total_entries": len(self.chain),
            "event_types": event_types,
            "first_timestamp": self.chain[0].timestamp if self.chain else None,
            "last_timestamp": self.chain[-1].timestamp if self.chain else None,
            "is_valid": self.verify(),
        }
    
    def to_dict(self) -> Dict:
        """Serialize ledger to dictionary."""
        return {
            "chain": [entry.to_dict() for entry in self.chain],
            "statistics": self.get_statistics(),
        }
    
    def save(self, filepath: str):
        """Save ledger to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'WormLedger':
        """Load ledger from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        ledger = cls.__new__(cls)  # Create without __init__
        ledger.chain = [LedgerEntry.from_dict(e) for e in data['chain']]
        
        # Verify integrity
        if not ledger.verify():
            raise ValueError("Loaded ledger failed integrity check!")
        
        return ledger
    
    def __len__(self) -> int:
        """Return number of entries in the ledger."""
        return len(self.chain)
    
    def __repr__(self) -> str:
        return f"WormLedger(entries={len(self.chain)}, valid={self.verify()})"
